"""静的非線形解析（プッシュオーバー）: 材端ばねモデル・事象追跡型の荷重増分法

1. 長期荷重（G+P）を荷重係数 0→1 で増分載荷
2. 水平外力（Ai 分布、増分中は分布形一定）を比例的に増加し、層間変形角の上限で終了

各ステップでは、いずれかのばねが次の折れ点（ひび割れ・降伏・除荷・反転）に達するまで
荷重を進める。ばねは折れ線なので各ステップは線形で厳密（不釣合い力なし）。
ばねの向き（載荷・除荷）は、仮定した向きで解いたモーメント増分の符号と一致するまで繰り返す。
P-Δ 効果、せん断破壊、2方向曲げの相互作用は考慮しない。
"""
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg as spla

from .hinge import Hinge
from .results import Result
from .skeleton import (ES, Skeleton, beam_yield_moment, column_yield_moment, crack_moment,
                       member_end_stiffness, sugano_alpha_y)

PLANE_NAME = ("X", "Y")   # 柱の面 0（xy 面, Mz）= X方向、面 1（xz 面, My）= Y方向


def spring_moment(face, plane, end):
    """可撓部端の材端力（局所, 12）からばねのモーメント（梁の面 1 は下端引張が正）"""
    if plane == 1:
        return face[4] if end == 0 else -face[10]
    return face[5] if end == 0 else -face[11]


@dataclass
class Level:
    name: str
    lower: object     # 下の床の節点（None = 基礎）
    upper: int
    h: float


@dataclass
class PushoverResult:
    direction: str
    steps: list = field(default_factory=list)     # dict: phase, lam, Q, drifts, events
    events: list = field(default_factory=list)    # dict: step, key, label, event, side, lam, Q, drift
    hinges: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    total_P: float = 0.0                          # 荷重係数 1 の水平外力の合計（= 1層の設計用層せん断力）
    total_W: float = 0.0
    levels: list = field(default_factory=list)
    final: Result = None                          # 最終状態の変位・部材端力
    gravity_disp: dict = None                     # 長期荷重載荷後の節点変位
    warnings: list = field(default_factory=list)

    @property
    def lateral(self):
        return [s for s in self.steps if s["phase"] == "水平"]

    def print_curve(self, every=10):
        """水平載荷の荷重–変形関係（事象が起きたステップと every ステップごと）"""
        print(f"\n■ プッシュオーバー {self.direction}  ΣP(λ=1) = {self.total_P / 1e3:.1f} kN"
              f"  ΣW = {self.total_W / 1e3:.1f} kN")
        names = [lv.name for lv in self.levels]
        print(f"{'step':>5} {'λ':>7} {'Q[kN]':>9} {'C':>6} " +
              " ".join(f"{n + ' R':>12}" for n in names) + "  事象")
        rows = self.lateral
        for k, s in enumerate(rows):
            if s["events"] or k % every == 0 or k == len(rows) - 1:
                ev = ", ".join(f"{e['label']} {e['event']}" for e in s["events"][:3])
                if len(s["events"]) > 3:
                    ev += f" ほか{len(s['events']) - 3}"
                print(f"{s['step']:>5} {s['lam']:7.3f} {s['Q'] / 1e3:9.1f} "
                      f"{s['Q'] / self.total_W if self.total_W else 0:6.3f} " +
                      " ".join(f"{'1/' + format(1 / abs(r), ',.0f') if r else '0':>12}"
                               for r in s["drifts"]) + f"  {ev}")

    def print_hinges(self, all_springs=False):
        """ばねごとの最終状態と、ひび割れ・降伏したステップ（既定は弾性のままのばねを省略）"""
        first = {}
        for e in self.events:
            first.setdefault((e["key"], e["event"]), e)
        print(f"\n■ 材端ばねの状態（{self.direction} 最終）")
        print(f"{'部材端':<24} {'状態':<6} {'ひび割れ(step, Q[kN])':>22} {'降伏(step, Q[kN], 層間変形角)':>30}")
        for key, h in self.hinges.items():
            if not all_springs and h.state == "弾性":
                continue
            c, y = first.get((key, "ひび割れ")), first.get((key, "降伏"))
            cs = f"{c['step']:>5}, {c['Q'] / 1e3:8.1f}" if c else "-"
            ys = (f"{y['step']:>5}, {y['Q'] / 1e3:8.1f}, 1/{1 / y['drift']:,.0f}"
                  if y and y["drift"] else (f"{y['step']:>5}, {y['Q'] / 1e3:8.1f}" if y else "-"))
            print(f"{self.labels[key]:<24} {h.state:<6} {cs:>22} {ys:>30}")
        for w in self.warnings:
            print(f"  [注意] {w}")


def pushover(model, hinges, gravity_cases, pattern, levels, d, max_drift, step_drift,
             labels=None, max_steps=20000):
    """事象追跡型の荷重増分解析（入力形式に依存しない本体）

    hinges        : {(eid, 面, 端): Hinge}（端 0 = i端, 1 = j端）
    gravity_cases : 長期荷重のケース名のリスト（節点荷重・部材荷重）
    pattern       : 水平外力 [(節点, F6)]（荷重係数 1 の値）
    levels        : [Level]（層間変形角の監視、下から）
    d             : 水平載荷の方向（0 = X, 1 = Y）
    戻り値: PushoverResult（final 以外。final は呼び出し側で設定）
    """
    labels = labels or {k: str(k) for k in hinges}
    idx, n = model._index()
    C, _ = model._constraint_matrix(idx, n)
    E = model.elements
    by_elem = defaultdict(list)
    for key in hinges:
        by_elem[key[0]].append(key)
    sigma = {key: 1 for key in hinges}
    u = np.zeros(n)
    face = {eid: np.zeros(12) for eid in E}
    node = {eid: np.zeros(12) for eid in E}
    res = PushoverResult("", hinges=hinges, labels=labels, levels=levels)
    lat_total = sum(F[d] for _, F in pattern)

    def solve(phase, tang):
        for eid, keys in by_elem.items():
            fs = np.zeros((2, 2))
            for key in keys:
                fs[key[1], key[2]] = tang[key][0]
            E[eid].springs = fs
        K, _, _ = model.assemble_K()
        F = np.zeros(n)
        eloads = defaultdict(lambda: (np.zeros(12), np.zeros(12)))
        if phase == "長期":
            for case in gravity_cases:
                for nid, f in model.nodal_loads.get(case, []):
                    F[6 * idx[nid]:6 * idx[nid] + 6] += f
                for eid, (pf, pr) in model._element_loads(case).items():
                    e = E[eid]
                    F[6 * idx[e.ni]:6 * idx[e.ni] + 6] += (e.T.T @ (e.H.T @ pf + pr))[:6]
                    F[6 * idx[e.nj]:6 * idx[e.nj] + 6] += (e.T.T @ (e.H.T @ pf + pr))[6:]
                    a, b = eloads[eid]
                    eloads[eid] = (a + pf, b + pr)
        else:
            for nid, f in pattern:
                F[6 * idx[nid]:6 * idx[nid] + 6] += f
        ur = spla.spsolve((C.T @ K @ C).tocsc(), C.T @ F)
        if not np.all(np.isfinite(ur)):
            raise RuntimeError("剛性マトリクスが特異です（崩壊機構の形成または拘束不足）")
        du = C @ ur
        dface, dnode = {}, {}
        for eid, e in E.items():
            dg = np.r_[du[6 * idx[e.ni]:6 * idx[e.ni] + 6], du[6 * idx[e.nj]:6 * idx[e.nj] + 6]]
            pf, pr = eloads[eid] if eid in eloads else (np.zeros(12), np.zeros(12))
            dface[eid] = e.k_flex @ (e.H @ (e.T @ dg)) - pf
            dnode[eid] = e.H.T @ dface[eid] - pr
        return du, dface, dnode

    def drifts(vec):
        out = []
        for lv in levels:
            lo = vec[6 * idx[lv.lower] + d] if lv.lower is not None else 0.0
            out.append((vec[6 * idx[lv.upper] + d] - lo) / lv.h)
        return out

    lam = {"長期": 0.0, "水平": 0.0}
    step = 0
    zero_steps = 0
    for phase in ("長期", "水平"):
        while True:
            if step >= max_steps:
                raise RuntimeError(f"ステップ数が上限 {max_steps} に達しました")
            # ---- ばねの向きの仮定と解の整合（最大 20 回）
            for _ in range(20):
                tang = {key: hinges[key].tangent(sigma[key]) for key in hinges}
                du, dface, dnode = solve(phase, tang)
                dm = {key: spring_moment(dface[key[0]], key[1], key[2]) for key in hinges}
                scale = max((abs(v) for v in dm.values()), default=0.0)
                changed = False
                for key, v in dm.items():
                    if abs(v) > 1e-9 * scale:
                        s = 1 if v > 0 else -1
                        if s != sigma[key]:
                            sigma[key] = s
                            changed = True
                if not changed:
                    break
            # ---- 次の事象までの荷重係数増分
            dl = math.inf
            cand = {}
            for key, v in dm.items():
                flex, lim, kind = tang[key]
                if abs(v) <= 1e-9 * scale or math.isinf(lim):
                    continue
                t = max(0.0, (lim - hinges[key].M) / v)
                cand[key] = t
                dl = min(dl, t)
            rate = drifts(du)
            cur = drifts(u)
            if phase == "長期":
                dl = min(dl, 1.0 - lam[phase])
            else:
                rmax = max(abs(r) for r in rate)
                if rmax > 0:
                    dl = min(dl, step_drift / rmax)
                    for c, r in zip(cur, rate):          # |層間変形角| が上限に達するまで
                        if r != 0 and r * c >= 0:
                            dl = min(dl, max(0.0, max_drift - abs(c)) / abs(r))
            if math.isinf(dl):
                raise RuntimeError("荷重増分が定まりません（外力 0 または全ばねが無限遠）")
            zero_steps = zero_steps + 1 if dl == 0 else 0
            if zero_steps > 1000:
                raise RuntimeError("荷重増分 0 のステップが続いています（事象の判定が収束しません）")
            # ---- 状態の更新
            u += dl * du
            for eid in E:
                face[eid] += dl * dface[eid]
                node[eid] += dl * dnode[eid]
            lam[phase] += dl
            step += 1
            Q = lam["水平"] * abs(lat_total)
            dr = drifts(u)
            evs = []
            for key, h in hinges.items():
                flex, lim, kind = tang[key]
                hit = key in cand and cand[key] <= dl * (1 + 1e-9) + 1e-300
                for ev in h.advance(dl * dm[key], flex, sigma[key], hit, lim, kind):
                    side = sigma[key]
                    rec = dict(step=step, key=key, label=labels[key], event=ev, side=side,
                               lam=lam[phase], Q=Q, drift=max((abs(x) for x in dr), default=0.0),
                               phase=phase)
                    evs.append(rec)
                    res.events.append(rec)
            res.steps.append(dict(step=step, phase=phase, lam=lam[phase], Q=Q, drifts=dr,
                                  events=evs))
            if phase == "長期" and lam[phase] >= 1.0 - 1e-12:
                break
            if phase == "水平" and max(abs(x) for x in dr) >= max_drift * (1 - 1e-9):
                break
        if phase == "長期":
            res.gravity_disp = {nid: u[6 * k:6 * k + 6].copy() for nid, k in idx.items()}

    res.total_P = lat_total
    res._state = (u, face, node, idx)
    return res


# ================================================================ Building から実行
def _parse_direction(direction):
    s = str(direction).strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    axis = s.lstrip("+-")
    if axis not in ("x", "y"):
        raise ValueError(f"載荷方向は +x, -x, +y, -y のいずれかです: {direction!r}")
    return ("+" if sign > 0 else "-") + axis, sign, {"x": 0, "y": 1}[axis], f"E{axis}"


def build_hinges(bld, rL, opts):
    """配筋のある部材に材端ばねを設定する（柱の軸力は長期荷重時の値で固定）"""
    m = bld.model
    sy_f, r = opts["sigma_y_factor"], opts["post_yield_ratio"]
    hinges, labels, warns = {}, {}, []
    for key, eid in sorted(bld.elem_ids.items(), key=lambda kv: kv[1]):
        e = m.elements[eid]
        rb = e.sec.rebar
        label = bld.label("elem", eid)
        if rb is None:
            warns.append(f"{label}: 配筋がないため弾性部材として扱います")
            continue
        s, mat = e.sec, e.mat
        Fc, n = mat.Fc, ES / mat.E
        sy = sy_f * rb.main.F
        Lf = e.Lf
        f_face = rL.forces[eid]["face"]
        if key[0] != "c":
            if not hasattr(rb, "end"):
                raise ValueError(f"{label}: 断面 {s.name} の配筋は柱用です")
            K0 = member_end_stiffness(mat.E * s.Iy, mat.G * s.Asz if e.shear_deform else None, Lf)
            d = s.D - rb.dt
            sk = {}
            for side, bars in ((1, rb.end.bottom), (-1, rb.end.top)):
                My = beam_yield_moment(bars.area, sy, d)
                ay = sugano_alpha_y(n, bars.area / (s.b * d), Lf / 2, s.D, d)
                sk[side] = Skeleton.from_member(crack_moment(Fc, s.b, s.D), My, ay, K0, r)
            for end, en in ((0, "i端"), (1, "j端")):
                hinges[(eid, 1, end)] = Hinge(sk[1], sk[-1])
                labels[(eid, 1, end)] = f"{label} {en}"
        else:
            if not hasattr(rb, "x"):
                raise ValueError(f"{label}: 断面 {s.name} の配筋は梁用です")
            # 面 0: X方向（幅 D, せい b, 主筋 x）、面 1: Y方向（幅 b, せい D, 主筋 y）
            planes = ((0, s.D, s.b, rb.x, mat.E * s.Iz, s.Asy),
                      (1, s.b, s.D, rb.y, mat.E * s.Iy, s.Asz))
            for end, en in ((0, "柱脚"), (1, "柱頭")):
                N = f_face[0] if end == 0 else -f_face[6]            # 圧縮正
                for plane, w, dep, bars, EI, As in planes:
                    d = dep - rb.dt
                    K0 = member_end_stiffness(EI, mat.G * As if e.shear_deform else None, Lf)
                    My = column_yield_moment(bars.area, rb.total_area, sy, w, dep, Fc, N)
                    tag = f"{label} {en} {PLANE_NAME[plane]}"
                    if My <= 0:
                        warns.append(f"{tag}: 長期軸力 {N / 1e3:.0f} kN で My ≦ 0 のため微小値とします")
                        My = 1e-3 * beam_yield_moment(bars.area, sy, d)
                    ay = sugano_alpha_y(n, bars.area / (w * d), Lf / 2, dep, d, N / (w * dep * Fc))
                    skel = Skeleton.from_member(crack_moment(Fc, w, dep, N), My, ay, K0, r)
                    hinges[(eid, plane, end)] = Hinge(skel, skel)
                    labels[(eid, plane, end)] = tag
    return hinges, labels, warns


def run_pushover(bld, direction="+x", results=None):
    """Building のプッシュオーバー解析（direction: "+x", "-x", "+y", "-y"）

    長期荷重ケースは design.long（既定 G, P）、水平外力は Ai 分布の地震荷重ケース Ex / Ey。
    非線形の設定は YAML の nonlinear（sigma_y_factor, post_yield_ratio, pushover）。
    """
    name, sign, d, case = _parse_direction(direction)
    m = bld.model
    opts = bld.nonlinear
    if case not in m.nodal_loads:
        raise ValueError(f"水平外力の荷重ケース {case} がありません（seismic.directions を確認）")
    cases = set(m.nodal_loads) | set(m.member_loads)
    long = bld.design["long"] or [c for c in ("G", "P") if c in cases]
    res = dict(results or {})
    for c in long:
        if c not in res:
            res[c] = m.analyze(c)
    rL = Result.combine("+".join(long), {res[c]: 1.0 for c in long}) if long else None
    if rL is None:
        raise ValueError("長期荷重のケース（G, P）がありません")
    hinges, labels, warns = build_hinges(bld, rL, opts)

    levels, prev, z_prev = [], None, bld.floors[0][1]
    for f, (fname, z) in enumerate(bld.floors[1:], 1):
        mst = bld.masters.get(f)
        if mst is None:
            raise ValueError(f"階 {fname} に剛床がないため層間変形角を監視できません")
        levels.append(Level(bld.floors[f - 1][0], prev, mst, z - z_prev))
        prev, z_prev = mst, z
    pattern = [(nid, sign * np.asarray(F, float)) for nid, F in m.nodal_loads[case]]
    po_opt = opts["pushover"]
    try:
        po = pushover(m, hinges, long, pattern, levels, d, po_opt["max_drift"], po_opt["step_drift"],
                      labels)
        u, face, node, idx = po._state
        disp = {nid: u[6 * k:6 * k + 6].copy() for nid, k in idx.items()}
        forces = {eid: {"face": face[eid].copy(), "node": node[eid].copy()} for eid in face}
        po.final = Result(m, f"pushover {name}", disp, forces, {})
    finally:
        for e in m.elements.values():
            e.springs = None
    po.direction = name
    po.total_P = abs(po.total_P)
    po.total_W = sum(w for w, _, _ in bld.weights.values()) if bld.weights else 0.0
    po.warnings = warns
    return po
