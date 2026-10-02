"""静的非線形解析（プッシュオーバー）: 材端ばねモデル・事象追跡型の荷重増分法

1. 長期荷重（G+P）を荷重係数 0→1 で増分載荷
2. 水平外力（Ai 分布、増分中は分布形一定）を比例的に増加し、層間変形角の上限で終了

各ステップでは、いずれかのばねが次の折れ点（ひび割れ・降伏・除荷・反転）に達するまで
荷重を進める。ばねは折れ線なので各ステップは線形で厳密（不釣合い力なし）。
ばねの向き（載荷・除荷）は、仮定した向きで解いたモーメント増分の符号と一致するまで繰り返す。
P-Δ 効果、せん断破壊、2方向曲げの相互作用は考慮しない。
"""
import copy
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
    unconverged: list = field(default_factory=list)   # ばねの載荷・除荷の判定が整合しなかったステップ
    mechanism_step: int = None                    # 崩壊機構が形成されたステップ（mechanism_check 時）
    mechanism_stories: list = field(default_factory=list)   # 機構で変形が集中する層
    states: dict = field(default_factory=dict)    # step -> (変位, {eid: 可撓部端の材端力})（keep_states 時）
    hinge_theta: dict = field(default_factory=dict)   # step -> {ばね: 回転角}（keep_states 時）
    hinge_M: dict = field(default_factory=dict)       # step -> {ばね: モーメント}（keep_states 時）

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


class Incremental:
    """事象追跡型の荷重増分解析の状態と載荷（入力形式に依存しない本体）

    状態（変位・部材端力・材端ばね・節点ばね）を保持し、run() で指定の荷重を指定の停止条件まで
    載せる。続けて別の荷重（除荷・再載荷・利用荷重など）を載せられ、snapshot() / restore() で
    状態を保存・復元できる。

    hinges        : {(eid, 面, 端): Hinge}（端 0 = i端, 1 = j端）
    levels        : [Level]（層間変形角の監視、下から）
    d             : 水平方向（0 = X, 1 = Y）。層間変形角とベースシアの方向
    shear_caps    : {(eid, 面): Qsu}  水平載荷中に |Q| が Qsu に達したら「せん断破壊」の事象
    mechanism_check: 降伏したばねを解放（ピン）した接線剛性で水平外力に対して不安定になったら
                    崩壊機構の形成とする（res.mechanism_step, res.mechanism_stories）
    keep_states   : 各ステップの変位・部材端力を res.states、ばねの回転角・モーメントを res.hinge_theta・res.hinge_M に保存
    node_springs  : {(節点, 自由度): EPSpring}  非線形の節点ばね（地盤ばね）。各ステップの接線剛性を
                    model.node_springs[節点][自由度] に設定する（呼び出し側で元の値に戻すこと）
    lateral_total : 基準の水平外力パターンの d 方向の合計（ベースシア Q = 正味の荷重係数 × これ）

    地盤変位（run の ground）: 節点ばね（model.node_springs、地盤ばね）の地盤側の端に強制変位 ug を与える。
    ばねの力 = k·(u − ug)。剛性方程式では等価節点力 k·Δug として右辺に加える（側方流動）
    """

    def __init__(self, model, hinges, levels, d, labels=None, max_steps=20000, shear_caps=None,
                 shear_labels=None, mechanism_check=False, keep_states=False, node_springs=None,
                 node_labels=None, lateral_total=0.0):
        self.model, self.hinges, self.levels, self.d = model, hinges, levels, d
        self.ns = dict(node_springs or {})
        self.ns_labels = node_labels or {k: str(k) for k in self.ns}
        self.ns_sigma = {k: 1 for k in self.ns}
        self.shear_caps = dict(shear_caps or {})
        self.shear_labels = shear_labels or {k: str(k) for k in self.shear_caps}
        self.sheared = set()
        self.ref_rate = None                 # 水平載荷の最初のステップの層間変形角増分（機構判定の基準）
        self.labels = labels or {k: str(k) for k in hinges}
        self.mechanism_check, self.keep_states, self.max_steps = mechanism_check, keep_states, max_steps
        self.idx, self.n = model._index()
        self.C, _ = model._constraint_matrix(self.idx, self.n)
        E = model.elements
        self.by_elem = defaultdict(list)
        for key in hinges:
            self.by_elem[key[0]].append(key)
        self.sigma = {key: 1 for key in hinges}
        self.u = np.zeros(self.n)
        self.ug = np.zeros(self.n)           # 節点ばねの地盤側の変位（地盤変位、u と同じ自由度番号）
        self.face = {eid: np.zeros(12) for eid in E}
        self.node = {eid: np.zeros(12) for eid in E}
        self.res = PushoverResult("", hinges=hinges, labels=self.labels, levels=levels)
        self.lat_total = lateral_total
        self.lam_lat = 0.0                   # 基準の水平外力の正味の荷重係数
        self.step = 0
        self.zero_steps = 0
        self.edofs = {eid: np.r_[6 * self.idx[e.ni] + np.arange(6), 6 * self.idx[e.nj] + np.arange(6)]
                      for eid, e in E.items()}
        self._eids = list(E)
        self._edof_arr = np.array([self.edofs[eid] for eid in self._eids])

    # ------------------------------------------------------------ 状態
    @property
    def state(self):
        return self.u, self.face, self.node, self.idx

    def disp(self, nid, dof):
        return self.u[6 * self.idx[nid] + dof]

    def drifts(self, vec=None):
        vec = self.u if vec is None else vec
        out = []
        for lv in self.levels:
            lo = vec[6 * self.idx[lv.lower] + self.d] if lv.lower is not None else 0.0
            out.append((vec[6 * self.idx[lv.upper] + self.d] - lo) / lv.h)
        return out

    _SNAP = ("u", "ug", "face", "node", "hinges", "ns", "sigma", "ns_sigma", "sheared", "ref_rate", "res",
             "lam_lat", "step", "zero_steps")

    def snapshot(self):
        """現在の状態の複製（restore で戻す）。結果の履歴も含む（モデル本体は含まない）"""
        inc_ref = self.res.__dict__.pop("inc", None)      # res.inc（自分自身）は複製しない
        try:
            return copy.deepcopy({k: getattr(self, k) for k in self._SNAP})
        finally:
            if inc_ref is not None:
                self.res.inc = inc_ref

    def restore(self, snap):
        snap = copy.deepcopy(snap)
        for k in self._SNAP:
            setattr(self, k, snap[k])
        self.res.hinges = self.hinges
        self.res.inc = self

    # ------------------------------------------------------------ 求解
    def _solve(self, cases, pattern, sign, tang, ground=()):
        m, E, idx, n = self.model, self.model.elements, self.idx, self.n
        for eid, keys in self.by_elem.items():
            fs = np.zeros((2, 2))
            for key in keys:
                fs[key[1], key[2]] = tang[key][0]
            E[eid].springs = fs
        K, _, _ = m.assemble_K()
        F = np.zeros(n)
        eloads = {}
        for case in cases:
            for nid, f in m.nodal_loads.get(case, []):
                F[6 * idx[nid]:6 * idx[nid] + 6] += sign * f
            for eid, (pf, pr) in m._element_loads(case).items():
                e = E[eid]
                pg = e.T.T @ (e.H.T @ pf + pr)
                F[6 * idx[e.ni]:6 * idx[e.ni] + 6] += sign * pg[:6]
                F[6 * idx[e.nj]:6 * idx[e.nj] + 6] += sign * pg[6:]
                a, b = eloads.get(eid, (np.zeros(12), np.zeros(12)))
                eloads[eid] = (a + sign * pf, b + sign * pr)
        for nid, f in pattern:
            F[6 * idx[nid]:6 * idx[nid] + 6] += sign * np.asarray(f, float)
        dug = np.zeros(n)
        for nid, dof, v in ground:                # 地盤変位: 等価節点力 k·Δug（k は現在の接線剛性）
            dug[6 * idx[nid] + dof] = sign * v
            F[6 * idx[nid] + dof] += m.node_springs[nid][dof] * sign * v
        ur = spla.spsolve((self.C.T @ K @ self.C).tocsc(), self.C.T @ F)
        if not np.all(np.isfinite(ur)):
            raise RuntimeError("剛性マトリクスが特異です（崩壊機構の形成または拘束不足）")
        du = self.C @ ur
        # 材端力の増分（全要素をまとめて計算）: face = k_face·dg、node = Hᵀ·face
        eids = self._eids
        kf = np.stack([E[eid].k_face for eid in eids])
        HT = np.stack([E[eid].H.T for eid in eids])
        fa = np.einsum("eij,ej->ei", kf, du[self._edof_arr])
        no = np.einsum("eij,ej->ei", HT, fa)
        dface, dnode = dict(zip(eids, fa)), dict(zip(eids, no))
        for eid, (pf, pr) in eloads.items():
            dface[eid] = dface[eid] - pf
            dnode[eid] = E[eid].H.T @ dface[eid] - pr
        return du, dface, dnode, dug

    # ------------------------------------------------------------ 載荷
    def run(self, phase, cases=(), pattern=(), sign=1.0, lam_target=None, max_drift=None, step_drift=None,
            drift_marks=(), disp_target=None, lateral=False, checks=None, ground=()):
        """荷重（cases の節点・部材荷重 + pattern の節点荷重、いずれも × sign）を荷重係数 0 から増やす

        停止条件（いずれか、到達でステップを区切って終了）:
            lam_target  : この載荷の荷重係数がこの値に達する
            max_drift   : 監視の |層間変形角| のいずれかがこの値に達する
            disp_target : (節点, 自由度, 値)  節点の変位がこの値に達する（変位制御）
        step_drift  : 1ステップの層間変形角増分の上限、drift_marks: 必ずステップを区切る層間変形角
        lateral     : 基準の水平外力の載荷（正味の荷重係数とベースシアを更新）
        checks      : せん断破壊・崩壊機構の判定を行う（省略時は lateral と同じ）
        ground      : [(節点, 自由度, 地盤変位)]  荷重係数 1 あたりの地盤変位（× sign）。節点ばねが必要
        戻り値: この載荷の最終の荷重係数
        """
        if lam_target is None and max_drift is None and disp_target is None:
            raise ValueError("停止条件（lam_target, max_drift, disp_target）を指定してください")
        checks = lateral if checks is None else checks
        ground = list(ground)
        for nid, dof, _ in ground:
            if nid not in self.model.node_springs:
                raise ValueError(f"地盤変位を与える節点 {nid} に節点ばねがありません")
        m, E, hinges, ns, res = self.model, self.model.elements, self.hinges, self.ns, self.res
        lam = 0.0
        while True:
            if self.step >= self.max_steps:
                raise RuntimeError(f"ステップ数が上限 {self.max_steps} に達しました")
            # ---- ばねの向きの仮定と解の整合（最大 20 回。整合しなければ警告を残して進む）
            for it in range(20):
                tang = {key: hinges[key].tangent(self.sigma[key]) for key in hinges}
                ntang = {key: ns[key].tangent(self.ns_sigma[key]) for key in ns}
                for (nid, dof), t in ntang.items():
                    m.node_springs[nid][dof] = t[0]
                du, dface, dnode, dug = self._solve(cases, pattern, sign, tang, ground)
                dm = {key: spring_moment(dface[key[0]], key[1], key[2]) for key in hinges}
                dF = {key: ntang[key][0] * (du - dug)[6 * self.idx[key[0]] + key[1]] for key in ns}
                scale = max((abs(v) for v in dm.values()), default=0.0)
                fscale = max((abs(v) for v in dF.values()), default=0.0)
                changed = False
                for key, v in dm.items():
                    if abs(v) > 1e-9 * scale:
                        s = 1 if v > 0 else -1
                        if s != self.sigma[key]:
                            self.sigma[key] = s
                            changed = True
                for key, v in dF.items():
                    if abs(v) > 1e-9 * fscale:
                        s = 1 if v > 0 else -1
                        if s != self.ns_sigma[key]:
                            self.ns_sigma[key] = s
                            changed = True
                if not changed:
                    break
            else:
                res.unconverged.append(self.step + 1)
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
            ncand = {}
            for key, v in dF.items():
                k_t, lim, kind = ntang[key]
                if abs(v) <= 1e-9 * fscale or math.isinf(lim):
                    continue
                t = max(0.0, (lim - ns[key].F) / v)
                ncand[key] = t
                dl = min(dl, t)
            if lam_target is not None:
                dl = min(dl, max(0.0, lam_target - lam))
            rate = self.drifts(du)
            cur = self.drifts()
            rmax = max((abs(r) for r in rate), default=0.0)
            if lateral and self.ref_rate is None:
                self.ref_rate = rmax
            if rmax > 0:
                if step_drift:
                    dl = min(dl, step_drift / rmax)
                cmax = max(abs(c) for c in cur)
                marks = [mk for mk in drift_marks if mk > cmax * (1 + 1e-9)]
                marks += [max_drift] if max_drift is not None else []
                for c, r in zip(cur, rate):          # |層間変形角| が上限・通過点に達するまで
                    if r != 0 and r * c >= 0:
                        for mk in marks:
                            dl = min(dl, max(0.0, mk - abs(c)) / abs(r))
            if disp_target is not None:              # 変位制御: 指定の変位に達するまで
                nid, dof, val = disp_target
                rd = du[6 * self.idx[nid] + dof]
                gap = val - self.disp(nid, dof)
                if rd != 0 and gap * rd >= 0:
                    dl = min(dl, gap / rd)
            if checks:
                for (eid, plane), cap in self.shear_caps.items():   # せん断力が Qsu に達するまで
                    if (eid, plane) in self.sheared:
                        continue
                    for k in ((1, 7) if plane == 0 else (2, 8)):
                        q, dq = self.face[eid][k], dface[eid][k]
                        if dq != 0 and q * dq >= 0:
                            dl = min(dl, max(0.0, cap - abs(q)) / abs(dq))
            if math.isinf(dl):
                raise RuntimeError("荷重増分が定まりません（外力 0、または停止条件に向かって変形しない）")
            self.zero_steps = self.zero_steps + 1 if dl == 0 else 0
            if self.zero_steps > 1000:
                raise RuntimeError("荷重増分 0 のステップが続いています（事象の判定が収束しません）")
            # ---- 状態の更新
            self.u += dl * du
            self.ug += dl * dug
            for eid in E:
                self.face[eid] += dl * dface[eid]
                self.node[eid] += dl * dnode[eid]
            lam += dl
            if lateral:
                self.lam_lat += sign * dl
            self.step += 1
            step = self.step
            Q = self.lam_lat * abs(self.lat_total)
            dr = self.drifts()
            drift_now = max((abs(x) for x in dr), default=0.0)
            evs = []
            for key, h in hinges.items():
                flex, lim, kind = tang[key]
                hit = key in cand and cand[key] <= dl * (1 + 1e-9) + 1e-300
                for ev in h.advance(dl * dm[key], flex, self.sigma[key], hit, lim, kind):
                    rec = dict(step=step, key=key, label=self.labels[key], event=ev, side=self.sigma[key],
                               lam=lam, Q=Q, drift=drift_now, phase=phase)
                    evs.append(rec)
                    res.events.append(rec)
            for key, sp_ in ns.items():
                k_t, lim, kind = ntang[key]
                hit = key in ncand and ncand[key] <= dl * (1 + 1e-9) + 1e-300
                for ev in sp_.advance(dl * dF[key], hit, lim):
                    rec = dict(step=step, key=key, label=self.ns_labels[key], event=ev, side=self.ns_sigma[key],
                               lam=lam, Q=Q, drift=drift_now, phase=phase)
                    evs.append(rec)
                    res.events.append(rec)
            if checks:
                for (eid, plane), cap in self.shear_caps.items():
                    if (eid, plane) in self.sheared:
                        continue
                    q = max(abs(self.face[eid][k]) for k in ((1, 7) if plane == 0 else (2, 8)))
                    if q >= cap * (1 - 1e-9):
                        self.sheared.add((eid, plane))
                        rec = dict(step=step, key=(eid, plane), label=self.shear_labels[(eid, plane)],
                                   event="せん断破壊", side=0, lam=lam, Q=Q, drift=drift_now, phase=phase)
                        evs.append(rec)
                        res.events.append(rec)
                if self.mechanism_check and res.mechanism_step is None and any(
                        any(h.yielded.values()) for h in hinges.values()):
                    rel = {}
                    for key, h in hinges.items():
                        flex, lim, kind = h.tangent(self.sigma[key])
                        if kind.startswith("post"):              # 降伏後の分岐 → 解放（ピン）
                            flex = 1e6 / h.sk[self.sigma[key]].ks3
                        rel[key] = (flex, lim, kind)
                    try:
                        mdu, _, _, _ = self._solve(cases, pattern, sign, rel, ground)
                        mrate = [abs(x) for x in self.drifts(mdu)]
                    except RuntimeError:
                        mrate = [math.inf] * len(self.levels)
                    if max(mrate) > 1e5 * self.ref_rate:
                        res.mechanism_step = step
                        top = max(mrate)
                        res.mechanism_stories = [lv.name for lv, x in zip(self.levels, mrate)
                                                 if x > 0.1 * top]
            res.steps.append(dict(step=step, phase=phase, lam=lam, Q=Q, drifts=dr, events=evs))
            if self.keep_states:
                res.states[step] = (self.u.copy(), {eid: self.face[eid].copy() for eid in E})
                res.hinge_theta[step] = {key: h.theta for key, h in hinges.items()}
                res.hinge_M[step] = {key: h.M for key, h in hinges.items()}
            if lam_target is not None and lam >= lam_target - 1e-12 * max(1.0, abs(lam_target)):
                break
            if max_drift is not None and drift_now >= max_drift * (1 - 1e-9):
                break
            if disp_target is not None:
                nid, dof, val = disp_target
                if abs(self.disp(nid, dof) - val) <= 1e-9 * max(abs(val), 1e-9):
                    break
        return lam

    def unload(self, pattern, phase="除荷", step_drift=None):
        """基準の水平外力を正味の荷重係数 0 まで戻す（除荷）。戻り値: 除荷した荷重係数"""
        lam0 = self.lam_lat
        if lam0 == 0:
            return 0.0
        return self.run(phase, pattern=pattern, sign=-math.copysign(1.0, lam0), lam_target=abs(lam0),
                        step_drift=step_drift, lateral=True, checks=False)


def residual_table(inc, pattern, peaks, step_drift=None, drift_marks=()):
    """最大の層間変形角（または変位）peaks まで水平外力で押し、それぞれ除荷した残留値の表

    inc は長期荷重を載荷済みの Incremental。peaks は昇順。各点で状態を保存して除荷し、
    元に戻して次の点へ進む（解析は 1本の載荷経路で済む）。
    戻り値: [dict(peak, Q_peak, lam_peak, residual: [各層の残留値], step_peak, step_residual)]
    """
    rows = []
    for pk in peaks:
        inc.run("水平", pattern=pattern, max_drift=pk, step_drift=step_drift, drift_marks=drift_marks,
                lateral=True)
        snap = inc.snapshot()
        Qp, lp, sp = inc.lam_lat * abs(inc.lat_total), inc.lam_lat, inc.step
        inc.unload(pattern, step_drift=step_drift)
        rows.append(dict(peak=pk, Q_peak=Qp, lam_peak=lp, residual=inc.drifts(), step_peak=sp,
                         step_residual=inc.step))
        inc.restore(snap)
    return rows


def pushover(model, hinges, gravity_cases, pattern, levels, d, max_drift, step_drift,
             labels=None, max_steps=20000, shear_caps=None, shear_labels=None,
             mechanism_check=False, drift_marks=(), keep_states=False,
             node_springs=None, node_labels=None):
    """長期荷重（gravity_cases）を荷重係数 0→1 で載荷した後、水平外力 pattern を層間変形角 max_drift まで
    比例増加する（Incremental の薄いラッパー。引数の意味は Incremental.run を参照）
    戻り値: PushoverResult（final 以外。final は呼び出し側で設定）
    """
    lat_total = sum(F[d] for _, F in pattern)
    inc = Incremental(model, hinges, levels, d, labels, max_steps, shear_caps, shear_labels,
                      mechanism_check, keep_states, node_springs, node_labels, lateral_total=lat_total)
    inc.run("長期", cases=gravity_cases, lam_target=1.0)
    inc.res.gravity_disp = {nid: inc.u[6 * k:6 * k + 6].copy() for nid, k in inc.idx.items()}
    inc.run("水平", pattern=pattern, max_drift=max_drift, step_drift=step_drift, drift_marks=drift_marks,
            lateral=True)
    res = inc.res
    res.total_P = lat_total
    res._state = inc.state
    res.inc = inc                        # 続けて除荷・再載荷するための状態
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


def run_pushover(bld, direction="+x", results=None, max_drift=None, **kw):
    """Building のプッシュオーバー解析（direction: "+x", "-x", "+y", "-y"）

    長期荷重ケースは design.long（既定 G, P）、水平外力は Ai 分布の地震荷重ケース Ex / Ey。
    非線形の設定は YAML の nonlinear（sigma_y_factor, post_yield_ratio, pushover）。
    kw は pushover() へ渡す（shear_caps, mechanism_check, drift_marks, keep_states など）。
    shear_caps のキーは (eid, 面)。
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
        po = pushover(m, hinges, long, pattern, levels, d, max_drift or po_opt["max_drift"],
                      po_opt["step_drift"], labels, **kw)
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
    if po.unconverged:
        po.warnings.append(f"ばねの載荷・除荷の判定が整合しなかったステップがあります: {po.unconverged[:10]}"
                           f"{' ほか' if len(po.unconverged) > 10 else ''}（結果の精度に注意）")
    return po
