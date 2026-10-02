"""桟橋のレベル2地震動に対するプッシュオーバー（桟橋編 P4）

    鋼管杭: 全塑性モーメント Mp(N) = Mp0·cos(π|N| / 2Ny)（薄肉円管の相関、腐食後の断面）の
            バイリニア骨格の材端ばね。N は長期荷重時の値で固定
            ヒンジ位置: 杭頭、仮想固定点（仮想固定点モデル）/ 海底面と地中部の分割点（地盤ばねモデル、
            海底面から inground_depth/β の深さまで）
    地盤ばね: 上限 pu（入力）があれば弾完全塑性（hinge.EPSpring）、なければ線形
    上部工の梁: 弾性。要求点で梁の曲げが Mu（示方書、材料係数 1.0）以下かを確認
    載荷: 長期荷重（level2.gravity）の後、上部工の重心に水平力 W（荷重係数 = 換算震度 kh）を比例増加
    判定: 要求点（照査用震度 kh または要求変位）で level2.criteria を確認

要求側と判定基準は入力（港湾基準の規定はプログラム作成者が確定できないため）。単位: mm, N
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .hinge import BilinearHinge, EPSpring
from .port_check import jsce_mu
from .pushover import Level, _parse_direction, pushover
from .results import Result
from .skeleton import Skeleton, member_end_stiffness

VERIFY_ITEMS = [
    "鋼管杭の全塑性モーメントと軸力の相関 Mp(N) = Mp0·cos(π|N|/2Ny)、Zp = (Do³ − Di³)/6（腐食後の断面）",
    "レベル2照査の要求側（照査用震度・要求変位）と判定基準（崩壊機構、地中部の全塑性、杭頭の塑性回転角）の扱い",
    "地盤ばねのモデル化（kh·D·負担長、上限 pu・杭先端の条件）",
    "海底面の位置のヒンジを地中部として扱うこと",
]


def pipe_mp(sec, fy, N):
    """鋼管（腐食後の断面）の全塑性モーメント Mp(N) [N·mm]"""
    Do, Di = sec.dims(corroded=True)
    Zp = (Do ** 3 - Di ** 3) / 6
    Ny = math.pi * (Do ** 2 - Di ** 2) / 4 * fy
    if abs(N) >= Ny:
        return 0.0
    return Zp * fy * math.cos(math.pi * abs(N) / (2 * Ny))


@dataclass
class Level2Result:
    direction: str
    po: object
    W: float
    demand: dict = field(default_factory=dict)        # kh, disp, step, t, reached
    criteria: list = field(default_factory=list)      # [(名前, OK?, 説明)]
    deck: dict = field(default_factory=dict)          # 梁の max(M/Mu) と部材
    kinds: dict = field(default_factory=dict)         # ばね -> "杭頭" / "地中部"
    soil: dict = field(default_factory=dict)          # (節点, 自由度) -> EPSpring（上限つきの地盤ばね）
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.criteria)

    @property
    def curve(self):
        """[(上部工の変位 [mm], 換算震度 = 水平力/W)]"""
        return [(0.0, 0.0)] + [(abs(s["drifts"][0]), s["lam"]) for s in self.po.lateral]

    def print(self):
        dm = self.demand
        print(f"\n■ レベル2 プッシュオーバー {self.direction}  W = {self.W / 1e3:,.1f} kN")
        kmax = max(k for _, k in self.curve)
        print(f"  最大の換算震度 {kmax:.3f}（上部工の変位 {max(abs(x) for x, _ in self.curve) / 1e3:.3f} m まで）")
        first = {}
        for e in self.po.events:
            if e["event"] == "降伏" and e["phase"] == "水平":
                first.setdefault(self.kinds.get(e["key"], ""), e)
        for kind in ("杭頭", "地中部"):
            if kind in first:
                e = first[kind]
                print(f"  {kind}の最初の全塑性: step {e['step']}, kh = {e['lam']:.3f}, 変位 {e['drift'] / 1e3:.3f} m"
                      f"（{e['label']}）")
        if self.po.mechanism_step:
            s = next(x for x in self.po.lateral if x["step"] == self.po.mechanism_step)
            print(f"  崩壊機構: step {self.po.mechanism_step}, kh = {s['lam']:.3f}")
        if dm.get("reached"):
            print(f"  要求点: kh = {dm['kh']:.3f}, 変位 {dm['disp'] / 1e3:.3f} m")
        else:
            print(f"  要求点に到達しません（{dm.get('note', '')}）")
        for name, ok, note in self.criteria:
            print(f"  {'OK' if ok else 'NG'}  {name}: {note}")
        if self.deck:
            print(f"  上部工の梁（弾性の前提）: max M/Mu = {self.deck['ratio']:.2f}（{self.deck['label']}）")
        for w in self.warnings:
            print(f"  [注意] {w}")


def build_pile_hinges(pier, rG, plane, r, depth_factor):
    """杭の塑性ヒンジ {(eid, 面, 端): Hinge}、ラベル、種類（杭頭 / 地中部）"""
    m = pier.model
    hinges, labels, kinds, warns = {}, {}, {}, []
    soil = pier.pile_model == "soil_spring"
    for (i, j), p in sorted(pier.piles.items(), key=lambda kv: kv[1]["eid"]):
        sec, mat = pier.pile_sec[(i, j)]
        tag = f"{pier.gx[i][0]}/{pier.gy[j][0]}"
        EI = mat.E * (sec.Iz if plane == 0 else sec.Iy)
        Lref = p["free_length"] + (1 / p["beta"] if soil else 0.0)     # 等価な片持ち長さ
        K0 = member_end_stiffness(EI, mat.G * sec.Asy, Lref)
        pos = [(p["eid"], 1, "杭頭", "杭頭"), (p["eid"], 0, "地中部", "海底面" if soil else "仮想固定点")]
        if soil:
            for k, sid in enumerate(p.get("segments", [])):
                if k == 0:
                    continue                                          # 上端は海底面（自由長部の i端）
                z_top = p["ground"][k][1]
                if p["seabed"] - z_top > depth_factor / p["beta"] + 1e-6:
                    break
                pos.append((sid, 1, "地中部", f"地中 z={z_top / 1e3:.2f}"))
        for eid, end, kind, where in pos:
            f = rG.forces[eid]["face"]
            N = f[0] if end == 0 else -f[6]                           # 圧縮正
            Mp = pipe_mp(sec, mat.fy, N)
            if Mp <= 0:
                warns.append(f"杭 {tag} {where}: 長期軸力が降伏軸力以上のため Mp を微小値とします")
                Mp = 1e-3 * pipe_mp(sec, mat.fy, 0.0)
            sk = Skeleton.from_member(Mp, Mp, 0.5, K0, r)
            key = (eid, plane, end)
            hinges[key] = BilinearHinge(sk)              # 鋼材: 移動硬化（除荷は 2Mp の幅で弾性）
            labels[key] = f"杭 {tag} {where}"
            kinds[key] = kind
    return hinges, labels, kinds, warns


def run_level2(pier, direction="+y", results=None):
    """レベル2地震動のプッシュオーバーと判定（direction: +x, -x, +y, -y）"""
    opt = pier.level2
    name, sign, d, _ = _parse_direction(direction)
    plane = 0 if d == 0 else 1
    m = pier.model
    if "地震用" not in pier.weights or pier.master() is None:
        raise ValueError("レベル2: 地震用重量（actions の D, Ls）と上部工の剛床が必要です")
    W, xg, yg = pier.weights["地震用"]
    mst = m.nodes[pier.master()]
    res = dict(results or {})
    gcases = list(opt["gravity"])
    for c in gcases:
        if c not in res:
            res[c] = m.analyze(c)
    rG = Result.combine("+".join(gcases), {res[c]: 1.0 for c in gcases})
    hinges, labels, kinds, warns = build_pile_hinges(pier, rG, plane, opt["post_yield_ratio"],
                                                     opt["inground_depth"])
    # 地盤ばね（上限 pu があれば弾完全塑性）
    ns, nlabels = {}, {}
    if pier.pile_model == "soil_spring":
        for (i, j), p in pier.piles.items():
            for nid, z, k, cap in p["ground"]:
                if cap:
                    ns[(nid, d)] = EPSpring(k, cap)
                    nlabels[(nid, d)] = f"地盤 {pier.gx[i][0]}/{pier.gy[j][0]} z={z / 1e3:.2f}"
    F = np.zeros(6)
    F[d] = sign * W
    F[5] = sign * ((xg - mst.x) * W if d == 1 else -(yg - mst.y) * W)
    pattern = [(pier.master(), F)]
    levels = [Level("上部工", None, pier.master(), 1.0)]      # 「層間変形角」= 上部工の変位 [mm]
    marks = (opt["target_displacement"],) if opt["target_displacement"] else ()
    saved = {nid: k.copy() for nid, k in m.node_springs.items()}
    try:
        po = pushover(m, hinges, gcases, pattern, levels, d, opt["max_displacement"], opt["step"], labels,
                      mechanism_check=True, drift_marks=marks, keep_states=True,
                      node_springs=ns, node_labels=nlabels)
    finally:
        m.node_springs.update(saved)
        for e in m.elements.values():
            e.springs = None
    po.direction = name
    u, face, node, idx = po._state                            # 最終状態（3D 表示・図用）
    po.final = Result(m, f"レベル2 {name}", {nid: u[6 * k:6 * k + 6].copy() for nid, k in idx.items()},
                      {eid: {"face": face[eid].copy(), "node": node[eid].copy()} for eid in face}, {},
                      {c: 1.0 for c in gcases})
    out = Level2Result(name, po, W, kinds=kinds, soil=ns, warnings=warns + po.warnings)
    _evaluate(pier, out, opt, gcases, res)
    return out


def _evaluate(pier, out, opt, gcases, res):
    """要求点の評価と判定"""
    po = out.po
    lat = po.lateral
    steps = [s["step"] for s in po.steps]
    prev_of = {s: steps[k - 1] for k, s in enumerate(steps) if k > 0}
    k_step, t = None, 1.0
    if opt["kh"] is not None:
        kh = opt["kh"]
        for s in lat:
            if s["lam"] >= kh * (1 - 1e-12):
                k_step = s["step"]
                p = next((x for x in po.steps if x["step"] == prev_of[k_step]), None)
                l0 = p["lam"] if p and p["phase"] == "水平" else 0.0
                t = (kh - l0) / (s["lam"] - l0) if s["lam"] > l0 else 1.0
                break
        note = f"換算震度の最大値 {max(x['lam'] for x in lat):.3f} < 照査用震度 {kh:g}"
    else:
        target = opt["target_displacement"]
        for s in lat:
            if abs(s["drifts"][0]) >= target * (1 - 1e-9):
                k_step = s["step"]
                break
        note = f"要求変位 {target / 1e3:g} m まで押せません"
    if k_step is None:
        out.demand = {"reached": False, "note": note}
        out.criteria.append(("要求点への到達", False, note))
        return
    s0 = next(x for x in po.steps if x["step"] == prev_of[k_step])
    s1 = next(x for x in po.steps if x["step"] == k_step)
    disp = abs(s0["drifts"][0] + t * (s1["drifts"][0] - s0["drifts"][0]))
    lam = (s0["lam"] if s0["phase"] == "水平" else 0.0) * (1 - t) + s1["lam"] * t
    out.demand = {"reached": True, "kh": lam, "disp": disp, "step": k_step, "t": t}

    def before(ev_step):
        return ev_step < k_step or (ev_step == k_step and t >= 1 - 1e-9)

    for c in opt["criteria"]:
        if c == "no_mechanism":
            ok = po.mechanism_step is None or not before(po.mechanism_step)
            out.criteria.append(("崩壊機構が形成されない", ok,
                                 "なし" if ok else f"step {po.mechanism_step} で形成"))
        elif c == "no_inground_plastic":
            evs = [e for e in po.events if e["event"] == "降伏" and out.kinds.get(e["key"]) == "地中部"
                   and before(e["step"])]
            out.criteria.append(("地中部が全塑性に達しない", not evs,
                                 "なし" if not evs else f"{evs[0]['label']}（step {evs[0]['step']}）ほか {len(evs)} 箇所"))
        elif isinstance(c, dict) and "pile_rotation" in c:
            lim = c["pile_rotation"]
            th0, th1 = po.hinge_theta[prev_of[k_step]], po.hinge_theta[k_step]
            rot = {k: abs(th0[k] + t * (th1[k] - th0[k])) for k in th1 if out.kinds.get(k) == "杭頭"}
            kmax = max(rot, key=rot.get)
            out.criteria.append((f"杭頭の塑性回転角 ≦ {lim:g} rad", rot[kmax] <= lim,
                                 f"最大 {rot[kmax]:.4f} rad（{po.labels[kmax]}）"))
    # 上部工の梁が弾性の範囲か（要求点の部材端力を線形補間して Mu と比較）
    m = pier.model
    u0, f0 = po.states[prev_of[k_step]]
    u1, f1 = po.states[k_step]
    forces = {}
    for eid in f1:
        f = f0[eid] + t * (f1[eid] - f0[eid])
        forces[eid] = {"face": f, "node": f}
    disp_n = {}
    rr = Result(m, "L2 要求点", disp_n, forces, {}, {c: 1.0 for c in gcases})
    worst = (0.0, "")
    pile_at = {pier.node_ids[ij]: sec for ij, (sec, _) in pier.pile_sec.items()}
    for key, eid in pier.elem_ids.items():
        if key[0] in ("p", "pg"):
            continue
        e = m.elements[eid]
        rb, s = e.sec.rebar, e.sec
        if rb is None:
            continue
        ri = pile_at[e.ni].D / 2 if e.ni in pile_at else 0.0
        rj = pile_at[e.nj].D / 2 if e.nj in pile_at else 0.0
        for x, bars in ((ri, rb.end), (e.L / 2, rb.center), (e.L - rj, rb.end)):
            M = -rr.section_forces_at(eid, x)[4]
            comp, tens = (bars.top, bars.bottom) if M >= 0 else (bars.bottom, bars.top)
            Mu = jsce_mu(s.b, s.D, comp.area, tens.area, rb.dt, rb.dt, e.mat.fck, rb.main.F, rb.main.F)
            if abs(M) / Mu > worst[0]:
                worst = (abs(M) / Mu, pier.label("elem", eid))
    if worst[1]:
        out.deck = {"ratio": worst[0], "label": worst[1]}
        if worst[0] > 1:
            out.warnings.append("上部工の梁が Mu を超えています（弾性の前提が成り立ちません）")
