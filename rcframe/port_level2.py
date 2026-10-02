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

from .fiber import PipeFibers, PipeMPhi, SteelLaw, local_buckling_strain
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
    "ファイバーの M–φ からヒンジを作る方法（最初の降伏 My で降伏、θu = (φu − φy)·Lp で Mu、局部座屈後の耐力低下は無視）",
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
    caps: dict = field(default_factory=dict)          # ばね -> PileCapacity（pile_hinge.model: fiber）
    damage: list = field(default_factory=list)        # 要求点の杭の損傷（pile_damage の戻り値）
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
        word = "降伏（M–φ の My）" if self.caps else "全塑性"
        for kind in ("杭頭", "地中部"):
            if kind in first:
                e = first[kind]
                print(f"  {kind}の最初の{word}: step {e['step']}, kh = {e['lam']:.3f}, 変位 {e['drift'] / 1e3:.3f} m"
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
        if self.damage:
            cnt = {lv: sum(r["level"] == lv for r in self.damage) for lv in ("局部座屈", "塑性化", "弾性")}
            top = self.damage[0]
            print(f"  杭の損傷（要求点）: 局部座屈 {cnt['局部座屈']}、塑性化 {cnt['塑性化']}、弾性 {cnt['弾性']}"
                  f"（最大ひずみ {top['strain']:.5f}、{top['label']}）")
        if self.deck:
            print(f"  上部工の梁（弾性の前提）: max M/Mu = {self.deck['ratio']:.2f}（{self.deck['label']}）")
        for w in self.warnings:
            print(f"  [注意] {w}")


@dataclass
class PileCapacity:
    """ファイバーの M–φ から作った杭の塑性ヒンジの諸元（利用可否 U3）

    ヒンジ（剛塑性 + 線形硬化）: 最初の降伏 My で降伏し、θu = (φu − φy)·Lp で Mu に達する
    （降伏後の勾配 (Mu − My)/θu）。θ ≧ θu を局部座屈とする。φy: 最初の降伏の曲率、φu: 限界曲率
    """
    N: float
    curve: object                 # fiber.MPhi
    calc: object                  # fiber.PipeMPhi（ひずみの逆算用）
    Lp: float

    @property
    def My(self):
        return self.curve.My

    @property
    def Mu(self):
        return self.curve.Mu

    @property
    def theta_u(self):
        return (self.curve.phi_u - self.curve.phi_y) * self.Lp

    def strain(self, theta, M):
        """ヒンジの回転角 θ・モーメント M から圧縮縁のひずみ（圧縮正）を推定する。
        θ > 0: φ = φy + |θ|/Lp、θ = 0: M–φ 関係で |M| に対応する曲率"""
        if abs(theta) > 0:
            phi = self.curve.phi_y + abs(theta) / self.Lp
        else:
            phi = self.curve.phi_at(abs(M))
            phi = self.curve.phi_y if phi is None else phi
        return -self.calc.strain_edges(phi, self.N)[0]

    def level(self, theta, M):
        """損傷の程度: "弾性" / "塑性化"（M ≧ My）/ "局部座屈"（θ ≧ θu）"""
        if abs(theta) >= self.theta_u * (1 - 1e-9) and (abs(theta) > 0 or abs(M) >= self.Mu * (1 - 1e-9)):
            return "局部座屈"
        return "塑性化" if abs(theta) > 0 or abs(M) >= self.My * (1 - 1e-9) else "弾性"


def pile_capacity(sec, mat, N, opt):
    """ファイバーの M–φ（腐食後の断面、軸力 N 圧縮正）から PileCapacity と注意を作る"""
    Do, Di = sec.dims(corroded=True)
    method, value = opt["ecu"]
    ecu, warns = local_buckling_strain(Do, (Do - Di) / 2, mat.fy, mat.E, method, value)
    calc = PipeMPhi(PipeFibers(Do, Di), SteelLaw(mat.E, mat.fy, opt["hardening"]))
    c = calc.curve(N, ecu)
    return PileCapacity(N, c, calc, opt["hinge_length"] * Do), warns + c.warnings


def build_pile_hinges(pier, rG, plane, r, depth_factor, hinge_opt=None, caps=None):
    """杭の塑性ヒンジ {(eid, 面, 端): Hinge}、ラベル、種類（杭頭 / 地中部）

    hinge_opt: level2.pile_hinge。model = fiber なら M–φ からヒンジを作り、caps に PileCapacity を入れる
    """
    m = pier.model
    hinges, labels, kinds, warns = {}, {}, {}, []
    soil = pier.pile_model == "soil_spring"
    fiber = hinge_opt is not None and hinge_opt["model"] == "fiber"
    cache = {}
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
            key = (eid, plane, end)
            if fiber:
                # 軸力は降伏軸力の 0.5% 刻みに丸めて M–φ を共有する（計算時間の短縮）
                Do, Di = sec.dims(corroded=True)
                dN = 0.005 * math.pi * (Do ** 2 - Di ** 2) / 4 * mat.fy
                Nq = round(N / dN) * dN
                ck = (id(sec), id(mat), Nq)
                if ck not in cache:
                    cache[ck] = pile_capacity(sec, mat, Nq, hinge_opt)
                    for w in cache[ck][1]:
                        if f"杭 {sec.name}: {w}" not in warns:
                            warns.append(f"杭 {sec.name}: {w}")
                cap = cache[ck][0]
                ks3 = r * K0
                if cap.theta_u > 0:
                    ks3 = max(ks3, (cap.Mu - cap.My) / cap.theta_u)
                sk = Skeleton(Mc=cap.My, My=cap.My, theta_y=0.0, ks2=math.inf, ks3=ks3)
                # θu を越えたら（局部座屈）勾配 r·K0（耐力低下は無視）
                if caps is not None:
                    caps[key] = cap
            else:
                Mp = pipe_mp(sec, mat.fy, N)
                if Mp <= 0:
                    warns.append(f"杭 {tag} {where}: 長期軸力が降伏軸力以上のため Mp を微小値とします")
                    Mp = 1e-3 * pipe_mp(sec, mat.fy, 0.0)
                sk = Skeleton.from_member(Mp, Mp, 0.5, K0, r)
            if fiber:
                hinges[key] = BilinearHinge(sk, cap.theta_u, r * K0)
            else:
                hinges[key] = BilinearHinge(sk)          # 鋼材: 移動硬化（除荷は 2Mp の幅で弾性）
            labels[key] = f"杭 {tag} {where}"
            kinds[key] = kind
    return hinges, labels, kinds, warns


def pile_damage(caps, hinges, labels, kinds, theta=None, moment=None):
    """ファイバーの諸元がある杭のヒンジの損傷の一覧
    theta / moment: {ばね: 値}（省略時はヒンジの現在の値）
    戻り値: [dict(key, label, kind, theta, theta_u, strain, level)]（局部座屈 → 塑性化 → 弾性の順）"""
    rows = []
    order = {"局部座屈": 0, "塑性化": 1, "弾性": 2}
    for key, cap in caps.items():
        th = hinges[key].theta if theta is None else theta[key]
        M = hinges[key].M if moment is None else moment[key]
        rows.append(dict(key=key, label=labels[key], kind=kinds.get(key, ""), theta=abs(th),
                         theta_u=cap.theta_u, strain=cap.strain(th, M), level=cap.level(th, M)))
    rows.sort(key=lambda x: (order[x["level"]], -x["strain"]))
    return rows


def inertia_pattern(pier, d, sign):
    """上部工の重心に作用する水平力 W（荷重係数 1 = 換算震度 1）を剛床のマスター節点へ移した節点力。
    戻り値: ([(マスター節点, 6成分)], W)"""
    W, xg, yg = pier.weights["地震用"]
    mst = pier.model.nodes[pier.master()]
    F = np.zeros(6)
    F[d] = sign * W
    F[5] = sign * ((xg - mst.x) * W if d == 1 else -(yg - mst.y) * W)   # 重心の偏心によるねじり
    return [(pier.master(), F)], W


def run_level2(pier, direction="+y", results=None):
    """レベル2地震動のプッシュオーバーと判定（direction: +x, -x, +y, -y）"""
    opt = pier.level2
    name, sign, d, _ = _parse_direction(direction)
    plane = 0 if d == 0 else 1
    m = pier.model
    if "地震用" not in pier.weights or pier.master() is None:
        raise ValueError("レベル2: 地震用重量（actions の D, Ls）と上部工の剛床が必要です")
    pattern, W = inertia_pattern(pier, d, sign)
    res = dict(results or {})
    gcases = list(opt["gravity"])
    for c in gcases:
        if c not in res:
            res[c] = m.analyze(c)
    rG = Result.combine("+".join(gcases), {res[c]: 1.0 for c in gcases})
    caps = {}
    hinges, labels, kinds, warns = build_pile_hinges(pier, rG, plane, opt["post_yield_ratio"],
                                                     opt["inground_depth"], opt["pile_hinge"], caps)
    # 地盤ばね（上限 pu があれば弾完全塑性）
    ns, nlabels = {}, {}
    if pier.pile_model == "soil_spring":
        for (i, j), p in pier.piles.items():
            for nid, z, k, cap in p["ground"]:
                if cap:
                    ns[(nid, d)] = EPSpring(k, cap)
                    nlabels[(nid, d)] = f"地盤 {pier.gx[i][0]}/{pier.gy[j][0]} z={z / 1e3:.2f}"
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
    out = Level2Result(name, po, W, kinds=kinds, soil=ns, caps=caps, warnings=warns + po.warnings)
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
    # 杭の損傷（ファイバーの諸元があるとき）: 要求点の回転角・モーメントを線形補間
    if out.caps:
        s0_, s1_ = prev_of[k_step], k_step
        th = {k: po.hinge_theta[s0_][k] + t * (po.hinge_theta[s1_][k] - po.hinge_theta[s0_][k]) for k in out.caps}
        mm = {k: po.hinge_M[s0_][k] + t * (po.hinge_M[s1_][k] - po.hinge_M[s0_][k]) for k in out.caps}
        out.damage = pile_damage(out.caps, po.hinges, po.labels, out.kinds, th, mm)
    if "no_local_buckling" in opt["criteria"]:
        lb = [r for r in out.damage if r["level"] == "局部座屈"]
        out.criteria.append(("杭が局部座屈に達しない", not lb,
                             "なし" if not lb else f"{lb[0]['label']} ほか {len(lb)} 箇所"))
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
