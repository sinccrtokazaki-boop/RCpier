"""桟橋の側方流動（地盤変位）シナリオ（利用可否 U2）

    地盤変位: 地盤ばねの地盤側の端に強制変位 ug(ζ) を与える（ζ = 海底面からの深さ）。
              ばねの力 = k·(u杭 − ug)。上限 pu があれば弾完全塑性なので、杭に作用する力は pu·D·負担長 で頭打ち
    分布形  : cos   ug = ug0·cos(πζ / 2H)（ζ ≦ H）
              linear ug = ug0·(1 − ζ/H)
              uniform ug = ug0
              ζ > H では 0（H: 流動する層の厚さ）。ug0 は杭の列ごとに与えられる
              H は depth（海底面からの厚さ、全列共通）または bottom（流動層の下端の標高、列ごとに H = 海底面 − bottom）
    流動層  : ζ < H の地盤ばねの剛性・上限に低減係数（kh_factor, pu_factor）を掛けられる（液状化の影響）
    慣性力  : 上部工の重心に kh·W（port_level2.inertia_pattern）
    載荷順序: simultaneous（同時に比例増加）/ ground_first（地盤変位 → 慣性力）/ inertia_first（慣性力 → 地盤変位）
              最後に慣性力を除荷し（unload）、残留変位・残留の塑性回転角を求める（pushover.Incremental）
    打ち切り: 上部工の変位が max_displacement（地盤変位を含む段階は + 海底面の地盤変位の最大値）に達したら止める

地盤変位の分布・低減係数・載荷順序は入力（FLIP 等の結果または基準で定めること）。単位: mm, N
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .grid import InputError, _check_keys, _num
from .hinge import EPSpring
from .port_level2 import build_pile_hinges, inertia_pattern, pile_damage
from .pushover import Incremental, Level, _parse_direction
from .results import Result

M = 1e3           # m → mm
SHAPES = ("cos", "linear", "uniform")
ORDERS = ("simultaneous", "ground_first", "inertia_first")
KEYS = ("direction", "surface", "depth", "bottom", "shape", "kh_factor", "pu_factor", "kh", "order", "unload",
        "step", "max_displacement")


def ground_profile(shape, ug0, H, zeta):
    """深さ zeta（海底面から下向き正）の地盤変位"""
    if zeta < -1e-9 or zeta > H * (1 + 1e-9):
        return 0.0
    if shape == "cos":
        return ug0 * math.cos(math.pi * min(zeta, H) / (2 * H))
    if shape == "linear":
        return ug0 * (1 - min(zeta, H) / H)
    if shape == "uniform":
        return ug0
    raise ValueError(f"分布形は {SHAPES} のいずれかです: {shape!r}")


@dataclass
class FlowCase:
    """側方流動のシナリオ（内部単位 mm）"""
    name: str
    direction: str = "+y"
    surface: dict = field(default_factory=dict)     # 杭の列（Y通りの番号）-> 海底面の地盤変位 ug0 [mm]
    depth: float = 0.0                              # 流動する層の厚さ H [mm]
    shape: str = "cos"
    kh_factor: float = 1.0
    pu_factor: float = 1.0
    kh: float = 0.0                                 # 慣性力の換算震度（負は流動と逆向き）
    order: str = "simultaneous"
    unload: bool = True
    step: float = None                              # 1ステップの上部工の変位増分の上限 [mm]（None: level2.step）
    max_displacement: float = None                  # 打ち切る上部工の変位 [mm]（None: level2.max_displacement）
    bottom: float = None                            # 流動層の下端の標高 [mm]（与えれば depth の代わりに列ごとの H）

    def layer(self, seabed):
        """杭の列の流動層の厚さ H [mm]（海底面の標高 seabed [mm]）"""
        return self.depth if self.bottom is None else max(seabed - self.bottom, 0.0)

    @property
    def layer_text(self):
        return f"層厚 {self.depth / M:g} m" if self.bottom is None else f"流動層の下端 {self.bottom / M:+g} m"


def read_flow_cases(pier, data):
    """YAML の lateral_flow（{名前: シナリオ}）を FlowCase のリストにする。単位: m"""
    if not data:
        return []
    if not isinstance(data, dict):
        raise InputError("lateral_flow: {シナリオ名: {...}} で指定してください")
    out = []
    for name, d in data.items():
        what = f"lateral_flow.{name}"
        d = d or {}
        _check_keys(d, KEYS, what)
        direction = str(d.get("direction", "+y"))
        if direction.lstrip("+-").lower() not in ("x", "y"):
            raise InputError(f"{what}.direction: +x, -x, +y, -y から指定してください: {direction!r}")
        if "surface" not in d:
            raise InputError(f"{what}: surface がありません")
        if ("depth" in d) == ("bottom" in d):
            raise InputError(f"{what}: depth（層厚）と bottom（下端の標高）のどちらか一方を指定してください")
        rows = sorted({j for (_, j) in pier.pile_sec})
        s = d["surface"]
        if isinstance(s, dict):
            surface = {}
            for row, v in s.items():
                if str(row) not in pier.iy:
                    raise InputError(f"{what}.surface: 未定義の Y通り（杭の列）です: {row!r}")
                surface[pier.iy[str(row)]] = _num(v, f"{what}.surface.{row}") * M
            surface = {j: surface.get(j, 0.0) for j in rows}
        else:
            surface = {j: _num(s, f"{what}.surface") * M for j in rows}
        depth = _num(d["depth"], f"{what}.depth") * M if "depth" in d else 0.0
        if "depth" in d and depth <= 0:
            raise InputError(f"{what}.depth: 正の値です")
        bottom = _num(d["bottom"], f"{what}.bottom") * M if "bottom" in d else None
        shape = str(d.get("shape", "cos"))
        if shape not in SHAPES:
            raise InputError(f"{what}.shape: {', '.join(SHAPES)} のいずれかです")
        order = str(d.get("order", "simultaneous"))
        if order not in ORDERS:
            raise InputError(f"{what}.order: {', '.join(ORDERS)} のいずれかです")
        fac = {}
        for k in ("kh_factor", "pu_factor"):
            fac[k] = _num(d.get(k, 1.0), f"{what}.{k}")
            if not 0 < fac[k] <= 1:
                raise InputError(f"{what}.{k}: 0 < 係数 ≦ 1 としてください")
        out.append(FlowCase(
            str(name), direction, surface, depth, shape, fac["kh_factor"], fac["pu_factor"],
            _num(d.get("kh", 0.0), f"{what}.kh"), order, bool(d.get("unload", True)),
            _num(d["step"], f"{what}.step") * M if d.get("step") is not None else None,
            _num(d["max_displacement"], f"{what}.max_displacement") * M
            if d.get("max_displacement") is not None else None, bottom))
    return out


@dataclass
class FlowResult:
    case: FlowCase
    inc: object                                       # pushover.Incremental（最終状態）
    W: float
    reached: bool = True                              # 打ち切り変位に達せず全段階を載荷できた
    phases: list = field(default_factory=list)        # [(段階名, 終了時の step)]
    deck: dict = field(default_factory=dict)          # peak, residual（上部工の変位 [mm]）、rot_residual（平面の回転 [rad]）
    hinge_max: dict = field(default_factory=dict)     # ばね -> 履歴中の最大 |θ|（塑性回転角）
    hinge_residual: dict = field(default_factory=dict)
    kinds: dict = field(default_factory=dict)         # ばね -> "杭頭" / "地中部"
    labels: dict = field(default_factory=dict)
    soil: dict = field(default_factory=dict)          # (節点, 自由度) -> EPSpring
    soil_force: dict = field(default_factory=dict)    # 節点 -> 地盤ばねの力 k·(u − ug)（載荷方向、最終状態）
    ground: dict = field(default_factory=dict)        # 節点 -> 与えた地盤変位 [mm]（最終状態）
    final: object = None                              # Result（最終状態の部材端力・変位）
    caps: dict = field(default_factory=dict)          # ばね -> PileCapacity（level2.pile_hinge.model: fiber）
    springs_used: dict = field(default_factory=dict)  # 節点 -> 解析に用いた地盤ばねの剛性（低減後、続けて載荷する用）
    gravity: list = field(default_factory=list)       # 先に載せた長期荷重のケース
    warnings: list = field(default_factory=list)

    @property
    def yielded(self):
        """全塑性に達したばねの [(ラベル, 種類)]"""
        return [(self.labels[k], self.kinds.get(k, "")) for k, h in self.inc.hinges.items() if h.state == "降伏"]

    def max_rotation(self, kind, residual=False):
        """種類（杭頭 / 地中部）ごとの最大の塑性回転角とラベル"""
        src = self.hinge_residual if residual else self.hinge_max
        vals = {k: v for k, v in src.items() if self.kinds.get(k) == kind}
        if not vals:
            return 0.0, ""
        k = max(vals, key=vals.get)
        return vals[k], self.labels[k]

    def damage(self, residual=False):
        """杭の損傷の一覧（port_level2.pile_damage）。residual: 最終状態、既定: 履歴中で最も厳しい状態
        （各ヒンジで |θ| が最大、θ = 0 のままなら |M| が最大のステップ）"""
        if not self.caps:
            return []
        h = self.inc.hinges
        if residual:
            return pile_damage(self.caps, h, self.labels, self.kinds)
        res = self.inc.res
        th, mm = {}, {}
        for k in self.caps:
            s = max(res.hinge_theta, key=lambda st: (abs(res.hinge_theta[st][k]), abs(res.hinge_M[st][k])))
            th[k], mm[k] = res.hinge_theta[s][k], res.hinge_M[s][k]
        return pile_damage(self.caps, h, self.labels, self.kinds, th, mm)

    def print(self):
        c = self.case
        print(f"\n■ 側方流動 {c.name}  方向 {c.direction}、分布 {c.shape}、{c.layer_text}、"
              f"慣性力 kh = {c.kh:g}（{c.order}）")
        print("  海底面の地盤変位: " + ", ".join(f"{v / M:g} m" for v in c.surface.values()))
        if not self.reached:
            print("  打ち切り変位に達したため、載荷を途中で終えました")
        dk = self.deck
        print(f"  上部工の変位: 最大 {dk['peak'] / M:.3f} m、残留 {dk['residual'] / M:.3f} m、"
              f"平面の残留回転 {dk['rot_residual']:.5f} rad")
        for kind in ("杭頭", "地中部"):
            v, lab = self.max_rotation(kind)
            r, _ = self.max_rotation(kind, residual=True)
            print(f"  {kind}の塑性回転角: 最大 {v:.4f} rad、残留 {r:.4f} rad（{lab or '-'}）")
        dm = self.damage()
        if dm:
            cnt = {lv: sum(r["level"] == lv for r in dm) for lv in ("局部座屈", "塑性化", "弾性")}
            print(f"  杭の損傷（最大時）: 局部座屈 {cnt['局部座屈']}、塑性化 {cnt['塑性化']}、弾性 {cnt['弾性']}"
                  f"（最大ひずみ {dm[0]['strain']:.5f}、{dm[0]['label']}）")
        ys = [s for s in self.soil.values() if s.yielded]
        if self.soil:
            print(f"  地盤ばねの降伏: {len(ys)} / {len(self.soil)}")
        for w in self.warnings:
            print(f"  [注意] {w}")


def run_flow(pier, case, results=None, hinge_opt=None):
    """側方流動のシナリオ case（FlowCase）を解析する

    hinge_opt: 杭のヒンジの設定（level2.pile_hinge と同じ形式）。省略時は pier.level2["pile_hinge"]
    地盤変位が 0 のシナリオ（慣性力のみ）は仮想固定点モデルでも解析できる（一括実行のふるい分け用）
    """
    if pier.pile_model != "soil_spring" and any(v != 0 for v in case.surface.values()):
        raise ValueError("側方流動の解析には地盤ばねモデル（piles.model: soil_spring）が必要です")
    if "地震用" not in pier.weights or pier.master() is None:
        raise ValueError("側方流動: 地震用重量（actions の D, Ls）と上部工の剛床が必要です")
    opt = pier.level2
    name, sign, d, _ = _parse_direction(case.direction)
    plane = 0 if d == 0 else 1
    step = case.step or opt["step"]
    dmax = case.max_displacement or opt["max_displacement"]
    m = pier.model
    res = dict(results or {})
    gcases = list(opt["gravity"])
    for c in gcases:
        if c not in res:
            res[c] = m.analyze(c)
    rG = Result.combine("+".join(gcases), {res[c]: 1.0 for c in gcases})
    # 地中部のヒンジは全ての分割点に置く（流動層の下端付近で全塑性になりうるため）
    caps = {}
    hinges, labels, kinds, warns = build_pile_hinges(pier, rG, plane, opt["post_yield_ratio"], math.inf,
                                                     hinge_opt or opt["pile_hinge"], caps)
    pattern, W = inertia_pattern(pier, d, sign * (1 if case.kh >= 0 else -1))
    saved = {nid: k.copy() for nid, k in m.node_springs.items()}
    ns, nlabels, ground = {}, {}, []
    for (i, j), p in pier.piles.items():
        tag = f"{pier.gx[i][0]}/{pier.gy[j][0]}"
        ug0 = case.surface.get(j, 0.0)
        H = case.layer(p["seabed"])
        for nid, z, k, cap in p["ground"]:
            zeta = p["seabed"] - z
            inside = zeta < H * (1 - 1e-9)
            if inside:                                            # 流動層の地盤ばねの低減
                m.node_springs[nid][:2] = saved[nid][:2] * case.kh_factor
            if cap:
                ns[(nid, d)] = EPSpring(m.node_springs[nid][d], cap * (case.pu_factor if inside else 1.0))
                nlabels[(nid, d)] = f"地盤 {tag} z={z / M:.2f}"
            ug = sign * ground_profile(case.shape, ug0, H, zeta) if H > 0 else 0.0
            if ug != 0:
                ground.append((nid, d, ug))
    k_lin = {nid: m.node_springs[nid][d] for nid in m.node_springs}       # 低減後の剛性（線形のばね）
    springs_used = {nid: k.copy() for nid, k in m.node_springs.items()}
    levels = [Level("上部工", None, pier.master(), 1.0)]                 # 「層間変形角」= 上部工の変位 [mm]
    inc = Incremental(m, hinges, levels, d, labels, mechanism_check=True, keep_states=True,
                      node_springs=ns, node_labels=nlabels, lateral_total=W)
    out = FlowResult(case, inc, W, kinds=kinds, labels=labels, soil=ns, caps=caps, warnings=list(warns),
                     springs_used=springs_used, gravity=gcases)
    kh = abs(case.kh)
    try:
        inc.run("長期", cases=gcases, lam_target=1.0)
        u0 = inc.u.copy()
        out.phases.append(("長期", inc.step))
        if case.order == "simultaneous" and kh > 0:
            todo = [("地盤変位+慣性力", dict(pattern=pattern, ground=[(n, f, v / kh) for n, f, v in ground],
                                        lam_target=kh, lateral=True))]
        else:
            g = ("地盤変位", dict(ground=ground, lam_target=1.0, checks=False))
            a = ("慣性力", dict(pattern=pattern, lam_target=kh, lateral=True))
            todo = [g] if kh == 0 else ([g, a] if case.order == "ground_first" else [a, g])
        for phase, kw in todo:
            if phase == "地盤変位" and not ground:
                continue
            # 打ち切り: 地盤変位を含む段階は 上部工の打ち切り変位 + 地盤変位の最大値（上部工は地盤に引きずられる）
            cap = dmax + (max((abs(v) for v in case.surface.values()), default=0.0) if "ground" in kw else 0.0)
            lam = inc.run(phase, max_drift=cap, step_drift=step, **kw)
            out.phases.append((phase, inc.step))
            if lam < kw["lam_target"] * (1 - 1e-9):
                out.reached = False
                out.warnings.append(f"{phase}: 上部工の変位が打ち切り値 {cap / M:g} m に達しました")
                break
        if case.unload and inc.lam_lat != 0:
            inc.unload(pattern, step_drift=step)
            out.phases.append(("除荷", inc.step))
        idx = inc.idx
        mi = 6 * idx[pier.master()]
        peak = max(abs(uu[mi + d] - u0[mi + d]) for uu, _ in inc.res.states.values())
        out.deck = {"peak": peak, "residual": (inc.u[mi + d] - u0[mi + d]) * sign,
                    "rot_residual": inc.u[mi + 5] - u0[mi + 5]}
        out.hinge_max = {k: max(abs(th[k]) for th in inc.res.hinge_theta.values()) for k in hinges}
        out.hinge_residual = {k: abs(h.theta) for k, h in hinges.items()}
        for nid in k_lin:
            rel = inc.u[6 * idx[nid] + d] - inc.ug[6 * idx[nid] + d]
            out.soil_force[nid] = ns[(nid, d)].F if (nid, d) in ns else k_lin[nid] * rel
            out.ground[nid] = inc.ug[6 * idx[nid] + d]
    finally:
        m.node_springs.update(saved)
        for e in m.elements.values():
            e.springs = None
    u, face, node = inc.u, inc.face, inc.node
    out.final = Result(m, f"側方流動 {case.name}", {nid: u[6 * k:6 * k + 6].copy() for nid, k in inc.idx.items()},
                       {eid: {"face": face[eid].copy(), "node": node[eid].copy()} for eid in face}, {},
                       {c: 1.0 for c in gcases})
    out.warnings += inc.res.warnings
    return out
