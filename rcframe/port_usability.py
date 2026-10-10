"""被災後の桟橋の利用可否の判定・許容荷重・超過確率（利用可否 U5）

許容荷重: 側方流動・地震の解析（port_flow.run_flow）の最終状態（除荷後の被災状態）から、利用時の荷重を
    単位荷重（上載荷重 1 kN/m²、接岸力・牽引力 1 kN）の荷重係数として事象追跡型で増やし（pushover.Incremental）、
    最初に限界に達する荷重係数を許容荷重とする。事象の間は応答が荷重係数に比例するので、限界の値は
    ステップ内の線形補間で厳密に求まる。限界（usability.allowable で選択）:
        pile       : "plastic"（杭のヒンジの塑性回転が増える）/ "buckling"（局部座屈に達する、または
                     局部座屈したヒンジの回転が増える）/ "none"
        deck_mu    : 上部工の梁の曲げ |M| が Mu（示方書、材料係数 1.0）に達する
        pile_axial : 杭の軸力が設計支持力 Rcd（押込み）・Rtd（引抜き）に達する（piles.rows の入力値）
    接岸力は防舷材ごと、牽引力は係船柱ごとに求め、最小値をとる。地震時の長期荷重のうち remove の
    ケース（既定 Ls）は利用時の荷重に置き換えるため、先に取り除く
判定（usability.limits）: 上部工の残留変位、杭の残留の塑性回転角、杭の損傷の程度、許容荷重と必要な荷重
    から「利用可 / 制限付き利用 / 利用不可」を決め、最も厳しいものをとる
超過確率: 一括実行（port_batch、scenarios.distributions で変数を確率分布から抽出）の結果の行から、
    判定ごとの確率、許容荷重が必要な荷重を下回る確率（標準誤差つき）、地盤変位に対する
    フラジリティ曲線 P = Φ(ln(x/θ)/β)（最尤法）

判定基準・必要な荷重は入力（ガイドライン・利用計画で定めること）。単位: YAML は m・rad・kN/m²・kN
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .grid import InputError, _check_keys, _num
from .port_level2 import deck_moment, deck_sections
from .port_loads import WeightSum, mooring_force
from .results import Result

M, KN, KN_M2 = 1e3, 1e3, 1e-3
LOADS = ("surcharge", "berthing", "tractive")
LOAD_NAME = {"surcharge": "上載荷重", "berthing": "接岸力", "tractive": "牽引力"}
LOAD_UNIT = {"surcharge": "kN/m²", "berthing": "kN", "tractive": "kN"}
STATUS = ("利用可", "制限付き利用", "利用不可")
VERIFY_ITEMS = [
    "利用可否の判定基準（残留変位・杭の塑性回転角・損傷の程度の閾値）。計算例の値は仮の値",
    "許容荷重の限界（杭の塑性化・局部座屈、梁の Mu、杭の支持力）と、利用時の地盤ばねを地震時と同じとする扱い",
]


# ============================================================ 入力
def _levels(d, what):
    """{limited: 値, unusable: 値} → (制限付きの下限, 利用不可の下限)"""
    _check_keys(d, ("limited", "unusable"), what)
    lo = _num(d["limited"], f"{what}.limited") if "limited" in d else math.inf
    hi = _num(d["unusable"], f"{what}.unusable") if "unusable" in d else math.inf
    if lo > hi:
        raise InputError(f"{what}: limited ≦ unusable としてください")
    return lo, hi


def read_usability(pier, d):
    """YAML の usability を読む（なければ None）"""
    if not d:
        return None
    _check_keys(d, ("limits", "required", "allowable", "remove"), "usability")
    lim = d.get("limits") or {}
    _check_keys(lim, ("residual_displacement", "pile_rotation", "pile_damage"), "usability.limits")
    limits = {}
    if "residual_displacement" in lim:
        lo, hi = _levels(lim["residual_displacement"], "usability.limits.residual_displacement")
        limits["residual_displacement"] = (lo * M, hi * M)
    if "pile_rotation" in lim:
        limits["pile_rotation"] = _levels(lim["pile_rotation"], "usability.limits.pile_rotation")
    pd = lim.get("pile_damage", {"塑性化": "limited", "局部座屈": "unusable"})
    _check_keys(pd, ("塑性化", "局部座屈"), "usability.limits.pile_damage")
    for k, v in pd.items():
        if v not in ("ok", "limited", "unusable"):
            raise InputError(f"usability.limits.pile_damage.{k}: ok / limited / unusable です")
    limits["pile_damage"] = dict(pd)
    req = d.get("required") or {}
    _check_keys(req, LOADS, "usability.required")
    required = {k: _num(v, f"usability.required.{k}") for k, v in req.items()}
    al = d.get("allowable") or {}
    _check_keys(al, ("pile", "deck_mu", "pile_axial", "max", "fenders", "bollards", "loads"), "usability.allowable")
    pile = str(al.get("pile", "plastic"))
    if pile not in ("plastic", "buckling", "none"):
        raise InputError("usability.allowable.pile: plastic / buckling / none です")
    mx = {"surcharge": 200.0, "berthing": 20000.0, "tractive": 20000.0}
    _check_keys(al.get("max") or {}, LOADS, "usability.allowable.max")
    mx.update({k: _num(v, f"usability.allowable.max.{k}") for k, v in (al.get("max") or {}).items()})
    loads = [str(x) for x in al.get("loads", list(required) or list(LOADS))]
    for x in loads:
        if x not in LOADS:
            raise InputError(f"usability.allowable.loads: {', '.join(LOADS)} から選んでください: {x!r}")
    fenders = []
    for f in al.get("fenders") or []:
        what = f"usability.allowable.fenders {f}"
        _check_keys(f, ("x", "z", "friction"), what)
        fenders.append(([_num(f["x"], what) * M, None, _num(f.get("z", 0.0), what) * M],
                        _num(f.get("friction", 0.0), what)))
    bollards = []
    for b in al.get("bollards") or []:
        what = f"usability.allowable.bollards {b}"
        _check_keys(b, ("x", "angle_h", "angle_v"), what)
        bollards.append((_num(b["x"], what) * M, _num(b.get("angle_h", 0), what), _num(b.get("angle_v", 0), what)))
    if pier is not None:                                 # 作用位置が上部工の範囲内か
        xa, xb = pier.gx[0][1], pier.gx[-1][1]
        for x, what in [(P[0], "fenders") for P, _ in fenders] + [(b[0], "bollards") for b in bollards]:
            if not xa <= x <= xb:
                raise InputError(f"usability.allowable.{what}: x = {x / M:g} m が上部工の範囲外です")
    remove = [str(c) for c in d.get("remove", ["Ls"])]
    return dict(limits=limits, required=required, pile=pile, deck_mu=bool(al.get("deck_mu", True)),
                pile_axial=bool(al.get("pile_axial", True)), max=mx, loads=loads,
                fenders=fenders, bollards=bollards, remove=remove)


# ============================================================ 許容荷重
@dataclass
class Allowable:
    kind: str
    value: float                  # [kN/m²] または [kN]
    capped: bool = False          # 上限 max まで限界に達しなかった
    governing: str = ""           # 限界の種類
    where: str = ""               # 位置（部材・防舷材など）


def _unit_cases(pier, opts):
    """単位荷重のケースを作る。戻り値: {種類: [(ケース名, 位置の説明)]}"""
    m = pier.model
    y_face = pier.gy[-1][1] + pier.overhang["sea"]
    out = {}
    if "surcharge" in opts["loads"]:
        pier._area_load("_利用_q", 1.0 * KN_M2, WeightSum())
        out["surcharge"] = [("_利用_q", "全面")]
    if "berthing" in opts["loads"]:
        specs = opts["fenders"] or [(list(P), mu) for P, mu in pier.fender_specs]
        out["berthing"] = []
        for k, (P, mu) in enumerate(specs):
            P = [P[0], y_face, P[2] if P[2] is not None else pier.deck_top]
            c = f"_利用_B{k}"
            pier._point_load(c, P, [mu * KN, -KN, 0.0], c)
            out["berthing"].append((c, f"防舷材 x={P[0] / M:g} m"))
    if "tractive" in opts["loads"]:
        specs = opts["bollards"] or list(pier.bollard_specs)
        out["tractive"] = []
        for k, (x, ah, av) in enumerate(specs):
            c = f"_利用_T{k}"
            pier._point_load(c, [x, y_face, pier.deck_top], mooring_force(KN, ah, av), c)
            out["tractive"].append((c, f"係船柱 x={x / M:g} m"))
    return {k: v for k, v in out.items() if v}


def _first_crossing(la, lb, va, vb, limit):
    """線形に変わる値 v（la → lb で va → vb）が limit を越える最初の荷重係数（越えなければ None）"""
    if va > limit:
        return la
    if vb > limit and vb != va:
        return la + (limit - va) / (vb - va) * (lb - la)
    return None


def _check_segment(pier, la, lb, fa, fb, case, base, opts, secs, piles):
    """ステップ [la, lb] の中で最初に限界に達する (荷重係数, 限界の種類, 位置)（なければ None）"""
    best = None

    def take(l, gov, where):
        nonlocal best
        if l is not None and (best is None or l < best[0]):
            best = (l, gov, where)
    if secs:
        m = pier.model
        ra = Result(m, "a", {}, {e: {"face": f, "node": f} for e, f in fa.items()}, {}, {**base, case: la})
        rb = Result(m, "b", {}, {e: {"face": f, "node": f} for e, f in fb.items()}, {}, {**base, case: lb})
        for s in secs:
            Ma, Mb = deck_moment(ra, s), deck_moment(rb, s)
            take(_first_crossing(la, lb, Ma / s["Mu_pos"], Mb / s["Mu_pos"], 1.0), "梁の曲げ M = Mu", s["label"])
            take(_first_crossing(la, lb, -Ma / s["Mu_neg"], -Mb / s["Mu_neg"], 1.0), "梁の曲げ M = Mu", s["label"])
    for eid, Rcd, Rtd, label in piles:
        Ca, Cb = fa[eid][0], fb[eid][0]                       # 杭の下端（i端）の軸力（圧縮正）
        if Rcd:
            take(_first_crossing(la, lb, Ca / Rcd, Cb / Rcd, 1.0), "杭の押込み N = Rcd", label)
        if Rtd:
            take(_first_crossing(la, lb, -Ca / Rtd, -Cb / Rtd, 1.0), "杭の引抜き N = Rtd", label)
    return best


def _limit_run(inc, pier, case, lam_max, base, opts, secs, piles, labels):
    """単位荷重 case を荷重係数 0 → lam_max まで増やし、最初に限界に達する荷重係数を返す"""
    res = inc.res
    lam, la = 0.0, 0.0
    fa = {e: f.copy() for e, f in inc.face.items()}
    buckled = {k for k, h in inc.hinges.items() if any(getattr(h, "buckled", {}).values())}
    chunk = lam_max / 50                                      # 限界に達したら止めるため小分けに載荷
    while lam < lam_max * (1 - 1e-12):
        s0 = inc.step
        th0 = {k: h.theta for k, h in inc.hinges.items()}
        try:
            inc.run(f"利用 {case}", cases=[case], lam_target=min(chunk, lam_max - lam), checks=False)
        except RuntimeError:                                  # 不安定（機構）: その時点までで打ち切り
            return la, "構造が不安定", ""
        prev_th = th0
        for st in [s for s in res.steps if s["step"] > s0]:
            lb = lam + st["lam"]
            fb = res.states[st["step"]][1]
            th = res.hinge_theta[st["step"]]
            # 杭のヒンジ: このステップで回転角が変わった（塑性化）・局部座屈の事象
            if opts["pile"] != "none":
                moved = [k for k in th if abs(th[k] - prev_th[k]) > 1e-12 * (1 + abs(prev_th[k]))]
                evs = [e for e in st["events"] if e["event"] == "局部座屈"]
                if opts["pile"] == "plastic" and moved:
                    return la, "杭の塑性化", labels[moved[0]]
                if opts["pile"] == "buckling":
                    hit = [k for k in moved if k in buckled]
                    if hit:
                        return la, "局部座屈した杭の塑性化", labels[hit[0]]
                    if evs:
                        return lb, "杭の局部座屈", evs[0]["label"]
            b = _check_segment(pier, la, lb, fa, fb, case, base, opts, secs, piles)
            if b:
                return b
            la, fa, prev_th = lb, fb, th
        lam += chunk if lam + chunk < lam_max else lam_max - lam
    return None


def allowable_loads(pier, fr, opts):
    """被災状態（FlowResult fr の最終状態）からの許容荷重 {種類: Allowable}。fr の状態は元に戻す"""
    m, inc = pier.model, fr.inc
    saved = {nid: k.copy() for nid, k in m.node_springs.items()}
    snap0 = inc.snapshot()
    cases = {}
    out = {}
    try:
        m.node_springs.update({nid: k.copy() for nid, k in fr.springs_used.items()})
        removed = [c for c in opts["remove"] if c in fr.gravity]
        for c in removed:
            inc.run(f"除去 {c}", cases=[c], sign=-1.0, lam_target=1.0, checks=False)
        base = {c: 1.0 for c in fr.gravity if c not in removed}
        snap = inc.snapshot()
        cases = _unit_cases(pier, opts)
        secs = deck_sections(pier, opts.get("rebar_loss")) if opts["deck_mu"] else []
        piles = []
        if opts["pile_axial"]:
            for (i, j), p in pier.piles.items():
                cap = pier.pile_capacity.get(j)
                if cap:
                    eid = p["elements"][-1]                   # 杭の最下部の要素
                    piles.append((eid, cap[0], cap[1], f"杭 {pier.gx[i][0]}/{pier.gy[j][0]}"))
        for kind, items in cases.items():
            best = None
            for case, where in items:
                inc.restore(snap)
                lmax = opts["max"][kind]
                r = _limit_run(inc, pier, case, lmax, base, opts, secs, piles, fr.labels)
                a = Allowable(kind, lmax, True, "", where) if r is None else \
                    Allowable(kind, r[0], False, r[1], f"{where}・{r[2]}" if r[2] else where)
                if best is None or a.value < best.value:
                    best = a
            out[kind] = best
    finally:
        m.node_springs.update(saved)
        for e in m.elements.values():
            e.springs = None
        for items in cases.values():
            for case, _ in items:
                m.member_loads.pop(case, None)
                m.nodal_loads.pop(case, None)
        inc.restore(snap0)
    return out


# ============================================================ 判定
def judge(row, opts):
    """結果の行（port_batch の列、または usability_row の戻り値）の判定。戻り値: (判定, [理由])"""
    lv, reasons = 0, []

    def up(level, why):
        nonlocal lv
        if level > 0:
            reasons.append(f"{STATUS[level]}: {why}")
        lv = max(lv, level)
    lim = opts["limits"]
    if "residual_displacement" in lim:
        lo, hi = lim["residual_displacement"]
        v = abs(row["deck_residual"]) * M
        up(2 if v >= hi else 1 if v >= lo else 0, f"上部工の残留変位 {v / M:.3f} m")
    if "pile_rotation" in lim:
        lo, hi = lim["pile_rotation"]
        v = max(row.get("head_rot_res") or 0.0, row.get("inground_rot_res") or 0.0)
        up(2 if v >= hi else 1 if v >= lo else 0, f"杭の残留の塑性回転角 {v:.4f} rad")
    code = {"ok": 0, "limited": 1, "unusable": 2}
    pd = lim["pile_damage"]
    if row.get("n_buckled"):
        up(code[pd["局部座屈"]], f"杭の局部座屈 {row['n_buckled']} 箇所")
    n_pl = row.get("n_plastic") if row.get("n_plastic") is not None else row.get("n_yield", 0)
    if n_pl:
        up(code[pd["塑性化"]], f"杭の塑性化 {n_pl} 箇所")
    for k, req in opts["required"].items():
        a = row.get(f"allow_{k}")
        if a is None:
            continue
        if a <= 0:
            up(2, f"{LOAD_NAME[k]}を載せられない")
        elif a < req:
            up(1, f"{LOAD_NAME[k]}の許容値 {a:g} {LOAD_UNIT[k]} < 必要な値 {req:g}")
    return STATUS[lv], reasons


def usability_row(pier, fr, opts):
    """FlowResult の許容荷重と判定を結果の行の列にする"""
    al = allowable_loads(pier, fr, opts)
    row = {}
    for k in LOADS:
        a = al.get(k)
        row[f"allow_{k}"] = None if a is None else a.value              # 単位荷重の係数 = kN/m²・kN
        row[f"allow_{k}_by"] = "" if a is None else ("上限まで限界なし" if a.capped else f"{a.governing}（{a.where}）")
    return row


# ============================================================ 超過確率
def final_rows(rows):
    """シナリオごとに 2段目があれば 2段目、なければ 1段目の行"""
    by = {}
    for r in rows:
        if r["id"] not in by or r["tier"] == "refine":
            by[r["id"]] = r
    return list(by.values())


def _prob(flags):
    n = len(flags)
    p = sum(flags) / n if n else math.nan
    return p, math.sqrt(p * (1 - p) / n) if n else math.nan


def fit_fragility(x, y):
    """フラジリティ曲線 P(y = 1 | x) = Φ(ln(x/θ)/β) の最尤推定。戻り値: (θ, β)（推定できなければ None）
    x > 0 のデータのみ用いる。全て 0 または全て 1 なら推定できない"""
    from scipy.optimize import minimize
    from scipy.stats import norm
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = x > 0
    x, y = x[ok], y[ok]
    if len(x) < 2 or y.min() == y.max():
        return None
    lx = np.log(x)

    def nll(p):
        lt, lb = p
        z = (lx - lt) / math.exp(lb)
        return -np.sum(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z))
    p0 = [np.median(lx), math.log(max(np.std(lx), 0.1))]
    r = minimize(nll, p0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    return math.exp(r.x[0]), math.exp(r.x[1])


def print_summary(s, opts):
    """probability_summary の表示"""
    print(f"\n■ 利用可否の集計（{s['n']} シナリオ）")
    for k, (p, se) in s["status"].items():
        print(f"  {k:<8} {p:6.1%}（標準誤差 {se:.1%}）")
    for k, (p, se) in s["allow"].items():
        print(f"  {LOAD_NAME[k]}の許容値 < 必要な値 {opts['required'][k]:g} {LOAD_UNIT[k]}: {p:6.1%}（標準誤差 {se:.1%}）")
    for k, f in s["fragility"].items():
        print(f"  フラジリティ（{k}）: " + ("推定できません（全て同じ判定）" if f is None else
                                          f"中央値 {f[0]:.3f} m、対数標準偏差 {f[1]:.3f}"))


def probability_summary(rows, opts, x_key="surface"):
    """結果の行（usability の列つき）から確率の集計。戻り値: dict
        n、status: {判定: (確率, 標準誤差)}、allow: {荷重: (P(許容 < 必要), 標準誤差)}、
        fragility: {"制限付き利用以上" / "利用不可": (θ, β) or None}（x_key の値に対して）"""
    fr = final_rows(rows)
    out = {"n": len(fr), "status": {}, "allow": {}, "fragility": {}}
    st = [r["status"] for r in fr]
    for s in STATUS:
        out["status"][s] = _prob([x == s for x in st])
    for k, req in opts["required"].items():
        vals = [r.get(f"allow_{k}") for r in fr if r.get(f"allow_{k}") is not None]
        out["allow"][k] = _prob([v < req for v in vals])
    xs = [float(r[x_key]) if not isinstance(r[x_key], str) else math.nan for r in fr]
    out["fragility"]["制限付き利用以上"] = fit_fragility(xs, [s != STATUS[0] for s in st])
    out["fragility"]["利用不可"] = fit_fragility(xs, [s == STATUS[2] for s in st])
    return out
