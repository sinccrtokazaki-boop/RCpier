"""劣化のモデル: 確率分布から供用年数ごとの劣化の変数を作る（変状連鎖の重点シナリオ D2）

YAML の scenarios.deterioration（sample 個を抽出。distributions と同時に使える）:
    age: [40, 60]                   供用年数 [年]（リストなら同じ抽出値で年数ごとのシナリオを作る）
    chloride:                       上部工の塩害 → rebar_loss_bottom（chloride_top があれば → rebar_loss_top）
        C0, D, cover, Clim, icorr   表面塩化物イオン濃度 [kg/m³]、見かけの拡散係数 [cm²/年]、かぶり [mm]（鉄筋の表面まで）、
                                    腐食発生限界濃度 [kg/m³]、腐食電流密度 [μA/cm²]
        Ci（既定 0）, bar           初期塩化物イオン濃度 [kg/m³]、鉄筋の径 [mm]（断面欠損率の計算）
    pile:                           杭の防食と腐食 → corrosion_head / sea / soil（boundary → corrosion_boundary）
        head / sea / soil: {rate, life（省略可）, factor（既定 0）}
                                    腐食速度 [mm/年]、防食の寿命 [年]、寿命までの腐食速度の比（電気防食 90% なら 0.1）
    scour:                          洗掘 → scour（ratio → scour_ratio）
        depth, tau（既定 10）, ratio 平衡の洗掘深さ [m]、時定数 [年]、列ごとの比（省略時は全ての列に depth）
各値は数値、または確率分布 {lognormal: {median, beta}} / {normal: {mean, sd}} / {uniform: [a, b]}

モデル:
    塩化物イオン（Fick の第 2 法則、表面濃度一定）: C(x, t) = Ci + (C0 − Ci)·erfc(x / (2√(D·t)))
        腐食の開始 t_i: C(cover, t_i) = Clim → t_i = x² / (4·D·z²)、z = erfc⁻¹((Clim − Ci)/(C0 − Ci))
        （Clim ≧ C0 なら腐食しない、Clim ≦ Ci なら t_i = 0）。示方書の照査式と同じ形（安全係数は掛けない）
    鉄筋の腐食（一様腐食）: 半径の減少 Δr = 0.0116·icorr·(t − t_i) [mm]（Faraday 則、1 μA/cm² ≒ 11.6 μm/年）
        断面欠損率 = 1 − ((bar − 2Δr)/bar)²（上限 0.99）。孔食・ひび割れによる加速は考えない
    杭の腐食: 腐食量 = rate·(factor·min(t, life) + max(t − life, 0))（life を省略すれば rate·t）
    洗掘: 深さ = depth·(1 − exp(−t/tau))（供用開始から平衡の深さに近づく）
抽出: 項目（chloride・chloride_top・pile・scour）ごとに別の乱数系列を使う（項目の組み合わせを変えても同じ抽出値）
単位: 年、mm、kg/m³、cm²/年、μA/cm²
"""
import math

import numpy as np
from scipy.special import erfc, erfcinv

from .grid import InputError, _check_keys, _num

FARADAY_MM = 0.0116                # 1 μA/cm² の腐食電流密度による鉄の侵食速度 [mm/年]
LOSS_MAX = 0.99                    # 鉄筋の断面欠損率の上限
CHLORIDE_KEYS = ("C0", "D", "cover", "Clim", "icorr", "Ci", "bar")
ZONES = ("head", "sea", "soil")
DISTS = ("lognormal", "normal", "uniform")


# ============================================================ モデル
def chloride(x, t, C0, D, Ci=0.0):
    """深さ x [mm]、時間 t [年] の塩化物イオン濃度 [kg/m³]（D: cm²/年）"""
    if t <= 0:
        return Ci
    return Ci + (C0 - Ci) * float(erfc((x / 10.0) / (2.0 * math.sqrt(D * t))))


def initiation_time(cover, C0, D, Clim, Ci=0.0):
    """鉄筋の位置（かぶり cover [mm]）で塩化物イオン濃度が Clim に達する時間 [年]（達しなければ inf）"""
    if Clim <= Ci:
        return 0.0
    if Clim >= C0:
        return math.inf
    z = float(erfcinv((Clim - Ci) / (C0 - Ci)))
    return (cover / 10.0) ** 2 / (4.0 * D * z * z)


def rebar_loss(t, t_i, icorr, bar):
    """腐食の開始 t_i からの一様腐食による鉄筋の断面欠損率（0〜LOSS_MAX）"""
    if not t > t_i:
        return 0.0
    dr = FARADAY_MM * icorr * (t - t_i)
    rem = max(bar - 2.0 * dr, 0.0) / bar
    return min(1.0 - rem * rem, LOSS_MAX)


def scour_depth(t, depth, tau):
    """供用年数 t の洗掘深さ [m]（平衡の深さ depth に時定数 tau で近づく）"""
    return depth * (1.0 - math.exp(-t / tau)) if t > 0 else 0.0


BLOCKS = ("chloride", "chloride_top", "pile", "scour")


def pile_loss(t, rate, life=None, factor=0.0):
    """防食の寿命 life [年] までは rate·factor、その後は rate で腐食するときの腐食量 [mm]"""
    if life is None:
        return rate * t
    return rate * (factor * min(t, life) + max(t - life, 0.0))


# ============================================================ 入力
def _read_value(v, what):
    """数値、または確率分布の指定"""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, dict) or len(set(v) & set(DISTS)) != 1:
        raise InputError(f"{what}: 数値、または {', '.join(DISTS)} のいずれかの確率分布で指定してください")
    _check_keys(v, DISTS, what)
    kind = next(k for k in DISTS if k in v)
    a = v[kind]
    if kind == "lognormal":
        _check_keys(a, ("median", "beta"), what)
        if _num(a.get("median"), what) <= 0 or _num(a.get("beta"), what) <= 0:
            raise InputError(f"{what}: median, beta は正の値です")
    elif kind == "normal":
        _check_keys(a, ("mean", "sd"), what)
        if _num(a.get("sd"), what) <= 0:
            raise InputError(f"{what}: sd は正の値です")
        _num(a.get("mean"), what)
    elif not isinstance(a, list) or len(a) != 2 or _num(a[0], what) >= _num(a[1], what):
        raise InputError(f"{what}: uniform は [下限, 上限] です")
    return dict(v)


def _draw(rng, v):
    if isinstance(v, float):
        return v
    if "lognormal" in v:
        a = v["lognormal"]
        return float(a["median"] * math.exp(a["beta"] * rng.standard_normal()))
    if "normal" in v:
        a = v["normal"]
        return float(a["mean"] + a["sd"] * rng.standard_normal())
    a = v["uniform"]
    return float(rng.uniform(a[0], a[1]))


def _read_chloride(d, what):
    _check_keys(d, CHLORIDE_KEYS, what)
    miss = [k for k in ("C0", "D", "cover", "Clim", "icorr", "bar") if k not in d]
    if miss:
        raise InputError(f"{what}: {', '.join(miss)} がありません")
    out = {k: _read_value(d[k], f"{what}.{k}") for k in CHLORIDE_KEYS if k in d}
    out.setdefault("Ci", 0.0)
    if not isinstance(out["bar"], float) or out["bar"] <= 0:
        raise InputError(f"{what}.bar: 鉄筋の径 [mm] を正の数値で指定してください")
    return out


def read_models(d):
    """scenarios.deterioration を読む（なければ None）。戻り値: dict(ages, chloride, chloride_top, pile, boundary)"""
    if not d:
        return None
    _check_keys(d, ("age",) + BLOCKS, "scenarios.deterioration")
    if "age" not in d:
        raise InputError("scenarios.deterioration: age（供用年数）がありません")
    ages = d["age"] if isinstance(d["age"], list) else [d["age"]]
    ages = [_num(a, "scenarios.deterioration.age") for a in ages]
    if not ages or any(a < 0 for a in ages) or len(set(ages)) != len(ages):
        raise InputError("scenarios.deterioration.age: 0 以上の重複しない年数です")
    out = dict(ages=ages, chloride=None, chloride_top=None, pile=None, boundary=None, scour=None)
    for k in ("chloride", "chloride_top"):
        if d.get(k):
            out[k] = _read_chloride(d[k], f"scenarios.deterioration.{k}")
    if d.get("pile"):
        p = d["pile"]
        what = "scenarios.deterioration.pile"
        _check_keys(p, ZONES + ("boundary",), what)
        zones = {}
        for z in ZONES:
            if z not in p:
                continue
            _check_keys(p[z], ("rate", "life", "factor"), f"{what}.{z}")
            if "rate" not in p[z]:
                raise InputError(f"{what}.{z}: rate（腐食速度 [mm/年]）がありません")
            zones[z] = {k: _read_value(p[z][k], f"{what}.{z}.{k}") for k in ("rate", "life") if k in p[z]}
            fac = _num(p[z].get("factor", 0.0), f"{what}.{z}.factor")
            if not 0 <= fac <= 1:
                raise InputError(f"{what}.{z}.factor: 0〜1 です")
            zones[z]["factor"] = fac
        if not zones:
            raise InputError(f"{what}: head / sea / soil のいずれかを指定してください")
        out["pile"] = zones
        out["boundary"] = _num(p.get("boundary", -1.0), f"{what}.boundary")
    if d.get("scour"):
        s, what = d["scour"], "scenarios.deterioration.scour"
        _check_keys(s, ("depth", "tau", "ratio"), what)
        if "depth" not in s:
            raise InputError(f"{what}: depth（平衡の洗掘深さ [m]）がありません")
        ratio = s.get("ratio")
        if ratio is not None and (not isinstance(ratio, dict) or not ratio):
            raise InputError(f"{what}.ratio: 列ごとの比 {{列: 比}} です")
        out["scour"] = dict(depth=_read_value(s["depth"], f"{what}.depth"), tau=_read_value(s.get("tau", 10.0), f"{what}.tau"),
                            ratio=None if ratio is None else {str(k): _num(v, f"{what}.ratio.{k}") for k, v in ratio.items()})
    if not any(out[k] for k in BLOCKS):
        raise InputError("scenarios.deterioration: chloride / chloride_top / pile / scour のいずれかを指定してください")
    return out


def made_keys(cfg):
    """deterioration から作るシナリオの変数の名前（base・distributions と重ねられない）"""
    keys = (["rebar_loss_bottom"] if cfg["chloride"] else []) + (["rebar_loss_top"] if cfg["chloride_top"] else [])
    if cfg["pile"]:
        keys += [f"corrosion_{z}" for z in cfg["pile"]] + ["corrosion_boundary"]
    if cfg["scour"]:
        keys += ["scour", "scour_ratio"]
    return keys


# ============================================================ 抽出と変数
def rngs(seed):
    """項目ごとの乱数系列"""
    return {b: np.random.default_rng([seed, 2, k]) for k, b in enumerate(BLOCKS)}


def draw(rng, cfg):
    """物理量を 1 組抽出する。rng: 項目ごとの乱数系列（rngs）。戻り値: {名前: 値}（chloride.C0、pile.head.rate など）"""
    s = {}
    for k in ("chloride", "chloride_top"):
        if cfg[k]:
            for n in CHLORIDE_KEYS:
                s[f"{k}.{n}"] = _draw(rng[k], cfg[k][n])
    for z, v in (cfg["pile"] or {}).items():
        for n in ("rate", "life"):
            if n in v:
                s[f"pile.{z}.{n}"] = _draw(rng["pile"], v[n])
    if cfg["scour"]:
        for n in ("depth", "tau"):
            s[f"scour.{n}"] = _draw(rng["scour"], cfg["scour"][n])
    return s


def params_at(sample, age, cfg):
    """抽出した物理量 sample と供用年数 age から劣化の変数（port_batch.DET_KEYS）と途中の値を求める。
    戻り値: (変数 dict, 途中の値 dict)。負になった値（正規分布の裾）は小さな正の値に丸める"""
    out, info = {}, {}
    for k, key in (("chloride", "rebar_loss_bottom"), ("chloride_top", "rebar_loss_top")):
        if not cfg[k]:
            continue
        g = {n: sample[f"{k}.{n}"] for n in CHLORIDE_KEYS}
        cover = max(g["cover"], 5.0)
        ti = initiation_time(cover, max(g["C0"], 1e-9), max(g["D"], 1e-9), max(g["Clim"], 1e-9), max(g["Ci"], 0.0))
        out[key] = rebar_loss(age, ti, max(g["icorr"], 0.0), g["bar"])
        info[f"{k}.t_init"] = ti
    if cfg["pile"]:
        for z, v in cfg["pile"].items():
            rate = max(sample[f"pile.{z}.rate"], 0.0)
            life = sample.get(f"pile.{z}.life")
            out[f"corrosion_{z}"] = pile_loss(age, rate, None if life is None else max(life, 0.0), v["factor"])
        out["corrosion_boundary"] = cfg["boundary"]
    if cfg["scour"]:
        out["scour"] = scour_depth(age, max(sample["scour.depth"], 0.0), max(sample["scour.tau"], 1e-9))
        if cfg["scour"]["ratio"]:
            out["scour_ratio"] = dict(cfg["scour"]["ratio"])
    return out, info


def scenario_params(cfg_det, base, seismic, rng):
    """deterioration の抽出 1 組から、年数ごとのシナリオの変数のリストを作る
    （base・地震の変数 seismic は年数によらず同じ。age と途中の値 _det を加える）"""
    sample = draw(rng, cfg_det)
    out = []
    for age in cfg_det["ages"]:
        p, info = params_at(sample, age, cfg_det)
        det = {**sample, **info}
        out.append({**base, **seismic, **p, "age": age,
                    "_det": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in det.items()}})
    return out
