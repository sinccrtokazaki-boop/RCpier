"""桟橋の利用可否シナリオの一括実行（利用可否 U4）

    シナリオの生成: YAML の scenarios（base の既定値 + grid の候補の組合せ、sample で無作為抽出）
    2段階の計算:
        1段目 ふるい分け（screen）: 地中部を粗く分割（screening.segment）、Mp(N) の cos 式のヒンジ。
              地盤変位 0（慣性力のみ）のシナリオは仮想固定点モデル
        2段目 詳細化（refine）: 入力の分割・ファイバーのヒンジ（εcu はシナリオの ecu）で計算し直す。
              対象は refine.when の条件（杭の降伏、上部工の残留変位の範囲）に当てはまるシナリオ
    並列実行（concurrent.futures.ProcessPoolExecutor、workers）。結果は 1シナリオ 1行の JSON Lines に
    逐次追記し、再実行時は計算済み（同じ id・段階）を飛ばす。CSV にも書き出す
    劣化の変数（DET_KEYS、変状連鎖の重点シナリオ）:
        rebar_loss_bottom / rebar_loss_top : 上部工の梁の主筋の断面欠損率（下端・上端、0〜1）→ 許容荷重の Mu
        corrosion_head / sea / soil        : 杭の外面の腐食量 [mm]（杭頭側・海中部・地中部）→ ヒンジの Mp・M–φ
        corrosion_boundary                 : 杭頭側の下端の標高 [m]（既定 −1.0 = L.W.L. − 1 m、L.W.L. ±0）
        scour（+ scour_ratio）             : 洗掘による海底面の低下 [m]（列ごと、またはスカラー × 列ごとの比）

使い方: python -m rcframe.port_batch 入力.yaml -o 出力の名前（.jsonl と .csv）[-j 並列数]
        [--cards フォルダ [--cards-all]]（判定カードと一覧 index.html、port_cards）
        [--report 報告書.html]（検討報告書、usability_report）
単位: YAML は m（地盤変位・層厚・残留変位）、結果の表も m・rad
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .grid import InputError, _check_keys, _num
from .port_flow import KEYS as FLOW_KEYS, read_flow_cases, run_flow
from .port_usability import judge, usability_row

M = 1e3
DET_KEYS = ("rebar_loss_bottom", "rebar_loss_top", "corrosion_head", "corrosion_sea", "corrosion_soil",
            "corrosion_boundary", "scour", "scour_ratio")
PARAMS = FLOW_KEYS + ("ecu", "surface_ratio") + DET_KEYS      # シナリオの変数（grid・base に書ける）
COLUMNS = ["id", "tier", "direction", "surface", "depth", "bottom", "shape", "kh_factor", "pu_factor", "kh", "order", "ecu",
           "rebar_loss_bottom", "rebar_loss_top", "corrosion_head", "corrosion_sea", "corrosion_soil", "scour",
           "reached", "deck_peak", "deck_residual", "deck_rot", "head_rot_max", "head_rot_res",
           "inground_rot_max", "inground_rot_res", "n_yield", "n_buckled", "n_plastic", "max_strain", "perforation",
           "soil_yield", "allow_surcharge", "allow_berthing", "allow_tractive", "status", "reasons",
           "allow_surcharge_by", "allow_berthing_by", "allow_tractive_by", "refine_reason", "seconds"]
DISTS = ("lognormal", "normal", "uniform", "choice")


# ============================================================ シナリオの生成
def read_scenarios(d):
    """scenarios の設定を読む。戻り値: dict(base, grid, sample, seed, screening, refine, workers)"""
    if not d:
        raise InputError("scenarios がありません")
    _check_keys(d, ("base", "grid", "distributions", "sample", "seed", "screening", "refine", "workers"),
                "scenarios")
    base = dict(d.get("base") or {})
    grid = dict(d.get("grid") or {})
    dists = dict(d.get("distributions") or {})
    for what, src in (("base", base), ("grid", grid), ("distributions", dists)):
        _check_keys(src, PARAMS, f"scenarios.{what}")
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            raise InputError(f"scenarios.grid.{k}: 候補をリストで与えてください")
    sample = d.get("sample")
    if sample is not None and (not isinstance(sample, int) or sample <= 0):
        raise InputError("scenarios.sample: 正の整数です")
    if dists:
        if grid:
            raise InputError("scenarios: grid と distributions はどちらか一方です")
        if sample is None:
            raise InputError("scenarios.distributions には sample（抽出する数）が必要です")
        for k, v in dists.items():
            _read_dist(v, f"scenarios.distributions.{k}")
    scr = dict(d.get("screening") or {})
    _check_keys(scr, ("segment",), "scenarios.screening")
    ref = d.get("refine", {"when": {"pile_yield": True}})
    if ref in ("all", "none"):
        ref = {"when": ref}
    _check_keys(ref, ("when", "segment"), "scenarios.refine")
    when = ref.get("when", {"pile_yield": True})
    if when not in ("all", "none"):
        _check_keys(when, ("pile_yield", "residual"), "scenarios.refine.when")
        if "residual" in when:
            lo, hi = (_num(x, "scenarios.refine.when.residual") for x in when["residual"])
            if not 0 <= lo < hi:
                raise InputError("scenarios.refine.when.residual: [下限, 上限]（m、下限 < 上限）です")
    return dict(base=base, grid=grid, dists=dists, sample=sample, seed=int(d.get("seed", 1)),
                screening=dict(segment=_num(scr.get("segment", 1.0), "scenarios.screening.segment")),
                refine=dict(when=when, segment=ref.get("segment")),
                workers=int(d.get("workers", 1)))


def _read_dist(v, what):
    """確率分布の指定 {lognormal: {median, beta}} / {normal: {mean, sd}} / {uniform: [a, b]} /
    {choice: [値...], p: [確率...]}（p は省略可）"""
    if not isinstance(v, dict) or not (set(v) & set(DISTS)):
        raise InputError(f"{what}: {', '.join(DISTS)} のいずれかで指定してください")
    _check_keys(v, DISTS + ("p",), what)
    kind = next(k for k in DISTS if k in v)
    a = v[kind]
    if kind == "lognormal":
        _check_keys(a, ("median", "beta"), what)
        if _num(a["median"], what) <= 0 or _num(a["beta"], what) <= 0:
            raise InputError(f"{what}: median, beta は正の値です")
    elif kind == "normal":
        _check_keys(a, ("mean", "sd"), what)
        if _num(a["sd"], what) <= 0:
            raise InputError(f"{what}: sd は正の値です")
    elif kind == "uniform":
        if not isinstance(a, list) or len(a) != 2 or _num(a[0], what) >= _num(a[1], what):
            raise InputError(f"{what}: uniform は [下限, 上限] です")
    else:
        if not isinstance(a, list) or not a:
            raise InputError(f"{what}: choice は候補のリストです")
        if "p" in v and (len(v["p"]) != len(a) or abs(sum(v["p"]) - 1) > 1e-9):
            raise InputError(f"{what}: p は候補と同じ数で、合計 1 です")
    return kind, a


def _draw(rng, v):
    kind, a = _read_dist(v, "")
    if kind == "lognormal":
        return float(a["median"] * math.exp(a["beta"] * rng.standard_normal()))
    if kind == "normal":
        return float(a["mean"] + a["sd"] * rng.standard_normal())
    if kind == "uniform":
        return float(rng.uniform(a[0], a[1]))
    i = rng.choice(len(a), p=v.get("p"))
    return a[int(i)]


def scenario_params(cfg):
    """シナリオの変数の [dict]。各 dict は base に grid の組合せ（sample があれば無作為に抽出）、
    または distributions から抽出した値（sample 個、seed で再現）を重ねたもの"""
    if cfg.get("dists"):
        rng = np.random.default_rng(cfg["seed"])
        keys = sorted(cfg["dists"])
        return [{**cfg["base"], **{k: _draw(rng, cfg["dists"][k]) for k in keys}} for _ in range(cfg["sample"])]
    keys = list(cfg["grid"])
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(cfg["grid"][k] for k in keys))]
    if cfg["sample"] is not None and cfg["sample"] < len(combos):
        combos = random.Random(cfg["seed"]).sample(combos, cfg["sample"])
    return [{**cfg["base"], **c} for c in combos]


def scenario_id(params):
    """シナリオの id（変数の値から決まる短いハッシュ。再実行で同じ id になる）"""
    s = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return "S" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def row_names(data):
    """入力 data の杭の列の名前"""
    return [str(k) for k in data["piles"]["rows"]]


def read_deterioration(params, rows):
    """シナリオの変数のうち劣化の変数（DET_KEYS）を読む。rows: 杭の列の名前
    戻り値: dict(rebar_loss=(下端, 上端) or None, corrosion={head, sea, soil [mm], boundary [mm]} or None,
                 scour={列の名前: 低下量 [m]} or None)"""
    p = params
    rl = None
    if "rebar_loss_bottom" in p or "rebar_loss_top" in p:
        rl = tuple(_num(p.get(k, 0.0), k) for k in ("rebar_loss_bottom", "rebar_loss_top"))
        if not all(0 <= v < 1 for v in rl):
            raise InputError("rebar_loss_bottom / rebar_loss_top: 断面欠損率は 0 以上 1 未満です")
    zones = {z: _num(p[f"corrosion_{z}"], f"corrosion_{z}") for z in ("head", "sea", "soil") if f"corrosion_{z}" in p}
    corr = None
    if zones:
        if any(v < 0 for v in zones.values()):
            raise InputError("corrosion_head / sea / soil: 腐食量は 0 以上です")
        corr = {**zones, "boundary": _num(p.get("corrosion_boundary", -1.0), "corrosion_boundary") * M}
    elif "corrosion_boundary" in p:
        raise InputError("corrosion_boundary は corrosion_head・corrosion_sea・corrosion_soil と一緒に指定します")
    sc = None
    s, ratio = p.get("scour"), p.get("scour_ratio")
    if s is not None:
        if isinstance(s, dict):
            sc = {str(k): _num(v, f"scour.{k}") for k, v in s.items()}
        elif ratio is not None:
            sc = {str(k): _num(s, "scour") * _num(v, f"scour_ratio.{k}") for k, v in ratio.items()}
        else:
            sc = {r: _num(s, "scour") for r in rows}
        bad = sorted(set(sc) - set(rows))
        if bad:
            raise InputError(f"scour: 杭の列にない名前です: {bad}")
        if any(v < 0 for v in sc.values()):
            raise InputError("scour: 海底面の低下量は 0 以上です")
    elif ratio is not None:
        raise InputError("scour_ratio は scour（スカラー）と一緒に指定します")
    return dict(rebar_loss=rl, corrosion=corr, scour=sc)


def apply_scour(data, scour):
    """杭の列の海底面（piles.rows.*.seabed）を scour [m] だけ下げた入力の複製"""
    d = copy.deepcopy(data)
    rows = d["piles"]["rows"]
    for k in rows:
        s = scour.get(str(k), 0.0)
        if s:
            row = dict(rows[k])
            row["seabed"] = _num(row["seabed"], f"piles.rows.{k}.seabed") - s
            if "tip" in row and row["seabed"] <= _num(row["tip"], f"piles.rows.{k}.tip"):
                raise InputError(f"scour: 列 {k} の海底面が杭先端より下になります")
            rows[k] = row
    return d


def to_flow_case(pier, params, name):
    """シナリオの変数（YAML の単位）を FlowCase と ecu の設定に変換する（劣化の変数は除く）。
    surface がスカラーで surface_ratio（列ごとの比）があれば、列ごとの値 = surface × 比"""
    p = {k: v for k, v in params.items() if k not in DET_KEYS}
    ecu = p.pop("ecu", "A")
    ratio = p.pop("surface_ratio", None)
    if ratio is not None and not isinstance(p.get("surface"), dict):
        p["surface"] = {r: _num(p.get("surface", 0.0), "surface") * _num(v, f"surface_ratio.{r}")
                        for r, v in ratio.items()}
    case = read_flow_cases(pier, {name: p})[0]
    e = str(ecu)
    ecu_opt = (e, None) if e in ("A", "B") else ("value", _num(ecu, "ecu"))
    return case, ecu_opt


# ============================================================ 1シナリオの計算（子プロセスでも動く）
_PIERS = {}
_PIERS_MAX = 6                    # キャッシュする Pier の数（洗掘のシナリオは海底面ごとに Pier が変わる）


def _pier(data, segment, model):
    """入力 data の杭の分割長・モデルを変えた Pier（プロセスごとにキャッシュ）"""
    from .pier import Pier
    key = (hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest(), segment, model)
    if key not in _PIERS:
        while len(_PIERS) >= _PIERS_MAX:
            _PIERS.pop(next(iter(_PIERS)))
        d = copy.deepcopy(data)
        d.pop("scenarios", None)
        d.pop("lateral_flow", None)
        d["piles"]["model"] = model
        if segment is not None:
            d["piles"]["segment"] = segment
        _PIERS[key] = Pier(d)
    return _PIERS[key]


def _summary(r, sid, tier, params, seconds):
    """FlowResult → 結果の表の 1行"""
    c = r.case
    head, head_r = r.max_rotation("杭頭"), r.max_rotation("杭頭", residual=True)
    ing, ing_r = r.max_rotation("地中部"), r.max_rotation("地中部", residual=True)
    dm = r.damage()
    surf = params.get("surface")
    sc = params.get("scour")
    return {
        "id": sid, "tier": tier, "direction": c.direction,
        "surface": surf if not isinstance(surf, dict) else json.dumps(surf, ensure_ascii=False),
        "depth": c.depth / M if c.bottom is None else None, "bottom": None if c.bottom is None else c.bottom / M,
        "shape": c.shape, "kh_factor": c.kh_factor, "pu_factor": c.pu_factor,
        "kh": c.kh, "order": c.order, "ecu": str(params.get("ecu", "A")),
        **{k: params.get(k) for k in ("rebar_loss_bottom", "rebar_loss_top", "corrosion_head", "corrosion_sea",
                                      "corrosion_soil")},
        "scour": sc if not isinstance(sc, dict) else json.dumps(sc, ensure_ascii=False),
        "reached": r.reached, "deck_peak": r.deck["peak"] / M, "deck_residual": r.deck["residual"] / M,
        "deck_rot": r.deck["rot_residual"], "head_rot_max": head[0], "head_rot_res": head_r[0],
        "inground_rot_max": ing[0], "inground_rot_res": ing_r[0], "n_yield": len(r.yielded),
        "n_buckled": sum(x["level"] == "局部座屈" for x in dm) if dm else None,
        "n_plastic": sum(x["level"] == "塑性化" for x in dm) if dm else None,
        "max_strain": max((x["strain"] for x in dm), default=None) if dm else None,
        "perforation": any("開孔" in w for w in r.warnings),
        "soil_yield": sum(s.yielded for s in r.soil.values()), "refine_reason": "",
        "seconds": round(seconds, 2),
    }


def analyze(data, sid, params, tier, segments):
    """1シナリオの解析。戻り値: (Pier, FlowResult)。tier = "screen"（粗い分割・cos 式）/ "refine"（入力の分割・ファイバー）
    劣化の変数: 洗掘は海底面を下げた入力で Pier を作り、杭の腐食はヒンジの設定（corrosion）で与える"""
    det = read_deterioration(params, row_names(data))
    if det["scour"]:
        data = apply_scour(data, det["scour"])
    ground = any(v != 0 for v in _surface_values(params))
    if tier == "screen":
        model = "soil_spring" if ground else "fixed_point"
        pier = _pier(data, segments["screen"] if ground else None, model)
        hinge = {"model": "cos", "ecu": ("A", None), "hardening": 0.01, "hinge_length": 1.0}
    else:
        pier = _pier(data, segments["refine"], "soil_spring")
        hinge = dict(pier.level2["pile_hinge"])
        hinge["model"] = "fiber"
    case, ecu = to_flow_case(pier, params, sid)
    if tier == "refine":
        hinge["ecu"] = ecu
    hinge["corrosion"] = det["corrosion"]
    return pier, run_flow(pier, case, hinge_opt=hinge)


def usability_opts(pier, params):
    """利用可否の判定・許容荷重の設定（pier.usability）に、シナリオの劣化の変数（梁の主筋の断面欠損）を加えたもの"""
    if not pier.usability:
        return None
    det = read_deterioration(params, [pier.gy[j][0] for j in pier.pile_rows])
    return {**pier.usability, "rebar_loss": det["rebar_loss"]}


def run_one(data, sid, params, tier, segments):
    """1シナリオの計算と結果の行（usability があれば許容荷重と判定を含む）"""
    t0 = time.time()
    pier, r = analyze(data, sid, params, tier, segments)
    row = _summary(r, sid, tier, params, time.time() - t0)
    if pier.usability:                                   # 許容荷重と判定（U5）
        opts = usability_opts(pier, params)
        row.update(usability_row(pier, r, opts))
        row["status"], reasons = judge(row, opts)
        row["reasons"] = " / ".join(reasons)
        row["seconds"] = round(time.time() - t0, 2)
    return row


def _surface_values(params):
    s = params.get("surface", 0.0)
    vals = list(s.values()) if isinstance(s, dict) else [s]
    ratio = params.get("surface_ratio")
    if ratio and not isinstance(s, dict):
        vals = [float(s) * float(v) for v in ratio.values()]
    return [float(v) for v in vals]


def refine_reason(row, when):
    """2段目に回す理由（空文字なら回さない）"""
    if when == "all":
        return "all"
    if when == "none":
        return ""
    reasons = []
    if when.get("pile_yield") and row["n_yield"] > 0:
        reasons.append("杭の降伏")
    if "residual" in when:
        lo, hi = when["residual"]
        if lo <= abs(row["deck_residual"]) <= hi:
            reasons.append("残留変位")
    if not row["reached"]:
        reasons.append("打ち切り")
    return "・".join(reasons)


# ============================================================ 一括実行
class Batch:
    """桟橋の入力（dict、scenarios を含む）からシナリオを作り、2段階で計算する"""

    def __init__(self, data):
        self.data = data
        self.cfg = read_scenarios(data.get("scenarios"))
        from .pier import Pier
        d = copy.deepcopy(data)
        d.pop("scenarios", None)
        self.pier = Pier(d)                                 # 入力の確認と列名の対応用
        self.scenarios = []
        for params in scenario_params(self.cfg):
            sid = scenario_id(params)
            to_flow_case(self.pier, params, sid)            # 入力の誤りをここで検出
            det = read_deterioration(params, row_names(data))
            if det["scour"]:
                apply_scour(data, det["scour"])
            self.scenarios.append((sid, params))
        seg = self.cfg["refine"]["segment"]
        self.segments = {"screen": self.cfg["screening"]["segment"],
                         "refine": seg if seg is not None else data["piles"].get("segment")}

    def run(self, out=None, workers=None, log=print):
        """全シナリオを計算する。out（拡張子なしのパス）があれば .jsonl に逐次追記し、.csv を書く。
        戻り値: 結果の行のリスト（1段目・2段目の両方）"""
        workers = workers or self.cfg["workers"]
        done = {}
        jl = Path(f"{out}.jsonl") if out else None
        if jl and jl.exists():
            for line in jl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[(row["id"], row["tier"])] = row
        rows = []

        def record(row):
            rows.append(row)
            if jl:
                with jl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        for tier in ("screen", "refine"):
            todo = []
            for sid, params in self.scenarios:
                if tier == "refine":
                    scr = next(r for r in rows if r["id"] == sid and r["tier"] == "screen")
                    if not scr["refine_reason"]:
                        continue
                if (sid, tier) in done:
                    rows.append(done[(sid, tier)])
                else:
                    todo.append((sid, params))
            log(f"{tier}: 計算 {len(todo)} 件（計算済み {sum(1 for k in done if k[1] == tier)} 件）")
            for row in self._map(todo, tier, workers):
                if tier == "screen":
                    row["refine_reason"] = refine_reason(row, self.cfg["refine"]["when"])
                record(row)
            # 計算済みの 1段目にも 2段目の判定を付け直す（条件を変えて再実行した場合）
            if tier == "screen":
                for r in rows:
                    if r["tier"] == "screen":
                        r["refine_reason"] = refine_reason(r, self.cfg["refine"]["when"])
        if out:
            try:
                write_csv(rows, f"{out}.csv")
            except PermissionError:                         # Excel などで開いている
                alt = f"{out}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                write_csv(rows, alt)
                log(f"[注意] {out}.csv を書き込めないため（他のプログラムで開いている）{alt} に書きました")
        return rows

    def _map(self, todo, tier, workers):
        """todo の各シナリオの結果の行を順に返す。1段目は ecu を使わないので、ecu 以外が同じシナリオは
        1回だけ計算して結果を写す"""
        groups = {}
        for sid, params in todo:
            key = scenario_id({k: v for k, v in params.items() if k != "ecu"}) if tier == "screen" else sid
            groups.setdefault(key, []).append((sid, params))
        reps = [g[0] for g in groups.values()]
        if workers <= 1 or len(reps) <= 1:
            results = (run_one(self.data, sid, params, tier, self.segments) for sid, params in reps)
        else:
            ex = ProcessPoolExecutor(max_workers=workers)
            futs = [ex.submit(run_one, self.data, sid, params, tier, self.segments) for sid, params in reps]
            results = (f.result() for f in futs)
        try:
            for g, row in zip(groups.values(), results):
                for sid, params in g:
                    yield {**row, "id": sid, "ecu": str(params.get("ecu", "A"))}
        finally:
            if workers > 1 and len(reps) > 1:
                ex.shutdown()


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="桟橋の利用可否シナリオの一括実行（利用可否 U4）")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default=None, help="出力の名前（.jsonl と .csv）。既定は入力名_scenarios")
    ap.add_argument("-j", "--workers", type=int, default=None)
    ap.add_argument("--cards", default=None, help="判定カードと一覧（index.html）を書き出すフォルダ")
    ap.add_argument("--cards-all", action="store_true", help="全シナリオのカードを作る（既定は利用不可・制限付き利用）")
    ap.add_argument("--report", default=None, help="検討報告書（HTML）の出力先")
    a = ap.parse_args(argv)
    data = yaml.safe_load(Path(a.input).read_text(encoding="utf-8"))
    out = a.out or str(Path(a.input).with_suffix("")) + "_scenarios"
    b = Batch(data)
    print(f"シナリオ {len(b.scenarios)} 件")
    t0 = time.time()
    rows = b.run(out, a.workers)
    print(f"完了: {len(rows)} 行、{time.time() - t0:.1f} 秒 → {out}.jsonl, {out}.csv")
    if b.pier.usability:
        from .port_usability import print_summary, probability_summary
        print_summary(probability_summary(rows, b.pier.usability), b.pier.usability)
    if a.cards:
        from .port_cards import write_cards
        which = "all" if a.cards_all or not b.pier.usability else ("制限付き利用", "利用不可")
        idx = write_cards(b, rows, a.cards, which, a.workers or b.cfg["workers"])
        print(f"判定カード → {idx}")
    if a.report:
        from .usability_report import write_usability_report
        p = write_usability_report(b, rows, a.report, Path(a.input).read_text(encoding="utf-8"))
        print(f"報告書 → {p}")


if __name__ == "__main__":
    main()
