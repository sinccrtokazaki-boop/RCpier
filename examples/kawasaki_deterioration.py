# -*- coding: utf-8 -*-
"""川崎港 東扇島 −12m 岸壁 桟橋部: 変状連鎖の重点シナリオの確率評価（D4）

  python examples/kawasaki_deterioration.py [-j 4] [ケース ...]

構造・地盤（仮定を含む）は examples/kawasaki_probability_L1.yaml。これに劣化のモデル（scenarios.deterioration）と
地震・土留部の確率分布を加えたケースを作り、kawasaki_deterioration/inputs/ に YAML を書いて一括実行する。
結果は kawasaki_deterioration/results/<ケース>.jsonl・.csv（再実行すると計算済みを飛ばす）、集計は summary.json、
報告書は 報告書.html（rcframe/deterioration_report.py）。

ケース（各 200 回の抽出。劣化の抽出値は項目ごとに共通、地震の抽出値は L1・L2 の中で共通）:
    normal_S1 / S2 / S4 / all : 常時（地震なし）。S1 塩害、S2 杭の防食工と腐食、S4 洗掘、all はその全部。供用 40・60 年
    L1_all / L2_all          : 地震後。劣化の全部 + 地震（S3）+ 土留部（S5）。供用 0（劣化なし）・40・60 年
確率分布の値は基準類の標準値と仮定（変状連鎖図の重点シナリオの検討で提案した値）。判定基準・必要な荷重も仮の値
"""
import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rcframe.deterioration_report import write_deterioration_report  # noqa: E402
from rcframe.port_batch import Batch  # noqa: E402
from rcframe.port_usability import STATUS, final_rows  # noqa: E402

BASE = HERE / "kawasaki_probability_L1.yaml"
OUT = HERE / "kawasaki_deterioration"

# ---- 劣化のモデル（S1・S2・S4）
CHLORIDE = {                                      # S1 上部工の塩害 → 梁の下端筋の断面欠損率
    "C0": {"lognormal": {"median": 13.0, "beta": 0.3}},     # 表面塩化物イオン濃度 [kg/m³]（示方書の飛沫帯）
    "D": {"lognormal": {"median": 1.0, "beta": 0.5}},       # 見かけの拡散係数 [cm²/年]（W/C 0.55 で約 0.7、ひび割れを見込む）
    "cover": {"normal": {"mean": 100.0, "sd": 10.0}},       # かぶり [mm]（入力の 100 mm = 主筋の表面まで）
    "Clim": {"lognormal": {"median": 1.6, "beta": 0.25}},   # 腐食発生限界濃度 [kg/m³]
    "icorr": {"lognormal": {"median": 2.0, "beta": 0.7}},   # 腐食電流密度 [μA/cm²]
    "bar": 22,                                              # 下端筋の径 [mm]（中央 D22、端部 D25 → 小さい方）
}
PILE = {                                          # S2 鋼管杭の防食工の劣化と腐食（港湾基準の腐食速度の標準値）
    "boundary": -1.0,                                       # 杭頭側の下端 = L.W.L. − 1 m（L.W.L. ±0 と仮定）
    "head": {"life": {"lognormal": {"median": 30.0, "beta": 0.5}},     # モルタルライニングの寿命 [年]
             "rate": {"lognormal": {"median": 0.2, "beta": 0.4}}},     # その後の腐食速度 [mm/年]（0.1〜0.3）
    "sea": {"life": {"lognormal": {"median": 30.0, "beta": 0.5}},      # 電気防食（陽極）の寿命 [年]
            "rate": {"lognormal": {"median": 0.15, "beta": 0.4}},      # 腐食速度 [mm/年]（0.1〜0.2）
            "factor": 0.1},                                            # 寿命までは防食率 90%
    "soil": {"rate": {"uniform": [0.02, 0.04]}},                       # 地中部（0.03）
}
SCOUR = {"depth": {"lognormal": {"median": 0.5, "beta": 0.8}},       # S4 平衡の洗掘深さ [m]（海側の列 S）
         "tau": 10.0, "ratio": {"S": 1.0, "C1": 0.5}}                 # 時定数 [年]、列 C1 は半分

# ---- 地震（S3）と土留部（S5）
SEISMIC = {
    "surface_ratio": {"choice": [{"L": 1.0, "C2": 1.0, "C1": 1.0, "S": 1.0}, {"L": 1.0, "C2": 0.8, "C1": 0.6, "S": 0.4}]},
    "kh_factor": {"uniform": [0.05, 0.2]}, "pu_factor": {"choice": [0.25, 0.5, 1.0]}, "kh": {"uniform": [0.0, 0.15]},
}
SURFACE = {"L1": 0.05, "L2": 0.5}                 # 陸側の列の海底面の地盤変位の中央値 [m]（β 0.7）
RETAIN = {"wall_ratio": {"uniform": [0.5, 1.5]},             # 土留部の移動 / 陸側の列の地盤変位
          "settlement_ratio": {"uniform": [0.2, 0.6]}}       # 背後の沈下 / 土留部の移動
RETAIN_LIMITS = {"bridge_opening": {"limited": 0.1, "unusable": 0.2},   # 渡版のかかり代 約 0.2 m（図64）
                 "apron_step": {"limited": 0.05, "unusable": 0.2}}     # 段差（仮の値）

CASES = {
    "normal_S1": dict(eq=None, blocks=["chloride"]),
    "normal_S2": dict(eq=None, blocks=["pile"]),
    "normal_S4": dict(eq=None, blocks=["scour"]),
    "normal_all": dict(eq=None, blocks=["chloride", "pile", "scour"]),
    "L1_all": dict(eq="L1", blocks=["chloride", "pile", "scour"]),
    "L2_all": dict(eq="L2", blocks=["chloride", "pile", "scour"]),
}
SAMPLE, SEED = 200, 1


def make_input(name):
    """ケースの入力（dict）"""
    c = CASES[name]
    d = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    for s in ("P12", "P16"):                      # 杭の腐食は劣化のモデルだけで与える（0 年 = 建設時）
        d["sections"][s]["corrosion"] = {"outer": 0.0, "inner": 0.0}
    det = {"age": [40, 60] if c["eq"] is None else [0, 40, 60]}
    det.update({b: copy.deepcopy({"chloride": CHLORIDE, "pile": PILE, "scour": SCOUR}[b]) for b in c["blocks"]})
    if c["eq"] is None:
        d["name"] = f"川崎港東扇島 −12m 岸壁 桟橋部（劣化 {name}、常時）"
        d["scenarios"] = {"base": {"direction": "+y", "shape": "cos", "order": "simultaneous", "surface": 0.0, "kh": 0.0,
                                   "bottom": -23.0, "ecu": "B"},           # bottom は地盤変位 0 なので効かない
                          "deterioration": det, "sample": SAMPLE, "seed": SEED,
                          "screening": {"segment": 1.0}, "refine": {"when": "all"}, "workers": 4}
        d["usability"]["required"] = {"surcharge": 20, "berthing": 500, "tractive": 350}   # 常時の上載荷重 20 kN/m²
    else:
        d["name"] = f"川崎港東扇島 −12m 岸壁 桟橋部（劣化と地震 {name}）"
        dists = {**copy.deepcopy(SEISMIC), "surface": {"lognormal": {"median": SURFACE[c["eq"]], "beta": 0.7}},
                 **copy.deepcopy(RETAIN)}
        d["scenarios"] = {"base": {"direction": "+y", "shape": "cos", "order": "simultaneous", "bottom": -23.0, "ecu": "B"},
                          "distributions": dists, "deterioration": det, "sample": SAMPLE, "seed": SEED,
                          "screening": {"segment": 1.0},
                          # 1段目の残留変位 0.5 m 以上は利用不可（閾値 0.2 m）に決まるので詳細化しない
                          "refine": {"when": {"pile_yield": True, "residual": [0.05, 3.0]}, "skip_above": 0.5},
                          "workers": 4}
        d["usability"]["limits"].update(copy.deepcopy(RETAIN_LIMITS))
        d["usability"]["retaining"] = {"row": "L"}
    return d


def summarize(rows):
    """供用年数ごとの判定の確率と、判定を決めた理由（利用不可・利用可（注意事項あり）の理由の種類の出現数）"""
    out = {}
    for age in sorted({r["age"] for r in rows}):
        g = [r for r in final_rows(rows) if r["age"] == age]
        n = len(g)
        st = Counter(r["status"] for r in g)
        why = Counter()
        for r in g:
            for x in (r.get("reasons") or "").split(" / "):
                if x:
                    level, _, body = x.partition(": ")
                    why[(level, body.split(" ")[0])] += 1
        out[str(age)] = dict(n=n, p={s: st.get(s, 0) / n for s in STATUS},
                             reasons={f"{a}: {b}": k for (a, b), k in why.most_common()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", default=list(CASES))
    ap.add_argument("-j", "--workers", type=int, default=4)
    a = ap.parse_args(argv)
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    sp = OUT / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for name in a.cases:
        d = make_input(name)
        (OUT / "inputs" / f"{name}.yaml").write_text(
            "# examples/kawasaki_deterioration.py が作った入力（直接編集しない）\n"
            + yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        b = Batch(d)
        t0 = time.time()
        print(f"== {name}: シナリオ {len(b.scenarios)} 件", flush=True)
        rows = b.run(str(OUT / "results" / name), a.workers, log=lambda s: print("  " + s, flush=True))
        summary[name] = summarize(rows)
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        for age, s in summary[name].items():
            print(f"  {age} 年: " + "、".join(f"{k} {v:.1%}" for k, v in s["p"].items())
                  + f"（{s['n']} 件、{time.time() - t0:.0f} 秒）", flush=True)
    print("報告書 →", write_deterioration_report(OUT, OUT / "報告書.html"))


if __name__ == "__main__":
    main()
