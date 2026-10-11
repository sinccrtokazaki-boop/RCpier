# -*- coding: utf-8 -*-
"""川崎港 東扇島 −12m 岸壁 桟橋部: ガイドラインに基づく利用可否の再計算（G4）

  python examples/kawasaki_guideline.py [-j 6] [ケース ...]

構造・地盤・劣化のモデル・地震と土留部の確率分布は D4（examples/kawasaki_deterioration.py の make_input）と同じで、
判定を「港湾施設の利用可否判断に係るガイドライン 第2版」（国土交通省港湾局、令和8年5月）に合わせる:
    利用方法（表 D-1）1-A・2-A・3-C、表 D-5 の条件（本震時の杭・上部工、余震 L1、利用時の荷重）、
    土留部の残留水平変位、渡版・段差は緊急復旧の注意事項。残留変位・塑性回転角の仮の閾値は使わない。
    地震後のケースは詳細化を省かない（残留変位が大きくても限界曲率・上部工を確かめる）
結果は kawasaki_guideline/results/<ケース>.jsonl・.csv、集計は summary.json、報告書は 報告書.html。
値のうち、想定船舶の牽引力・接岸力、余震の震度、土留部・渡版・段差の閾値は仮定（README の G4 を参照）
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
sys.path.insert(0, str(HERE))

import kawasaki_deterioration as kd  # noqa: E402
from rcframe.deterioration_report import write_deterioration_report  # noqa: E402
from rcframe.port_batch import Batch  # noqa: E402
from rcframe.port_usability import METHOD_SEP, STATUS, final_rows  # noqa: E402

OUT = HERE / "kawasaki_guideline"
CASES = kd.CASES

# 利用方法（ガイドライン 表 3-1・表 3-2・表 D-1・表 D-2）。接岸力・牽引力は 1,000〜3,000 GT 級（巡視船・貨物船）を想定（仮定）
METHODS = {
    "1-A": {"phase": 1, "label": "必要最小限の人員・物資の輸送（極めて慎重な作業、載荷重なし）",
            "surcharge": 0, "berthing": 500, "tractive": 350},
    "2-A": {"phase": 2, "label": "緊急物資の陸揚げ（地震時の載荷重）", "surcharge": 10, "berthing": 500, "tractive": 350},
    "3-C": {"phase": 3, "label": "災害前の日常的な利用（常時の載荷重）", "surcharge": 20, "berthing": 500, "tractive": 350},
}
AFTERSHOCK = 0.15                                 # 余震として想定する L1 の照査用震度（設計震度と同じと仮定）
EQ_LIMITS = {
    "wall_move": {"limited": 0.30, "unusable": 0.50},   # 土留部（控え直杭式矢板と仮定）: 表 A-1 の 30 cm、表 A-6 の d/H 5%（H ≈ 10 m）
    "apron_step": {"limited": 0.10},                    # 表 A-1 の段差 3〜10 cm → 超えれば緊急復旧（段差の解消）
    "bridge_opening": {"limited": 0.20},                # 渡版のかかり代 約 0.2 m（仮定）→ 超えれば渡版の仮設
}


def make_input(name):
    """ケースの入力（dict）: D4 の入力の判定の設定を置き換える"""
    d = kd.make_input(name)
    u = d["usability"]
    u.pop("required", None)
    u["methods"] = copy.deepcopy(METHODS)
    lim = u["limits"]
    for k in ("residual_displacement", "pile_rotation", "bridge_opening", "apron_step"):
        lim.pop(k, None)
    d["name"] = d["name"].rstrip("）") + "、ガイドライン第2版の判定）"
    if CASES[name]["eq"] is not None:
        u["required"] = {"aftershock": AFTERSHOCK}
        lim.update(copy.deepcopy(EQ_LIMITS))
        d["scenarios"]["refine"] = {"when": {"pile_yield": True, "residual": [0.05, 3.0]}}   # skip_above なし
    return d


def summarize(rows, methods):
    """供用年数ごとの判定・利用方法の確率と、不可の理由（「〜で不可」の対象と項目の最初の語）の出現数"""
    out = {}
    for age in sorted({r["age"] for r in rows}):
        g = [r for r in final_rows(rows) if r["age"] == age]
        n = len(g)
        st = Counter(r["status"] for r in g)
        why = Counter()
        for r in g:
            for x in (r.get("reasons") or "").split(" / "):
                who, _, body = x.partition(": ")
                if body and who != "使える利用方法":
                    why[f"{who}: {body.split(' ')[0]}"] += 1
        out[str(age)] = dict(n=n, p={s: st.get(s, 0) / n for s in STATUS},
                             methods={m: sum(m in (r.get("methods_ok") or "").split(METHOD_SEP) for r in g) / n
                                      for m in methods},
                             reasons=dict(why.most_common()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", default=list(CASES))
    ap.add_argument("-j", "--workers", type=int, default=6)
    a = ap.parse_args(argv)
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    sp = OUT / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for name in a.cases:
        d = make_input(name)
        (OUT / "inputs" / f"{name}.yaml").write_text(
            "# examples/kawasaki_guideline.py が作った入力（直接編集しない）\n"
            + yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        b = Batch(d)
        t0 = time.time()
        print(f"== {name}: シナリオ {len(b.scenarios)} 件", flush=True)
        rows = b.run(str(OUT / "results" / name), a.workers, log=lambda s: print("  " + s, flush=True))
        summary[name] = summarize(rows, METHODS)
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        for age, s in summary[name].items():
            print(f"  {age} 年: " + "、".join(f"{k} {v:.1%}" for k, v in s["p"].items())
                  + "；" + "、".join(f"{k} {v:.1%}" for k, v in s["methods"].items())
                  + f"（{s['n']} 件、{time.time() - t0:.0f} 秒）", flush=True)
    print("報告書 →", write_deterioration_report(OUT, OUT / "報告書.html"))


if __name__ == "__main__":
    main()
