"""変状連鎖の重点シナリオの確率評価の報告書（D5、rcframe/deterioration_report.py）のテスト:
小さな桟橋で常時・地震後の一括実行を行い、報告書の確率が結果の行の集計と一致し、外部参照がないことを確かめる"""
import copy
import re
from collections import Counter

import pytest
import yaml

from rcframe.deterioration_report import build_deterioration_report, load_study, probs, write_deterioration_report
from rcframe.port_batch import Batch
from rcframe.port_usability import STATUS

from test_deterioration import _data
from test_deterioration_model import _det


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    (root / "inputs").mkdir()
    (root / "results").mkdir()
    det = _det()
    det["chloride"]["icorr"] = 20.0                              # 欠損を大きくして判定に差を出す
    normal = _data()
    normal["name"] = "小さな桟橋（常時）"
    normal["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "surface": 0.0, "kh": 0.0},
                           "deterioration": det, "sample": 4, "seed": 3, "refine": {"when": "none"}}
    eq = _data()
    eq["name"] = "小さな桟橋（地震後）"
    eq["usability"]["limits"].update({"bridge_opening": {"limited": 0.1, "unusable": 0.2},
                                      "apron_step": {"limited": 0.05, "unusable": 0.2}})
    eq["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "kh_factor": 0.1,
                                "surface_ratio": {"L": 1.0, "S": 0.6}},
                       "distributions": {"surface": {"lognormal": {"median": 0.3, "beta": 0.7}},
                                         "kh": {"uniform": [0.0, 0.15]}, "wall_ratio": {"uniform": [0.5, 1.5]},
                                         "settlement_ratio": {"uniform": [0.2, 0.6]}},
                       "deterioration": {**copy.deepcopy(det), "age": [0, 60]}, "sample": 4, "seed": 3,
                       "refine": {"when": "none"}}
    for name, d in (("normal_all", normal), ("L1_all", eq)):
        (root / "inputs" / f"{name}.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        Batch(d).run(str(root / "results" / name), workers=1, log=lambda *a: None)
    return root


def test_load_study_kinds_and_labels(study):
    cases = load_study(study)
    assert [c["name"] for c in cases] == ["normal_all", "L1_all"]              # 常時 → 地震後
    n, e = cases
    assert not n["eq"] and n["scenarios"] == ["S1", "S2"] and n["label"] == "常時・S1＋S2"
    assert e["eq"] and e["scenarios"] == ["S1", "S2", "S3", "S5"] and e["ages"] == [0, 60]
    assert len(n["rows"]) == 8 and len(e["rows"]) == 8


def test_report_numbers_match_rows(study):
    """表の判定の確率（年数ごと）は結果の行の集計と一致する"""
    html = build_deterioration_report(study)
    for c in load_study(study):
        for a in c["ages"]:
            g = [r for r in c["rows"] if r["age"] == a]
            cnt = Counter(r["status"] for r in g)
            p = probs(g)
            for s in STATUS:
                assert p[s][0] == cnt[s] / len(g)
            cells = "".join(f"<td class=n>{p[s][0]:.1%}</td>" for s in STATUS)
            assert cells in html or "".join(f"<td class=n>{p[s][0]:.1%}±{p[s][1]:.1%}</td>" for s in STATUS) in html


def test_report_structure_and_no_external_refs(study, tmp_path):
    p = write_deterioration_report(study, tmp_path / "報告書.html")
    html = p.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and html.count("<h2>") == 9
    assert 'id="chain"' in html and "No.1328" in html                          # 変状連鎖図と出典
    assert not re.search(r'(src|href)="(https?:)?//', html)                    # 外部参照なし
    assert "<script src" not in html and "@import" not in html
    assert html.count("<section") == html.count("</section>")
