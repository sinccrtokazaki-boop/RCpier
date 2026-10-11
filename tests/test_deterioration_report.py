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


# ============================================================ ガイドラインによる判定（G4）
@pytest.fixture(scope="module")
def study_g(tmp_path_factory):
    """利用方法（G3）・余震（G2）を使う小さな常時・地震後の一括実行"""
    root = tmp_path_factory.mktemp("study_g")
    (root / "inputs").mkdir()
    (root / "results").mkdir()
    det = _det()
    det["chloride"]["icorr"] = 20.0
    methods = {"1-A": {"phase": 1, "label": "必要最小限", "surcharge": 0, "berthing": 100, "tractive": 100},
               "3-C": {"phase": 3, "label": "日常", "surcharge": 10, "berthing": 300, "tractive": 300}}
    normal = _data()
    normal["name"] = "小さな桟橋（常時）"
    normal["usability"].pop("required")
    normal["usability"]["methods"] = copy.deepcopy(methods)
    normal["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "surface": 0.0, "kh": 0.0},
                           "deterioration": det, "sample": 4, "seed": 3, "refine": {"when": "none"}}
    eq = _data()
    eq["name"] = "小さな桟橋（地震後）"
    eq["usability"].pop("required")
    eq["usability"]["methods"] = copy.deepcopy(methods)
    eq["usability"]["required"] = {"aftershock": 0.1}
    eq["usability"]["limits"] = {"wall_move": {"limited": 0.2, "unusable": 0.4}, "apron_step": {"limited": 0.05},
                                 "bridge_opening": {"limited": 0.1}}
    eq["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "kh_factor": 0.1,
                                "surface_ratio": {"L": 1.0, "S": 0.6}},
                       "distributions": {"surface": {"lognormal": {"median": 0.3, "beta": 0.7}},
                                         "kh": {"uniform": [0.0, 0.15]}, "wall_ratio": {"uniform": [0.5, 1.5]},
                                         "settlement_ratio": {"uniform": [0.2, 0.6]}},
                       "deterioration": {**copy.deepcopy(det), "age": [0, 60]}, "sample": 6, "seed": 3,
                       "refine": {"when": {"pile_yield": True, "residual": [0.05, 3.0]}}}
    for name, d in (("normal_all", normal), ("L1_all", eq)):
        (root / "inputs" / f"{name}.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        Batch(d).run(str(root / "results" / name), workers=1, log=lambda *a: None)
    return root


def test_guideline_sections_and_numbers(study_g):
    from rcframe.deterioration_report import _pse, _res, _usable
    html = build_deterioration_report(study_g)
    assert html.count("<h2>") == 11 and "ガイドラインによる判定" in html and "残留水平変位による判断基準" in html
    assert "ガイドライン第2版" in html and "skip_above" not in html
    for c in load_study(study_g):
        for a in c["ages"]:
            g = [r for r in c["rows"] if r["age"] == a]
            for m in ("1-A", "3-C"):                                  # 利用方法ごとの使える割合 = 行の集計
                assert f"<td class=n>{_pse(sum(_usable(r, m) for r in g), len(g))}</td>" in html
            if c["eq"]:
                for m in ("1-A", "3-C"):                              # 閾値 A = 使えない抽出の残留水平変位の最小
                    bad = [_res(r) for r in g if not _usable(r, m)]
                    if bad:
                        assert f"{min(bad):.3f}（{len(bad)} 件）" in html


def test_yield_displacement_closed_form(study_g):
    """表 A-1 の目安: 腐食なしでは δy = fy·Z·L²/(6EI) = fy·L²/(3·E·D)（Z = 2I/D）"""
    from rcframe.deterioration_report import _yield_disp
    from rcframe.pier import Pier
    c = [c for c in load_study(study_g) if c["eq"]][0]
    p = Pier({k: v for k, v in c["data"].items() if k != "scenarios"})
    dy = _yield_disp(p, 0.0)
    for (i, j), info in p.piles.items():
        sec, mat = p.pile_sec[(i, j)]
        D = sec.dims(corroded=False)[0]
        L = p.z_soffit - (info["seabed"] - 1 / info["beta"])
        assert dy[p.gy[j][0]] == pytest.approx(mat.fy * L ** 2 / (3 * mat.E * D) / 1e3, rel=1e-12)
    d5 = _yield_disp(p, 5.0)                                          # 腐食で My が下がれば δy も小さい
    assert all(d5[k] < dy[k] for k in dy)
