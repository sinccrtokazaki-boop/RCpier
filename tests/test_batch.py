"""利用可否 U4（一括実行）のテスト: 単独の計算との一致・再開・並列・剛性のキャッシュ"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import InputError, Model, Pier, run_flow, run_level2
from rcframe.materials import Concrete, RectSection
from rcframe.port_batch import Batch, read_scenarios, refine_reason, scenario_id, scenario_params, to_flow_case

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _data():
    """2 × 2 本の杭の小さな桟橋と、地盤変位 2 × 慣性力 2 × εcu 2 = 8 シナリオ"""
    d = yaml.safe_load((EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8"))
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0, "pu": 300},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0, "pu": 300}}
    d.pop("combinations")
    d.pop("lateral_flow")
    d.pop("usability")                                   # 判定・許容荷重（U5）は test_usability で確認
    d["actions"]["EQ"]["directions"] = ["y"]
    d["level2"]["step"] = 0.02
    d["scenarios"] = {
        "base": {"direction": "+y", "depth": 8.0, "shape": "cos", "kh_factor": 0.1,
                 "surface_ratio": {"L": 1.0, "S": 0.6}},
        "grid": {"surface": [0.0, 1.0], "kh": [0.0, 0.2], "ecu": ["A", "B"]},
        "refine": {"when": {"pile_yield": True}},
    }
    return d


@pytest.fixture(scope="module")
def serial():
    b = Batch(_data())
    return b, b.run(workers=1, log=lambda *a: None)


def _strip(rows):
    return sorted(({k: v for k, v in r.items() if k != "seconds"} for r in rows),
                  key=lambda r: (r["id"], r["tier"]))


# ============================================================ シナリオの生成
def test_scenario_generation_and_ids():
    """grid の組合せ（2·2·2 = 8）に base を重ねる。id は変数から決まり一意・再現可能。sample は seed で再現"""
    cfg = read_scenarios(_data()["scenarios"])
    ps = scenario_params(cfg)
    assert len(ps) == 8 and all(p["depth"] == 8.0 and p["kh_factor"] == 0.1 for p in ps)
    ids = [scenario_id(p) for p in ps]
    assert len(set(ids)) == 8 and ids == [scenario_id(p) for p in scenario_params(cfg)]
    cfg["sample"], cfg["seed"] = 3, 7
    a, b = scenario_params(cfg), scenario_params(cfg)
    assert len(a) == 3 and a == b
    cfg["seed"] = 8
    assert len(scenario_params(cfg)) == 3


def test_surface_ratio_to_flow_case():
    """surface（スカラー）× surface_ratio（列ごと）= 列ごとの海底面の地盤変位 [mm]"""
    p = Pier(_data())
    case, ecu = to_flow_case(p, {"surface": 0.5, "depth": 8.0, "surface_ratio": {"L": 1.0, "S": 0.6},
                                 "ecu": 0.02}, "x")
    assert case.surface == {p.iy["L"]: pytest.approx(500.0), p.iy["S"]: pytest.approx(300.0)}
    assert ecu == ("value", 0.02)


# ============================================================ 計算
def test_screen_matches_direct_run(serial):
    """1段目の行は、粗い分割（1 m）・cos 式のヒンジで run_flow を直接実行した結果と一致"""
    b, rows = serial
    d = copy.deepcopy(_data())
    d["piles"]["segment"] = 1.0
    p = Pier(d)
    sid, params = next((s, q) for s, q in b.scenarios if q["surface"] == 1.0 and q["kh"] == 0.2)
    case, _ = to_flow_case(p, params, sid)
    r = run_flow(p, case)
    row = next(x for x in rows if x["id"] == sid and x["tier"] == "screen")
    assert row["deck_residual"] == pytest.approx(r.deck["residual"] / 1e3, rel=1e-12)
    assert row["n_yield"] == len(r.yielded) > 0
    assert row["refine_reason"] == "杭の降伏"


def test_inertia_only_screen_uses_fixed_point(serial):
    """地盤変位 0 のシナリオの 1段目は仮想固定点モデル: 最大変位は P4（仮想固定点）の kh = 0.2 の要求点と一致"""
    b, rows = serial
    sid = next(s for s, q in b.scenarios if q["surface"] == 0.0 and q["kh"] == 0.2)
    row = next(x for x in rows if x["id"] == sid and x["tier"] == "screen")
    d = copy.deepcopy(_data())
    d["piles"]["model"] = "fixed_point"
    d["level2"].update(kh=0.2, criteria=[])
    l2 = run_level2(Pier(d), "+y")
    u0 = l2.po.gravity_disp[Pier(d).master()][1]
    assert row["deck_peak"] * 1e3 == pytest.approx(l2.demand["disp"] - u0, rel=1e-9)
    assert row["deck_residual"] == pytest.approx(0.0, abs=1e-9)          # 弾性なので除荷で戻る


def test_refine_and_dedup(serial):
    """杭が降伏した 1段目だけが 2段目（ファイバー）に回る。1段目は ecu によらず同じ結果を写す"""
    b, rows = serial
    scr = [r for r in rows if r["tier"] == "screen"]
    ref = [r for r in rows if r["tier"] == "refine"]
    assert len(scr) == 8
    assert {r["id"] for r in ref} == {r["id"] for r in scr if r["n_yield"] > 0} and ref
    assert all(r["n_buckled"] is not None for r in ref) and all(r["n_buckled"] is None for r in scr)
    by = {}
    for r in scr:
        key = (r["surface"], r["kh"])
        by.setdefault(key, []).append(r)
    for g in by.values():
        assert len(g) == 2 and g[0]["deck_residual"] == g[1]["deck_residual"] and g[0]["ecu"] != g[1]["ecu"]
    # 2段目は εcu の式で結果が変わる（式A の方が局部座屈が多い）
    ra = {(r["surface"], r["kh"]): r for r in ref if r["ecu"] == "A"}
    rb = {(r["surface"], r["kh"]): r for r in ref if r["ecu"] == "B"}
    for k in ra:
        assert ra[k]["n_buckled"] >= rb[k]["n_buckled"]


def test_refine_reason_rule():
    row = {"n_yield": 0, "deck_residual": 0.3, "reached": True}
    assert refine_reason(row, {"pile_yield": True}) == ""
    assert refine_reason(row, {"residual": [0.1, 0.5]}) == "残留変位"
    assert refine_reason({**row, "n_yield": 2}, {"pile_yield": True, "residual": [0.1, 0.5]}) == "杭の降伏・残留変位"
    assert refine_reason({**row, "reached": False}, {"pile_yield": True}) == "打ち切り"
    assert refine_reason(row, "all") == "all" and refine_reason(row, "none") == ""
    assert refine_reason({**row, "n_yield": 2}, {"pile_yield": True}, skip_above=0.3) == ""       # 0.3 m 以上は回さない
    assert refine_reason({**row, "n_yield": 2}, {"pile_yield": True}, skip_above=0.5) == "杭の降伏"


def test_resume_and_parallel(serial, tmp_path):
    """結果を逐次書き出し、再実行では計算済みを飛ばす。並列（2 プロセス）でも結果は同じ"""
    b, rows = serial
    out = tmp_path / "res"
    logs = []
    r1 = Batch(_data()).run(out, workers=2, log=logs.append)
    assert _strip(r1) == _strip(rows)
    assert (tmp_path / "res.csv").exists()
    n_lines = len((tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines())
    assert n_lines == len(rows)
    logs.clear()
    r2 = Batch(_data()).run(out, workers=2, log=logs.append)
    assert all("計算 0 件" in s for s in logs)
    assert _strip(r2) == _strip(rows)
    assert len((tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines()) == n_lines


@pytest.mark.parametrize("patch, msg", [
    (lambda s: s["grid"].update(foo=[1]), "foo"),
    (lambda s: s["grid"].update(kh=0.2), "リスト"),
    (lambda s: s.update(sample=0), "sample"),
    (lambda s: s.update(refine={"when": {"residual": [0.5, 0.1]}}), "residual"),
    (lambda s: s["grid"].update(shape=["sin"]), "shape"),
])
def test_scenario_input_errors(patch, msg):
    d = _data()
    patch(d["scenarios"])
    with pytest.raises(InputError, match=msg):
        Batch(d)


# ============================================================ 剛性のキャッシュ
def test_element_stiffness_cache_consistent():
    """ばねの柔性を A → B → A と変えても、キャッシュから戻した剛性は新しく計算した値と一致"""
    def elem(fs):
        m = Model()
        m.add_node(1, 0, 0, 0)
        m.add_node(2, 1000, 500, 3000)
        e = m.add_element(1, 1, 2, Concrete("Fc24", 24), RectSection("C", 600, 500), "column",
                          rigid_i=200.0, beta=15.0)
        e.springs = fs
        m.assemble_K()
        return m, e
    A = np.array([[1e-12, 0.0], [0.0, 3e-12]])
    B = np.array([[0.0, 2e-12], [5e-13, 0.0]])
    m, e = elem(A)
    for fs in (B, A, None, B):
        e.springs = fs
        m.assemble_K()
        _, fresh = elem(fs)
        np.testing.assert_array_equal(e.k_global, fresh.k_global)
        np.testing.assert_array_equal(e.k_face, fresh.k_face)
    m.nodes[2].x = 1500.0                                 # 形状が変われば座標変換も計算し直す
    m.assemble_K()
    m2, fresh = elem(B)
    m2.nodes[2].x = 1500.0
    m2.assemble_K()
    np.testing.assert_allclose(e.k_global, fresh.k_global, rtol=1e-14, atol=1e-6)
