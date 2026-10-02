"""層別の地盤ばね・法線方向の両端の張出し・流動層の下端の標高のテスト"""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import FlowCase, InputError, Pier, run_flow
from rcframe.port_flow import ground_profile

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _data():
    d = yaml.safe_load((EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8"))
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0}}
    for k in ("combinations", "lateral_flow", "scenarios", "usability"):
        d.pop(k, None)
    d["actions"]["EQ"]["directions"] = ["y"]
    return d


def test_layered_soil_springs():
    """層別: 節点の水平ばね k = kh(層)·D·負担長、上限 = pu(層)·D·負担長。層が 1つで同じ値なら元のモデルと一致"""
    d = _data()
    d["piles"]["rows"]["L"]["layers"] = [{"bottom": -10.0, "kh": 30000, "pu": 300},
                                         {"bottom": -15.0, "kh": 10000, "pu": 150},
                                         {"bottom": -99.0, "kh": 40000}]
    p = Pier(d)
    info = p.pile_info("B1", "L")
    D = p.pile_sec[(0, 0)][0].D
    for nid, z, k, cap in info["ground"]:
        lt = 250.0 if z in (info["ground"][0][1], info["ground"][-1][1]) else 500.0
        kh, pu = (30000, 300) if z > -10000 + 1e-6 else (10000, 150) if z > -15000 + 1e-6 else (40000, None)
        assert k == pytest.approx(kh * 1e-6 * D * lt)
        assert (cap is None) if pu is None else cap == pytest.approx(pu * 1e-3 * D * lt)
        np.testing.assert_allclose(p.model.node_springs[nid][:2], [k, k])
    # 1層で行の kh と同じ → 元のモデルと同じばね
    d2 = _data()
    d2["piles"]["rows"]["L"]["layers"] = [{"bottom": -99.0, "kh": 20000}]
    a, b = Pier(_data()), Pier(d2)
    for n, k in a.model.node_springs.items():
        np.testing.assert_array_equal(k, b.model.node_springs[n])


def test_layer_input_errors():
    d = _data()
    d["piles"]["rows"]["L"]["layers"] = [{"bottom": -10.0, "kh": 30000}, {"bottom": -5.0, "kh": 10000}]
    with pytest.raises(InputError, match="bottom"):
        Pier(d)
    d = _data()
    d["piles"]["rows"]["L"].pop("kh")
    d["piles"]["rows"]["L"]["layers"] = [{"bottom": -99.0, "kh": 25000}]
    p = Pier(d)                                                  # kh を省略すると最上層の kh で β を求める
    assert p.pile_rows[p.iy["L"]][1] == pytest.approx(25000e-6)


def test_end_overhang_weight():
    """法線方向の両端の張出し: 重量 = Σ q·面積、重心は張出しを含む上部工の図心（張出しが同じなら中央）"""
    d0 = _data()
    d1 = copy.deepcopy(d0)
    d1["deck"]["overhang"].update(start=2.2, end=2.2)
    p0, p1 = Pier(d0), Pier(d1)
    W0, x0, y0 = p0.weights["D"]
    W1, x1, y1 = p1.weights["D"]
    q = p1.gamma_rc * p1.slab_t + 2.0e-3                         # 床版 + 舗装 [N/mm²]
    assert W1 - W0 == pytest.approx(q * 2200 * 7000 * 2, rel=1e-9)
    assert x1 == pytest.approx(3000.0) and x0 == pytest.approx(3000.0)
    # 片側だけなら重心が張出し側へ寄る
    d2 = copy.deepcopy(d0)
    d2["deck"]["overhang"].update(end=2.2)
    W2, x2, _ = Pier(d2).weights["D"]
    assert x2 == pytest.approx((W0 * 3000.0 + q * 2200 * 7000 * (6000 + 1100)) / W2, rel=1e-9)
    # 鉛直荷重の合計 = 重量（荷重ケース D の支点反力・ばね反力の合計）
    r = p1.model.analyze("D")
    Rz = sum(v[2] for v in r.reactions.values()) + sum(v[2] for v in r.spring_forces.values())
    assert Rz == pytest.approx(W1, rel=1e-9)


def test_flow_bottom_elevation():
    """bottom（流動層の下端の標高）: 列ごとに H = 海底面 − bottom。海底面より上の bottom は流動なし"""
    d = _data()
    d["lateral_flow"] = {"F": {"surface": 0.3, "bottom": -10.0, "shape": "linear"}}
    p = Pier(d)
    c = p.lateral_flow[0]
    assert c.layer(-4000.0) == 6000.0 and c.layer(-8000.0) == 2000.0 and c.layer(-12000.0) == 0.0
    r = run_flow(p, c)
    for (i, j), info in p.piles.items():
        H = info["seabed"] + 10000.0                         # 海底面 − bottom
        for nid, z, _, _ in info["ground"]:
            ug = r.inc.ug[6 * r.inc.idx[nid] + 1]
            assert ug == pytest.approx(ground_profile("linear", 300.0, H, info["seabed"] - z), abs=1e-9)
    # 全列で海底面が同じなら depth = 海底面 − bottom と同じ
    d2 = _data()
    d2["piles"]["rows"]["S"]["seabed"] = -4.0
    d2["lateral_flow"] = {"A": {"surface": 0.3, "bottom": -10.0}, "B": {"surface": 0.3, "depth": 6.0}}
    p2 = Pier(d2)
    ra, rb = run_flow(p2, p2.lateral_flow[0]), run_flow(p2, p2.lateral_flow[1])
    np.testing.assert_allclose(ra.inc.u, rb.inc.u, rtol=0, atol=1e-9 * abs(rb.inc.u).max())
    with pytest.raises(InputError, match="どちらか一方"):
        d3 = _data()
        d3["lateral_flow"] = {"F": {"surface": 0.3, "bottom": -10.0, "depth": 5.0}}
        Pier(d3)
