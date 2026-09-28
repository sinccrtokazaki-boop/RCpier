"""桟橋編 P2（荷重の生成と組合せ）のテスト: 手計算との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import InputError, Pier, load_model
from rcframe.port_loads import lever_split, mooring_force, point_to_nodes

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
kN = 1e3


def _data():
    return yaml.safe_load((EXAMPLES / "pier_example.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pier():
    p = load_model(EXAMPLES / "pier_example.yaml")
    return p, p.analyze_cases()


def _vsum(res, case):
    """荷重の合計（反力の合計の逆符号）[N, N·mm]"""
    return -sum(res[case].reactions.values())


def _member_total(p, case, eid):
    """部材 eid に載る荷重ケース case の鉛直荷重の合計 [N]"""
    tot = 0.0
    for e, w, prof in p.model.member_loads[case]:
        if e != eid:
            continue
        L = p.model.elements[e].L if hasattr(p.model.elements[e], "L") else None
        if prof is None:
            n0, n1 = p.model.nodes[p.model.elements[e].ni], p.model.nodes[p.model.elements[e].nj]
            tot += -w[2] * math.dist(n0.xyz, n1.xyz)
        else:
            xs, fs = zip(*prof)
            tot += -w[2] * np.trapezoid(fs, xs)
    return tot


# ============================================================ 計算の部品
def test_lever_and_point_to_nodes():
    """位置 x の力を両端へてこの原理で配分、偏心 r = P − Q による r × F を加える"""
    assert lever_split(15, 12, 18) == pytest.approx((0.5, 0.5))
    assert lever_split(12, 12, 18) == (1.0, 0.0)
    with pytest.raises(ValueError):
        lever_split(20, 12, 18)
    # 防舷材: P = (12, 15.5, 0.5) m、梁軸 (12, 14, 2.25) m、F = (375, −1500, 0) kN
    f0, f1 = point_to_nodes([12, 15.5, 0.5], [375, -1500, 0], [12, 14, 2.25], [18, 14, 2.25])
    # r = (0, 1.5, −1.75): M = (ry·Fz − rz·Fy, rz·Fx − rx·Fz, rx·Fy − ry·Fx)
    assert f0 == pytest.approx([375, -1500, 0, -1.75 * 1500, -1.75 * 375, -1.5 * 375])
    assert f1 == pytest.approx([0] * 6)


def test_mooring_components():
    """P = 700, 水平角 30°, 仰角 15°: Tx = P cos15 sin30, Ty = P cos15 cos30, Tz = P sin15"""
    c, s = math.cos(math.radians(15)), math.sin(math.radians(15))
    assert mooring_force(700, 30, 15) == pytest.approx([700 * c * 0.5, 700 * c * math.sqrt(3) / 2, 700 * s])


# ============================================================ 固定荷重・上載荷重
def test_dead_load_total_and_center(pier):
    """D = (γ·t + 舗装)·(42×14 + 42×1.5) + γ·b·(D − t)·梁の総延長 238 m
    重心の y: 区画と梁は y = 7 m、張出し部は y = 14 + 1.5/2 m"""
    p, res = pier
    q = 24.5 * 0.40 + 2.0
    slab, over = q * 42 * 14, q * 42 * 1.5
    beams = 24.5 * 1.2 * (1.5 - 0.40) * (42 * 3 + 14 * 8)
    D = slab + over + beams
    assert _vsum(res, "D")[2] == pytest.approx(-D * kN, rel=1e-9)
    W, xg, yg = p.weights["D"]
    assert W == pytest.approx(D * kN, rel=1e-12)
    assert (xg, yg) == pytest.approx((21000, ((slab + beams) * 7 + over * 14.75) / D * 1e3), rel=1e-12)


def test_surcharge_total_and_distribution(pier):
    """L = 20 kN/m² × 651 m²。梁 1本の負担:
    中間の Y方向梁 B2/L-M（7 m、両側に 6×7 の区画）: 台形 h = 3 m ×2 側 → 面積 2×3×(7−3) = 24 m² → 480 kN
    海側の X方向梁 S/B1-B2（6 m、片側の区画）: 三角形 3×3 = 9 m² → 180 kN + 張出し 20×1.5×6 = 180 kN"""
    p, res = pier
    assert _vsum(res, "L")[2] == pytest.approx(-20 * 651 * kN, rel=1e-9)
    assert _member_total(p, "L", p.beam("B2", "L", "M")) == pytest.approx(480 * kN, rel=1e-12)
    assert _member_total(p, "L", p.beam("S", "B1", "B2")) == pytest.approx(360 * kN, rel=1e-12)
    assert _vsum(res, "Ls")[2] == pytest.approx(-10 * 651 * kN, rel=1e-9)


# ============================================================ 接岸力・牽引力
def test_berthing(pier):
    """防舷材 2基（x = 12, 30 m、R = 1500 kN、μ = 0.25）:
    合計 Fy = −3000 kN、Fx = 2×375 kN。x = 12 m は B3 の節点に全量、偏心モーメントは手計算どおり"""
    p, res = pier
    s = _vsum(res, "B")
    assert (s[0], s[1], s[2]) == pytest.approx((750 * kN, -3000 * kN, 0), abs=1e-3)
    loads = {n: F for n, F in p.model.nodal_loads["B"]}
    F = loads[p.node("B3", "S")]
    exp = np.array([375, -1500, 0, -1.75 * 1500, -1.75 * 375, -1.5 * 375]) * np.r_[[kN] * 3, [1e6] * 3]
    assert F == pytest.approx(exp, rel=1e-12)


def test_berthing_eccentric_twists_deck():
    """防舷材 1基を端寄り（x = 6 m）に置くと、上部工は +Z 軸まわりに回転（Mz = (6 − 21)·(−R) > 0）
    中間位置 x = 9 m は B2, B3 に半分ずつ"""
    d = _data()
    d["actions"]["B"] = {"fenders": [{"x": 9.0, "z": 0.5, "R": 1000}]}
    p = Pier(d)
    loads = dict(p.model.nodal_loads["B"])
    assert loads[p.node("B2", "S")][1] == pytest.approx(-500 * kN)
    assert loads[p.node("B3", "S")][1] == pytest.approx(-500 * kN)
    r = p.model.analyze("B")
    assert r.disp[p.master()][5] > 0


def test_mooring(pier):
    """係船柱 2基（水平角 ±30°、仰角 15°）: ΣTx = 0, ΣTy = 2·700 cos15 cos30, ΣTz = 2·700 sin15"""
    p, res = pier
    c, s = math.cos(math.radians(15)), math.sin(math.radians(15))
    t = _vsum(res, "T")
    assert (t[0], t[1], t[2]) == pytest.approx((0, 2 * 700 * c * math.sqrt(3) / 2 * kN, 2 * 700 * s * kN),
                                               abs=1e-3)


# ============================================================ レベル1地震動
def test_level1_seismic(pier):
    """EQ: 慣性力 = kh × (D + Ls)、重心に作用（剛床の図心とのずれ分の Mz）"""
    p, res = pier
    W = p.weights["D"][0] + p.weights["Ls"][0]
    xg, yg = p.weights["地震用"][1:]
    assert p.weights["地震用"][0] == pytest.approx(W)
    assert yg == pytest.approx((p.weights["D"][0] * p.weights["D"][2] + p.weights["Ls"][0] * 7750) / W)
    F = 0.15 * W
    assert _vsum(res, "EQx")[0] == pytest.approx(F, rel=1e-9)
    assert _vsum(res, "EQy")[1] == pytest.approx(F, rel=1e-9)
    mst = p.model.nodes[p.master()]
    ex = dict(p.model.nodal_loads["EQx"])[p.master()]
    assert ex[5] == pytest.approx(-(yg - mst.y) * F)          # 重心が海側 → X方向の地震でねじれ
    ey = dict(p.model.nodal_loads["EQy"])[p.master()]
    assert ey[5] == pytest.approx(0, abs=1e-6)                 # X方向は対称


def test_vertical_seismic():
    """kv > 0 なら EQz = kv × (D + Ls) の荷重（下向き）"""
    d = _data()
    d["actions"]["EQ"]["kv"] = 0.1
    p = Pier(d)
    r = {c: p.model.analyze(c) for c in ("D", "Ls", "EQz")}
    assert _vsum(r, "EQz")[2] == pytest.approx(0.1 * (_vsum(r, "D")[2] + _vsum(r, "Ls")[2]), rel=1e-9)


# ============================================================ 組合せ
def test_combinations_linear(pier):
    """組合せの結果 = Σ 係数 × 各ケースの結果"""
    p, res = pier
    cr = p.analyze_combinations(res)
    assert set(cr) == set(p.combinations)
    n = p.node("B5", "S")
    for name, dct in p.combinations.items():
        exp = sum(f * res[c].disp[n] for c, f in dct["factors"].items())
        np.testing.assert_allclose(cr[name].disp[n], exp, rtol=1e-12, atol=1e-12)
    assert p.combinations["L1地震_-Y"]["factors"]["EQy"] == -1.0
    assert cr["L1地震_+Y"].disp[p.master()][1] > 0 > cr["L1地震_-Y"].disp[p.master()][1]


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["actions"]["B"]["fenders"].append({"x": 50, "z": 0, "R": 1}), "範囲外"),
    (lambda d: d["actions"]["B"]["fenders"].append({"x": 5, "R": 1}), "z がありません"),
    (lambda d: d["beams"].append({"at": "B2/L-M", "section": "none"}), "区画"),
    (lambda d: (d["grids"].update(y={"S": 0}), d["piles"].update(rows={"S": {"seabed": -12, "kh": 15000}}),
                d["beams"].append({"at": "S/B1-B2", "section": "none"})), "張出し"),   # 区画なし・張出しのみ
    (lambda d: d["actions"].pop("D"), "actions.D が必要"),
    (lambda d: d["combinations"].update(X={"state": "永続", "factors": {"Q": 1}}), "荷重ケース"),
    (lambda d: d["combinations"].update(X={"state": "常時", "factors": {"D": 1}}), "state"),
    (lambda d: d["deck"].update(slab_thickness=5.0), "slab_thickness"),
    (lambda d: d["actions"]["EQ"].update(directions=["z"]), "directions"),
])
def test_load_input_errors(patch, msg):
    d = copy.deepcopy(_data())
    patch(d)
    with pytest.raises(InputError, match=msg):
        Pier(d)
