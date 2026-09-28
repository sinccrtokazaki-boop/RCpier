"""荷重の自動生成（フェーズ3）のテスト: 理論解・手計算との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from rcframe import Building, Concrete, InputError, Model, RectSection, load_building
from rcframe.element import distributed_equivalent_loads, udl_equivalent_loads
from rcframe.loads import tributary_profiles

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FC24 = Concrete("Fc24", 24)
SEC = RectSection("G", 400, 700)


def _stiff(sec=SEC, mat=FC24, shear=True):
    return dict(EIy=mat.E * sec.Iy, EIz=mat.E * sec.Iz,
                GAy=mat.G * sec.Asy if shear else None, GAz=mat.G * sec.Asz if shear else None)


# ============================================================ 分布荷重の等価節点荷重
def test_distributed_matches_udl():
    """全長に一様な profile は既存の等分布荷重と一致（剛域・せん断変形あり）"""
    L, ai, aj = 6000.0, 300.0, 150.0
    w = (1.0, -2.0, -3.0)
    p, r = distributed_equivalent_loads(w, [(0, 1), (L, 1)], L, ai, aj, **_stiff())
    pu, ru = udl_equivalent_loads(w, L - ai - aj, ai, aj)
    np.testing.assert_allclose(p, pu, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(r, ru, rtol=1e-12, atol=1e-6)


@pytest.mark.parametrize("shear", [False, True])
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5])
def test_symmetric_trapezoid_fixed_end_moment(alpha, shear):
    """両端固定・対称台形荷重: M = qL²/12·(1 − 2α² + α³)（α=0.5 の三角形で 5qL²/96）
    対称荷重なのでせん断変形の有無によらない"""
    L, q = 6000.0, 30.0
    a = alpha * L
    prof = [(0, 0), (a, 1), (L - a, 1), (L, 0)]
    p, r = distributed_equivalent_loads((0, 0, -q), prof, L, 0, 0, **_stiff(shear=shear))
    M = q * L ** 2 / 12 * (1 - 2 * alpha ** 2 + alpha ** 3)
    assert p[4] == pytest.approx(M, rel=1e-12)          # ry = −dw/dx 基準で下向き荷重は i端 +
    assert p[10] == pytest.approx(-M, rel=1e-12)
    assert p[2] == pytest.approx(-q * (L - a) / 2, rel=1e-12)
    assert np.all(r == 0)
    if alpha == 0.5:
        assert p[4] == pytest.approx(5 * q * L ** 2 / 96, rel=1e-12)


def test_triangle_fixed_end_euler():
    """両端固定・三角形荷重 0→q（せん断変形無視）: Mi = qL²/30, Mj = qL²/20, Ri = 3qL/20, Rj = 7qL/20"""
    L, q = 5000.0, 20.0
    p, _ = distributed_equivalent_loads((0, q, 0), [(0, 0), (L, 1)], L, 0, 0, **_stiff(shear=False))
    assert p[1] == pytest.approx(3 * q * L / 20, rel=1e-12)
    assert p[7] == pytest.approx(7 * q * L / 20, rel=1e-12)
    assert p[5] == pytest.approx(q * L ** 2 / 30, rel=1e-12)       # rz = dv/dx 基準
    assert p[11] == pytest.approx(-q * L ** 2 / 20, rel=1e-12)


def _cantilever(n_div, prof, q, L, ai, aj):
    """片持ち梁（x=0 固定）を n_div 分割して解く。荷重は全体 −Z に q × prof"""
    m = Model()
    xs = np.linspace(0, L, n_div + 1)
    for k, x in enumerate(xs):
        m.add_node(k, x, 0, 0, fix=[1] * 6 if k == 0 else None)
    for k in range(n_div):
        m.add_element(k, k, k + 1, FC24, SEC,
                      rigid_i=ai if k == 0 else 0.0, rigid_j=aj if k == n_div - 1 else 0.0)
        x0, x1 = xs[k], xs[k + 1]
        cut = [(x0, np.interp(x0, *zip(*prof)))] + [(x, f) for x, f in prof if x0 < x < x1] \
            + [(x1, np.interp(x1, *zip(*prof)))]
        m.add_member_load("W", k, [0, 0, -q], [(x - x0, f) for x, f in cut])
    return m.analyze("W"), n_div


def test_cantilever_asymmetric_load_exact():
    """非対称荷重・剛域・せん断変形あり: 先端たわみが数値積分による厳密解と一致し、
    分割数によらない（等価節点荷重が厳密であることの確認）"""
    L, ai, aj, q = 6000.0, 400.0, 250.0, 25.0
    prof = [(0, 0.3), (1500, 1.0), (4500, 0.2), (L, 0.8)]    # 剛域上にも荷重あり
    f = lambda x: q * np.interp(x, *zip(*prof))
    EI, GA = FC24.E * SEC.Iy, FC24.G * SEC.Asz
    Lf = L - ai - aj
    # 可撓部 s∈[0, Lf] の片持ち梁。j側剛域の荷重は可撓部先端へ力とモーメントで伝わる
    Pj = quad(f, L - aj, L)[0]
    Mj = quad(lambda x: f(x) * (x - (L - aj)), L - aj, L)[0]
    opts = dict(points=[1500 - ai, 4500 - ai], limit=200)
    w_f = (quad(lambda s: f(s + ai) * (s ** 2 * (3 * Lf - s) / (6 * EI) + s / GA), 0, Lf, **opts)[0]
           + Pj * (Lf ** 3 / (3 * EI) + Lf / GA) + Mj * Lf ** 2 / (2 * EI))
    t_f = (quad(lambda s: f(s + ai) * s ** 2 / (2 * EI), 0, Lf, **opts)[0]
           + Pj * Lf ** 2 / (2 * EI) + Mj * Lf / EI)
    exact = -(w_f + t_f * aj)                     # 下向き荷重 → 負のたわみ
    for n in (1, 3, 7):
        r, tip = _cantilever(n, prof, q, L, ai, aj)
        assert r.disp[tip][2] == pytest.approx(exact, rel=1e-9)
        total = quad(f, 0, L, points=[1500, 4500])[0]
        assert sum(r.reactions.values())[2] == pytest.approx(total, rel=1e-9)


# ============================================================ 床荷重の分配
def test_tributary_kikko():
    """6000×7000 区画の亀甲分割: X方向の辺（短辺）は三角形 h=3000、Y方向の辺は台形 h=3000"""
    p = tributary_profiles(6000, 7000, {"x0", "x1", "y0", "y1"})
    assert p["x0"] == [(0, 0), (3000, 3000), (3000, 3000), (6000, 0)]
    assert p["y1"] == [(0, 0), (3000, 3000), (4000, 3000), (7000, 0)]
    area = sum(np.trapezoid([f for _, f in pr], [x for x, _ in pr]) for pr in p.values())
    assert area == pytest.approx(6000 * 7000)


def test_tributary_one_way_and_error():
    """対向 2辺のみ → 一方向スラブ（負担幅 = スパン/2）、それ以外はエラー"""
    p = tributary_profiles(6000, 7000, {"y0", "y1"})
    assert p == {"y0": [(0, 3000), (7000, 3000)], "y1": [(0, 3000), (7000, 3000)]}
    with pytest.raises(ValueError, match="分配できません"):
        tributary_profiles(6000, 7000, {"x0", "y0", "y1"})


# ============================================================ 建物全体（計算例）
# 手計算（two_story_auto_loads.yaml）
#   床面積 12×7 = 84 m²
#   梁自重  0.4×0.7×24 = 6.72 kN/m、1層分 X方向 6m×4本 + Y方向 7m×3本 = 45 m → 302.4 kN
#   柱自重  0.6×0.6×24 = 8.64 kN/m、1F柱 4m×6本 = 207.36 kN、2F柱 3.5m×6本 = 181.44 kN
DL = 6.0 * 84 + 7.0 * 84
LL_FRAME = 1.8 * 84 + 0.65 * 84
BEAMS, COL1, COL2 = 302.4, 207.36, 181.44
W2 = (6.0 + 0.8) * 84 + BEAMS + COL1 / 2 + COL2 / 2              # 1068.0 kN
WR = (7.0 + 0.3) * 84 + BEAMS + COL2 / 2 + 150                   # 1156.32 kN


@pytest.fixture(scope="module")
def auto():
    b = load_building(EXAMPLES / "two_story_auto_loads.yaml")
    return b, {c: b.model.analyze(c) for c in ("G", "P", "Ex", "Ey")}


def test_vertical_equilibrium(auto):
    """鉛直反力の合計 = 床荷重 + 部材自重（G）、積載荷重（P）"""
    _, r = auto
    assert sum(r["G"].reactions.values())[2] == pytest.approx((DL + 2 * BEAMS + COL1 + COL2) * 1e3, rel=1e-9)
    assert sum(r["P"].reactions.values())[2] == pytest.approx(LL_FRAME * 1e3, rel=1e-9)


def test_seismic_weights_and_ai(auto):
    """地震用重量・T・Rt・Ai・Qi の手計算との照合"""
    b, _ = auto
    s = b.seismic
    assert W2 == pytest.approx(1068.0) and WR == pytest.approx(1156.32)
    assert b.weights[1][0] == pytest.approx(W2 * 1e3, rel=1e-12)
    assert b.weights[2][0] == pytest.approx(WR * 1e3, rel=1e-12)
    assert b.weights[1][1:] == pytest.approx((6000, 3500))        # 対称なので重心 = 図心
    T = 0.02 * 7.5                                                # 0.15 s < Tc = 0.6 s
    assert s["T"] == pytest.approx(T) and s["Rt"] == 1.0
    a2 = WR / (W2 + WR)
    Ai2 = 1 + (1 / math.sqrt(a2) - a2) * 2 * T / (1 + 3 * T)
    Q1 = 0.2 * 1.0 * (W2 + WR)
    Q2 = 0.2 * Ai2 * WR
    r1, r2 = s["rows"]
    assert (r1["Ai"], r2["Ai"]) == pytest.approx((1.0, Ai2), rel=1e-12)
    assert (r1["Q"], r2["Q"]) == pytest.approx((Q1 * 1e3, Q2 * 1e3), rel=1e-12)
    assert (r1["P"], r2["P"]) == pytest.approx(((Q1 - Q2) * 1e3, Q2 * 1e3), rel=1e-12)


@pytest.mark.parametrize("case, d", [("Ex", 0), ("Ey", 1)])
def test_story_shear(auto, case, d):
    """各層の柱のせん断力の合計 = 層せん断力 Qi、ねじれなし（重心 = 図心）"""
    b, r = auto
    m = b.model
    for k, row in enumerate(b.seismic["rows"]):
        cols = [eid for (t, f, *_), eid in b.elem_ids.items() if t == "c" and f == k]
        shear = sum((m.elements[e].R.T @ r[case].forces[e]["node"][0:3])[d] for e in cols)
        assert abs(shear) == pytest.approx(row["Q"], rel=1e-9)
        assert r[case].disp[b.masters[k + 1]][5] == pytest.approx(0, abs=1e-12)


# ============================================================ 重心の偏り・偶発偏心
ECC = {
    "materials": {"Fc24": {"Fc": 24}},
    "sections": {"C1": {"b": 600, "D": 600}, "G1": {"b": 400, "D": 700}},
    "defaults": {"material": "Fc24"},
    "grids": {"x": {"X1": 0, "X2": 6000, "X3": 12000}, "y": {"Y1": 0, "Y2": 7000}},
    "floors": {"1F": 0, "2F": 4000},
    "columns": [{"at": "*", "section": "C1"}],
    "beams": [{"at": "*", "section": "G1"}],
    "floor_loads": {"A": {"DL": 6.0}, "B": {"DL": 12.0}},
    "floor_use": [{"area": "X1-X2/Y1-Y2", "use": "A"}, {"area": "X2-X3/Y1-Y2", "use": "B"}],
    "self_weight": {"gamma": 0},
    "seismic": {"eccentricity": 0.05},
}


def test_mass_center_and_eccentricity():
    """重量 252 kN（x=3000）と 504 kN（x=9000）→ xg = 7000、図心 6000 から 1000 ずれる
    Ey: Mz = 1000·P、Ex±e: Mz = ∓0.05×7000·P = ∓350·P"""
    b = Building(copy.deepcopy(ECC))
    m = b.model
    mst = b.masters[1]
    assert (m.nodes[mst].x, m.nodes[mst].y) == (6000, 3500)
    W, xg, yg = b.weights[1]
    assert (W, xg, yg) == pytest.approx(((6 + 12) * 42 * 1e3, 7000, 3500))
    P = 0.2 * W
    loads = {c: sum(F for n, F in m.nodal_loads[c] if n == mst) for c in m.nodal_loads}
    assert loads["Ex"] == pytest.approx([P, 0, 0, 0, 0, 0])
    assert loads["Ey"] == pytest.approx([0, P, 0, 0, 0, 1000 * P])
    assert loads["Ex+e"][5] == pytest.approx(-350 * P)
    assert loads["Ex-e"][5] == pytest.approx(350 * P)
    assert loads["Ey+e"][5] == pytest.approx((1000 + 0.05 * 12000) * P)
    # 重い側（X3 側）が大きく動く向きにねじれる: Ey で rz > 0
    assert b.model.analyze("Ey").disp[mst][5] > 0


def test_manual_loads_are_added():
    """手入力の loads は自動生成のケースに加算される"""
    d = copy.deepcopy(ECC)
    d["loads"] = {"G": {"beams": [{"at": "Y1/*", "w": [0, 0, -10]}]}}
    d["self_weight"] = {"gamma": 24}
    b = Building(d)
    total = sum(b.model.analyze("G").reactions.values())[2]
    sw = 0.4 * 0.7 * 24 * (12 * 2 + 7 * 3) + 0.6 * 0.6 * 24 * 4 * 6    # 自重 [kN]
    assert total == pytest.approx((18 * 42 + sw + 10 * 12) * 1e3, rel=1e-9)


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["floor_use"].append({"use": "C"}), "未定義の用途"),
    (lambda d: d["beams"].append({"at": "Y2/X1-X2", "section": "none"}), "分配できません"),
    (lambda d: d["seismic"].update({"soil": 4}), "地盤種別"),
    (lambda d: d["seismic"].update({"directions": ["z"]}), "directions"),
    (lambda d: d.update({"openings": [{"area": "*"}]}), "剛床がない"),
    (lambda d: d["floor_loads"].update({"D": {"LL_frame": 1}}), "DL がありません"),
])
def test_load_input_errors(patch, msg):
    d = copy.deepcopy(ECC)
    patch(d)
    with pytest.raises(InputError, match=msg):
        Building(d)
