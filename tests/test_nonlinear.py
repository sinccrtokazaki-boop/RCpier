"""非線形解析（フェーズ5）のテスト: 理論解・手計算との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import Building, Concrete, InputError, Model, RectSection, Result, run_pushover
from rcframe.element import local_stiffness, local_stiffness_springs
from rcframe.hinge import Hinge
from rcframe.pushover import Level, pushover
from rcframe.skeleton import (Skeleton, beam_yield_moment, column_yield_moment, crack_moment,
                              member_end_stiffness, sugano_alpha_y)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FC24 = Concrete("Fc24", 24)


# ============================================================ ばね付き部材
def test_rigid_springs_equal_elastic():
    """ばね柔性 0 の剛性は弾性要素と完全に一致"""
    s = RectSection("G", 400, 700, phi_y=1.5)
    args = (FC24.E, FC24.G, s.A, s.Iy, s.Iz, s.J, 5400.0, s.Asy, s.Asz)
    np.testing.assert_allclose(local_stiffness_springs(*args, fs=np.zeros((2, 2))),
                               local_stiffness(*args), rtol=1e-10, atol=1e-6)


def _fixed_beam_with_springs(fs, w=30.0, L=6000.0):
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, L, 0, 0, fix=[1] * 6)
    e = m.add_element(1, 1, 2, FC24, RectSection("G", 400, 700), shear_deform=False)
    e.springs = np.array(fs, float)
    m.add_member_load("W", 1, [0, 0, -w])
    return m.analyze("W"), e


def test_fixed_end_moment_with_springs():
    """両端に回転ばね k（せん断変形無視）: 端部 M = (wL²/12) / (1 + 2EI/(kL))"""
    L, w = 6000.0, 30.0
    EI = FC24.E * RectSection("G", 400, 700).Iy
    k = 2 * EI / L
    r, _ = _fixed_beam_with_springs([[0, 0], [1 / k, 1 / k]])
    M = w * L ** 2 / 12 / (1 + 2 * EI / (k * L))
    si, sj = r.section_forces(1, "node")
    assert (si[4], sj[4]) == pytest.approx((M, M), rel=1e-9)
    assert r.section_forces_at(1, L / 2)[4] == pytest.approx(M - w * L ** 2 / 8, rel=1e-9)


def test_pin_spring_propped_cantilever():
    """j端のばね柔性を極大（ピン）にすると一端固定・他端ピン: 固定端 M = wL²/8、ピン端 0"""
    L, w = 6000.0, 30.0
    EI = FC24.E * RectSection("G", 400, 700).Iy
    r, _ = _fixed_beam_with_springs([[0, 0], [0, 1e9 * L / EI]])
    si, sj = r.section_forces(1, "node")
    assert si[4] == pytest.approx(w * L ** 2 / 8, rel=1e-6)
    assert abs(sj[4]) < 1e-6 * w * L ** 2


# ============================================================ 骨格曲線の略算式
def test_skeleton_formulas():
    Fc = 24.0
    assert crack_moment(Fc, 400, 700) == pytest.approx(0.56 * math.sqrt(24) * 400 * 700 ** 2 / 6)
    assert crack_moment(Fc, 600, 600, 1000e3) == pytest.approx(
        0.56 * math.sqrt(24) * 600 ** 3 / 6 + 1000e3 * 600 / 6)
    assert beam_yield_moment(2026.8, 379.5, 630) == pytest.approx(0.9 * 2026.8 * 379.5 * 630)
    # 柱 600×600, at = 5-D25 = 2533.5, ag = 16-D25 = 8107.2, σy = 379.5
    at, ag, sy, b, D = 2533.5, 8107.2, 379.5, 600.0, 600.0
    base = 0.8 * at * sy * D
    assert column_yield_moment(at, ag, sy, b, D, Fc, -500e3) == pytest.approx(base + 0.4 * -500e3 * D)
    N = 1500e3                                                  # ≦ 0.4bDFc = 3456 kN
    assert column_yield_moment(at, ag, sy, b, D, Fc, N) == pytest.approx(
        base + 0.5 * N * D * (1 - N / (b * D * Fc)))
    N, Nmax, Nb = 5000e3, b * D * Fc + ag * sy, 0.4 * b * D * Fc
    assert column_yield_moment(at, ag, sy, b, D, Fc, N) == pytest.approx(
        (base + 0.12 * b * D ** 2 * Fc) * (Nmax - N) / (Nmax - Nb))
    # 区分の境界で連続
    for Nx in (0.0, Nb):
        assert column_yield_moment(at, ag, sy, b, D, Fc, Nx * (1 - 1e-12)) == \
            pytest.approx(column_yield_moment(at, ag, sy, b, D, Fc, Nx * (1 + 1e-12) + 1e-9), rel=1e-6)
    assert column_yield_moment(at, ag, sy, b, D, Fc, Nmax * 1.01) == 0
    ay = sugano_alpha_y(8.0, 0.008, 2700, 700, 630, 0.1)
    assert ay == pytest.approx((0.043 + 1.64 * 8 * 0.008 + 0.043 * 2700 / 700 + 0.033) * 0.81)
    EI, L = 1e14, 5000.0
    assert member_end_stiffness(EI, None, L) == pytest.approx(6 * EI / L)


def test_skeleton_member_trilinear():
    """ばね + 弾性部材（K0）の M–θ が部材のトリリニア（降伏点 My/(αy·K0), 降伏後 r·K0）に一致"""
    K0, Mc, My, ay, r = 2e11, 100e6, 300e6, 0.3, 0.01
    sk = Skeleton.from_member(Mc, My, ay, K0, r)
    assert sk.theta_y + My / K0 == pytest.approx(My / (ay * K0))
    # 降伏後: 部材の接線 = 1/(1/K0 + 1/ks3) = r·K0
    assert 1 / (1 / K0 + 1 / sk.ks3) == pytest.approx(r * K0)
    bi = Skeleton.from_member(400e6, My, ay, K0, r)             # Mc > My → バイリニア
    assert (bi.Mc, bi.theta_y) == (My, 0.0)


# ============================================================ 履歴則
def test_hinge_peak_oriented_path():
    """ひび割れ → 降伏 → 降伏後 → 除荷（剛） → 反対側のひび割れ点へ向かう直線 → 反転 → 最大点へ"""
    sk = Skeleton(Mc=100.0, My=200.0, theta_y=0.002, ks2=50000.0, ks3=1000.0)
    h = Hinge(sk, sk)

    def go(sigma, dM=None):
        flex, lim, kind = h.tangent(sigma)
        step = (lim - h.M) if dM is None else dM
        hit = dM is None
        return (flex, lim, kind), h.advance(step, flex, sigma, hit, lim, kind)

    (f, lim, kind), ev = go(1)
    assert (f, lim, kind, ev) == (0.0, 100.0, "skeleton", ["ひび割れ"])
    (f, lim, kind), ev = go(1)
    assert (f, lim, kind, ev) == (1 / 50000, 200.0, "yield", ["降伏"])
    assert h.theta == pytest.approx(0.002)
    (f, lim, kind), _ = go(1, 10.0)
    assert (f, kind) == (1 / 1000, "post") and (h.theta, h.M) == pytest.approx((0.012, 210.0))
    assert h.peak[1] == pytest.approx((0.012, 210.0))
    (f, lim, kind), _ = go(-1)                                  # 除荷: 剛で M = 0 へ
    assert (f, lim, kind) == (0.0, 0.0, "zero") and h.theta == pytest.approx(0.012)
    (f, lim, kind), ev = go(-1, -50.0)                          # (0.012, 0) → (0, −100) の直線
    assert (f, lim, kind) == (pytest.approx(0.012 / 100), -100.0, "peak")
    assert (h.theta, h.M) == pytest.approx((0.006, -50.0)) and ev == []
    (f, lim, kind), _ = go(1)                                   # 反転: 剛で M = 0 へ
    assert (f, kind) == (0.0, "zero") and h.anchor[1] == pytest.approx(0.006)
    (f, lim, kind), _ = go(1)                                   # (0.006, 0) → 最大点 (0.012, 210)
    assert (f, lim, kind) == (pytest.approx(0.006 / 210), 210.0, "peak")
    assert (h.theta, h.M) == pytest.approx((0.012, 210.0))
    (f, lim, kind), _ = go(1, 5.0)                              # 骨格曲線に復帰
    assert kind == "post" and h.M == pytest.approx(215.0)
    assert h.cracked[-1] is False and h.state == "降伏"


# ============================================================ 単純な架構のプッシュオーバー
def test_cantilever_trilinear():
    """片持ち柱: ひび割れ P = Mc/L、降伏 P = My/L、降伏時の頂部変位 = 弾性分 + θs_y·L"""
    L, P = 3000.0, 100e3
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, 0, 0, L, fix=[0, 1, 1, 1, 0, 1])   # 頂部は X 方向の並進と Y 軸まわりの回転が自由
    m.add_element(1, 1, 2, FC24, RectSection("C", 600, 600), "column")
    s = RectSection("C", 600, 600)
    EI, GA = FC24.E * s.Iz, FC24.G * s.Asy
    K0 = member_end_stiffness(EI, GA, L)
    Mc, My = 150e6, 450e6
    sk = Skeleton.from_member(Mc, My, 0.3, K0, 0.01)
    hinges = {(1, 0, 0): Hinge(sk, sk)}
    m.assemble_K()
    po = pushover(m, hinges, [], [(2, np.array([P, 0, 0, 0, 0, 0]))],
                  [Level("1F", None, 2, L)], 0, 0.02, 1.0)
    ev = {e["event"]: e for e in po.events}
    assert ev["ひび割れ"]["lam"] * P == pytest.approx(Mc / L, rel=1e-9)
    assert ev["降伏"]["lam"] * P == pytest.approx(My / L, rel=1e-9)
    step_y = next(s for s in po.steps if any(e["event"] == "降伏" for e in s["events"]))
    Py = My / L
    delta = Py * (L ** 3 / (3 * EI) + L / GA) + sk.theta_y * L
    assert step_y["drifts"][0] * L == pytest.approx(delta, rel=1e-9)
    # ひび割れ時の変位は弾性のまま
    step_c = next(s for s in po.steps if any(e["event"] == "ひび割れ" for e in s["events"]))
    assert step_c["drifts"][0] * L == pytest.approx((Mc / L) * (L ** 3 / (3 * EI) + L / GA), rel=1e-9)


def test_portal_frame_mechanism():
    """門形ラーメン（剛な梁、柱の上下にバイリニアばね）: 崩壊荷重 P = 4Mp/h（仮想仕事法）"""
    h, span, Mp = 3000.0, 6000.0, 300e6
    m = Model()
    for nid, x, z in ((1, 0, 0), (2, span, 0), (3, 0, h), (4, span, h)):
        m.add_node(nid, x, 0, z, fix=[1] * 6 if z == 0 else [0, 1, 0, 1, 0, 1])
    col = RectSection("C", 600, 600)
    m.add_element(1, 1, 3, FC24, col, "column")
    m.add_element(2, 2, 4, FC24, col, "column")
    m.add_element(3, 3, 4, FC24, RectSection("G", 3000, 3000))    # ほぼ剛な梁
    m.assemble_K()
    K0 = member_end_stiffness(FC24.E * col.Iz, FC24.G * col.Asy, h)
    sk = Skeleton.from_member(Mp, Mp, 0.5, K0, 1e-6)
    hinges = {(eid, 0, end): Hinge(sk, sk) for eid in (1, 2) for end in (0, 1)}
    po = pushover(m, hinges, [], [(3, np.array([1e3, 0, 0, 0, 0, 0]))],
                  [Level("1F", None, 3, h)], 0, 0.03, 0.01)
    assert sum(e["event"] == "降伏" for e in po.events) == 4
    Pu = po.steps[-1]["lam"] * 1e3
    assert Pu == pytest.approx(4 * Mp / h, rel=1e-3)
    assert Pu > 4 * Mp / h                                      # 降伏後剛性の分だけ上回る


# ============================================================ 2層の計算例
def _design_data():
    return yaml.safe_load((EXAMPLES / "two_story_design.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pushed():
    b = Building(_design_data())
    return b, {d: run_pushover(b, d) for d in ("+x", "-x")}


def test_pushover_symmetry_and_drift(pushed):
    """X方向に対称な建物: +x と −x の最終ベースシアが一致、最終の層間変形角は 1/50"""
    _, po = pushed
    qp, qm = po["+x"].steps[-1]["Q"], po["-x"].steps[-1]["Q"]
    assert qp == pytest.approx(qm, rel=1e-6)
    assert max(abs(x) for x in po["+x"].steps[-1]["drifts"]) == pytest.approx(0.02, rel=1e-9)
    assert po["+x"].steps[-1]["drifts"][0] == pytest.approx(-po["-x"].steps[-1]["drifts"][0], rel=1e-6)


def test_pushover_story_shear_equilibrium(pushed):
    """最終状態で各層の柱のせん断力の合計 = 層せん断力（外力の上階の合計）"""
    b, po = pushed
    r = po["+x"]
    m = b.model
    lam = r.steps[-1]["lam"]
    for k, row in enumerate(b.seismic["rows"]):
        cols = [eid for (t, f, *_), eid in b.elem_ids.items() if t == "c" and f == k]
        shear = sum((m.elements[e].R.T @ r.final.forces[e]["node"][0:3])[0] for e in cols)
        assert abs(shear) == pytest.approx(lam * row["Q"], rel=1e-8)


def test_springs_reset_after_pushover(pushed):
    """解析後はばねが外れ、線形解析の結果は変わらない"""
    b, _ = pushed
    assert all(e.springs is None for e in b.model.elements.values())
    ref = Building(_design_data()).model.analyze("Ex")
    r = b.model.analyze("Ex")
    for nid in ref.disp:
        np.testing.assert_allclose(r.disp[nid], ref.disp[nid], rtol=1e-12, atol=1e-15)


def test_light_gravity_matches_linear():
    """ひび割れが生じない軽い長期荷重: 長期載荷後の変位は線形解析と一致し、
    水平載荷の最初のステップ（事象前）の剛性も線形解析と一致"""
    d = _design_data()
    d["floor_loads"] = {"事務室": {"DL": 0.5, "LL_frame": 0.1, "LL_seismic": 0.1},
                        "屋上": {"DL": 0.5, "LL_frame": 0.1, "LL_seismic": 0.1}}
    d["self_weight"] = {"gamma": 1.0}
    b = Building(d)
    po = run_pushover(b, "+x")
    assert not [e for e in po.events if e["phase"] == "長期"]
    rG, rP, rE = (b.model.analyze(c) for c in ("G", "P", "Ex"))
    for nid, u in po.gravity_disp.items():
        np.testing.assert_allclose(u, rG.disp[nid] + rP.disp[nid], rtol=1e-9, atol=1e-12)
    s1 = po.lateral[0]
    lvl = [lv for lv in po.levels]
    lin = [(rE.disp[lv.upper][0] - (rE.disp[lv.lower][0] if lv.lower else 0)) / lv.h for lv in lvl]
    grav = [(po.gravity_disp[lv.upper][0] - (po.gravity_disp[lv.lower][0] if lv.lower else 0)) / lv.h
            for lv in lvl]
    for dr, l0, g0 in zip(s1["drifts"], lin, grav):
        assert dr - g0 == pytest.approx(s1["lam"] * l0, rel=1e-9)


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d.update(nonlinear={"post_yield_ratio": 0}), "post_yield_ratio"),
    (lambda d: d.update(nonlinear={"pushover": {"directions": ["+z"]}}), "directions"),
    (lambda d: d.update(nonlinear={"sigma_y": 1.1}), "未知の項目"),
])
def test_nonlinear_input_errors(patch, msg):
    d = _design_data()
    patch(d)
    with pytest.raises(InputError, match=msg):
        Building(d)
