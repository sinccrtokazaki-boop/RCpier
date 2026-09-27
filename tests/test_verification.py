"""理論解との照合テスト（pytest で実行）"""
import numpy as np
import pytest
from rcframe import Model, Concrete, RectSection

FC24 = Concrete("Fc24", 24)


def test_cantilever_with_shear():
    """片持ち柱の頂部水平変位: PL^3/3EI + κPL/GA"""
    sec = RectSection("C", 600, 600)
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, 0, 0, 3000)
    m.add_element(1, 1, 2, FC24, sec, kind="column")
    P, L = 100e3, 3000
    m.add_nodal_load("P", 2, [P, 0, 0, 0, 0, 0])
    r = m.analyze("P")
    E, G = FC24.E, FC24.G
    exact = P * L**3 / (3 * E * sec.Iz) + sec.kappa * P * L / (G * sec.A)
    assert r.disp[2][0] == pytest.approx(exact, rel=1e-9)
    # 柱脚モーメント = P*L
    si, _ = r.section_forces(1, "node")
    assert abs(si[5]) == pytest.approx(P * L, rel=1e-9)


def test_fixed_beam_udl():
    """両端固定梁・等分布荷重: 端部M=wL^2/12, 中央たわみ=wL^4/384EI + κwL^2/8GA"""
    sec = RectSection("G", 400, 700)
    w, L = 30.0, 6000
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, L / 2, 0, 0)
    m.add_node(3, L, 0, 0, fix=[1] * 6)
    for eid, (a, b) in enumerate([(1, 2), (2, 3)], 1):
        m.add_element(eid, a, b, FC24, sec)
        m.add_member_load("D", eid, [0, 0, -w])
    r = m.analyze("D")
    E, G = FC24.E, FC24.G
    exact = w * L**4 / (384 * E * sec.Iy) + sec.kappa * w * L**2 / (8 * G * sec.A)
    assert -r.disp[2][2] == pytest.approx(exact, rel=1e-9)
    si, _ = r.section_forces(1)
    assert abs(si[4]) == pytest.approx(w * L**2 / 12, rel=1e-9)


def test_rigid_zone_cantilever():
    """柱脚に剛域 a を持つ片持ち柱 = 長さ L-a の片持ち柱"""
    sec = RectSection("C", 600, 600)
    L, a, P = 3000, 500, 100e3
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, 0, 0, L)
    m.add_element(1, 1, 2, FC24, sec, kind="column", rigid_i=a)
    m.add_nodal_load("P", 2, [0, P, 0, 0, 0, 0])
    r = m.analyze("P")
    Lf = L - a
    exact = P * Lf**3 / (3 * FC24.E * sec.Iy) + sec.kappa * P * Lf / (FC24.G * sec.A)
    assert r.disp[2][1] == pytest.approx(exact, rel=1e-9)
    # 節点位置の柱脚モーメントは P*L、フェイス位置は P*(L-a)
    assert abs(r.section_forces(1, "node")[0][4]) == pytest.approx(P * L, rel=1e-9)
    assert abs(r.section_forces(1, "face")[0][4]) == pytest.approx(P * Lf, rel=1e-9)


def _frame(diaphragm):
    """2x2スパン1層の立体ラーメン"""
    col = RectSection("C", 600, 600)
    bm = RectSection("G", 400, 700, phi_y=2.0)
    m = Model()
    nid = 0
    top = {}
    for i, x in enumerate([0, 6000, 12000]):
        for j, y in enumerate([0, 6000, 12000]):
            m.add_node(100 + 10 * i + j, x, y, 0, fix=[1] * 6)
            m.add_node(200 + 10 * i + j, x, y, 3500)
            top[(i, j)] = 200 + 10 * i + j
    eid = 1
    for i in range(3):
        for j in range(3):
            m.add_element(eid, 100 + 10 * i + j, 200 + 10 * i + j, FC24, col, "column"); eid += 1
    for i in range(3):
        for j in range(3):
            if i < 2:
                m.add_element(eid, top[(i, j)], top[(i + 1, j)], FC24, bm); eid += 1
            if j < 2:
                m.add_element(eid, top[(i, j)], top[(i, j + 1)], FC24, bm); eid += 1
    m.auto_rigid_zones()
    if diaphragm:
        m.add_node(900, 6000, 6000, 3500)
        m.add_diaphragm(900, list(top.values()))
    return m, top


def test_equilibrium_3d():
    """反力合計 = 荷重合計（部材荷重＋節点荷重）"""
    m, top = _frame(False)
    for e in m.elements.values():
        if e.kind == "beam":
            m.add_member_load("D", e.id, [0, 0, -30])
    m.add_nodal_load("D", top[(2, 2)], [50e3, -20e3, 0, 0, 0, 0])
    r = m.analyze("D")
    tot = sum(r.reactions.values())
    total_w = sum(30 * e.L for e in m.elements.values() if e.kind == "beam")
    assert tot[0] == pytest.approx(-50e3, rel=1e-8)
    assert tot[1] == pytest.approx(20e3, rel=1e-8)
    assert tot[2] == pytest.approx(total_w, rel=1e-8)


def test_rigid_diaphragm():
    """剛床: 重心への水平力 → 全節点の水平変位が一致・ねじれなし"""
    m, top = _frame(True)
    m.add_nodal_load("E", 900, [500e3, 0, 0, 0, 0, 0])
    r = m.analyze("E")
    ux = [r.disp[n][0] for n in top.values()]
    assert np.ptp(ux) < 1e-9 * max(ux)
    assert abs(r.disp[900][5]) < 1e-12
    assert sum(r.reactions.values())[0] == pytest.approx(-500e3, rel=1e-8)


def test_auto_rigid_zone_values():
    """梁端剛域 = 柱幅/2 - 梁せい/4 = 300-175 = 125, 柱頭 = 梁せい/2 - 柱せい/4 = 350-150 = 200"""
    m, _ = _frame(False)
    beam = next(e for e in m.elements.values() if e.kind == "beam")
    colm = next(e for e in m.elements.values() if e.kind == "column")
    assert beam.rigid_i == pytest.approx(125)
    assert colm.rigid_j == pytest.approx(200)
    assert colm.rigid_i == 0.0
