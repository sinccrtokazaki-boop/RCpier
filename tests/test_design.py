"""断面算定（フェーズ4）のテスト: 手計算・理論値との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import Building, Concrete, InputError, Model, RectSection, Result, STEELS, run_checks
from rcframe.design import (allowable_moment, alpha_factor, beam_shear_long, beam_shear_short,
                            column_shear_short, concrete_fc, concrete_fs)
from rcframe.materials import Bars, Hoop

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
n = 15.0


# ============================================================ 許容応力度
def test_allowable_stresses():
    """RC規準 6条: 鉄筋（径による長期 ft の切替え）・せん断補強筋・コンクリート"""
    s345, s295, s390 = STEELS["SD345"], STEELS["SD295"], STEELS["SD390"]
    assert (s345.ft(25, "long"), s345.ft(29, "long"), s345.ft(29, "short")) == (215, 195, 345)
    assert (s295.ft(25, "long"), s295.ft(25, "short")) == (195, 295)
    assert (s390.ft(22, "long"), s390.ft(32, "long"), s390.ft(32, "short")) == (215, 195, 390)
    assert (s295.wft("long"), s295.wft("short"), s390.wft("short")) == (195, 295, 390)
    assert (concrete_fc(24, "long"), concrete_fc(24, "short")) == pytest.approx((8, 16))
    assert concrete_fs(24, "long") == pytest.approx(0.73)             # min(0.8, 0.73)
    assert concrete_fs(24, "short") == pytest.approx(1.095)
    assert concrete_fs(18, "long") == pytest.approx(0.6)              # min(0.6, 0.67)


def test_bar_parse():
    assert Bars.parse("4-D25").area == pytest.approx(4 * 506.7)
    h = Hoop.parse("2-D13@150")
    assert h.pw(400) == pytest.approx(2 * 126.7 / (400 * 150))
    for bad in ("4-D26", "D25", "4D25", "2-D10@0"):
        with pytest.raises(ValueError):
            Bars.parse(bad) if "@" not in bad else Hoop.parse(bad)


# ============================================================ 曲げ（梁 13条・柱 14条）
B, D, DT = 400.0, 700.0, 70.0
d = D - DT


def _single(at, fc, ft):
    """単筋梁: xn = d·np(−1 + √(1 + 2/(np)))、Ma = min(fc·b·xn/2·(d−xn/3), at·ft·(d−xn/3))"""
    p = at / (B * d)
    xn = d * n * p * (-1 + math.sqrt(1 + 2 / (n * p)))
    return fc * B * xn / 2 * (d - xn / 3), at * ft * (d - xn / 3)


@pytest.mark.parametrize("bars, governs", [("3-D25", "steel"), ("6-D25", "concrete")])
def test_single_reinforced_beam(bars, governs):
    """釣合い鉄筋比（at_b = b·xnb·fc/2/ft ≒ 1679 mm²）の下では鉄筋、上ではコンクリートで決まる"""
    at = Bars.parse(bars).area
    Mc, Ms = _single(at, 8.0, 215.0)
    assert (Ms < Mc) == (governs == "steel")
    Ma = allowable_moment(B, D, 0.0, at, DT, 0.0, 8.0, 215.0, 215.0)
    assert Ma == pytest.approx(min(Mc, Ms), rel=1e-9)


def _double(ac, at, fc, ft, dc=DT):
    """複筋梁の手計算: 中立軸 (b/2)xn² + n(ac+at)xn − n(ac·dc + at·d) = 0
    圧縮縁応力 σc を 3つの許容条件の最小で決め、引張鉄筋まわりのモーメントを求める"""
    a2, a1, a0 = B / 2, n * (ac + at), -n * (ac * dc + at * d)
    xn = (-a1 + math.sqrt(a1 ** 2 - 4 * a2 * a0)) / (2 * a2)
    sc = min(fc, ft * xn / (n * (d - xn)), ft * xn / (n * abs(xn - dc)))
    C = B * xn * sc / 2
    Cs = ac * n * sc * (xn - dc) / xn
    return C * (d - xn / 3) + Cs * (d - dc)


@pytest.mark.parametrize("top, bot, fc, ft", [
    ("2-D25", "4-D25", 8, 215),        # 長期・鉄筋で決まる
    ("4-D25", "8-D25", 8, 215),        # 長期・コンクリートで決まる
    ("4-D25", "4-D25", 16, 345),       # 短期・対称配筋
])
def test_double_reinforced_beam(top, bot, fc, ft):
    ac, at = Bars.parse(top).area, Bars.parse(bot).area
    Ma = allowable_moment(B, D, ac, at, DT, 0.0, fc, ft, ft)
    assert Ma == pytest.approx(_double(ac, at, fc, ft), rel=1e-9)


# 柱 600×600、1面 5-D25（対称配筋）、長期 Fc24・SD345
CB = CD = 600.0
CA = Bars.parse("5-D25").area
CFC, CFT = 8.0, 215.0
cd = CD - DT


def test_column_zero_axial_equals_beam():
    """N = 0 の柱の Ma は同じ配筋の複筋梁の Ma に一致"""
    Ma = allowable_moment(CB, CD, CA, CA, DT, 0.0, CFC, CFT, CFT)
    a2, a1, a0 = CB / 2, n * 2 * CA, -n * CA * (DT + cd)
    xn = (-a1 + math.sqrt(a1 ** 2 - 4 * a2 * a0)) / (2 * a2)
    sc = min(CFC, CFT * xn / (n * (cd - xn)))
    hand = CB * xn * sc / 2 * (cd - xn / 3) + CA * n * sc * (xn - DT) / xn * (cd - DT)
    assert Ma == pytest.approx(hand, rel=1e-9)


def test_column_balanced_point():
    """釣合い点: 圧縮縁 fc と引張鉄筋 ft が同時に達する xnb = n·fc·d/(n·fc + ft)
    N_b = C + Cs − T、M_b = 断面中心まわり"""
    xnb = n * CFC * cd / (n * CFC + CFT)
    C = CB * xnb * CFC / 2
    Cs = CA * n * CFC * (xnb - DT) / xnb
    T = CA * CFT
    Nb = C + Cs - T
    Mb = C * (CD / 2 - xnb / 3) + Cs * (CD / 2 - DT) + T * (cd - CD / 2)
    assert allowable_moment(CB, CD, CA, CA, DT, Nb, CFC, CFT, CFT) == pytest.approx(Mb, rel=1e-9)


def test_column_axial_limits():
    """純圧縮 fc·(bD + n·ag)、純引張 ft·ag を超えると Ma = 0。途中では正"""
    Nmax = CFC * (CB * CD + n * 2 * CA)
    Nmin = -CFT * 2 * CA
    Ma = lambda N: allowable_moment(CB, CD, CA, CA, DT, N, CFC, CFT, CFT)
    assert Ma(Nmax * 1.0001) == 0 and Ma(Nmin * 1.0001) == 0
    assert 0 < Ma(0.999 * Nmax) < Ma(0.5 * Nmax)
    assert 0 < Ma(0.999 * Nmin) < Ma(0.0)
    # Ma は釣合い点（この断面の長期では Nb ≒ 103 kN）まで増加し、その後減少する
    xnb = n * CFC * cd / (n * CFC + CFT)
    Nb = CB * xnb * CFC / 2 + CA * n * CFC * (xnb - DT) / xnb - CA * CFT
    assert Nb == pytest.approx(103e3, rel=0.01)
    assert Ma(0.0) < Ma(0.5 * Nb) < Ma(Nb)
    assert Ma(Nb) > Ma(Nb + 500e3) > Ma(Nb + 1000e3)


# ============================================================ せん断（15条）
def test_alpha_and_shear():
    assert alpha_factor(0.0, 100e3, 600) == 2.0          # 4/(0+1) = 4 → 上限 2
    assert alpha_factor(3 * 100e3 * 600, 100e3, 600) == 1.0
    assert alpha_factor(2 * 100e3 * 600, 100e3, 600) == pytest.approx(4 / 3)
    assert alpha_factor(1e9, 0.0, 600) == 1.0
    j = 7 / 8 * d
    assert beam_shear_long(B, d, 0.73, 1.5) == pytest.approx(B * j * 1.5 * 0.73)
    # pw は 1.2% で頭打ち、0.2% 未満は補強筋の寄与 0
    assert beam_shear_short(B, d, 1.095, 1.0, 295, 0.015) == \
        pytest.approx(B * j * (2 / 3 * 1.095 + 0.5 * 295 * 0.010))
    assert beam_shear_short(B, d, 1.095, 1.2, 295, 0.001) == pytest.approx(B * j * 2 / 3 * 1.2 * 1.095)
    assert column_shear_short(CB, cd, 1.095, 295, 0.0042) == \
        pytest.approx(CB * 7 / 8 * cd * (1.095 + 0.5 * 295 * 0.0022))


# ============================================================ 部材中間の断面力
FC24 = Concrete("Fc24", 24)


def _fixed_beam(L, w, profile=None, ai=0.0, aj=0.0):
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, L, 0, 0, fix=[1] * 6)
    m.add_element(1, 1, 2, FC24, RectSection("G", 400, 700), rigid_i=ai, rigid_j=aj)
    m.add_member_load("W", 1, [0, 0, -w], profile)
    return m.analyze("W")


def test_midspan_moment_udl():
    """両端固定・等分布: 端部 wL²/12（上端引張 My > 0）、中央 −wL²/24"""
    L, w = 6000.0, 30.0
    r = _fixed_beam(L, w)
    assert r.section_forces_at(1, 0)[4] == pytest.approx(w * L ** 2 / 12, rel=1e-9)
    assert r.section_forces_at(1, L / 2)[4] == pytest.approx(-w * L ** 2 / 24, rel=1e-9)
    assert r.section_forces_at(1, L / 2)[2] == pytest.approx(0, abs=1e-6)
    assert r.section_forces_at(1, L)[4] == pytest.approx(w * L ** 2 / 12, rel=1e-9)


def test_midspan_moment_trapezoid():
    """両端固定・対称台形（立上り a）: 中央 = 単純梁の中央 M0 − 固定端 M"""
    L, w, a = 6000.0, 30.0, 2000.0
    r = _fixed_beam(L, w, [(0, 0), (a, 1), (L - a, 1), (L, 0)])
    R = w * (L - a) / 2
    M0 = R * L / 2 - w * a / 2 * (L / 2 - 2 * a / 3) - w * (L / 2 - a) ** 2 / 2
    al = a / L
    Mend = w * L ** 2 / 12 * (1 - 2 * al ** 2 + al ** 3)
    assert r.section_forces_at(1, L / 2)[4] == pytest.approx(-(M0 - Mend), rel=1e-9)


def test_section_forces_at_faces_and_combine():
    """剛域端・節点で section_forces（face/node）に一致し、組合せ結果でも線形に重なる"""
    L, ai, aj = 6000.0, 300.0, 200.0
    r1 = _fixed_beam(L, 30.0, [(0, 0.5), (2500, 1.0), (L, 0.2)], ai, aj)
    for r in (r1,):
        si, sj = r.section_forces(1, "face")
        np.testing.assert_allclose(r.section_forces_at(1, ai), si, rtol=1e-9, atol=1e-3)
        np.testing.assert_allclose(r.section_forces_at(1, L - aj), sj, rtol=1e-9, atol=1e-3)
        ni, nj = r.section_forces(1, "node")
        np.testing.assert_allclose(r.section_forces_at(1, 0), ni, rtol=1e-9, atol=1e-3)
        np.testing.assert_allclose(r.section_forces_at(1, L), nj, rtol=1e-9, atol=1e-3)
    rc = Result.combine("1.5W", {r1: 1.5})
    np.testing.assert_allclose(rc.section_forces_at(1, 2345.0), 1.5 * r1.section_forces_at(1, 2345.0),
                               rtol=1e-12)


# ============================================================ 検定の流れ（計算例）
@pytest.fixture(scope="module")
def design():
    from rcframe import load_building
    b = load_building(EXAMPLES / "two_story_design.yaml")
    res = {c: b.model.analyze(c) for c in ("G", "P", "Ex", "Ey")}
    return b, res, run_checks(b, res)


def test_check_beam_row_by_hand(design):
    """2F梁 X2/Y1-Y2 の i端: G+P−Ey の M・Ma・QD・QA を独立に計算して照合"""
    b, res, rep = design
    eid = b.beam("X2", "Y1", "Y2", "2F")
    e = b.model.elements[eid]
    x = e.rigid_i
    sL = res["G"].section_forces_at(eid, x) + res["P"].section_forces_at(eid, x)
    sE = res["Ey"].section_forces_at(eid, x)
    M = -(sL[4] - sE[4])
    assert M < 0                                           # 端部は上端引張
    rb = e.sec.rebar
    top, bot = rb.end.top.area, rb.end.bottom.area
    Ma = allowable_moment(400, 700, bot, top, 70, 0, 16, 345, 345)
    QD = abs(sL[2] - 1.5 * sE[2])
    QA = beam_shear_short(400, 630, concrete_fs(24, "short"), alpha_factor(M, QD, 630), 295,
                          rb.end.stirrup.pw(400))
    rows = {r.item: r for r in rep.rows if r.eid == eid and r.pos == "i端" and r.combo == "G+P-Ey"}
    assert rows["曲げ"].demand == pytest.approx(abs(M), rel=1e-12)
    assert rows["曲げ"].capacity == pytest.approx(Ma, rel=1e-12)
    assert rows["曲げ"].note == "上端引張"
    assert rows["せん断"].demand == pytest.approx(QD, rel=1e-12)
    assert rows["せん断"].capacity == pytest.approx(QA, rel=1e-12)


def test_check_column_row_by_hand(design):
    """1F柱 X2/Y1 柱脚 X方向: 長期の N と Ma(N)"""
    b, res, rep = design
    eid = b.column("X2", "Y1", "1F")
    e = b.model.elements[eid]
    s = res["G"].section_forces_at(eid, e.rigid_i) + res["P"].section_forces_at(eid, e.rigid_i)
    N = -s[0]
    assert N > 0
    Ma = allowable_moment(600, 600, CA, CA, 70, N, 8, 215, 215)
    row = next(r for r in rep.rows if r.eid == eid and r.pos == "柱脚" and r.direction == "X"
               and r.item == "曲げ" and r.term == "長期")
    assert (row.N, row.demand, row.capacity) == pytest.approx((N, abs(s[5]), Ma), rel=1e-12)


def test_governing_and_ok(design):
    """検定比の最大行の選択と、計算例がすべて OK であること"""
    b, _, rep = design
    for (eid, item), g in rep.governing().items():
        assert g.ratio == max(r.ratio for r in rep.rows if r.eid == eid and r.item == item)
    assert rep.ng == [] and rep.warnings == []
    assert 0.5 < rep.max_ratio < 1.0
    # 検定の対象: 柱 12本 × 2位置 × 2方向 × 5組合せ × 2 + 梁 14本 × 3位置 × 5組合せ × 2
    assert len(rep.rows) == 12 * 2 * 2 * 5 * 2 + 14 * 3 * 5 * 2


def _example_data():
    return yaml.safe_load((EXAMPLES / "two_story_design.yaml").read_text(encoding="utf-8"))


def test_ng_and_warnings():
    """配筋を減らすと NG と構造規定の警告が出る"""
    d = _example_data()
    d["sections"]["C1"]["rebar"].update(x="2-D13", y="2-D13", hoop_x="2-D10@200")
    rep = run_checks(Building(d))
    assert any(r.eid == 1 and r.item == "曲げ" for r in rep.ng)
    assert any("主筋比 pg" in w for w in rep.warnings)
    assert any("帯筋比 pw" in w and "X方向" in w for w in rep.warnings)


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["sections"]["C1"]["rebar"].update(x="5-D26"), "主筋は"),
    (lambda d: d["sections"]["C1"]["rebar"].update(main="SD999"), "未定義の鋼種"),
    (lambda d: d["sections"]["G1"]["rebar"].pop("center"), "center"),
    (lambda d: d["sections"]["C1"]["rebar"].update(hoop_y="2-D13"), "せん断補強筋は"),
    (lambda d: d["design"].update(n=15), "未知の項目"),
])
def test_rebar_input_errors(patch, msg):
    d = _example_data()
    patch(d)
    with pytest.raises(InputError, match=msg):
        Building(d)


def test_rebar_kind_mismatch():
    """柱用の配筋を梁に使うと検定時にエラー"""
    d = _example_data()
    d["sections"]["G1"]["rebar"] = copy.deepcopy(d["sections"]["C1"]["rebar"])
    with pytest.raises(ValueError, match="柱用"):
        run_checks(Building(d))
