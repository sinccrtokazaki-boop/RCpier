"""保有水平耐力（フェーズ6）のテスト: 手計算・理論値との照合"""
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import Building, Concrete, InputError, Model, RectSection, run_ultimate
from rcframe.hinge import Hinge
from rcframe.pushover import Level, pushover
from rcframe.skeleton import Skeleton, member_end_stiffness
from rcframe.ultimate import (beam_class, column_class, eccentricity, fe_factor, fs_factor,
                              member_group, qsu_arakawa, stiffness_ratios)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FC24 = Concrete("Fc24", 24)


# ============================================================ 式・表
def test_qsu_arakawa():
    """荒川式: {k·pt^0.23·(Fc+18)/(M/Qd+0.12) + 0.85√(pw·σwy) + 0.1σ0}·b·j"""
    b, d, Fc = 600.0, 530.0, 24.0
    pt, pw, swy, s0 = 0.008, 0.0042, 295.0, 1.5
    j = 7 / 8 * d
    for kind, k in (("mean", 0.068), ("min", 0.053)):
        hand = (k * 0.8 ** 0.23 * 42 / (2.0 + 0.12) + 0.85 * math.sqrt(pw * swy) + 0.1 * s0) * b * j
        assert qsu_arakawa(b, d, pt, 2.0, pw, swy, s0, Fc, kind) == pytest.approx(hand)
    # M/(Qd) は 1〜3、pw は 1.2% で頭打ち
    q = lambda r, p: qsu_arakawa(b, d, pt, r, p, swy, 0.0, Fc)
    assert q(0.5, pw) == pytest.approx(q(1.0, pw)) and q(5.0, pw) == pytest.approx(q(3.0, pw))
    assert q(2.0, 0.02) == pytest.approx(q(2.0, 0.012))


@pytest.mark.parametrize("args, expect", [
    ((2.5, 0.35, 0.008, 0.100), "FA"),
    ((2.49, 0.35, 0.008, 0.100), "FB"),
    ((2.0, 0.45, 0.010, 0.125), "FB"),
    ((2.0, 0.451, 0.010, 0.125), "FC"),
    ((1.5, 0.55, 0.02, 0.15), "FC"),
    ((1.5, 0.56, 0.02, 0.15), "FD"),
    ((6.0, 0.1, 0.005, 0.151), "FD"),
])
def test_column_class_boundaries(args, expect):
    assert column_class(*args) == expect


def test_beam_class_and_shear_failure():
    assert [beam_class(t) for t in (0.15, 0.2, 0.3)] == ["FA", "FB", "FC"]
    assert beam_class(0.01, shear_failure=True) == "FD"
    assert column_class(6.0, 0.1, 0.005, 0.05, shear_failure=True) == "FD"


@pytest.mark.parametrize("gamma, expect", [
    ({"FA": 0.5, "FC": 0.2, "FB": 0.3}, "A"),
    ({"FA": 0.5, "FC": 0.21, "FB": 0.29}, "B"),
    ({"FA": 0.49, "FB": 0.51}, "B"),
    ({"FC": 0.49, "FB": 0.51}, "B"),
    ({"FC": 0.5, "FB": 0.5}, "C"),
    ({"FA": 0.9, "FD": 0.1}, "D"),               # FD があれば安全側に D
    ({"FD": 1.0}, "D"),
])
def test_member_group(gamma, expect):
    assert member_group(gamma) == expect


def test_fs_fe():
    assert fs_factor(0.6) == 1.0 and fs_factor(0.3) == pytest.approx(1.5)
    assert fs_factor(0.6 - 1e-12) == pytest.approx(1.0)
    assert fe_factor(0.15) == 1.0 and fe_factor(0.225) == pytest.approx(1.25)
    assert fe_factor(0.3 - 1e-12) == pytest.approx(1.5) and fe_factor(0.4) == 1.5
    assert stiffness_ratios([1000, 3000]) == pytest.approx([0.5, 1.5])


def test_eccentricity_by_hand():
    """4本の柱（6 m × 6 m）、Y=0 側の柱の Dx が大きい:
    yk = (1·6000·2)/6 = 2000, xk = 3000, KR = 48e6 + 54e6 = 102e6,
    rex = √(102e6/6), ey = 1000 → Rex = 1000/rex ≒ 0.2425, Rey = 0"""
    cols = [(0, 0, 2, 1), (6000, 0, 2, 1), (0, 6000, 1, 2), (6000, 6000, 1, 2)]
    e = eccentricity(cols, 3000, 3000)
    assert (e["xk"], e["yk"], e["KR"]) == pytest.approx((3000, 2000, 102e6))
    assert e["rex"] == pytest.approx(math.sqrt(17e6))
    assert e["Rex"] == pytest.approx(1000 / math.sqrt(17e6))
    assert e["Rey"] == 0
    assert fe_factor(e["Rex"]) == pytest.approx(1 + 0.5 * (1000 / math.sqrt(17e6) - 0.15) / 0.15)


# ============================================================ 崩壊機構・せん断破壊の検出
def _portal():
    h, span, Mp = 3000.0, 6000.0, 300e6
    m = Model()
    for nid, x, z in ((1, 0, 0), (2, span, 0), (3, 0, h), (4, span, h)):
        m.add_node(nid, x, 0, z, fix=[1] * 6 if z == 0 else [0, 1, 0, 1, 0, 1])
    col = RectSection("C", 600, 600)
    m.add_element(1, 1, 3, FC24, col, "column")
    m.add_element(2, 2, 4, FC24, col, "column")
    m.add_element(3, 3, 4, FC24, RectSection("G", 3000, 3000))
    K0 = member_end_stiffness(FC24.E * col.Iz, FC24.G * col.Asy, h)
    sk = Skeleton.from_member(Mp, Mp, 0.5, K0, 1e-3)
    hinges = {(eid, 0, end): Hinge(sk, sk) for eid in (1, 2) for end in (0, 1)}
    return m, hinges, h, Mp


def test_mechanism_detected_at_last_hinge():
    """門形ラーメン: 4つ目のヒンジが降伏したステップで崩壊機構（1層）を検出"""
    m, hinges, h, Mp = _portal()
    po = pushover(m, hinges, [], [(3, np.array([1e3, 0, 0, 0, 0, 0]))],
                  [Level("1F", None, 3, h)], 0, 0.03, 0.01, mechanism_check=True)
    last_yield = max(e["step"] for e in po.events if e["event"] == "降伏")
    assert po.mechanism_step == last_yield
    assert po.mechanism_stories == ["1F"]
    assert sum(e["event"] == "降伏" for e in po.events if e["step"] <= po.mechanism_step) == 4


def test_shear_failure_event():
    """柱のせん断力が Qsu に達した荷重でせん断破壊の事象
    左右の柱の分担はほぼ P/2 ずつ（梁の軸変形のため加力側の柱がわずかに大きく、
    梁の軸剛性/柱の水平剛性 ≒ 3e-3 程度の差）"""
    m, hinges, h, Mp = _portal()
    Qsu = 0.3 * 2 * Mp / h                         # 曲げ降伏時のせん断力 2Mp/h より小さい
    caps = {(1, 0): Qsu, (2, 0): Qsu}
    po = pushover(m, hinges, [], [(3, np.array([1e3, 0, 0, 0, 0, 0]))],
                  [Level("1F", None, 3, h)], 0, 0.03, 0.01, shear_caps=caps, keep_states=True)
    ev = [e for e in po.events if e["event"] == "せん断破壊"]
    assert len(ev) == 2
    assert ev[0]["lam"] * 1e3 == pytest.approx(2 * Qsu, rel=5e-3)
    assert ev[0]["lam"] * 1e3 < 2 * Qsu < ev[1]["lam"] * 1e3
    assert not [e for e in po.events if e["event"] == "降伏" and e["step"] < ev[0]["step"]]
    _, face = po.states[ev[0]["step"]]
    assert max(abs(face[1][1]), abs(face[2][1])) == pytest.approx(Qsu, rel=1e-9)


# ============================================================ 2層の計算例
def _data():
    return yaml.safe_load((EXAMPLES / "two_story_design.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def example():
    b = Building(_data())
    return b, run_ultimate(b, directions=("+x", "-x", "+y"))


def test_example_table(example):
    """Qud = Qi/C0（1F は ΣW）、Qu = λu·Qi、対称なので Re = 0・±x で一致、すべて OK"""
    b, rep = example
    rows = b.seismic["rows"]
    W = sum(w for w, _, _ in b.weights.values())
    st = {(s.direction, s.story): s for s in rep.stories}
    assert st[("+x", "1F")].Qud == pytest.approx(W, rel=1e-12)
    assert st[("+x", "2F")].Qud == pytest.approx(rows[1]["Q"] / 0.2, rel=1e-12)
    for s in rep.stories:
        k = [r["story"] for r in rows].index(s.story)
        lam = next(x["lam"] for x in rep.pushovers[s.direction].lateral if x["step"] == s.step)
        assert s.Qu == pytest.approx(lam * rows[k]["Q"], rel=1e-12)
        assert s.Re == pytest.approx(0, abs=1e-9) and s.Fe == 1.0
        assert s.ok
    assert st[("+x", "1F")].Qu == pytest.approx(st[("-x", "1F")].Qu, rel=1e-6)
    assert all(s.group == "A" and s.Ds == 0.30 for s in rep.stories)
    assert st[("+x", "1F")].reason == "層間変形角 1/100"
    assert st[("+x", "1F")].drift == pytest.approx(0.01, rel=1e-9)


def test_example_stiffness_ratio(example):
    """剛性率: rs = h/δ（線形 Ex）の平均に対する比"""
    b, rep = example
    r = b.model.analyze("Ex")
    m1, m2 = b.masters[1], b.masters[2]
    d1 = r.disp[m1][0] / 4000
    d2 = (r.disp[m2][0] - r.disp[m1][0]) / 3500
    rs = [1 / d1, 1 / d2]
    ecc = {(e["story"], e["dir"]): e for e in rep.ecc}
    assert ecc[("1F", "X")]["Rs"] == pytest.approx(rs[0] / (sum(rs) / 2), rel=1e-9)


def test_shear_critical_columns():
    """帯筋を減らし主筋を増やした柱（Qsu < Qmu）と強い梁: 柱が曲げ降伏より先にせん断破壊
    → 種別 FD、部材群 D、Ds = 0.45、Qu は最初の柱のせん断破壊の時点"""
    d = _data()
    d["sections"]["C1"]["rebar"].update(x="7-D29", y="7-D29", hoop_x="2-D10@300",
                                        hoop_y="2-D10@300")
    for p in ("end", "center"):
        d["sections"]["G1"]["rebar"][p].update(top="6-D29", bottom="6-D29")
    b = Building(d)
    rep = run_ultimate(b, directions=("+x",))
    cols = [m for m in rep.members["X"] if m.kind == "柱"]
    assert all(m.shear_failure and m.cls == "FD" for m in cols)
    s1 = next(s for s in rep.stories if s.story == "1F")
    assert s1.group == "D" and s1.Ds == 0.45
    assert s1.reason == "柱のせん断破壊"
    po = rep.pushovers["+x"]
    first = min(e["step"] for e in po.events if e["event"] == "せん断破壊" and "柱" in e["label"])
    first_yield = min(e["step"] for e in po.events if e["event"] == "降伏" and e["phase"] == "水平")
    assert s1.step == first < first_yield
    # Qu 時の柱のせん断力は Qsu 以下（破壊した柱で等しい）
    _, face = po.states[s1.step]
    for m in cols:
        assert abs(face[m.eid][1]) <= m.Qsu * (1 + 1e-9)


def test_beam_yield_limits_column_shear():
    """柱が Qsu < Qmu（FD）でも、梁が先に降伏して柱のせん断力が Qsu に達しなければ
    Qu は層間変形角の上限で決まる（種別は FD のまま）"""
    d = _data()
    d["sections"]["C1"]["rebar"].update(x="7-D29", y="7-D29", hoop_x="2-D10@300",
                                        hoop_y="2-D10@300")
    rep = run_ultimate(Building(d), directions=("+x",))
    s1 = next(s for s in rep.stories if s.story == "1F")
    assert s1.reason == "層間変形角 1/100" and s1.group == "D"
    assert not [e for e in rep.pushovers["+x"].events
                if e["event"] == "せん断破壊" and "柱" in e["label"] and e["step"] <= s1.step]


def test_ds_override_and_input_errors():
    d = _data()
    d["ultimate"] = {"ds_override": {"1F": 0.4}}
    b = Building(d)
    rep = run_ultimate(b, directions=("+x",))
    assert next(s for s in rep.stories if s.story == "1F").Ds == 0.4
    for bad, msg in (({"qsu": "max"}, "min または mean"), ({"ds_override": {"RF": 0.3}}, "層"),
                     ({"drift": 0.01}, "未知の項目")):
        d = _data()
        d["ultimate"] = bad
        with pytest.raises(InputError, match=msg):
            Building(d)
