"""利用可否 U2（地盤変位の入力・側方流動）のテスト: 理論解・釣合いとの照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import FlowCase, InputError, Model, Pier, load_model, run_flow, run_level2
from rcframe.hinge import EPSpring
from rcframe.port_flow import ground_profile
from rcframe.pushover import Incremental, Level
from rcframe.steel import PipeSection, Steel, chang_beta

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
STEEL = Steel("S", fy=315)
PIPE = PipeSection("P", 1000, 14)
KH = 20000e-6                      # kh = 20000 kN/m³ → N/mm³


def _data(name="pier_level2.yaml"):
    return yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))


def _pile(L, dz, head_fix, tip_fix, ep=None):
    """鉛直な 1本の杭（下端 z = 0 → 上端 z = L）。各節点に水平地盤ばね k = kh·D·負担長
    ep: 上限 [N/mm]（地盤反力度×D）を与えると X方向のばねを EPSpring にする。戻り値: (model, 節点, 要素, ns)"""
    m = Model()
    n = round(L / dz)
    nodes = list(range(1, n + 2))
    for k, nid in enumerate(nodes):
        fix = tip_fix if k == 0 else (head_fix if k == n else None)
        m.add_node(nid, 0, 0, k * dz, fix=fix)
        lt = dz / 2 if k in (0, n) else dz
        kk = KH * PIPE.D * lt
        m.add_node_spring(nid, [kk, kk, 0, 0, 0, 0])
    ns = {}
    if ep:
        for k, nid in enumerate(nodes):
            lt = dz / 2 if k in (0, n) else dz
            ns[(nid, 0)] = EPSpring(m.node_springs[nid][0], ep * lt, r=1e-9)
    for k in range(n):
        m.add_element(k + 1, nodes[k], nodes[k + 1], STEEL, PIPE, "column", shear_deform=False)
    return m, nodes, list(range(1, n + 1)), ns


def _push_ground(m, nodes, ug, ns=None):
    inc = Incremental(m, {}, [Level("頭", None, nodes[-1], 1.0)], 0, node_springs=ns)
    inc.run("地盤変位", ground=[(nid, 0, ug(k)) for k, nid in enumerate(nodes)], lam_target=1.0)
    return inc


# ============================================================ 分布形
def test_ground_profile_shapes():
    """cos: ug0·cos(πζ/2H)、linear: ug0(1 − ζ/H)、uniform: ug0。ζ > H と海底面より上は 0"""
    H = 8000.0
    assert ground_profile("cos", 1.0, H, 0.0) == pytest.approx(1.0)
    assert ground_profile("cos", 1.0, H, 4000.0) == pytest.approx(math.cos(math.pi / 4))
    assert ground_profile("cos", 1.0, H, H) == pytest.approx(0.0, abs=1e-15)
    assert ground_profile("linear", 2.0, H, 2000.0) == pytest.approx(1.5)
    assert ground_profile("uniform", 2.0, H, H) == 2.0
    for s in ("cos", "linear", "uniform"):
        assert ground_profile(s, 1.0, H, H + 1.0) == 0.0
        assert ground_profile(s, 1.0, H, -1.0) == 0.0
    with pytest.raises(ValueError):
        ground_profile("sin", 1.0, H, 0.0)


# ============================================================ 単杭（理論解）
def test_free_pile_uniform_ground_rigid_translation():
    """杭頭が自由な杭に一様な地盤変位: 杭は地盤と一緒に剛体移動し、ばねの力・曲げは 0"""
    m, nodes, elems, _ = _pile(10000.0, 500.0, None, [0, 0, 1, 0, 0, 1])
    inc = _push_ground(m, nodes, lambda k: 300.0)
    for nid in nodes:
        assert inc.disp(nid, 0) == pytest.approx(300.0, rel=1e-9)
    assert max(abs(f).max() for f in inc.face.values()) < 1e-6 * 300.0 * KH * PIPE.D * 10000.0


def test_fixed_head_uniform_ground_matches_chang():
    """杭頭固定（回転・水平変位とも拘束）、一様な地盤変位 ug: 杭頭を −ug 押し込んだのと同じ。
    Chang の解（杭頭固定、突出なし）: H = 4EIβ³·ug、M0 = H/(2β)（分割 0.25 m、杭長 5/β で 1% 以内）"""
    EI = STEEL.E * PIPE.Iy
    b2 = chang_beta(KH, PIPE.D, EI)                                  # β = (kh·D / 4EI)^(1/4)
    L = round(5 / b2 / 250) * 250.0
    m, nodes, elems, _ = _pile(L, 250.0, [1] * 6, [0, 0, 1, 0, 0, 1])
    ug = 100.0
    inc = _push_ground(m, nodes, lambda k: ug)
    H = 4 * EI * b2 ** 3 * ug
    top = inc.face[elems[-1]]
    shear = math.hypot(top[7], top[8])
    moment = math.hypot(top[10], top[11])
    # 杭頭の節点のばね（負担長 Δz/2）の力は固定点へ直接流れるので、杭頭の反力 = 部材のせん断力 + そのばねの力
    k_head = m.node_springs[nodes[-1]][0] * ug
    assert shear + k_head == pytest.approx(H, rel=1e-2)
    assert moment == pytest.approx(H / (2 * b2), rel=1e-2)
    # 釣合い: 地盤ばねの力の合計 Σk(ug − u) = 杭頭の反力
    ksum = sum(m.node_springs[n][0] * (ug - inc.disp(n, 0)) for n in nodes)
    assert ksum == pytest.approx(shear + k_head, rel=1e-9)


def test_capped_soil_fixed_fixed_pile():
    """両端固定の杭、地盤変位が大きく全ての地盤ばねが上限 pu·D·負担長 に達する: 杭には等分布荷重 q = pu·D
    （節点集中）。端モーメント = Σ P·a·b²/L²（両端固定梁の集中荷重の重ね合わせ）、端のせん断 = 全荷重/2"""
    L, dz, q = 10000.0, 250.0, 0.3 * PIPE.D            # pu = 300 kN/m² → q = pu·D [N/mm]
    m, nodes, elems, ns = _pile(L, dz, [1] * 6, [1] * 6, ep=q)
    inc = _push_ground(m, nodes, lambda k: 1e3, ns)
    assert all(s.yielded and s.F == pytest.approx(-s.cap, rel=1e-5) for s in ns.values())
    P = q * dz
    Mend = sum(P * a * (L - a) ** 2 / L ** 2 for a in (k * dz for k in range(1, len(nodes) - 1)))
    top = inc.face[elems[-1]]
    assert math.hypot(top[10], top[11]) == pytest.approx(Mend, rel=1e-5)
    assert Mend == pytest.approx(q * L ** 2 / 12, rel=2e-3)            # 連続な等分布荷重の qL²/12 に近い
    assert math.hypot(top[7], top[8]) == pytest.approx(P * (len(nodes) - 2) / 2, rel=1e-5)


def test_ground_requires_node_spring():
    m, nodes, _, _ = _pile(2000.0, 500.0, None, [0, 0, 1, 0, 0, 1])
    m.add_node(99, 5000, 0, 0, fix=[1] * 6)
    inc = Incremental(m, {}, [Level("頭", None, nodes[-1], 1.0)], 0)
    with pytest.raises(ValueError, match="節点ばね"):
        inc.run("地盤変位", ground=[(99, 0, 1.0)], lam_target=1.0)


# ============================================================ 桟橋
@pytest.fixture(scope="module")
def pier():
    return load_model(EXAMPLES / "pier_level2.yaml")


def _case(pier, ug, **kw):
    rows = sorted({j for (_, j) in pier.pile_sec})
    kw.setdefault("depth", 8000.0)
    return FlowCase("t", "+y", {j: ug for j in rows}, **kw)


def test_zero_ground_matches_level2(pier):
    """地盤変位 0・換算震度 0.25 の同時載荷は、P4 のレベル2プッシュオーバーの要求点（kh = 0.25）と一致"""
    r = run_flow(pier, _case(pier, 0.0, kh=0.25, unload=False))
    l2 = run_level2(pier, "+y")
    u0 = r.inc.res.states[1][0][6 * r.inc.idx[pier.master()] + 1]       # 長期荷重時の上部工の変位
    assert r.deck["peak"] + u0 == pytest.approx(l2.demand["disp"], rel=1e-9)


def test_uniform_ground_over_full_length_moves_rigidly(pier):
    """杭先端より深い層全体が一様に動く（慣性力なし）: 桟橋全体が地盤と一緒に剛体移動し、
    上部工の変位 = ug0、杭は塑性化しない（地盤ばねの低減係数によらない）"""
    for f in (1.0, 0.1):
        r = run_flow(pier, _case(pier, 500.0, depth=60000.0, shape="uniform", kh_factor=f))
        assert r.deck["peak"] == pytest.approx(500.0, rel=1e-9)
        assert r.deck["residual"] == pytest.approx(500.0, rel=1e-9)
        assert not r.yielded and max(r.hinge_max.values()) == 0.0


def test_elastic_order_independent_and_superposition(pier):
    """弾性範囲: 載荷順序によらず同じ最終状態。除荷後は地盤変位のみの解と一致。地盤変位に比例"""
    finals = {}
    for order in ("simultaneous", "ground_first", "inertia_first"):
        r = run_flow(pier, _case(pier, 20.0, kh=0.02, order=order, unload=False))
        assert not r.yielded
        finals[order] = r.inc.u.copy()
    for u in finals.values():
        np.testing.assert_allclose(u, finals["simultaneous"], rtol=0, atol=1e-9 * abs(u).max())
    a = run_flow(pier, _case(pier, 20.0, kh=0.02))                       # 除荷あり
    g = run_flow(pier, _case(pier, 20.0))                                # 地盤変位のみ
    g2 = run_flow(pier, _case(pier, 40.0))
    np.testing.assert_allclose(a.inc.u, g.inc.u, rtol=0, atol=1e-9 * abs(g.inc.u).max())
    assert g2.deck["residual"] == pytest.approx(2 * g.deck["residual"], rel=1e-9)
    # 上部工は地盤と同じ向きに動く（杭が海底面で傾くので、突出長の分だけ海底面の地盤変位より大きくなりうる）
    assert g.deck["residual"] > 0


def test_plastic_flow_with_soil_cap_equilibrium():
    """地盤反力の上限あり・大きな側方流動: 地盤ばねと地中部が降伏し残留変位が残る。
    釣合い: 上部工の水平方向の Σ（地盤ばねの力）= 慣性力（除荷前 kh·W、除荷後 0）"""
    d = _data()
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0, "pu": 150},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0, "pu": 150}}
    d.pop("combinations")
    d["actions"]["EQ"]["directions"] = ["y"]
    d["lateral_flow"] = {"F1": {"surface": {"L": 2.0, "S": 1.2}, "depth": 10.0, "shape": "cos",
                                "kh": 0.15, "order": "simultaneous", "kh_factor": 0.2, "pu_factor": 0.5}}
    p = Pier(d)
    k_before = {nid: k.copy() for nid, k in p.model.node_springs.items()}
    case = p.lateral_flow[0]
    assert case.surface == {p.iy["L"]: 2000.0, p.iy["S"]: 1200.0}
    r = run_flow(p, case)
    assert r.reached and [ph for ph, _ in r.phases] == ["長期", "地盤変位+慣性力", "除荷"]
    assert any(s.yielded for s in r.soil.values())
    assert any(kind == "地中部" for _, kind in r.yielded)
    assert r.deck["residual"] > 0 and r.deck["peak"] >= r.deck["residual"]
    total = sum(r.soil_force.values())
    assert abs(total) < 1e-6 * sum(abs(v) for v in r.soil_force.values())     # 除荷後は 0
    for (nid, _), s in r.soil.items():                                        # 流動層の上限は低減
        zeta = next(p_["seabed"] - z for p_ in p.piles.values() for n, z, _, _ in p_["ground"] if n == nid)
        cap0 = next(c for p_ in p.piles.values() for n, _, _, c in p_["ground"] if n == nid)
        assert s.cap == pytest.approx(cap0 * (0.5 if zeta < 10000 * (1 - 1e-9) else 1.0))
        i = 6 * r.inc.idx[nid] + 1
        rel = abs(r.inc.u[i] - r.inc.ug[i])                                   # 降伏後は剛性 r·k
        assert abs(s.F) <= (s.cap + s.r * s.k * rel) * (1 + 1e-9)
    for nid, k in p.model.node_springs.items():                               # 解析後は剛性が元に戻る
        np.testing.assert_array_equal(k, k_before[nid])
    # 除荷前（unload: false）は Σ = kh·W
    r2 = run_flow(p, FlowCase(**{**case.__dict__, "unload": False}))
    assert sum(r2.soil_force.values()) == pytest.approx(0.15 * r2.W, rel=1e-6)


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["lateral_flow"]["F"].pop("depth"), "depth"),
    (lambda d: d["lateral_flow"]["F"].update(shape="sin"), "shape"),
    (lambda d: d["lateral_flow"]["F"].update(order="random"), "order"),
    (lambda d: d["lateral_flow"]["F"].update(kh_factor=0), "kh_factor"),
    (lambda d: d["lateral_flow"]["F"].update(surface={"Z": 1.0}), "surface"),
    (lambda d: d["lateral_flow"]["F"].update(direction="+z"), "direction"),
    (lambda d: d["lateral_flow"]["F"].update(foo=1), "foo"),
])
def test_flow_input_errors(patch, msg):
    d = _data()
    d["lateral_flow"] = {"F": {"surface": 0.5, "depth": 8.0}}
    d = copy.deepcopy(d)
    patch(d)
    with pytest.raises(InputError, match=msg):
        Pier(d)


def test_flow_requires_soil_spring_model():
    d = _data("pier_example.yaml")
    d["lateral_flow"] = {"F": {"surface": 0.5, "depth": 8.0}}
    p = Pier(d)
    with pytest.raises(ValueError, match="地盤ばねモデル"):
        run_flow(p, p.lateral_flow[0])
