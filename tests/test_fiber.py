"""利用可否 U3（鋼管のファイバーモデルによる M–φ）のテスト: 理論値・手計算との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import InputError, Model, Pier, run_level2
from rcframe.fiber import PipeFibers, PipeMPhi, SteelLaw, local_buckling_strain, pipe_mphi
from rcframe.hinge import BilinearHinge
from rcframe.port_level2 import pile_capacity
from rcframe.pushover import Incremental, Level
from rcframe.skeleton import Skeleton
from rcframe.steel import PipeSection, Steel

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

E, FY = 2.0e5, 315.0


def _calc(Do=1000.0, t=14.0, H=0.0, n_circ=72, n_rad=4):
    return PipeMPhi(PipeFibers(Do, Do - 2 * t, n_circ, n_rad), SteelLaw(E, FY, H))


def test_local_buckling_strain_formulas():
    """式A 1.14/(D/t) − 0.012、式B 0.44·t/R − 0.0026（R = (D − t)/2）、下限 εy、D/t > 100 で注意"""
    ecu, w = local_buckling_strain(1000, 14, FY, E, "A")
    assert ecu == pytest.approx(1.14 / (1000 / 14) - 0.012) and not w          # = 0.003960
    assert ecu == pytest.approx(0.003960, abs=1e-6)
    ecu, w = local_buckling_strain(1000, 14, FY, E, "B")
    assert ecu == pytest.approx(0.44 * 14 / 493 - 0.0026) and not w            # = 0.009895
    assert ecu == pytest.approx(0.009895, abs=1e-6)
    ecu, w = local_buckling_strain(1500, 12, FY, E, "A")                      # D/t = 125: 式は負 → εy
    assert ecu == pytest.approx(FY / E) and len(w) == 2 and "100" in w[0]
    assert local_buckling_strain(1000, 14, FY, E, "value", 0.02)[0] == 0.02
    with pytest.raises(ValueError):
        local_buckling_strain(1000, 14, FY, E, "C")
    with pytest.raises(ValueError):
        local_buckling_strain(1000, 14, FY, E, "value")


def test_steel_law():
    law = SteelLaw(E, FY, 0.01)
    assert law.stress(0.001) == pytest.approx(200.0)
    assert law.stress(-0.01) == pytest.approx(-(FY + 0.01 * E * (0.01 - FY / E)))


def test_elastic_moment_and_mesh_convergence():
    """弾性範囲: M = EI·φ（I = π(Do⁴ − Di⁴)/64）。分割を細かくすると誤差が小さくなる"""
    errs = []
    for nc, nr in ((24, 1), (72, 4), (144, 8)):
        c = _calc(n_circ=nc, n_rad=nr)
        phi = 0.5 * FY / E / 500
        _, M = c.state(phi, 0.0)
        errs.append(abs(M / (E * c.f.I * phi) - 1))
    assert errs[1] < 2e-3 and errs[2] < errs[1] < errs[0]
    assert sum(_calc().f.A) == pytest.approx(math.pi * (1000 ** 2 - 972 ** 2) / 4, rel=1e-12)   # 面積は厳密


def test_first_yield_with_axial_force():
    """最初の降伏（圧縮縁 = εy）: My = (fy − N/A)·Z。軸力は一定に保たれる"""
    c = _calc()
    A = c.f.area
    N = 0.3 * FY * A
    mp = c.curve(N, 0.01)
    assert mp.My == pytest.approx((FY - N / A) * c.f.Z, rel=2e-3)
    comp, tens = c.strain_edges(mp.phi_y, N)
    assert -comp == pytest.approx(FY / E, rel=1e-9) and abs(tens) < FY / E
    assert c._axial(*[c.state(mp.phi_y, N)[0], mp.phi_y]) == pytest.approx(-N, rel=1e-9)


def test_full_plastic_pure_bending():
    """N = 0、完全弾塑性: φ → 大で M → Zp·fy（Zp = (Do³ − Di³)/6）。剛塑性の分布でも Zp·fy"""
    c = _calc()
    Mp = c.f.Zp * FY
    assert c.full_plastic(0.0) == pytest.approx(Mp, rel=1e-3)
    _, M = c.state(100 * FY / E / 500, 0.0)
    assert M == pytest.approx(Mp, rel=2e-3) and M <= Mp * (1 + 1e-12)


@pytest.mark.parametrize("n_ratio", [0.3, 0.6])
def test_thin_wall_interaction(n_ratio):
    """薄肉の極限（D/t = 500）: 全塑性モーメントの相関 Mp(N) = Zp·fy·cos(π·N/2Ny)"""
    c = _calc(t=2.0, n_circ=360, n_rad=1)
    Ny = FY * c.f.area
    assert c.full_plastic(n_ratio * Ny) == pytest.approx(
        c.f.Zp * FY * math.cos(math.pi * n_ratio / 2), rel=3e-3)


def test_limit_curvature_pure_bending():
    """N = 0 では中立軸が中心のまま（対称）なので φu = εcu/(Do/2)。M–φ は単調増加で φu で終わる"""
    c = _calc(H=0.01)
    mp = c.curve(0.0, 0.004)
    assert mp.phi_u == pytest.approx(0.004 / 500, rel=1e-9)
    assert mp.phi[-1] == pytest.approx(mp.phi_u) and mp.M[-1] == pytest.approx(mp.Mu)
    assert all(b > a for a, b in zip(mp.M, mp.M[1:]))
    assert mp.moment_at(mp.phi_y) == pytest.approx(mp.My, rel=1e-9)
    assert mp.phi_at(mp.My) == pytest.approx(mp.phi_y, rel=1e-9)
    assert mp.phi_at(2 * mp.Mu) is None


def test_pipe_mphi_uses_corroded_section():
    """腐食後の断面（Do = 996, t = 12）で εcu（式A）を算定。式A は D/t = 83 で εcu ≒ 1.1εy と小さく、
    全塑性モーメントに達する前に限界に達する（Mu < Mp）"""
    sec = PipeSection("P", 1000, 14, corrosion_outer=2.0)
    mp = pipe_mphi(sec, Steel("S", FY), 0.0)
    assert mp.ecu == pytest.approx(1.14 / (996 / 12) - 0.012)
    assert mp.phi_u == pytest.approx(mp.ecu / 498, rel=1e-9)
    assert mp.Mu < mp.Mp
    mpb = pipe_mphi(sec, Steel("S", FY), 0.0, ecu_method="B")
    assert mpb.ecu > mp.ecu and mpb.Mu > mp.Mu


# ============================================================ 局部座屈の折れ点をもつ鋼材のヒンジ
def test_bilinear_hinge_with_buckling_point():
    """My で降伏 → 勾配 ks3 で θu まで（M = My + ks3·θu で事象「局部座屈」）→ 以後 勾配 ks4。
    除荷は 2My の幅を剛で戻り、降伏線 M = g(θ) − My で反対側に降伏する"""
    sk = Skeleton(Mc=100.0, My=100.0, theta_y=0.0, ks2=math.inf, ks3=1000.0)
    h = BilinearHinge(sk, theta_u=0.01, ks4=10.0)
    f, lim, kind = h.tangent(1)
    assert h.advance(lim, f, 1, True, lim, kind) == ["降伏"]
    f, lim, kind = h.tangent(1)
    assert (f, kind) == (1e-3, "post_u") and lim == pytest.approx(110.0)
    assert h.advance(lim - h.M, f, 1, True, lim, kind) == ["局部座屈"]
    assert (h.theta, h.M) == pytest.approx((0.01, 110.0)) and h.buckled[1] and h.state == "降伏"
    f, lim, kind = h.tangent(1)
    assert f == pytest.approx(0.1) and math.isinf(lim)
    h.advance(1.0, f, 1)                                         # θ = 0.11, M = 111
    assert (h.theta, h.M) == pytest.approx((0.11, 111.0))
    f, lim, kind = h.tangent(-1)                                 # 除荷: g(0.11) − My = 11 − 100
    assert (f, kind) == (0.0, "elastic") and lim == pytest.approx(11.0 - 100.0)


def test_pile_capacity_rotation_limit():
    """θu = (φu − φy)·Lp。θ = θu で推定する圧縮縁のひずみ = εcu、損傷の程度の区分"""
    sec = PipeSection("P", 1000, 14, corrosion_outer=2.0)
    opt = {"ecu": ("B", None), "hardening": 0.01, "hinge_length": 1.0}
    cap, _ = pile_capacity(sec, Steel("S", FY), 2e6, opt)
    c = cap.curve
    assert cap.Lp == 996.0 and cap.theta_u == pytest.approx((c.phi_u - c.phi_y) * 996.0)
    assert cap.strain(cap.theta_u, cap.Mu) == pytest.approx(c.ecu, rel=1e-9)
    assert cap.strain(0.0, cap.My) == pytest.approx(FY / E, rel=1e-6)     # 圧縮縁が先に降伏（軸圧縮）
    assert cap.level(0.0, 0.5 * cap.My) == "弾性"
    assert cap.level(0.0, cap.My) == "塑性化"
    assert cap.level(0.5 * cap.theta_u, cap.My) == "塑性化"
    assert cap.level(cap.theta_u, cap.Mu) == "局部座屈"


def test_fiber_hinge_cantilever():
    """片持ち柱（せん断変形なし）の柱脚にファイバーのヒンジ、頂部に水平力 P:
    降伏 P = My/L、局部座屈（θ = θu）で P = Mu/L、そのときの頂部変位 = PL³/(3EI) + θu·L"""
    sec = PipeSection("P", 1000, 14)
    mat = Steel("S", FY)
    L = 8000.0
    cap, _ = pile_capacity(sec, mat, 0.0, {"ecu": ("B", None), "hardening": 0.01, "hinge_length": 1.0})
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, 0, 0, L, fix=[0, 1, 1, 1, 0, 1])
    m.add_element(1, 1, 2, mat, sec, "column", shear_deform=False)
    sk = Skeleton(Mc=cap.My, My=cap.My, theta_y=0.0, ks2=math.inf, ks3=(cap.Mu - cap.My) / cap.theta_u)
    h = BilinearHinge(sk, cap.theta_u, 1e-6 * sk.ks3)
    inc = Incremental(m, {(1, 0, 0): h}, [Level("頂部", None, 2, 1.0)], 0, lateral_total=1.0)
    P = np.array([1.0, 0, 0, 0, 0, 0])
    inc.run("水平", pattern=[(2, P)], max_drift=3000.0, lateral=True)
    ev = {e["event"]: e for e in inc.res.events}
    assert ev["降伏"]["lam"] == pytest.approx(cap.My / L, rel=1e-9)
    assert ev["局部座屈"]["lam"] == pytest.approx(cap.Mu / L, rel=1e-9)
    d = cap.Mu / L * L ** 3 / (3 * mat.E * sec.Iy) + cap.theta_u * L
    assert ev["局部座屈"]["drift"] == pytest.approx(d, rel=1e-9)


# ============================================================ レベル2（ファイバーのヒンジ）
def _l2data():
    return yaml.safe_load((EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8"))


def _small(d):
    """計算時間を短くするため 2 × 2 本の杭の桟橋にする"""
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0}}
    d.pop("combinations")
    d.pop("lateral_flow", None)
    d["actions"]["EQ"]["directions"] = ["y"]
    d["level2"]["step"] = 0.02
    return d


def test_level2_with_fiber_hinges():
    """pile_hinge.model: fiber。杭頭のヒンジの降伏モーメントは M–φ の My、要求点の損傷と局部座屈の判定。
    要求点を大きくすると局部座屈の事象が起き、判定が NG になる"""
    d = _small(_l2data())
    d["level2"]["pile_hinge"] = {"model": "fiber", "ecu": "B"}
    d["level2"]["criteria"] = ["no_local_buckling"]
    d["level2"]["kh"] = 0.25
    r = run_level2(Pier(d), "+y")
    assert r.ok and r.damage and all(x["level"] == "弾性" for x in r.damage)
    for key, cap in r.caps.items():
        assert r.po.hinges[key].My == pytest.approx(cap.My)
        assert r.po.hinges[key].theta_u == pytest.approx(cap.theta_u)
    d["level2"]["kh"] = None
    d["level2"]["target_displacement"] = 0.5
    r = run_level2(Pier(d), "+y")
    assert any(e["event"] == "局部座屈" for e in r.po.events)
    assert not r.ok and r.criteria[-1][0] == "杭が局部座屈に達しない"
    assert r.damage[0]["level"] == "局部座屈" and r.damage[0]["strain"] >= r.caps[r.damage[0]["key"]].curve.ecu


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["level2"].update(pile_hinge={"model": "fem"}), "pile_hinge.model"),
    (lambda d: d["level2"].update(pile_hinge={"model": "fiber", "ecu": -0.01}), "ecu"),
    (lambda d: d["level2"].update(pile_hinge={"model": "fiber", "hinge_length": 0}), "hinge_length"),
    (lambda d: d["level2"].update(criteria=["no_local_buckling"]), "fiber"),
])
def test_pile_hinge_input_errors(patch, msg):
    d = copy.deepcopy(_l2data())
    patch(d)
    with pytest.raises(InputError, match=msg):
        Pier(d)
