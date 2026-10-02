"""U1（除荷・状態の引き継ぎ・変位制御）のテスト: 理論値との照合"""
import math

import numpy as np
import pytest

from rcframe import Concrete, Model, RectSection
from rcframe.hinge import BilinearHinge, Hinge
from rcframe.pushover import Incremental, Level, residual_table
from rcframe.skeleton import Skeleton, member_end_stiffness

FC24 = Concrete("Fc24", 24)
COL = RectSection("C", 600, 600)
L = 3000.0
P0 = np.array([1e3, 0, 0, 0, 0, 0])           # 基準の水平外力（荷重係数 1 で 1 kN）


def _cantilever():
    """片持ち柱（柱脚にトリリニアの材端ばね）。頂部は X・Z 方向の並進と Y 軸まわりの回転が自由"""
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, 0, 0, L, fix=[0, 1, 0, 1, 0, 1])
    m.add_element(1, 1, 2, FC24, COL, "column")
    K0 = member_end_stiffness(FC24.E * COL.Iz, FC24.G * COL.Asy, L)
    sk = Skeleton.from_member(150e6, 450e6, 0.3, K0, 0.01)
    hinges = {(1, 0, 0): Hinge(sk, sk)}
    inc = Incremental(m, hinges, [Level("1F", None, 2, L)], 0, lateral_total=P0[0])
    inc.run("長期", lam_target=1.0)
    return m, inc


def _k_elastic():
    """片持ち柱の弾性の水平剛性 1/(L³/(3EI) + L/(G·As))"""
    return 1.0 / (L ** 3 / (3 * FC24.E * COL.Iz) + L / (FC24.G * COL.Asy))


def test_residual_displacement_cantilever():
    """降伏後に除荷: 除荷中は材端ばねが剛なので部材は弾性的に戻り、残留変位 = δmax − Pmax/K（弾性）"""
    m, inc = _cantilever()
    inc.run("水平", pattern=[(2, P0)], max_drift=0.02, lateral=True)
    dmax, Pmax = inc.disp(2, 0), inc.lam_lat * 1e3
    assert Pmax > 450e6 / L                                     # 降伏している
    inc.unload([(2, P0)])
    assert inc.lam_lat == pytest.approx(0.0, abs=1e-12)
    assert inc.disp(2, 0) == pytest.approx(dmax - Pmax / _k_elastic(), rel=1e-9)
    assert inc.res.steps[-1]["phase"] == "除荷" and inc.res.steps[-1]["Q"] == pytest.approx(0, abs=1e-6)


def test_reload_returns_to_peak_then_skeleton():
    """除荷後に再載荷: 元の最大点（δmax, Pmax）に戻り、その後は単調載荷の骨格曲線に復帰"""
    m, inc = _cantilever()
    pat = [(2, P0)]
    inc.run("水平", pattern=pat, max_drift=0.01, lateral=True)
    dmax, lmax = inc.disp(2, 0), inc.lam_lat
    inc.unload(pat)
    inc.run("再載荷", pattern=pat, disp_target=(2, 0, dmax), lateral=True)
    assert inc.disp(2, 0) == pytest.approx(dmax, rel=1e-9)
    assert inc.lam_lat == pytest.approx(lmax, rel=1e-9)
    inc.run("再載荷", pattern=pat, max_drift=0.015, lateral=True)
    _, mono = _cantilever()                                     # 単調載荷の同じ変位での荷重
    mono.run("水平", pattern=pat, max_drift=0.015, lateral=True)
    assert inc.lam_lat == pytest.approx(mono.lam_lat, rel=1e-9)
    assert inc.disp(2, 0) == pytest.approx(mono.disp(2, 0), rel=1e-9)


def test_displacement_control_matches_load_path():
    """変位制御: 指定の変位に正確に止まり、そのときの荷重は層間変形角で止めた場合と同じ"""
    _, a = _cantilever()
    a.run("水平", pattern=[(2, P0)], disp_target=(2, 0, 25.0), lateral=True)
    _, b = _cantilever()
    b.run("水平", pattern=[(2, P0)], max_drift=25.0 / L, lateral=True)
    assert a.disp(2, 0) == pytest.approx(25.0, rel=1e-12)
    assert a.lam_lat == pytest.approx(b.lam_lat, rel=1e-9)


def test_snapshot_restore_and_residual_table():
    """残留変位の表: 状態の保存・復元で 1本の載荷経路から複数の最大点の残留値を得る。
    各行は、その最大点まで単独で押して除荷した結果と一致する"""
    _, inc = _cantilever()
    peaks = [0.0003, 0.005, 0.01, 0.02]                         # 0.0003 はひび割れ前（約 0.0006）
    rows = residual_table(inc, [(2, P0)], peaks)
    assert [r["peak"] for r in rows] == peaks
    for r in rows:
        _, one = _cantilever()
        one.run("水平", pattern=[(2, P0)], max_drift=r["peak"], lateral=True)
        Qp = one.lam_lat * 1e3
        one.unload([(2, P0)])
        assert r["Q_peak"] == pytest.approx(Qp, rel=1e-9)
        assert r["residual"][0] == pytest.approx(one.drifts()[0], rel=1e-9, abs=1e-15)
    # 弾性範囲（ひび割れ前）の最大点では残留 0、以降は単調増加
    res = [r["residual"][0] for r in rows]
    assert abs(res[0]) < 1e-12 and all(b > a for a, b in zip(res[1:], res[2:]))
    # 表の作成後、状態は最後の最大点（除荷前）に戻っている
    assert inc.drifts()[0] == pytest.approx(0.02, rel=1e-9)


def test_portal_unload_after_mechanism():
    """門形ラーメン（剛な梁、柱の上下に鋼材の BilinearHinge）: 崩壊後に除荷すると、各ヒンジは 2Mp の幅で
    剛のまま戻るので構造は弾性的に応答し、残留変位 = δmax − Pmax/K（弾性）"""
    h, span, Mp = 3000.0, 6000.0, 300e6
    m = Model()
    for nid, x, z in ((1, 0, 0), (2, span, 0), (3, 0, h), (4, span, h)):
        m.add_node(nid, x, 0, z, fix=[1] * 6 if z == 0 else [0, 1, 0, 1, 0, 1])
    m.add_element(1, 1, 3, FC24, COL, "column")
    m.add_element(2, 2, 4, FC24, COL, "column")
    m.add_element(3, 3, 4, FC24, RectSection("G", 3000, 3000))
    m.add_nodal_load("U", 3, P0)
    k_el = 1e3 / m.analyze("U").disp[3][0]                      # 弾性の水平剛性（線形解析）
    K0 = member_end_stiffness(FC24.E * COL.Iz, FC24.G * COL.Asy, h)
    sk = Skeleton.from_member(Mp, Mp, 0.5, K0, 1e-3)
    hinges = {(eid, 0, end): BilinearHinge(sk) for eid in (1, 2) for end in (0, 1)}
    inc = Incremental(m, hinges, [Level("1F", None, 3, h)], 0, lateral_total=1e3)
    inc.run("長期", lam_target=1.0)
    inc.run("水平", pattern=[(3, P0)], max_drift=0.03, lateral=True)
    assert sum(e["event"] == "降伏" for e in inc.res.events) == 4
    dmax, Pmax = inc.disp(3, 0), inc.lam_lat * 1e3
    inc.unload([(3, P0)])
    assert inc.disp(3, 0) == pytest.approx(dmax - Pmax / k_el, rel=1e-9)
    assert all(h.state == "降伏" for h in inc.hinges.values())  # ヒンジの履歴は残る


def test_use_load_after_damage():
    """状態の引き継ぎ: 被災状態（除荷後）から別の荷重（ここでは鉛直荷重）を載せられ、
    降伏したばねの状態と残留変位を保ったまま弾性的に応答する"""
    m, inc = _cantilever()
    pat = [(2, P0)]
    inc.run("水平", pattern=pat, max_drift=0.01, lateral=True)
    inc.unload(pat)
    ur = inc.disp(2, 0)
    m.add_nodal_load("V", 2, [0, 0, -500e3, 0, 0, 0])
    inc.run("利用", cases=["V"], lam_target=1.0)
    assert inc.disp(2, 0) == pytest.approx(ur, rel=1e-9)        # 鉛直荷重では水平変位は変わらない
    assert inc.disp(2, 2) == pytest.approx(-500e3 * L / (FC24.E * COL.A), rel=1e-9)
    assert inc.hinges[(1, 0, 0)].state == "降伏"                 # 降伏の履歴は残る（向きは符号の定義による）


def test_bilinear_hinge_kinematic():
    """鋼材の BilinearHinge: My で降伏 → 降伏線 M = My + ks3·θ に沿って硬化 → 除荷は剛のまま
    2My 戻り、M = ks3·θ − My で反対側に再降伏（移動硬化）"""
    sk = Skeleton(Mc=100.0, My=100.0, theta_y=0.0, ks2=math.inf, ks3=1000.0)
    h = BilinearHinge(sk)
    f, lim, kind = h.tangent(1)
    assert (f, lim, kind) == (0.0, 100.0, "elastic")
    assert h.advance(100.0, f, 1, True, lim, kind) == ["降伏"]
    f, lim, kind = h.tangent(1)
    assert (f, kind) == (1 / 1000.0, "post")
    h.advance(20.0, f, 1)                                        # θ = 0.02, M = 120
    assert (h.theta, h.M) == pytest.approx((0.02, 120.0))
    f, lim, kind = h.tangent(-1)                                 # 除荷: 剛で M = 20 − 100 = −80 まで
    assert (f, lim, kind) == (0.0, pytest.approx(-80.0), "elastic")
    h.advance(lim - h.M, f, -1, True, lim, kind)
    assert h.theta == pytest.approx(0.02) and h.yielded[-1]      # 反対側の降伏
    f, lim, kind = h.tangent(-1)
    assert kind == "post"
    with pytest.raises(ValueError):
        BilinearHinge(Skeleton.from_member(50.0, 100.0, 0.3, 1e4, 0.01))   # トリリニアは不可


def test_stop_condition_required():
    _, inc = _cantilever()
    with pytest.raises(ValueError):
        inc.run("水平", pattern=[(2, P0)])
