"""桟橋編 P4（地盤ばねモデル・レベル2プッシュオーバー）のテスト: 理論解・手計算との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import InputError, Model, Pier, load_model, run_level2, run_pier_checks
from rcframe.hinge import EPSpring
from rcframe.port_level2 import pipe_mp
from rcframe.steel import PipeSection

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _data(name="pier_level2.yaml"):
    return yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))


# ============================================================ 部品
def test_pipe_full_plastic_moment():
    """Mp(N) = Zp·fy·cos(π|N|/2Ny)、Zp = (Do³ − Di³)/6（腐食後: Do = 996, Di = 972）"""
    sec = PipeSection("P", 1000, 14, corrosion_outer=2.0)
    Zp = (996 ** 3 - 972 ** 3) / 6
    Ny = math.pi * (996 ** 2 - 972 ** 2) / 4 * 315
    assert pipe_mp(sec, 315, 0.0) == pytest.approx(Zp * 315)
    assert pipe_mp(sec, 315, 0.3 * Ny) == pytest.approx(Zp * 315 * math.cos(0.15 * math.pi))
    assert pipe_mp(sec, 315, -0.3 * Ny) == pytest.approx(pipe_mp(sec, 315, 0.3 * Ny))
    assert pipe_mp(sec, 315, 1.01 * Ny) == 0.0


def test_ep_spring():
    """弾完全塑性ばね: 上限まで k、上限で降伏（事象）、以後 r·k、除荷は k"""
    s = EPSpring(10.0, 100.0, r=1e-3)
    k, lim, kind = s.tangent(1)
    assert (k, lim, kind) == (10.0, 100.0, "elastic")
    assert s.advance(100.0, hit=True, limit=100.0) == ["地盤降伏"]
    k, lim, kind = s.tangent(1)
    assert kind == "plastic" and k == pytest.approx(0.01) and math.isinf(lim)
    s.advance(0.01 * 5)
    k, lim, _ = s.tangent(-1)                       # 除荷は弾性、反対側の上限まで
    assert k == 10.0 and lim == -100.0


# ============================================================ 地盤ばねモデル
def test_soil_spring_single_pile_matches_chang():
    """杭頭固定・突出長 h の杭（せん断変形なし）: 地盤ばねモデルの杭頭変位が
    Chang の解 δ = P{(1 + βh)³ + 2}/(12EIβ³) に一致（分割 0.25 m、杭長 5/β で 0.5% 以内）"""
    d = _data()
    d["piles"]["segment"] = 0.25
    for row in d["piles"]["rows"].values():
        row.pop("tip")
    p = Pier(d)
    info = p.pile_info("B2", "L")
    src = p.model
    ids = info["elements"]
    m = Model()
    for n in {n for k in ids for n in (src.elements[k].ni, src.elements[k].nj)}:
        nd = src.nodes[n]
        m.add_node(n, nd.x, nd.y, nd.z, fix=nd.fix)
        if n in src.node_springs:
            m.add_node_spring(n, src.node_springs[n])
    head = src.elements[info["eid"]].nj
    m.nodes[head].fix = [0, 1, 1, 1, 1, 1]
    for k in ids:
        el = src.elements[k]
        m.add_element(k, el.ni, el.nj, el.mat, el.sec, "column", rigid_j=el.rigid_j, shear_deform=False)
    m.add_nodal_load("P", head, [100e3, 0, 0, 0, 0, 0])
    r = m.analyze("P")
    e = src.elements[info["eid"]]
    b, h = info["beta"], info["free_length"]
    chang = 100e3 * ((1 + b * h) ** 3 + 2) / (12 * e.mat.E * e.sec.Iy * b ** 3)
    assert r.disp[head][0] == pytest.approx(chang, rel=5e-3)
    assert -sum(f[0] for f in r.spring_forces.values()) == pytest.approx(100e3, rel=1e-9)   # 釣合い


def test_soil_model_structure_and_p3_checks():
    """地盤ばねモデル: 杭先端は鉛直・ねじりを拘束、地中部の分割、P3 の照査は地中部の分割点も対象"""
    p = load_model(EXAMPLES / "pier_level2.yaml")
    info = p.pile_info("B1", "L")
    assert info["tip"] == -24000 and len(info["segments"]) == 40          # (−4 − (−24)) / 0.5
    assert p.model.nodes[info["tip_node"]].fix == [0, 0, 1, 0, 0, 1]
    ks = [k for _, _, k, _ in info["ground"]]
    assert ks[0] == pytest.approx(ks[1] / 2) and ks[-1] == pytest.approx(ks[1] / 2)   # 両端は負担長 1/2
    assert ks[1] == pytest.approx(20000e-6 * 1000 * 500)                             # kh·D·Δz
    rep = run_pier_checks(p)
    assert any(r.pos.startswith("地中") for r in rep.rows if r.item == "軸力+曲げ")
    assert not rep.warnings or "配筋" not in " ".join(rep.warnings)      # 杭の要素を梁として扱わない


# ============================================================ 崩壊荷重（仮想仕事法）
def test_fixed_point_collapse_load():
    """仮想固定点モデル: 全杭の杭頭と仮想固定点が全塑性になる崩壊荷重 = Σ 2Mp(N)/Lf（Lf: 自由長）"""
    d = _data("pier_example.yaml")
    d["level2"] = {"directions": ["+y"], "kh": 0.1, "max_displacement": 2.0, "step": 0.05,
                   "post_yield_ratio": 1e-6, "criteria": ["no_mechanism"]}
    p = Pier(d)
    r = run_level2(p, "+y")
    Pu = 0.0
    for (i, j), info in p.piles.items():
        key = (info["eid"], 1, 1)                  # 杭頭のばね（軸力は杭に沿って一定なので固定点側も同じ Mp）
        Pu += 2 * r.po.hinges[key].sk[1].My / info["free_length"]
    assert r.po.mechanism_step is not None
    lam_final = r.po.lateral[-1]["lam"]
    assert lam_final * r.W == pytest.approx(Pu, rel=2e-3) and lam_final * r.W > Pu
    assert sum(e["event"] == "降伏" for e in r.po.events) == 2 * len(p.piles)


# ============================================================ レベル2（計算例）
@pytest.fixture(scope="module")
def level2():
    p = load_model(EXAMPLES / "pier_level2.yaml")
    return p, run_level2(p, "+y")


def test_level2_demand_and_criteria(level2):
    """要求点は換算震度 kh = 0.25 の点（ステップ内を線形補間）、判定 3 項目がすべて OK"""
    p, r = level2
    assert r.W == pytest.approx(p.weights["地震用"][0])
    assert r.demand["reached"] and r.demand["kh"] == pytest.approx(0.25, rel=1e-9)
    assert [ok for _, ok, _ in r.criteria] == [True, True, True] and r.ok
    assert 0 < r.deck["ratio"] < 1
    # 杭頭が地中部より先に全塑性になる
    first = {}
    for e in r.po.events:
        if e["event"] == "降伏":
            first.setdefault(r.kinds[e["key"]], e["step"])
    assert first["杭頭"] < first["地中部"]


def test_level2_equilibrium(level2):
    """最終状態: 水平力 λ·W = 地盤ばねの反力の合計（杭先端は水平方向に拘束しない）"""
    p, r = level2
    u, _ = r.po.states[r.po.steps[-1]["step"]]
    idx, _ = p.model._index()
    ky = sum(k * u[6 * idx[nid] + 1] for info in p.piles.values() for nid, _, k, _ in info["ground"])
    assert ky == pytest.approx(r.po.lateral[-1]["lam"] * r.W, rel=1e-6)


def test_level2_target_displacement_and_not_reached():
    """要求変位で指定した場合は、その変位でステップが区切られる。届かない照査用震度は NG"""
    d = _data()
    d["level2"].pop("kh")
    d["level2"].update(target_displacement=0.10, max_displacement=0.15)
    r = run_level2(Pier(d), "+y")
    assert r.demand["disp"] == pytest.approx(100.0, rel=1e-9)
    d = _data()
    d["level2"].update(kh=5.0, max_displacement=0.1)
    r = run_level2(Pier(d), "+y")
    assert not r.demand["reached"] and not r.ok


def test_soil_yield_events():
    """地盤反力の上限 pu を与えると地盤ばねが降伏し、釣合い: λ·W = Σ（地盤ばねの力）"""
    d = _data()
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0, "pu": 150},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0, "pu": 150}}
    d.pop("combinations")
    d["actions"]["EQ"]["directions"] = ["y"]
    d["level2"].update(kh=0.3, max_displacement=0.3, step=0.01, criteria=["no_mechanism"])
    p = Pier(d)
    k_before = {nid: k.copy() for nid, k in p.model.node_springs.items()}
    r = run_level2(p, "+y")
    assert any(e["event"] == "地盤降伏" for e in r.po.events)
    # 全ての地盤ばねが上限つき（非線形）なので、ばねの力の合計 = 水平力 λ·W
    assert sum(s.F for s in r.soil.values()) == pytest.approx(r.po.lateral[-1]["lam"] * r.W, rel=1e-6)
    assert any(s.yielded for s in r.soil.values())
    # 解析後は地盤ばねの剛性が元に戻っている（線形解析に影響しない）
    for nid, k in p.model.node_springs.items():
        np.testing.assert_array_equal(k, k_before[nid])


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["level2"].update(target_displacement=0.2), "どちらか一方"),
    (lambda d: d["level2"].update(criteria=["no_collapse"]), "criteria"),
    (lambda d: d["level2"].update(gravity={"D": 1.2}), "1.0 のみ"),
    (lambda d: d["level2"].update(directions=["+z"]), "directions"),
    (lambda d: d["piles"].update(model="spring"), "piles.model"),
    (lambda d: d["piles"]["rows"]["L"].update(tip=-2.0), "杭先端"),
])
def test_level2_input_errors(patch, msg):
    d = copy.deepcopy(_data())
    patch(d)
    with pytest.raises(InputError, match=msg):
        Pier(d)
