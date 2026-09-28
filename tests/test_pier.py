"""桟橋編 P1（直杭式横桟橋のモデル生成）のテスト: 理論値・手計算との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import Building, InputError, JsceConcrete, Model, Pier, PipeSection, Steel, load_model
from rcframe.steel import chang_beta

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _data():
    return yaml.safe_load((EXAMPLES / "pier_example.yaml").read_text(encoding="utf-8"))


# ============================================================ 材料・断面
def test_pipe_section_properties():
    """φ1000 × t14: 腐食前 A = π(1000² − 972²)/4, I = π(1000⁴ − 972⁴)/64
    腐食後（外面 2 mm）: 外径 996, 内径 972"""
    p = PipeSection("P", 1000, 14, corrosion_outer=2.0)
    a = p.props(False)
    assert a["A"] == pytest.approx(math.pi * (1000 ** 2 - 972 ** 2) / 4)
    assert a["I"] == pytest.approx(math.pi * (1000 ** 4 - 972 ** 4) / 64)
    assert a["Z"] == pytest.approx(a["I"] / 500)
    assert (a["J"], a["As"]) == pytest.approx((2 * a["I"], a["A"] / 2))
    c = p.props(True)
    assert c["A"] == pytest.approx(math.pi * (996 ** 2 - 972 ** 2) / 4)
    assert (c["D"], c["t"]) == pytest.approx((996, 12))
    assert p.Iy == a["I"]                                    # 解析の剛性は腐食前（既定）
    assert PipeSection("P", 1000, 14, 2.0, stiffness="corroded").Iy == pytest.approx(c["I"])
    for bad in (dict(D=20, t=10), dict(D=1000, t=14, corrosion_outer=10, corrosion_inner=5)):
        with pytest.raises(ValueError):
            PipeSection("P", **bad)


def test_jsce_concrete_ec():
    """Ec は f'ck の表から線形補間（24 → 25, 27 → 26.5 kN/mm²）"""
    assert JsceConcrete("C", 24).E == pytest.approx(25000)
    assert JsceConcrete("C", 27).E == pytest.approx(26500)
    assert JsceConcrete("C", 24, E=30000).E == 30000
    with pytest.raises(ValueError):
        JsceConcrete("C", 100)
    s = Steel("S", 315)
    assert s.G == pytest.approx(2.0e5 / 2.6)


# ============================================================ 仮想固定点
def test_virtual_fixed_point_by_hand():
    """β = (kh·D / 4EI)^(1/4) を kN・m 単位で手計算: φ1000×t14, kh = 20000 kN/m³ → 1/β ≒ 3.81 m
    仮想固定点 = 海底面 − 1/β、杭長 = model_level − 固定点"""
    pier = load_model(EXAMPLES / "pier_example.yaml")
    I_m4 = math.pi * (1.0 ** 4 - 0.972 ** 4) / 64
    EI = 2.0e8 * I_m4                                        # kN·m²
    for row, seabed, kh in (("L", -4.0, 20000), ("S", -12.0, 15000)):
        beta = (kh * 1.0 / (4 * EI)) ** 0.25                  # 1/m
        p = pier.pile_info("B3", row)
        assert p["beta"] * 1e3 == pytest.approx(beta, rel=1e-12)
        assert p["z_fix"] / 1e3 == pytest.approx(seabed - 1 / beta, rel=1e-12)
        assert p["length"] / 1e3 == pytest.approx(2.25 - (seabed - 1 / beta), rel=1e-12)
        assert p["free_length"] / 1e3 == pytest.approx(1.5 - (seabed - 1 / beta), rel=1e-12)
    assert 1 / pier.pile_info("B1", "L")["beta"] / 1e3 == pytest.approx(3.81, abs=0.005)
    assert chang_beta(16.0, 2.0, 2.0) == pytest.approx(math.sqrt(2))   # (16·2/(4·2))^(1/4) = 4^(1/4)


def _pile_stiffness(p, sec, mat):
    """杭頭固定（回転拘束）の水平剛性: 1 / (Lf³/(12EI) + Lf/(G·As))、Lf = 自由長（剛域を除く）"""
    Lf = p["free_length"]
    return 1.0 / (Lf ** 3 / (12 * mat.E * sec.Iy) + Lf / (mat.G * sec.Asy))


def test_single_pile_fixed_head_stiffness():
    """杭 1本（仮想固定点で固定、杭頭の回転を拘束）の水平剛性が理論式と一致"""
    pier = load_model(EXAMPLES / "pier_example.yaml")
    p = pier.pile_info("B2", "M")
    e = pier.model.elements[p["eid"]]
    m = Model()
    m.add_node(1, 0, 0, p["z_fix"], fix=[1] * 6)
    m.add_node(2, 0, 0, pier.z_model, fix=[0, 1, 1, 1, 1, 1])
    m.add_element(1, 1, 2, e.mat, e.sec, "column", rigid_j=e.rigid_j)
    m.add_nodal_load("P", 2, [100e3, 0, 0, 0, 0, 0])
    r = m.analyze("P")
    assert 100e3 / r.disp[2][0] == pytest.approx(_pile_stiffness(p, e.sec, e.mat), rel=1e-9)


def _small_pier():
    """梁を十分剛にした 2×2 の桟橋（陸側・海側で杭長が異なる）"""
    d = _data()
    for k in ("actions", "combinations", "checks"):   # P2 以降の荷重生成・照査は使わない（防舷材が範囲外になる）
        d.pop(k, None)
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["sections"]["GR"] = {"b": 6000, "D": 6000, "material": "C24"}
    d["beams"] = [{"at": "*", "section": "GR"}]
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000}, "S": {"seabed": -12.0, "kh": 15000}}
    d["loads"] = {"H": {"deck": [0, 1000, 0]}}
    pier = Pier(d)
    ks = {ij: _pile_stiffness(p, pier.pile_sec[ij][0], pier.pile_sec[ij][1]) for ij, p in pier.piles.items()}
    return pier, ks


def test_deck_stiffness_and_distribution():
    """上部工の節点の回転（rx, ry）を拘束すると杭頭は回転しないので、
    上部工の水平剛性 = Σk、杭のせん断力の配分 = k/Σk が厳密に成り立つ"""
    pier, ks = _small_pier()
    for nid in pier.node_ids.values():
        pier.model.nodes[nid].fix = [0, 0, 0, 1, 1, 0]
    r = pier.model.analyze("H")
    K = sum(ks.values())
    assert 1000e3 / r.disp[pier.master()][1] == pytest.approx(K, rel=1e-9)
    shear = {ij: abs(r.section_forces(p["eid"], "face")[1][2]) for ij, p in pier.piles.items()}
    for ij in pier.piles:
        assert shear[ij] / 1000e3 == pytest.approx(ks[ij] / K, rel=1e-9)
    assert shear[(0, 0)] > shear[(0, 1)]                                # 短い陸側の杭が多く負担


def test_deck_rocking_reduces_stiffness():
    """回転を拘束しない実際のモデル: 水平力による転倒で杭に軸力が生じ、杭の軸変形で上部工が傾いて
    杭頭の回転拘束が不完全になるため、水平剛性は Σk よりやや小さい（この例で数 %）"""
    pier, ks = _small_pier()
    r = pier.model.analyze("H")
    K = sum(ks.values())
    k_deck = 1000e3 / r.disp[pier.master()][1]
    assert 0.95 * K < k_deck < K
    assert abs(r.disp[pier.node("B1", "L")][3]) > 0                     # 上部工が X 軸まわりに傾く
    shear = [abs(r.section_forces(p["eid"], "face")[1][2]) for p in pier.piles.values()]
    assert sum(shear) == pytest.approx(1000e3, rel=1e-9)                # 釣合い


# ============================================================ 計算例
def test_example_model():
    """8本 × 3列: 杭 24、梁 7×3 + 8×2 = 37、節点 = 上部工 24 + 固定点 24 + マスター 1"""
    pier = load_model(EXAMPLES / "pier_example.yaml")
    m = pier.model
    assert isinstance(pier, Pier)
    assert len(pier.piles) == 24
    assert sum(1 for k in pier.elem_ids if k[0] != "p") == 37
    assert len(m.nodes) == 49 and len(m.diaphragms) == 1
    mst = m.nodes[pier.master()]
    assert (mst.x, mst.y, mst.z) == (21000, 7000, 2250)
    e = m.elements[pier.pile("B1", "L")]
    assert e.rigid_j == pytest.approx(750) and e.rigid_i == 0
    assert m.nodes[e.ni].fix == [1] * 6


def test_example_equilibrium():
    """H: 水平反力の合計 = −1000 kN、W: 鉛直反力の合計 = 50 kN/m × 梁の総延長 238 m"""
    pier = load_model(EXAMPLES / "pier_example.yaml")
    rH = pier.model.analyze("H")
    assert sum(rH.reactions.values())[1] == pytest.approx(-1000e3, rel=1e-9)
    rW = pier.model.analyze("W")
    assert sum(rW.reactions.values())[2] == pytest.approx(50 * (42 * 3 + 14 * 8) * 1e3, rel=1e-9)


def test_pile_override_none():
    d = _data()
    d["piles"]["overrides"] = [{"at": "B1/S", "section": "none"}]
    pier = Pier(d)
    assert len(pier.piles) == 23
    with pytest.raises(KeyError):
        pier.pile("B1", "S")
    pier.node("B1", "S")                  # 梁があるので節点は残る


def test_load_model_dispatch():
    assert isinstance(load_model(EXAMPLES / "two_story_frame.yaml"), Building)
    d = _data()
    with pytest.raises(InputError, match="type"):
        Building(d)


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["piles"]["rows"]["S"].update(seabed=2.0), "海底面"),
    (lambda d: d["piles"]["rows"]["S"].update(kh=0), "kh"),
    (lambda d: d["piles"]["rows"].pop("M"), "seabed・kh がありません"),
    (lambda d: d["piles"].update(section="G1"), "鋼管"),
    (lambda d: d["beams"].append({"at": "L/*", "section": "P1"}), "鋼管断面は使えません"),
    (lambda d: d["deck"].update(soffit=2.5), "top ≧ model_level ≧ soffit"),
    (lambda d: d["sections"]["P1"]["pipe"].update(t=600), "D > 2t"),
    (lambda d: d["sections"]["G1"].update(material="C99"), "材料"),
    (lambda d: d["materials"]["C24"].update(fck=120), "Ec の表"),
    (lambda d: d["piles"]["rows"].update(Q={"seabed": -5, "kh": 1}), "未定義の Y通り"),
])
def test_pier_input_errors(patch, msg):
    d = copy.deepcopy(_data())
    patch(d)
    with pytest.raises(InputError, match=msg):
        Pier(d)
