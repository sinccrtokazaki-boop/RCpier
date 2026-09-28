"""桟橋編 P3（照査）のテスト: 手計算との照合"""
import copy
import math
from pathlib import Path

import pytest
import yaml

from rcframe import InputError, Pier, load_model, run_pier_checks
from rcframe.materials import BAR_AREA
from rcframe.port_check import (cracked_steel_stress, crack_width, jsce_mu, jsce_vcd, jsce_vsd,
                                jsce_vwcd, pile_ratio, pile_shear_ratio)
from rcframe.steel import PipeSection

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FCD = 24 / 1.3
B, D, DT = 1200.0, 1500.0, 100.0
d = D - DT


def _data():
    return yaml.safe_load((EXAMPLES / "pier_example.yaml").read_text(encoding="utf-8"))


# ============================================================ 曲げ耐力
def test_mu_single_reinforced():
    """単鉄筋: x = As·fyd/(k1·f'cd·b·β)、Mu = As·fyd·(d − βx/2)（k1 = 0.85, β = 0.8）"""
    As = 6 * BAR_AREA[29]
    x = As * 345 / (0.85 * FCD * B * 0.8)
    assert 0.0035 * (d - x) / x > 345 / 2.0e5                  # 引張鉄筋は降伏
    assert jsce_mu(B, D, 0.0, As, DT, DT, FCD, 345, 345) == pytest.approx(As * 345 * (d - 0.4 * x), rel=1e-9)


def test_mu_double_compression_steel_elastic():
    """複筋（圧縮鉄筋は弾性、引張鉄筋は降伏）: ひずみの適合条件から
    k1·f'cd·b·β·x² + (A's·Es·ε'cu − As·fyd)·x − A's·Es·ε'cu·dc = 0"""
    As, Asc, dc = 10 * BAR_AREA[32], 4 * BAR_AREA[29], 100.0
    Es, ecu = 2.0e5, 0.0035
    a2, a1, a0 = 0.85 * FCD * B * 0.8, Asc * Es * ecu - As * 345, -Asc * Es * ecu * dc
    x = (-a1 + math.sqrt(a1 ** 2 - 4 * a2 * a0)) / (2 * a2)
    sc = Es * ecu * (x - dc) / x
    assert x > dc and 0 < sc < 345                               # 前提の確認
    Mu = 0.85 * FCD * B * 0.8 * x * (d - 0.4 * x) + Asc * sc * (d - dc)
    assert jsce_mu(B, D, Asc, As, dc, DT, FCD, 345, 345) == pytest.approx(Mu, rel=1e-9)


# ============================================================ せん断耐力
def test_shear_capacities():
    """Vcd = βd·βp·fvcd·bw·d/γb、Vsd = Aw·fwyd·(d/1.15)/(s·γb)、Vwcd = 1.25√f'cd·bw·d/γb"""
    fvcd = 0.20 * FCD ** (1 / 3)
    bd, bp = (1000 / d) ** 0.25, (100 * 0.003) ** (1 / 3)
    assert jsce_vcd(B, d, 0.003, FCD, 1.3) == pytest.approx(bd * bp * fvcd * B * d / 1.3)
    # 上限: βd ≦ 1.5（d = 100）、βp ≦ 1.5（pw = 5%）、fvcd ≦ 0.72（f'cd = 80/1.3）
    assert jsce_vcd(B, 100, 0.003, FCD, 1.0) == pytest.approx(1.5 * bp * fvcd * B * 100)
    assert jsce_vcd(B, d, 0.05, FCD, 1.0) == pytest.approx(bd * 1.5 * fvcd * B * d)
    assert jsce_vcd(B, d, 0.003, 80 / 1.3, 1.0) == pytest.approx(bd * bp * 0.72 * B * d)
    aw = 4 * BAR_AREA[16]
    assert jsce_vsd(aw, 345, 200, d, 1.1) == pytest.approx(aw * 345 * d / 1.15 / (200 * 1.1))
    assert jsce_vwcd(B, d, FCD, 1.3) == pytest.approx(1.25 * math.sqrt(FCD) * B * d / 1.3)
    assert jsce_vwcd(B, d, 50.0, 1.0) == pytest.approx(7.8 * B * d)          # fwcd ≦ 7.8


# ============================================================ ひび割れ幅
def test_crack_width_by_hand():
    """ひび割れ断面の σse と w（計算例の梁の中央を想定した数値）"""
    As, Asc, M, n = 10 * BAR_AREA[32], 6 * BAR_AREA[29], 600e6, 2.0e5 / 25000
    a2, a1, a0 = B / 2, n * (As + Asc), -n * (Asc * DT + As * d)
    x = (-a1 + math.sqrt(a1 ** 2 - 4 * a2 * a0)) / (2 * a2)
    Icr = B * x ** 3 / 3 + n * As * (d - x) ** 2 + n * Asc * (x - DT) ** 2
    sig = n * M * (d - x) / Icr
    assert cracked_steel_stress(M, B, D, Asc, As, DT, DT, n) == pytest.approx(sig, rel=1e-12)
    c, cs, phi = 70.0, (1200 - 140 - 32) / 9, 32.0
    k2 = 15 / (24 + 20) + 0.7
    w = 1.1 * 1.0 * k2 * 1.0 * (4 * c + 0.7 * (cs - phi)) * (sig / 2.0e5 + 150e-6)   # k3 = 5·3/15 = 1
    assert crack_width(sig, c, cs, phi, 24, 150e-6) == pytest.approx(w, rel=1e-12)


# ============================================================ 鋼管杭
def test_pile_ratios_corroded():
    """腐食後（外面 2 mm）の断面: N/(A·fyd) + M/(Z·fyd)、せん断 (2Q/A)/(fyd/√3)"""
    p = PipeSection("P", 1000, 14, corrosion_outer=2.0).props(corroded=True)
    A = math.pi * (996 ** 2 - 972 ** 2) / 4
    Z = math.pi * (996 ** 4 - 972 ** 4) / 64 / 498
    assert pile_ratio(1000e3, 1000e6, p["A"], p["Z"], 315) == pytest.approx(1000e3 / (A * 315) + 1000e6 / (Z * 315))
    assert pile_shear_ratio(300e3, p["A"], 315) == pytest.approx(2 * 300e3 / A / (315 / math.sqrt(3)))


# ============================================================ 計算例
@pytest.fixture(scope="module")
def example():
    p = load_model(EXAMPLES / "pier_example.yaml")
    res = p.analyze_cases()
    return p, res, run_pier_checks(p, res)


def test_example_all_ok(example):
    p, _, rep = example
    assert rep.ng == [] and 0.5 < rep.max_ratio < 1.0
    items = {r.item for r in rep.rows}
    assert items == {"曲げ", "せん断", "せん断(腹部)", "ひび割れ幅", "軸力+曲げ", "杭せん断", "押込み", "引抜き"}
    assert {r.combo for r in rep.rows if r.item == "ひび割れ幅"} == {"永続_常時"}
    assert {r.combo for r in rep.rows if r.item == "曲げ"} == set(p.combinations)   # 永続・変動の全組合せ


def test_beam_end_section_at_pile_face(example):
    """梁端の検定断面は杭の前面（節点から杭の半径 500 mm）"""
    p, res, rep = example
    eid = p.beam("L", "B1", "B2")
    comb = p.analyze_combinations(res)["永続_常時"]
    row = next(r for r in rep.rows if r.eid == eid and r.pos == "i端" and r.item == "曲げ"
               and r.combo == "永続_常時")
    assert row.demand == pytest.approx(abs(comb.section_forces_at(eid, 500.0)[4]), rel=1e-12)


def test_pile_row_matches_forces(example):
    """杭の照査値 = 組合せの断面力（腐食後の断面）から手計算した値"""
    p, res, rep = example
    eid = p.pile("B7", "L")
    comb = p.analyze_combinations(res)["変動_接岸"]
    sf = comb.section_forces_at(eid, 0.0)
    cp = p.model.elements[eid].sec.props(corroded=True)
    exp = pile_ratio(-sf[0], math.hypot(sf[4], sf[5]), cp["A"], cp["Z"], 315.0)
    row = next(r for r in rep.rows if r.eid == eid and r.pos == "固定点" and r.item == "軸力+曲げ"
               and r.combo == "変動_接岸")
    assert row.demand == pytest.approx(exp, rel=1e-12)


def test_ng_detection():
    """配筋を減らすとひび割れ幅・曲げが NG、支持力を小さくすると押込みが NG"""
    d = _data()
    d["sections"]["G1"]["rebar"]["center"]["bottom"] = "4-D22"
    d["piles"]["rows"]["M"]["Rcd"] = 1000
    rep = run_pier_checks(Pier(d))
    ng_items = {r.item for r in rep.ng}
    assert {"ひび割れ幅", "曲げ", "押込み"} <= ng_items


def test_checks_defaults_and_errors():
    d = _data()
    d.pop("checks")
    p = Pier(d)
    assert p.checks["concrete"]["gamma_c"] == 1.3 and p.checks["crack"]["combinations"] == ["永続_常時"]
    for patch, msg in ((lambda x: x["checks"]["crack"].update(combinations=["無い"]), "組合せ"),
                       (lambda x: x["checks"].update(gamma=1.0), "未知の項目"),
                       (lambda x: x["sections"]["G1"].update(rebar={"main": "SD345", "shear": "SD345", "dt": 100,
                                                                    "x": "5-D25", "y": "5-D25",
                                                                    "hoop_x": "2-D13@100",
                                                                    "hoop_y": "2-D13@100"}), "end / center")):
        dd = copy.deepcopy(_data())
        patch(dd)
        with pytest.raises(InputError, match=msg):
            Pier(dd)
