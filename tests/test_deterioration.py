"""劣化の変数（変状連鎖の重点シナリオ D1）のテスト: 梁の主筋の断面欠損 → Mu、杭の部位ごとの腐食 → ヒンジ、
洗掘 → 海底面の低下"""
import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from rcframe import InputError, Pier
from rcframe.port_batch import Batch, apply_scour, read_deterioration, row_names
from rcframe.port_check import jsce_mu
from rcframe.port_level2 import (build_pile_hinges, corroded_section, corrosion_zone, deck_sections, pipe_mp)
from rcframe.results import Result

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _data(usability=True):
    """2 × 2 本の杭の小さな桟橋（test_batch と同じ）。杭 φ1000 t14、腐食代 2 mm、上部工の下面 +1.5"""
    d = yaml.safe_load((EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8"))
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0, "pu": 300},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0, "pu": 300}}
    d.pop("combinations")
    d.pop("lateral_flow")
    if not usability:
        d.pop("usability")
    else:
        d["usability"]["allowable"].update(pile_axial=False, fenders=[{"x": 3.0, "z": 1.0}], bollards=[{"x": 3.0}])
    d["actions"]["EQ"]["directions"] = ["y"]
    d["level2"]["step"] = 0.02
    return d


# ============================================================ 梁の主筋の断面欠損
def test_deck_mu_with_rebar_loss():
    """下端筋の断面欠損率 0.3 で、上端圧縮の Mu（下端引張）は手計算 As·fy·(d − a/2) と一致して約 0.7 倍、
    下端圧縮の Mu はほとんど変わらない"""
    p = Pier(_data())
    p.model.assemble_K()                                  # 要素の長さ L を求める
    s0 = deck_sections(p)
    s1 = deck_sections(p, (0.3, 0.0))
    assert len(s0) == len(s1) > 0
    e = p.model.elements[s0[0]["eid"]]
    rb, sec = e.sec.rebar, e.sec
    bars = rb.end if s0[0]["x"] < e.L / 4 else rb.center
    # 手計算（圧縮鉄筋を無視した長方形の応力ブロック、k1 = 0.85、β = 0.8 は a = β·x に含まれる）
    for loss, s in ((0.0, s0[0]), (0.3, s1[0])):
        As = bars.bottom.area * (1 - loss)
        a = As * rb.main.F / (0.85 * e.mat.fck * sec.b)
        assert s["Mu_pos"] == pytest.approx(As * rb.main.F * (sec.D - rb.dt - a / 2), rel=0.03)
        assert s["Mu_pos"] == pytest.approx(jsce_mu(sec.b, sec.D, bars.top.area, As, rb.dt, rb.dt, e.mat.fck,
                                                    rb.main.F, rb.main.F), rel=1e-12)
    for a0, a1 in zip(s0, s1):
        assert 0.69 < a1["Mu_pos"] / a0["Mu_pos"] < 0.75
        assert a1["Mu_neg"] == pytest.approx(a0["Mu_neg"], rel=0.02)
    s2 = deck_sections(p, (0.0, 0.4))                     # 上端筋の欠損は下端圧縮の Mu を下げる
    assert all(b["Mu_neg"] < a["Mu_neg"] for a, b in zip(s0, s2))


# ============================================================ 杭の部位ごとの腐食
def test_corrosion_zone_and_clip():
    assert corrosion_zone(1500.0, -4000.0, -1000.0) == "head"
    assert corrosion_zone(-1000.0, -4000.0, -1000.0) == "head"           # 境界は杭頭側
    assert corrosion_zone(-4000.0, -4000.0, -1000.0) == "sea"            # 海底面のヒンジは海中部
    assert corrosion_zone(-4500.0, -4000.0, -1000.0) == "soil"
    p = Pier(_data(False))
    sec = next(iter(p.pile_sec.values()))[0]
    s, w = corroded_section(sec, 3.0)
    assert w is None and s.dims(corroded=True) == (sec.D - 6.0, sec.D - 2 * sec.t)
    assert sec.corrosion_outer == 2.0                                     # もとの断面は変えない
    s, w = corroded_section(sec, 50.0)                                    # 板厚 14 mm を越える → 残り 1 mm
    assert "開孔" in w and (s.D - 2 * s.corrosion_outer - s.dims(corroded=True)[1]) / 2 == pytest.approx(1.0)


def _gravity(pier):
    gc = list(pier.level2["gravity"])
    return Result.combine("+".join(gc), {pier.model.analyze(c): 1.0 for c in gc})


def test_pile_hinges_use_zone_corrosion():
    """杭頭のヒンジは杭頭側、海底面のヒンジは海中部、地中のヒンジは地中部の腐食量の断面の Mp(N)。
    一様な腐食量の断面で作ったヒンジ・手計算 Zp·fy·cos(π|N|/2Ny) と一致"""
    p = Pier(_data(False))
    rG = _gravity(p)
    corr = {"head": 4.0, "sea": 1.0, "soil": 0.5, "boundary": -1000.0}
    opt = {"model": "cos", "corrosion": corr}
    hz, labels, kinds, warns = build_pile_hinges(p, rG, 1, 0.01, math.inf, opt)
    assert not warns
    zone_of = {k: ("head" if lab.endswith("杭頭") else "sea" if lab.endswith("海底面") else "soil")
               for k, lab in labels.items()}
    assert set(zone_of.values()) == {"head", "sea", "soil"}
    saved = dict(p.pile_sec)
    for zone in ("head", "sea", "soil"):
        p.pile_sec = {ij: (replace(sec, corrosion_outer=corr[zone]), mat) for ij, (sec, mat) in saved.items()}
        hu, *_ = build_pile_hinges(p, rG, 1, 0.01, math.inf, {"model": "cos"})
        for k, z in zone_of.items():
            if z == zone:
                assert hz[k].My == pytest.approx(hu[k].My, rel=1e-12)
    p.pile_sec = saved
    # 手計算（杭頭のヒンジの 1 つ）
    k = next(k for k, z in zone_of.items() if z == "head")
    sec, mat = next(iter(saved.values()))
    f = rG.forces[k[0]]["face"]
    N = f[0] if k[2] == 0 else -f[6]
    Do, Di = sec.D - 8.0, sec.D - 2 * sec.t
    Ny = math.pi * (Do ** 2 - Di ** 2) / 4 * mat.fy
    assert hz[k].My == pytest.approx((Do ** 3 - Di ** 3) / 6 * mat.fy * math.cos(math.pi * abs(N) / (2 * Ny)), rel=1e-12)
    # 腐食量を与えない部位は断面の値（2 mm）のまま
    hp, *_ = build_pile_hinges(p, rG, 1, 0.01, math.inf, {"model": "cos", "corrosion": {"head": 4.0, "boundary": -1000.0}})
    h0, *_ = build_pile_hinges(p, rG, 1, 0.01, math.inf, {"model": "cos"})
    assert all(hp[k].My == pytest.approx(h0[k].My) for k, z in zone_of.items() if z != "head")
    assert all(hp[k].My < h0[k].My for k, z in zone_of.items() if z == "head")
    # 板厚に達する腐食量は開孔の注意
    _, _, _, w = build_pile_hinges(p, rG, 1, 0.01, math.inf, {"model": "cos", "corrosion": {"head": 30.0, "boundary": -1000.0}})
    assert any("開孔" in x for x in w)


# ============================================================ 洗掘・入力
def test_read_deterioration_and_scour():
    d = _data(False)
    names = row_names(d)
    det = read_deterioration({"scour": 1.0, "scour_ratio": {"S": 1.0, "L": 0.5}, "rebar_loss_bottom": 0.2,
                              "corrosion_head": 3.0}, names)
    assert det["scour"] == {"S": 1.0, "L": 0.5} and det["rebar_loss"] == (0.2, 0.0)
    assert det["corrosion"] == {"head": 3.0, "boundary": -1000.0}
    assert read_deterioration({"scour": 0.4}, names)["scour"] == {"L": 0.4, "S": 0.4}
    assert read_deterioration({}, names) == dict(rebar_loss=None, corrosion=None, scour=None)
    d2 = apply_scour(d, det["scour"])
    assert d2["piles"]["rows"]["S"]["seabed"] == pytest.approx(-9.0) and d["piles"]["rows"]["S"]["seabed"] == -8.0
    p0, p1 = Pier(d), Pier(d2)
    for ij, info in p0.piles.items():                     # 自由長が洗掘の分だけ長くなる
        ds = {"L": 500.0, "S": 1000.0}[p0.gy[ij[1]][0]]
        assert p1.piles[ij]["free_length"] - info["free_length"] == pytest.approx(ds)


@pytest.mark.parametrize("params, msg", [
    ({"rebar_loss_bottom": 1.0}, "断面欠損率"),
    ({"corrosion_sea": -1.0}, "腐食量"),
    ({"corrosion_boundary": -1.0}, "corrosion_boundary"),
    ({"scour_ratio": {"S": 1.0}}, "scour_ratio"),
    ({"scour": {"X": 1.0}}, "列にない"),
    ({"scour": -0.5}, "0 以上"),
])
def test_deterioration_input_errors(params, msg):
    with pytest.raises(InputError, match=msg):
        read_deterioration(params, ["L", "S"])


def test_scour_below_tip_error():
    with pytest.raises(InputError, match="杭先端"):
        apply_scour(_data(False), {"L": 17.0})


# ============================================================ 一括実行
def test_batch_with_deterioration():
    """洗掘で上部工の変位が増え、下端筋の欠損で許容荷重は増えない。劣化の変数は結果の列に残る。
    杭頭の腐食量が板厚に達すると開孔の列が立つ"""
    d = _data()
    d["scenarios"] = {
        "base": {"direction": "+y", "depth": 8.0, "shape": "cos", "surface": 0.0, "kh": 0.2,
                 "corrosion_head": 3.0, "corrosion_sea": 2.0, "corrosion_soil": 1.0},
        "grid": {"scour": [0.0, 2.0], "rebar_loss_bottom": [0.0, 0.6]},
        "refine": {"when": "none"},
    }
    rows = Batch(d).run(workers=1, log=lambda *a: None)
    assert len(rows) == 4 and all(r["corrosion_head"] == 3.0 and r["perforation"] is False for r in rows)
    by = {(r["scour"], r["rebar_loss_bottom"]): r for r in rows}
    for rl in (0.0, 0.6):
        assert by[(2.0, rl)]["deck_peak"] > by[(0.0, rl)]["deck_peak"] * 1.02
    for sc in (0.0, 2.0):
        for k in ("allow_surcharge", "allow_berthing", "allow_tractive"):
            assert by[(sc, 0.6)][k] <= by[(sc, 0.0)][k] * (1 + 1e-9)
        assert by[(sc, 0.6)]["allow_surcharge"] < 0.5 * by[(sc, 0.0)]["allow_surcharge"]     # 梁の Mu で決まる
        assert "梁の曲げ" in by[(sc, 0.6)]["allow_surcharge_by"]
    assert by[(2.0, 0.0)]["allow_berthing"] < by[(0.0, 0.0)]["allow_berthing"]          # 突出長が増える
    d2 = copy.deepcopy(d)
    d2["scenarios"]["base"]["corrosion_head"] = 30.0
    d2["scenarios"]["grid"] = {"scour": [0.0]}
    r2 = Batch(d2).run(workers=1, log=lambda *a: None)
    assert r2[0]["perforation"] is True
