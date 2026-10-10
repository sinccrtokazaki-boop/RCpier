"""余震の照査（ガイドライン 第2版 表 D-5 の余震時の条件、G2）のテスト:
本震後の状態から慣性力 kh·W を両向きに漸増し、限界に最初に達する kh（allow_aftershock）を求める"""
import math

import pytest

from rcframe import FlowCase, Pier, run_flow
from rcframe.grid import InputError
from rcframe.port_cards import allowable_bars
from rcframe.port_level2 import deck_moment, deck_sections, inertia_pattern
from rcframe.port_usability import STATUS, allowable_loads, judge, read_usability, usability_row
from rcframe.pushover import spring_moment

from test_usability import _data, _drop, _undamaged


def _opts(p, **kw):
    return {**p.usability, "loads": ["aftershock"], "max": {"aftershock": 5.0}, **kw}


def _linear_inertia(p):
    """D のみの状態と、上部工の重心の慣性力 kh = 1（+y）の線形解析"""
    m = p.model
    rD = m.analyze("D")
    for nid, F in inertia_pattern(p, 1, 1.0)[0]:
        m.add_nodal_load("_u", nid, F)
    return rD, m.analyze("_u")


def _first(lims_and_values):
    """[(限界 − 長期の値, 単位あたりの値)] → 両向き（±1）で最初に限界に達する係数"""
    lam = math.inf
    for gap, unit in lims_and_values:
        for s in (1, -1):
            if unit * s != 0 and gap / (unit * s) > 0:
                lam = min(lam, gap / (unit * s))
    return lam


def test_aftershock_pile_yield_matches_linear():
    """仮想固定点モデル・杭の塑性化のみ: 許容震度 = 両向きの min (±My − M_D)/M_kh（線形の重ね合わせ）"""
    p = Pier(_data("fixed_point"))
    fr = _undamaged(p)
    al = allowable_loads(p, fr, _opts(p, deck_mu=False, pile_axial=False))["aftershock"]
    rD, ru = _linear_inertia(p)
    items = []
    for key, h in fr.inc.hinges.items():
        eid, plane, end = key
        MD = spring_moment(rD.forces[eid]["face"], plane, end)
        Mu = spring_moment(ru.forces[eid]["face"], plane, end)
        items += [(h.My - MD, Mu), (-h.My - MD, Mu)]
    _drop(p)
    assert not al.capped and al.governing == "杭の塑性化" and "本震と" in al.where
    assert al.value == pytest.approx(_first(items), rel=1e-6)


def test_aftershock_deck_mu_matches_linear():
    """杭の限界なし・梁の Mu のみ: 許容震度 = 両向きの min (±Mu − M_D)/M_kh。
    杭が先に降伏すると応力が再配分されるので、杭の降伏強度を大きくして弾性のままにする"""
    d = _data("fixed_point")
    d["materials"]["SKK490"]["fy"] = 5000
    p = Pier(d)
    al = allowable_loads(p, _undamaged(p), _opts(p, pile="none", pile_axial=False))["aftershock"]
    rD, ru = _linear_inertia(p)
    items = []
    for s in deck_sections(p):
        MD, Mk = deck_moment(rD, s), deck_moment(ru, s)
        items += [(s["Mu_pos"] - MD, Mk), (-s["Mu_neg"] - MD, Mk)]
    _drop(p)
    assert al.governing == "梁の曲げ M = Mu"
    assert al.value == pytest.approx(_first(items), rel=1e-9)


def test_damaged_state_reduces_aftershock():
    """側方流動で被災すると、杭の降伏で決まる許容震度は被災前より小さく、本震と同じ向きで決まる。
    算定後の状態は元のまま"""
    p = Pier(_data())
    rows = sorted({j for (_, j) in p.pile_sec})
    a0 = allowable_loads(p, _undamaged(p), _opts(p))["aftershock"]
    fr = run_flow(p, FlowCase("1", "+y", {j: 300.0 for j in rows}, 8000.0, kh_factor=0.1))
    u0 = fr.inc.u.copy()
    a1 = allowable_loads(p, fr, _opts(p))["aftershock"]
    assert 0 <= a1.value < a0.value
    assert "本震と同じ向き" in a1.where
    assert (fr.inc.u == u0).all()
    row = usability_row(p, fr, _opts(p))                  # 結果の行の列
    assert row["allow_aftershock"] == pytest.approx(a1.value) and "本震と同じ向き" in row["allow_aftershock_by"]
    bars = allowable_bars(row, {"required": {"aftershock": 0.15}})       # 判定カードの棒グラフ（震度の表示）
    assert f"kh = {a1.value:.3f} / kh = 0.150" in bars


# ============================================================ 入力と判定
def test_read_and_judge_aftershock():
    d = {"required": {"tractive": 300, "aftershock": 0.15}}
    o = read_usability(None, d)
    assert o["loads"] == ["tractive", "aftershock"] and o["max"]["aftershock"] == 1.0
    assert o["limits"]["aftershock"] == "unusable"
    base = {"deck_residual": 0.0, "allow_tractive": 1000.0}
    assert judge({**base, "allow_aftershock": 0.2}, o)[0] == STATUS[0]
    st, why = judge({**base, "allow_aftershock": 0.1}, o)
    assert st == STATUS[2] and "余震" in why[0] and "kh = 0.100" in why[0] and "kh = 0.150" in why[0]
    o2 = read_usability(None, {**d, "limits": {"aftershock": "limited"}})
    assert judge({**base, "allow_aftershock": 0.0}, o2)[0] == STATUS[1]   # 0 でも設定どおり（載せられない扱いにしない）
    with pytest.raises(InputError, match="aftershock"):
        read_usability(None, {**d, "limits": {"aftershock": "x"}})
