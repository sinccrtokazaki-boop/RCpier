"""本震時の記録（ガイドライン 第2版 表 D-5 の本震時の条件、G1）のテスト:
上部工の梁の M/Mu（履歴中・残留）、杭の軸力 / 設計支持力、杭の残留モーメント / My、判定の設定"""
import csv
import math

import pytest

from rcframe.grid import InputError
from rcframe.pier import Pier
from rcframe.port_batch import COLUMNS, Batch
from rcframe.port_flow import read_flow_cases, run_flow
from rcframe.port_level2 import deck_moment, deck_sections
from rcframe.port_usability import STATUS, judge, read_usability
from rcframe.results import Result

from test_deterioration import _data

RCD, RTD = 4000.0, 1500.0                    # 設計支持力 [kN]（仮の値）


def _pier_with_capacity():
    d = _data()
    for row in d["piles"]["rows"].values():
        row.update(Rcd=RCD, Rtd=RTD)
    return Pier(d)


@pytest.fixture(scope="module")
def flow():
    p = _pier_with_capacity()
    case = read_flow_cases(p, {"c": {"direction": "+y", "surface": 0.4, "depth": 6.0, "shape": "cos",
                                     "kh": 0.15, "kh_factor": 0.2}})[0]
    return p, run_flow(p, case)


def _ratio(M, s):
    return M / s["Mu_pos"] if M >= 0 else -M / s["Mu_neg"]


def test_deck_ratio_matches_section_forces(flow):
    """履歴中・残留の M/Mu は、各状態の材端力から results.section_forces_at（deck_moment）で求めた値の最大と一致する"""
    p, fr = flow
    secs = deck_sections(p)
    assert secs
    st = fr.inc.res.states
    s0, s1 = fr.phases[0][1], fr.phases[-1][1]
    worst = 0.0
    for s in range(s0, s1 + 1):
        if s not in st:
            continue
        forces = {e: {"face": f, "node": f} for e, f in st[s][1].items()}
        r = Result(p.model, "", {}, forces, {}, {c: 1.0 for c in fr.gravity})
        worst = max(worst, max(_ratio(deck_moment(r, sec), sec) for sec in secs))
    hist, label = fr.deck_ratio(p, secs)
    assert hist == pytest.approx(worst, rel=1e-10) and label
    res, _ = fr.deck_ratio(p, secs, residual=True)
    assert res == pytest.approx(max(_ratio(deck_moment(fr.final, sec), sec) for sec in secs), rel=1e-10)
    assert 0 < res <= hist
    # 主筋の断面欠損で Mu が下がれば M/Mu は大きくなる
    assert fr.deck_ratio(p, deck_sections(p, (0.5, 0.5)))[0] > hist


def test_pile_axial_ratio_gravity_matches_linear():
    """慣性力・地盤変位なし（長期のみ）の軸力 / Rcd は、線形解析の長期の軸力から求めた値と一致する"""
    p = _pier_with_capacity()
    case = read_flow_cases(p, {"c": {"direction": "+y", "surface": 0.0, "depth": 6.0, "kh": 0.0}})[0]
    fr = run_flow(p, case)
    assert len(fr.phases) == 1                                           # 長期の段階だけ
    m = p.model
    rG = Result.combine("+".join(fr.gravity), {m.analyze(c): 1.0 for c in fr.gravity})
    Ns = [rG.forces[pp["elements"][-1]]["face"][0] for pp in p.piles.values()]
    assert all(N > 0 for N in Ns)                                        # 圧縮
    ratio, label = fr.pile_axial_ratio(p)
    assert ratio == pytest.approx(max(Ns) / (RCD * 1e3), rel=1e-6) and "押込み" in label


def test_pile_axial_ratio_history_and_none(flow):
    p, fr = flow
    hist, _ = fr.pile_axial_ratio(p)
    res, _ = fr.pile_axial_ratio(p, residual=True)
    assert hist >= res > 0
    p0 = Pier(_data())                                                   # 支持力の入力がなければ None
    case = read_flow_cases(p0, {"c": {"direction": "+y", "surface": 0.0, "depth": 6.0, "kh": 0.1}})[0]
    assert run_flow(p0, case).pile_axial_ratio(p0) == (None, "")


def test_residual_moment_ratio(flow):
    """杭のヒンジの残留 |M|/My の最大は、最終状態のヒンジのモーメントから直接求めた値と一致する"""
    p, fr = flow
    v, label = fr.residual_moment_ratio()
    h = fr.inc.hinges
    assert v == pytest.approx(max(abs(x.M) / x.My for x in h.values()), rel=1e-12) and label
    last = fr.inc.res.hinge_M[fr.phases[-1][1]]
    assert v == pytest.approx(max(abs(last[k]) / h[k].My for k in h), rel=1e-12)


# ============================================================ 判定
def _opts(**limits):
    return read_usability(None, {"limits": limits})


def test_judge_deck_and_axial():
    base = {"deck_residual": 0.0}
    assert judge({**base, "deck_mu_max": 0.99}, _opts())[0] == STATUS[0]
    st, why = judge({**base, "deck_mu_max": 1.02}, _opts())
    assert st == STATUS[2] and "上部工の梁が Mu" in why[0] and "1.02" in why[0]
    assert judge({**base, "deck_mu_max": 1.02}, _opts(deck_mu="ok"))[0] == STATUS[0]
    st, why = judge({**base, "pile_axial_max": 1.1}, _opts())                # 既定は制限付き（フェーズ 1 は特性値）
    assert st == STATUS[1] and "設計支持力" in why[0]
    assert judge({**base, "pile_axial_max": None, "deck_mu_max": None}, _opts())[0] == STATUS[0]


def test_judge_pile_head_buckling():
    row = {"deck_residual": 0.0, "n_buckled": 3, "n_buckled_head": 2, "n_plastic": 1}
    st, why = judge(row, _opts())                                            # 既定: 杭頭も 局部座屈 と同じ
    assert st == STATUS[2] and "利用不可: 杭の局部座屈 3 箇所（杭頭 2、地中部 1）" in why
    o = _opts(pile_damage={"杭頭の局部座屈": "limited"})
    assert o["limits"]["pile_damage"] == {"塑性化": "limited", "局部座屈": "unusable", "杭頭の局部座屈": "limited"}
    st, why = judge(row, o)                                                   # 地中部の 1 箇所で利用不可
    assert st == STATUS[2] and "利用不可: 杭の局部座屈（地中部）1 箇所" in why
    assert "制限付き利用: 杭の局部座屈（杭頭）2 箇所" in why
    assert judge({**row, "n_buckled": 2}, o)[0] == STATUS[1]                  # 杭頭だけなら制限付き
    st, why = judge({**row, "n_buckled_head": None}, o)                       # 位置が不明なら全て地中部の扱い
    assert st == STATUS[2] and "杭の局部座屈 3 箇所" in why[0]
    # 局部座屈だけを指定すると杭頭も同じ設定
    assert _opts(pile_damage={"局部座屈": "limited"})["limits"]["pile_damage"]["杭頭の局部座屈"] == "limited"


@pytest.mark.parametrize("lim, msg", [({"deck_mu": "bad"}, "deck_mu"), ({"pile_axial": 1}, "pile_axial"),
                                      ({"pile_damage": {"杭頭": "ok"}}, "pile_damage"),
                                      ({"pile_damage": {"杭頭の局部座屈": "no"}}, "杭頭の局部座屈")])
def test_limits_input_errors(lim, msg):
    with pytest.raises(InputError, match=msg):
        read_usability(None, {"limits": lim})


# ============================================================ 一括実行の列
def test_batch_rows_have_main_shock_columns(tmp_path):
    d = _data()
    for row in d["piles"]["rows"].values():
        row.update(Rcd=RCD, Rtd=RTD)
    d["scenarios"] = {"base": {"direction": "+y", "depth": 6.0, "shape": "cos", "kh_factor": 0.2, "surface": 0.3},
                      "grid": {"kh": [0.0, 0.15]}, "refine": {"when": "all"}}
    rows = Batch(d).run(str(tmp_path / "r"), workers=1, log=lambda *a: None)
    assert {"deck_mu_max", "deck_mu_res", "pile_axial_max", "pile_m_res", "n_buckled_head"} <= set(COLUMNS)
    for r in rows:
        assert 0 < r["deck_mu_res"] <= r["deck_mu_max"] + 1e-12
        assert 0 < r["pile_axial_max"] and 0 <= r["pile_m_res"]
        if r["tier"] == "refine":
            assert 0 <= r["n_buckled_head"] <= r["n_buckled"]
        else:
            assert r["n_buckled_head"] is None
    with open(tmp_path / "r.csv", encoding="utf-8-sig") as f:
        head = next(csv.reader(f))
    assert "deck_mu_max" in head and "pile_axial_max" in head
    assert not any(math.isnan(r["deck_mu_max"]) for r in rows)
