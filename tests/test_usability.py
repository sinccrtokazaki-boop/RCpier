"""利用可否 U5（判定・許容荷重・超過確率）のテスト: 線形解析・理論値との照合"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import norm

from rcframe import FlowCase, InputError, Pier, Result, run_flow
from rcframe.port_batch import Batch, read_scenarios, scenario_params
from rcframe.port_level2 import deck_moment, deck_sections
from rcframe.port_loads import WeightSum
from rcframe.port_usability import (STATUS, allowable_loads, final_rows, fit_fragility, judge,
                                    probability_summary, read_usability, usability_row)
from rcframe.pushover import spring_moment

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _data(model="soil_spring"):
    d = yaml.safe_load((EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8"))
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["model"] = model
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0, "Rcd": 5000, "Rtd": 2000},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0, "Rcd": 6000, "Rtd": 2500}}
    d.pop("combinations")
    d.pop("lateral_flow")
    d.pop("scenarios")
    d["actions"]["EQ"]["directions"] = ["y"]
    d["level2"]["step"] = 0.02
    d["usability"] = {
        "required": {"surcharge": 10, "berthing": 300, "tractive": 300},
        "limits": {"residual_displacement": {"limited": 0.1, "unusable": 0.5},
                   "pile_rotation": {"limited": 0.005, "unusable": 0.02}},
        "allowable": {"fenders": [{"x": 3.0, "z": 1.0}], "bollards": [{"x": 3.0}]},
    }
    return d


def _undamaged(p):
    """地盤変位・慣性力なし（長期荷重のみ）の状態"""
    rows = sorted({j for (_, j) in p.pile_sec})
    return run_flow(p, FlowCase("0", "+y", {j: 0.0 for j in rows}, 8000.0))


def _linear(p, case_fn):
    """D のみ（Ls を除いた状態）と、単位荷重のケース "_u" の線形解析（"_u" は _drop で消す）"""
    m = p.model
    rD = m.analyze("D")
    case_fn()
    return rD, m.analyze("_u")


def _drop(p):
    p.model.member_loads.pop("_u", None)
    p.model.nodal_loads.pop("_u", None)


# ============================================================ 許容荷重（弾性範囲 = 線形解析）
def test_allowable_surcharge_deck_mu_matches_linear():
    """杭の限界なし・梁の Mu のみ: 許容上載荷重 = min over 断面 (±Mu − M_D)/M_q（線形の重ね合わせ）"""
    p = Pier(_data())
    opts = {**p.usability, "pile": "none", "pile_axial": False, "loads": ["surcharge"],
            "max": {"surcharge": 1e4, "berthing": 1, "tractive": 1}}
    fr = _undamaged(p)
    al = allowable_loads(p, fr, opts)["surcharge"]
    rD, ru = _linear(p, lambda: p._area_load("_u", 1e-3, WeightSum()))
    lam = math.inf
    for s in deck_sections(p):
        MD, Mq = deck_moment(rD, s), deck_moment(ru, s)
        for lim in (s["Mu_pos"], -s["Mu_neg"]):
            if Mq != 0 and (lim - MD) / Mq > 0:
                lam = min(lam, (lim - MD) / Mq)
    _drop(p)
    assert not al.capped and al.governing == "梁の曲げ M = Mu"
    assert al.value == pytest.approx(lam, rel=1e-9)


def test_allowable_axial_matches_linear():
    """梁・杭の曲げの限界なし・杭の支持力のみ: 許容上載荷重 = min (Rcd − N_D)/N_q（押込み）"""
    p = Pier(_data())
    opts = {**p.usability, "pile": "none", "deck_mu": False, "loads": ["surcharge"],
            "max": {"surcharge": 1e5, "berthing": 1, "tractive": 1}}
    al = allowable_loads(p, _undamaged(p), opts)["surcharge"]
    rD, ru = _linear(p, lambda: p._area_load("_u", 1e-3, WeightSum()))
    lam = math.inf
    for (i, j), info in p.piles.items():
        eid = info["elements"][-1]
        Rcd = p.pile_capacity[j][0]
        ND, Nq = rD.forces[eid]["face"][0], ru.forces[eid]["face"][0]
        lam = min(lam, (Rcd - ND) / Nq)
    _drop(p)
    assert al.governing == "杭の押込み N = Rcd"
    assert al.value == pytest.approx(lam, rel=1e-9)


def test_allowable_berthing_pile_yield_matches_linear():
    """仮想固定点モデル・杭の塑性化のみ: 許容接岸力 = min over ヒンジ (±Mp − M_D)/M_B（ヒンジの位置の材端モーメント）"""
    p = Pier(_data("fixed_point"))
    opts = {**p.usability, "deck_mu": False, "pile_axial": False, "loads": ["berthing"],
            "max": {"surcharge": 1, "berthing": 1e6, "tractive": 1}}
    fr = _undamaged(p)
    al = allowable_loads(p, fr, opts)["berthing"]
    y_face = p.gy[-1][1] + p.overhang["sea"]
    rD, ru = _linear(p, lambda: p._point_load("_u", [3000.0, y_face, 1000.0], [0.0, -1e3, 0.0], "u"))
    lam = math.inf
    for key, h in fr.inc.hinges.items():
        eid, plane, end = key
        MD = spring_moment(rD.forces[eid]["face"], plane, end)
        MB = spring_moment(ru.forces[eid]["face"], plane, end)
        for lim in (h.My, -h.My):
            if MB != 0 and (lim - MD) / MB > 0:
                lam = min(lam, (lim - MD) / MB)
    _drop(p)
    assert al.governing == "杭の塑性化"
    assert al.value == pytest.approx(lam, rel=1e-6)


def test_allowable_restores_state():
    """許容荷重の算定後: 解析の状態・モデルの荷重ケース・地盤ばねは元のまま"""
    p = Pier(_data())
    fr = _undamaged(p)
    u0 = fr.inc.u.copy()
    cases = set(p.model.member_loads) | set(p.model.nodal_loads)
    k0 = {n: k.copy() for n, k in p.model.node_springs.items()}
    allowable_loads(p, fr, p.usability)
    np.testing.assert_array_equal(fr.inc.u, u0)
    assert set(p.model.member_loads) | set(p.model.nodal_loads) == cases
    for n, k in p.model.node_springs.items():
        np.testing.assert_array_equal(k, k0[n])


def test_damaged_state_reduces_allowable():
    """側方流動で被災すると、杭の塑性化で決まる許容牽引力は被災前より小さい"""
    p = Pier(_data())
    rows = sorted({j for (_, j) in p.pile_sec})
    opts = {**p.usability, "loads": ["tractive"]}
    a0 = allowable_loads(p, _undamaged(p), opts)["tractive"].value
    fr = run_flow(p, FlowCase("1", "+y", {j: 300.0 for j in rows}, 8000.0, kh_factor=0.1))
    a1 = allowable_loads(p, fr, opts)["tractive"].value
    assert 0 <= a1 < a0


# ============================================================ 判定
def test_judge_boundaries():
    opts = read_usability(None, _data()["usability"])
    base = {"deck_residual": 0.0, "head_rot_res": 0.0, "inground_rot_res": 0.0, "n_yield": 0,
            "allow_surcharge": 50, "allow_berthing": 1000, "allow_tractive": 1000}
    assert judge(base, opts) == ("利用可", [])
    assert judge({**base, "deck_residual": 0.0999}, opts)[0] == "利用可"
    assert judge({**base, "deck_residual": 0.1}, opts)[0] == STATUS[1]
    assert judge({**base, "deck_residual": -0.5}, opts)[0] == "利用不可"
    assert judge({**base, "inground_rot_res": 0.005}, opts)[0] == STATUS[1]
    assert judge({**base, "n_yield": 2}, opts)[0] == STATUS[1]                 # 1段目: 降伏 = 塑性化
    assert judge({**base, "n_buckled": 1, "n_plastic": 0}, opts)[0] == "利用不可"
    assert judge({**base, "allow_berthing": 299.9}, opts)[0] == STATUS[1]
    assert judge({**base, "allow_surcharge": 0.0}, opts)[0] == "利用不可"
    s, why = judge({**base, "deck_residual": 0.2, "n_buckled": 3, "n_plastic": 1}, opts)
    assert s == "利用不可" and len(why) == 3


@pytest.mark.parametrize("patch, msg", [
    (lambda u: u["limits"].update(residual_displacement={"limited": 0.5, "unusable": 0.1}), "limited"),
    (lambda u: u["allowable"].update(pile="strain"), "pile"),
    (lambda u: u["required"].update(crane=10), "crane"),
    (lambda u: u["limits"].update(pile_damage={"塑性化": "bad"}), "pile_damage"),
    (lambda u: u["allowable"].update(fenders=[{"x": 21.0}]), "範囲外"),
])
def test_usability_input_errors(patch, msg):
    d = _data()
    patch(d["usability"])
    with pytest.raises(InputError, match=msg):
        Pier(d)


# ============================================================ 確率
def test_fragility_fit_recovers_parameters():
    """既知の θ = 0.5, β = 0.4 から作った 0/1 のデータから最尤法で θ, β が戻る"""
    rng = np.random.default_rng(3)
    x = np.exp(rng.uniform(math.log(0.05), math.log(5.0), 6000))
    y = rng.uniform(size=x.size) < norm.cdf(np.log(x / 0.5) / 0.4)
    th, b = fit_fragility(x, y)
    assert th == pytest.approx(0.5, rel=0.05) and b == pytest.approx(0.4, rel=0.08)
    assert fit_fragility([1, 2], [1, 1]) is None


def test_distribution_sampling():
    """lognormal の抽出: ln x の平均 = ln(median)、標準偏差 = β。seed で再現。choice の確率"""
    cfg = read_scenarios({"base": {"depth": 8.0},
                          "distributions": {"surface": {"lognormal": {"median": 0.5, "beta": 0.6}},
                                            "kh": {"uniform": [0.0, 0.2]},
                                            "shape": {"choice": ["cos", "linear"], "p": [0.75, 0.25]}},
                          "sample": 4000, "seed": 2})
    ps = scenario_params(cfg)
    lx = np.log([q["surface"] for q in ps])
    assert lx.mean() == pytest.approx(math.log(0.5), abs=0.03) and lx.std() == pytest.approx(0.6, rel=0.04)
    assert all(0 <= q["kh"] <= 0.2 for q in ps) and all(q["depth"] == 8.0 for q in ps)
    assert np.mean([q["shape"] == "cos" for q in ps]) == pytest.approx(0.75, abs=0.03)
    assert ps == scenario_params(cfg)
    with pytest.raises(InputError, match="sample"):
        read_scenarios({"distributions": {"kh": {"uniform": [0, 0.2]}}})
    with pytest.raises(InputError, match="どちらか一方"):
        read_scenarios({"grid": {"kh": [0.1]}, "distributions": {"kh": {"uniform": [0, 0.2]}}, "sample": 3})


def test_probability_summary_synthetic():
    """判定の確率 = 件数の比（標準誤差 √(p(1−p)/n)）。2段目の行があればそちらを用いる"""
    opts = read_usability(None, _data()["usability"])
    rows = []
    for k in range(10):
        st = STATUS[0] if k < 6 else STATUS[1] if k < 9 else STATUS[2]
        rows.append({"id": f"S{k}", "tier": "screen", "status": STATUS[0], "surface": 0.1 * (k + 1),
                     "allow_berthing": 1000.0})
        rows.append({"id": f"S{k}", "tier": "refine", "status": st, "surface": 0.1 * (k + 1),
                     "allow_berthing": 100.0 if k >= 8 else 1000.0})
    s = probability_summary(rows, {**opts, "required": {"berthing": 300}})
    assert s["n"] == 10 and len(final_rows(rows)) == 10
    assert s["status"][STATUS[1]][0] == pytest.approx(0.3)
    assert s["status"]["利用不可"] == pytest.approx((0.1, math.sqrt(0.1 * 0.9 / 10)))
    assert s["allow"]["berthing"][0] == pytest.approx(0.2)
    th, b = s["fragility"]["注意事項あり以上"]
    assert 0.6 < th < 0.7                                        # 0.6 と 0.7 の間で切り替わる（完全分離）


def test_batch_with_usability():
    """一括実行の行に許容荷重と判定が付き、行の値から判定し直しても同じ"""
    d = _data()
    d["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "kh_factor": 0.1},
                      "grid": {"surface": [0.0, 0.3]}, "refine": "all"}
    b = Batch(d)
    rows = b.run(workers=1, log=lambda *a: None)
    assert len(rows) == 4
    opts = b.pier.usability
    for r in rows:
        assert r["status"] in STATUS and r["allow_tractive"] is not None
        assert judge(r, opts)[0] == r["status"]
    s = probability_summary(rows, opts)
    assert s["n"] == 2
