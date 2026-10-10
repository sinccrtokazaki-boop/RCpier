"""劣化のモデル（変状連鎖の重点シナリオ D2、rcframe/port_deterioration.py）のテスト:
Fick の拡散と腐食の開始・鉄筋の断面欠損・杭の腐食量を手計算と照合し、一括実行のシナリオの作り方を確認する"""
import copy
import json
import math

import numpy as np
import pytest

from rcframe import InputError
from rcframe import port_deterioration as dm
from rcframe.port_batch import Batch, read_scenarios, scenario_params

from test_deterioration import _data


# ============================================================ モデル（手計算）
def test_chloride_profile_and_initiation_time():
    """C(x, t) = Ci + (C0 − Ci)·erfc(x/(2√(Dt)))。Clim = Ci + (C0 − Ci)·erfc(1) なら t_i = x²/(4D) ちょうど"""
    C0, D, Ci, x = 13.0, 1.0, 0.3, 70.0                       # x = 7 cm
    assert dm.chloride(x, 25.0, C0, D, Ci) == pytest.approx(Ci + (C0 - Ci) * math.erfc(0.7), rel=1e-12)
    assert dm.chloride(0.0, 10.0, C0, D) == pytest.approx(C0)
    assert dm.chloride(x, 0.0, C0, D, Ci) == Ci
    Clim = Ci + (C0 - Ci) * math.erfc(1.0)
    assert dm.initiation_time(x, C0, D, Clim, Ci) == pytest.approx(7.0 ** 2 / 4.0, rel=1e-10)    # 12.25 年
    # 逆関数の確認：t_i で鉄筋の位置の濃度が Clim
    ti = dm.initiation_time(100.0, 13.0, 0.8, 1.6)
    assert dm.chloride(100.0, ti, 13.0, 0.8) == pytest.approx(1.6, rel=1e-9)
    # かぶり 2 倍 → t_i は 4 倍、D 2 倍 → 1/2
    assert dm.initiation_time(200.0, 13.0, 0.8, 1.6) == pytest.approx(4 * ti)
    assert dm.initiation_time(100.0, 13.0, 1.6, 1.6) == pytest.approx(ti / 2)
    assert dm.initiation_time(100.0, 13.0, 1.0, 14.0) == math.inf and dm.initiation_time(100.0, 13.0, 1.0, 0.2, 0.3) == 0.0


def test_rebar_and_pile_loss():
    """icorr = 2 μA/cm² で 20 年 → Δr = 0.0116·2·20 = 0.464 mm、D22 の欠損率 1 − (21.072/22)²。
    杭: 寿命 30 年・0.2 mm/年・寿命まで 0 → 40 年で 2.0 mm、防食率 90%（0.1）・0.15 mm/年 → 0.15·(3 + 10) = 1.95 mm"""
    assert dm.rebar_loss(40.0, 20.0, 2.0, 22.0) == pytest.approx(1 - ((22 - 2 * 0.464) / 22) ** 2, rel=1e-12)
    assert dm.rebar_loss(15.0, 20.0, 2.0, 22.0) == 0.0 and dm.rebar_loss(40.0, math.inf, 2.0, 22.0) == 0.0
    assert dm.rebar_loss(500.0, 0.0, 10.0, 22.0) == dm.LOSS_MAX
    assert dm.pile_loss(40.0, 0.2, 30.0, 0.0) == pytest.approx(2.0)
    assert dm.pile_loss(40.0, 0.15, 30.0, 0.1) == pytest.approx(1.95)
    assert dm.pile_loss(20.0, 0.15, 30.0, 0.1) == pytest.approx(0.3)
    assert dm.pile_loss(40.0, 0.03) == pytest.approx(1.2)


# ============================================================ シナリオの作り方
def _det():
    return {"age": [40, 60],
            "chloride": {"C0": {"lognormal": {"median": 13.0, "beta": 0.3}}, "D": {"lognormal": {"median": 1.0, "beta": 0.5}},
                         "cover": {"normal": {"mean": 100.0, "sd": 10.0}}, "Clim": {"lognormal": {"median": 1.6, "beta": 0.25}},
                         "icorr": {"lognormal": {"median": 2.0, "beta": 0.7}}, "bar": 22},
            "pile": {"boundary": -1.0,
                     "head": {"life": {"lognormal": {"median": 30.0, "beta": 0.5}}, "rate": {"lognormal": {"median": 0.2, "beta": 0.4}}},
                     "sea": {"life": {"lognormal": {"median": 30.0, "beta": 0.5}}, "rate": {"lognormal": {"median": 0.15, "beta": 0.4}},
                             "factor": 0.1},
                     "soil": {"rate": {"uniform": [0.02, 0.04]}}}}


def _scen(det=True, n=50, seed=3):
    s = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos"},
         "distributions": {"surface": {"lognormal": {"median": 0.5, "beta": 0.7}}, "kh": {"uniform": [0.0, 0.15]}},
         "sample": n, "seed": seed}
    if det:
        s["deterioration"] = _det()
    return s


def test_scenarios_per_age_with_common_draws():
    """抽出 1 組ごとに年数の数だけシナリオ。同じ組の年数どうしは抽出値が同じで、劣化は年数とともに増える。
    地震の変数の抽出値は deterioration の有無で変わらない。seed で再現"""
    ps = scenario_params(read_scenarios(_scen()))
    assert len(ps) == 100
    plain = scenario_params(read_scenarios(_scen(det=False)))
    for k in range(50):
        a, b = ps[2 * k], ps[2 * k + 1]
        assert (a["age"], b["age"]) == (40, 60)
        assert {x: a["_det"][x] for x in a["_det"] if not x.endswith("t_init")} == \
            {x: b["_det"][x] for x in b["_det"] if not x.endswith("t_init")}
        assert a["surface"] == b["surface"] == plain[k]["surface"] and a["kh"] == plain[k]["kh"]
        for key in ("rebar_loss_bottom", "corrosion_head", "corrosion_sea", "corrosion_soil"):
            assert b[key] >= a[key]
        assert a["corrosion_boundary"] == -1.0
        # 変数は抽出値から手計算で再現できる
        g = a["_det"]
        ti = dm.initiation_time(max(g["chloride.cover"], 5.0), g["chloride.C0"], g["chloride.D"], g["chloride.Clim"])
        assert a["rebar_loss_bottom"] == pytest.approx(dm.rebar_loss(40, ti, g["chloride.icorr"], 22.0))
        assert a["corrosion_sea"] == pytest.approx(dm.pile_loss(40, g["pile.sea.rate"], g["pile.sea.life"], 0.1))
        assert a["corrosion_soil"] == pytest.approx(40 * g["pile.soil.rate"])
    assert ps == scenario_params(read_scenarios(_scen()))
    assert ps != scenario_params(read_scenarios(_scen(seed=4)))
    json.dumps(ps)                                             # 結果に書ける（inf を含まない）


def test_sample_statistics():
    """抽出値の統計量: ln C0 の平均 ≒ ln 13・標準偏差 ≒ 0.3、地中部の腐食速度は一様分布 [0.02, 0.04]"""
    cfg = read_scenarios(_scen(n=4000))
    ps = scenario_params(cfg)[::2]
    c0 = np.log([p["_det"]["chloride.C0"] for p in ps])
    assert c0.mean() == pytest.approx(math.log(13.0), abs=0.02) and c0.std() == pytest.approx(0.3, rel=0.05)
    rs = np.array([p["_det"]["pile.soil.rate"] for p in ps])
    assert rs.min() >= 0.02 and rs.max() <= 0.04 and rs.mean() == pytest.approx(0.03, rel=0.02)


@pytest.mark.parametrize("patch, msg", [
    (lambda s: s["deterioration"].pop("age"), "age"),
    (lambda s: s["deterioration"].update(age=[40, 40]), "重複"),
    (lambda s: s["deterioration"]["chloride"].pop("bar"), "bar"),
    (lambda s: s["deterioration"]["chloride"].update(D={"gamma": 1}), "確率分布"),
    (lambda s: s["deterioration"]["pile"]["sea"].update(factor=1.5), "factor"),
    (lambda s: s["deterioration"]["pile"]["soil"].pop("rate"), "rate"),
    (lambda s: s["base"].update(corrosion_head=1.0), "deterioration から作る"),
    (lambda s: s.pop("sample"), "sample"),
    (lambda s: s.update(grid={"kh": [0.1]}), "grid"),
])
def test_deterioration_input_errors(patch, msg):
    s = copy.deepcopy(_scen())
    patch(s)
    with pytest.raises(InputError, match=msg):
        read_scenarios(s)


# ============================================================ 一括実行
def test_batch_with_deterioration_models():
    """小さな桟橋で 3 組 × 2 年数。結果の行に年数・腐食の開始・抽出値が残り、60 年の許容荷重は 40 年以下"""
    d = _data()
    d["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "surface": 0.0, "kh": 0.0},
                      "deterioration": _det(), "sample": 3, "seed": 5, "refine": {"when": "none"}}
    d["scenarios"]["deterioration"]["chloride"]["icorr"] = 20.0      # 欠損を大きくして効き方を見る
    rows = Batch(d).run(workers=1, log=lambda *a: None)
    assert len(rows) == 6 and {r["age"] for r in rows} == {40, 60}
    for r in rows:
        det = json.loads(r["det"])
        assert r["t_init"] == det["chloride.t_init"] and r["rebar_loss_bottom"] >= 0
    rows.sort(key=lambda r: (r["det"], r["age"]))
    for a, b in zip(rows[::2], rows[1::2]):
        assert json.loads(a["det"])["chloride.C0"] == json.loads(b["det"])["chloride.C0"]
        assert b["allow_surcharge"] <= a["allow_surcharge"] * (1 + 1e-9)
