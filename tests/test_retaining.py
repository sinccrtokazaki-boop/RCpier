"""土留部と渡版の判定（変状連鎖の重点シナリオ D3、後処理）のテスト"""
import copy

import pytest

from rcframe import InputError, Pier
from rcframe.port_batch import Batch
from rcframe.port_usability import STATUS, judge, read_retaining, read_usability

from test_deterioration import _data

LIMITS = {"bridge_opening": {"limited": 0.1, "unusable": 0.2}, "bridge_closing": {"limited": 0.3, "unusable": 0.5},
          "apron_step": {"limited": 0.05, "unusable": 0.2}}


def _opts():
    return read_usability(None, {"limits": LIMITS})


@pytest.mark.parametrize("row, status, word", [
    ({"bridge_rel": 0.05, "apron_step": 0.0}, "利用可", None),
    ({"bridge_rel": 0.10, "apron_step": 0.0}, STATUS[1], "渡版の開き 0.100"),
    ({"bridge_rel": 0.25, "apron_step": 0.0}, "利用不可", "渡版の開き"),
    ({"bridge_rel": -0.25, "apron_step": 0.0}, "利用可", None),             # 寄る方向は bridge_closing で判定
    ({"bridge_rel": -0.35, "apron_step": 0.0}, STATUS[1], "土留部が上部工に寄る 0.350"),
    ({"bridge_rel": 0.0, "apron_step": 0.06}, STATUS[1], "段差 0.060"),
    ({"bridge_rel": 0.0, "apron_step": 0.20}, "利用不可", "段差"),
])
def test_judge_boundaries(row, status, word):
    st, reasons = judge({"deck_residual": 0.0, **row}, _opts())
    assert st == status
    if word:
        assert any(word in r for r in reasons)


def test_read_retaining_errors():
    assert read_retaining({"wall_ratio": 0.8, "settlement_ratio": 0.4}) == (0.8, None, 0.4)
    with pytest.raises(InputError, match="0 以上"):
        read_retaining({"wall_ratio": -1})
    with pytest.raises(InputError, match="どちらか"):
        read_retaining({"settlement": 0.1, "settlement_ratio": 0.4})
    d = _data()
    d["usability"]["retaining"] = {"row": "X"}
    with pytest.raises(InputError, match="杭の列"):
        Pier(d)


def _batch_data(**base):
    d = _data()
    d["usability"]["limits"].update(LIMITS)
    d["usability"]["limits"].pop("residual_displacement")
    d["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "kh_factor": 0.1, "kh": 0.0,
                               "surface_ratio": {"L": 1.0, "S": 0.6}, **base},
                      "grid": {"surface": [0.0, 0.5]}, "refine": {"when": "none"}}
    return d


def test_batch_retaining_values():
    """土留部の移動 = 比 × 土留部の列（既定は海底面の最も高い列 L）の地盤変位、開き = 上部工の残留変位 − 移動、
    段差 = 比 × 移動。地盤変位 0 なら移動・段差 0。判定の理由に入る"""
    rows = Batch(_batch_data(wall_ratio=0.8, settlement_ratio=0.5)).run(workers=1, log=lambda *a: None)
    by = {r["surface"]: r for r in rows}
    r0, r1 = by[0.0], by[0.5]
    assert r0["wall_move"] == 0.0 and r0["apron_step"] == 0.0 and r0["bridge_rel"] == pytest.approx(r0["deck_residual"])
    assert r1["wall_move"] == pytest.approx(0.8 * 0.5)
    assert r1["bridge_rel"] == pytest.approx(r1["deck_residual"] - 0.4)
    assert r1["apron_step"] == pytest.approx(0.2)
    assert r1["status"] == "利用不可" and "段差 0.200" in r1["reasons"]
    # 土留部の列を S（地盤変位 0.6 × 0.5 = 0.3）にすると移動は 0.8 × 0.3
    d = _batch_data(wall_ratio=0.8, settlement=0.01)
    d["usability"]["retaining"] = {"row": "S"}
    r = next(x for x in Batch(d).run(workers=1, log=lambda *a: None) if x["surface"] == 0.5)
    assert r["wall_move"] == pytest.approx(0.24) and r["apron_step"] == 0.01


def test_no_retaining_keeps_old_rows():
    """土留部の変数も閾値もなければ列を作らない（これまでの結果と同じ）"""
    d = _data()
    d["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "shape": "cos", "kh": 0.0},
                      "grid": {"surface": [0.0]}, "refine": {"when": "none"}}
    r = Batch(d).run(workers=1, log=lambda *a: None)[0]
    assert "bridge_rel" not in r
