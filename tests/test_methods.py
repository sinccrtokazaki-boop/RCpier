"""利用方法ごとの判定（ガイドライン 第2版 表 D-1・表 D-5、G3）のテスト:
項目の段階 limited = フェーズ 2・3 は不可、unusable = 全て不可、施設の利用の観点は緊急復旧の注意事項、利用時の荷重"""
import csv

import pytest

from rcframe.grid import InputError
from rcframe.port_batch import Batch
from rcframe.port_usability import STATUS, final_rows, judge, judge_detail, probability_summary, read_usability

from test_deterioration import _data

METHODS = {"1-A": {"phase": 1, "label": "必要最小限の輸送", "surcharge": 0, "berthing": 300, "tractive": 200},
           "2-A": {"phase": 2, "surcharge": 10, "berthing": 300, "tractive": 200},
           "3-C": {"phase": 3, "label": "日常の利用", "surcharge": 20, "berthing": 500, "tractive": 350}}
LIMITS = {"wall_move": {"limited": 0.3, "unusable": 0.5}, "apron_step": {"limited": 0.10},
          "bridge_opening": {"limited": 0.2}}
GOOD = {"deck_residual": 0.05, "n_yield": 0, "n_plastic": 0, "n_buckled": 0, "n_buckled_head": 0, "deck_mu_max": 0.5,
        "allow_surcharge": 50.0, "allow_berthing": 1000.0, "allow_tractive": 800.0, "allow_aftershock": 0.3,
        "wall_move": 0.1, "bridge_rel": -0.05, "apron_step": 0.02}


def _opts(**kw):
    d = {"methods": METHODS, "required": {"aftershock": 0.15}, "limits": dict(LIMITS)}
    d["limits"].update(kw.pop("limits", {}))
    d.update(kw)
    return read_usability(None, d)


def test_read_methods():
    o = _opts()
    assert list(o["methods"]) == ["1-A", "2-A", "3-C"]
    assert o["methods"]["3-C"] == {"phase": 3, "label": "日常の利用", "loads": {"surcharge": 20, "berthing": 500, "tractive": 350}}
    assert o["loads"] == ["surcharge", "berthing", "tractive", "aftershock"]     # 判定に使う荷重（LOADS の順）
    assert o["limits"]["wall_move"] == (300.0, 500.0)


@pytest.mark.parametrize("patch, msg", [
    ({"methods": {"x": {"surcharge": 1}}}, "phase"),
    ({"methods": {"x": {"phase": 4}}}, "phase"),
    ({"methods": {"a・b": {"phase": 1}}}, "名前"),
    ({"methods": {"x": {"phase": 1, "berthing": -1}}}, "0 以上"),
    ({"methods": {"x": {"phase": 1, "speed": 1}}}, "speed"),
    ({"required": {"aftershock": 0.15, "tractive": 100}}, "required は aftershock だけ"),
    ({"allowable": {"loads": ["surcharge"]}}, "berthing"),
])
def test_methods_input_errors(patch, msg):
    d = {"methods": METHODS, "required": {"aftershock": 0.15}, **patch}
    with pytest.raises(InputError, match=msg):
        read_usability(None, d)


def _j(**row):
    return judge_detail({**GOOD, **row}, _opts())


def test_all_methods_usable():
    st, why, ok = _j()
    assert (st, ok) == (STATUS[0], ["1-A", "2-A", "3-C"]) and why == ["使える利用方法: 1-A・2-A・3-C"]


@pytest.mark.parametrize("row, ok, line", [
    ({"n_yield": 3, "n_plastic": 3}, ["1-A"], "2-A・3-C で不可: 杭の塑性化 3 箇所"),                  # 本震で降伏
    ({"n_buckled": 2, "n_buckled_head": 2}, [], "全ての利用方法で不可: 杭の局部座屈 2 箇所（杭頭 2、地中部 0）"),
    ({"allow_tractive": 300.0}, ["1-A", "2-A"], "3-C で不可: 牽引力の許容値 300 kN < 350 kN"),
    ({"allow_surcharge": 0.0}, ["1-A"], "2-A で不可: 上載荷重の許容値 0 kN/m² < 10 kN/m²"),
    ({"wall_move": 0.35}, ["1-A"], "2-A・3-C で不可: 土留部の残留水平変位 0.350 m"),
    ({"wall_move": 0.6}, [], "全ての利用方法で不可: 土留部の残留水平変位 0.600 m"),
    ({"allow_aftershock": 0.1}, [], "全ての利用方法で不可: 余震（L1 相当）に対する余裕の不足: 許容 kh = 0.100 < 想定 kh = 0.150"),
    ({"deck_mu_max": 1.1}, [], "全ての利用方法で不可: 本震時に上部工の梁が Mu に達した（M/Mu = 1.10）"),
    ({"n_buckled": None, "n_buckled_head": None, "n_plastic": None, "n_yield": 2}, [],     # ふるい分けだけで降伏
     "全ての利用方法で不可: 杭が降伏し、限界曲率の超過は未確認（詳細化していない）"),
])
def test_method_failures(row, ok, line):
    st, why, got = _j(**row)
    assert got == ok
    assert st == (STATUS[2] if not ok else STATUS[1])
    assert why[0] == f"使える利用方法: {'・'.join(ok) or 'なし'}" and line in why


def test_facility_use_is_a_note():
    """段差・渡版の開き（施設の利用の観点）は、全ての利用方法が使えても緊急復旧の注意事項で 注意事項あり"""
    st, why, ok = _j(apron_step=0.15, bridge_rel=0.25)
    assert st == STATUS[1] and ok == ["1-A", "2-A", "3-C"]
    assert "注意事項: 土留部の背後の段差 0.150 m（緊急復旧: 段差の解消）" in why
    assert "注意事項: 渡版の開き 0.250 m（上部工 − 土留部）（緊急復旧: 渡版の仮設）" in why
    o = read_usability(None, {"methods": METHODS, "limits": {"apron_step": {"limited": 0.1, "unusable": 0.3}}})
    st, why, ok = judge_detail({**GOOD, "apron_step": 0.35}, o)                   # unusable を与えれば全て不可
    assert st == STATUS[2] and not ok


def test_head_buckling_setting_with_methods():
    row = {"n_buckled": 2, "n_buckled_head": 2, "n_plastic": 1, "n_yield": 3}
    assert _j(**row)[2] == []                                                       # 既定: 杭頭も不可
    o = _opts(limits={"pile_damage": {"杭頭の局部座屈": "limited"}})
    st, why, ok = judge_detail({**GOOD, **row}, o)                                  # フェーズ 1 は使える
    assert ok == ["1-A"] and "2-A・3-C で不可: 杭の局部座屈（杭頭）2 箇所" in why
    assert judge({**GOOD, **row}, o)[0] == STATUS[1]


def test_old_status_and_summary():
    rows = [{"id": "a", "tier": "screen", "status": "制限付き利用", "surface": 0.1},       # G3 より前の結果
            {"id": "b", "tier": "screen", "status": STATUS[0], "surface": 0.2, "methods_ok": "1-A・2-A・3-C"},
            {"id": "c", "tier": "screen", "status": STATUS[1], "surface": 0.3, "methods_ok": "1-A"},
            {"id": "d", "tier": "screen", "status": STATUS[2], "surface": 0.4, "methods_ok": ""}]
    fr = final_rows(rows)
    assert [r["status"] for r in fr] == [STATUS[1], STATUS[0], STATUS[1], STATUS[2]]
    assert rows[0]["status"] == "制限付き利用"                                       # 元の行は変えない
    s = probability_summary(rows, _opts())
    assert s["methods"]["1-A"][0] == pytest.approx(0.5) and s["methods"]["3-C"][0] == pytest.approx(0.25)
    assert s["status"][STATUS[1]][0] == pytest.approx(0.5)


def test_batch_with_methods(tmp_path):
    d = _data()
    d["usability"].pop("required")
    d["usability"]["methods"] = {"1-A": {"phase": 1, "surcharge": 0, "berthing": 100, "tractive": 100},
                                 "3-C": {"phase": 3, "surcharge": 10, "berthing": 300, "tractive": 300}}
    d["usability"]["required"] = {"aftershock": 0.1}
    d["scenarios"] = {"base": {"direction": "+y", "depth": 6.0, "shape": "cos", "kh_factor": 0.2, "kh": 0.1},
                      "grid": {"surface": [0.0, 0.5]}, "refine": {"when": "all"}}
    rows = Batch(d).run(str(tmp_path / "r"), workers=1, log=lambda *a: None)
    o = read_usability(None, d["usability"])
    for r in final_rows(rows):
        st, why, ok = judge_detail(r, o)
        assert (r["status"], r["methods_ok"]) == (st, "・".join(ok))
        assert r["reasons"].startswith("使える利用方法: ") and r["allow_aftershock"] is not None
    with open(tmp_path / "r.csv", encoding="utf-8-sig") as f:
        assert "methods_ok" in next(csv.reader(f))
