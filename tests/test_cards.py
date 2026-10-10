"""利用可否 U6（判定カード・一覧の HTML）のテスト"""
import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

from rcframe.port_batch import Batch
from rcframe.port_cards import card_one, damage_levels, index_html, write_cards
from rcframe.port_usability import STATUS, final_rows, probability_summary

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
VOID = {"meta", "br", "hr", "img", "input", "link", "line", "circle", "rect", "polygon", "polyline", "path"}


def _data():
    d = yaml.safe_load((EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8"))
    d["grids"] = {"x": {"B1": 0, "B2": 6}, "y": {"L": 0, "S": 7}}
    d["piles"]["rows"] = {"L": {"seabed": -4.0, "kh": 20000, "tip": -20.0, "pu": 300, "Rcd": 5000, "Rtd": 2000},
                          "S": {"seabed": -8.0, "kh": 20000, "tip": -24.0, "pu": 300, "Rcd": 6000, "Rtd": 2500}}
    d.pop("combinations")
    d.pop("lateral_flow")
    d["actions"]["EQ"]["directions"] = ["y"]
    d["level2"]["step"] = 0.02
    d["scenarios"] = {"base": {"direction": "+y", "depth": 8.0, "kh_factor": 0.1, "kh": 0.05,
                               "surface_ratio": {"L": 1.0, "S": 0.6}},
                      "grid": {"surface": [0.05, 0.3, 0.8]}, "refine": {"when": {"pile_yield": True}}}
    return d


@pytest.fixture(scope="module")
def batch():
    b = Batch(_data())
    return b, b.run(workers=1, log=lambda *a: None)


class _Check(HTMLParser):
    """開始・終了タグの対応と外部参照の有無を調べる"""

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.external = [], [], []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("src", "href") and v and re.match(r"(https?:)?//", v):
                self.external.append(v)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


def _well_formed(html):
    c = _Check()
    c.feed(html)
    return not c.errors and not c.stack and not c.external, c


def test_card_matches_batch_row(batch):
    """カードのために解析し直した値は一括実行の行と一致し、カードに判定・理由・数値が入る"""
    b, rows = batch
    params = dict(b.scenarios)
    for r in final_rows(rows):
        html, again = card_one(b.data, r["id"], params[r["id"]], r["tier"], b.segments, r)
        for k in ("deck_residual", "deck_peak", "head_rot_res", "inground_rot_res", "n_yield",
                  "allow_surcharge", "allow_berthing", "allow_tractive"):
            assert again[k] == pytest.approx(r[k], rel=1e-9, abs=1e-12), k
        assert again["status"] == r["status"] and again["reasons"] == r["reasons"]
        assert r["status"] in html and f"{r['deck_residual']:.3f} m" in html
        for why in filter(None, r["reasons"].split(" / ")):
            assert escape(why) in html
        ok, c = _well_formed(html)
        assert ok, (c.errors, c.stack, c.external)


def test_damage_levels_screen(batch):
    """ふるい分け（cos 式のヒンジ）では降伏 = 塑性化"""
    from rcframe.port_batch import analyze
    b, rows = batch
    sid, params = next((s, q) for s, q in b.scenarios if q["surface"] == 0.8)
    pier, fr = analyze(b.data, sid, params, "screen", b.segments)
    lv = damage_levels(fr)
    assert {k for k, v in lv.items() if v == "塑性化"} == {k for k, h in fr.inc.hinges.items() if h.state == "降伏"}
    assert "局部座屈" not in lv.values()


def test_index_counts_and_links(batch, tmp_path):
    """一覧の件数・割合は集計と一致し、カードへのリンクは書き出したファイルを指す"""
    b, rows = batch
    idx = write_cards(b, rows, tmp_path, "all", workers=1, log=lambda *a: None)
    html = idx.read_text(encoding="utf-8")
    ok, c = _well_formed(html)
    assert ok, (c.errors, c.stack, c.external)
    fr = final_rows(rows)
    links = re.findall(r'href="(card_[^"]+\.html)"', html)
    assert len(links) == len(fr) and all((tmp_path / f).exists() for f in links)
    s = probability_summary(rows, b.pier.usability)
    for k, (p, _) in s["status"].items():
        n = sum(r["status"] == k for r in fr)
        assert re.search(rf">{k}</span></td><td class=n>{n}</td><td class=n>{p:.1%}</td>", html)
    assert html.count('<tr data-st="') == len(fr)


def test_write_cards_filter(batch, tmp_path):
    """既定は 利用可（注意事項あり）・利用不可 のシナリオだけカードを作る"""
    b, rows = batch
    write_cards(b, rows, tmp_path, (STATUS[1], STATUS[2]), workers=1, log=lambda *a: None)
    n = sum(r["status"] != STATUS[0] for r in final_rows(rows))
    assert len(list(tmp_path.glob("card_*.html"))) == n and (tmp_path / "index.html").exists()
