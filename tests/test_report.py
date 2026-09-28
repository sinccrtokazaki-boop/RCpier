"""計算書（フェーズ7）のテスト"""
import json
import math
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from rcframe import Concrete, Model, RectSection, load_building
from rcframe import svg
from rcframe.report import _compute, build_report

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class Collect(HTMLParser):
    """タグ・id・script を集める（パースできること自体も確認）"""

    def __init__(self):
        super().__init__()
        self.ids, self.tags, self.scripts, self._in = [], [], [], None
        self.attrs = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        self.tags.append(tag)
        self.attrs.append((tag, d))
        if "id" in d:
            self.ids.append(d["id"])
        if tag == "script":
            self._in = d.get("id", "")
            self.scripts.append([self._in, ""])

    def handle_endtag(self, tag):
        if tag == "script":
            self._in = None

    def handle_data(self, data):
        if self._in is not None:
            self.scripts[-1][1] += data


@pytest.fixture(scope="module")
def design():
    b = load_building(EXAMPLES / "two_story_design.yaml")
    ctx = _compute(b)
    html = build_report(b, ctx)
    p = Collect()
    p.feed(html)
    return b, ctx, html, p


def test_structure_and_self_contained(design):
    """全章が出力され、外部参照（URL・外部スクリプト・スタイルシート）がない"""
    _, _, html, p = design
    for sid in ("sec-model", "sec-loads", "sec-analysis", "sec-drift", "sec-check", "sec-pushover",
                "sec-ultimate", "sec-appendix"):
        assert sid in p.ids
    assert "http:" not in html and "https:" not in html
    assert not [a for t, a in p.attrs if t in ("script", "img") and "src" in a]
    assert "link" not in p.tags
    assert html.startswith("<!DOCTYPE html>") and 'charset="utf-8"' in html


def test_viewer_data(design):
    """3D 表示のデータ: 節点（剛床のマスターを除く）・要素・荷重ケースの変位・ヒンジ"""
    b, ctx, _, p = design
    data = json.loads(next(s for sid, s in p.scripts if sid == "model-data"))
    m = b.model
    used = {n for e in m.elements.values() for n in (e.ni, e.nj)}
    assert set(map(int, data["nodes"])) == used
    assert len(data["elems"]) == len(m.elements)
    for c in ("G", "P", "Ex", "Ey", "G+P"):
        assert c in data["cases"]
    n = str(b.masters and b.node("X3", "Y2", "RF"))
    assert data["cases"]["Ex"][n][0] == pytest.approx(ctx["lin"]["Ex"].disp[int(n)][0], rel=1e-3)
    assert set(data["hinges"]) == {"+x", "-x", "+y", "-y"}
    assert all(h[3] in ("c", "y") for h in data["hinges"]["+x"])


def test_numbers_in_tables(design):
    """保有水平耐力・検定比・層間変形角の値が計算結果と同じ表記で載っている"""
    b, ctx, html, _ = design
    for s in ctx["ult"].stories:
        assert f"{s.Qu / 1e3:,.1f}" in html and f"{s.ratio:.2f}" in html
    assert f"最大検定比 <b>{ctx['check'].max_ratio:.2f}</b>" in html
    for name, du, h, ang in ctx["lin"]["Ex"].story_drifts(b.drift_levels(), 0):
        assert f"1/{1 / ang:,.0f}" in html
    assert "すべて OK" in html


def test_sections_omitted_without_rebar():
    """配筋のない入力では断面検定・プッシュオーバー・保有水平耐力の章を出さない"""
    b = load_building(EXAMPLES / "two_story_frame.yaml")
    html = build_report(b)
    p = Collect()
    p.feed(html)
    assert "sec-analysis" in p.ids and "sec-appendix" in p.ids
    for sid in ("sec-check", "sec-pushover", "sec-ultimate"):
        assert sid not in p.ids
    assert "v-hinge" not in p.ids


def test_cover_from_yaml():
    """report の項（表紙）"""
    text = (EXAMPLES / "two_story_frame.yaml").read_text(encoding="utf-8")
    text += "\nreport: {title: テスト計算書 <A&B>, author: 設計者, date: 2026-09-28}\n"
    from rcframe import building_from_yaml
    html = build_report(building_from_yaml(text))
    assert "<h1>テスト計算書 &lt;A&amp;B&gt;</h1>" in html     # エスケープされる
    assert "設計者" in html and "2026-09-28" in html


def test_svg_numbers_finite(design):
    """SVG の座標に NaN・inf がない"""
    _, _, html, _ = design
    for m_ in re.finditer(r'\s(?:x1|y1|x2|y2|cx|cy|x|y|width|height)="([^"]+)"', html):
        assert math.isfinite(float(m_.group(1)))
    for m_ in re.finditer(r'points="([^"]+)"', html):
        for v in re.split(r"[ ,]", m_.group(1)):
            assert math.isfinite(float(v))


def test_moment_diagram_proportional():
    """両端固定梁・等分布: 図の偏位は |M| に比例（端部 wL²/12 : 中央 wL²/24 = 2 : 1）、
    端部は上側（上端引張）、中央は下側に描く"""
    L, w = 6000.0, 30.0
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, L, 0, 0, fix=[1] * 6)
    e = m.add_element(1, 1, 2, Concrete("Fc24", 24), RectSection("G", 400, 700))
    m.add_member_load("W", 1, [0, 0, -w])
    r = m.analyze("W")
    Ms, vs = svg.member_moment_offsets(r, e, 1, [0.0, L / 2])
    scale = 1e-6
    pts = svg.moment_diagram_points((0, 0), (L, 0), [0, 0.5], [(v[0] * scale, v[2] * scale) for v in vs])
    end, mid = pts[1][1], pts[2][1]
    assert end == pytest.approx(w * L ** 2 / 12 * scale, rel=1e-9)
    assert mid == pytest.approx(-w * L ** 2 / 24 * scale, rel=1e-9)
    assert end / -mid == pytest.approx(2.0, rel=1e-9)


def test_column_moment_tension_side():
    """片持ち柱の頂部に +X の水平力: 柱脚の引張側は −X 側に描く"""
    m = Model()
    m.add_node(1, 0, 0, 0, fix=[1] * 6)
    m.add_node(2, 0, 0, 3000)
    e = m.add_element(1, 1, 2, Concrete("Fc24", 24), RectSection("C", 600, 600), "column")
    m.add_nodal_load("P", 2, [100e3, 0, 0, 0, 0, 0])
    r = m.analyze("P")
    Ms, vs = svg.member_moment_offsets(r, e, 0, [0.0])
    assert abs(Ms[0]) == pytest.approx(100e3 * 3000, rel=1e-9)
    assert vs[0][0] < 0 and abs(vs[0][1]) < 1e-9 * abs(vs[0][0])
    # Y方向の加力（xz 面）でも引張側（−Y）に描く
    m.nodal_loads.clear()
    m.add_nodal_load("P", 2, [0, 100e3, 0, 0, 0, 0])
    r = m.analyze("P")
    _, vs = svg.member_moment_offsets(r, e, 1, [0.0])
    assert vs[0][1] < 0


def test_cli(tmp_path):
    """python -m rcframe.report で出力（警告なし）"""
    out = tmp_path / "r.html"
    cp = subprocess.run([sys.executable, "-m", "rcframe.report", str(EXAMPLES / "two_story_frame.yaml"),
                         "-o", str(out)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert cp.returncode == 0, cp.stderr
    assert "RuntimeWarning" not in cp.stderr
    assert out.exists() and out.stat().st_size > 10000
