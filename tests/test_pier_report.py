"""桟橋編 P5（計算書）のテスト"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rcframe import Pier, build_report, load_model, model_from_yaml
from rcframe.pier_report import _compute, build_pier_report
from rcframe import svg

from test_report import Collect

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="module")
def fixed_point():
    p = load_model(EXAMPLES / "pier_example.yaml")
    ctx = _compute(p)
    html = build_pier_report(p, ctx)
    c = Collect()
    c.feed(html)
    return p, ctx, html, c


@pytest.fixture(scope="module")
def level2():
    """地盤ばね + レベル2（テスト時間短縮のため 1方向・粗いステップ）"""
    text = (EXAMPLES / "pier_level2.yaml").read_text(encoding="utf-8")
    d = yaml.safe_load(text)
    d["level2"].update(directions=["+y"], step=0.02, max_displacement=0.3)
    p = Pier(d)
    p.source_text = text
    ctx = _compute(p)
    html = build_pier_report(p, ctx)
    c = Collect()
    c.feed(html)
    return p, ctx, html, c


def test_sections_fixed_point(fixed_point):
    """仮想固定点・照査あり・レベル2なし: レベル2の章は出さない。外部参照なし"""
    _, _, html, c = fixed_point
    for sid in ("sec-outline", "sec-loads", "sec-analysis", "sec-checks", "sec-appendix"):
        assert sid in c.ids
    assert "sec-level2" not in c.ids
    assert "http:" not in html and "https:" not in html
    assert not [a for t, a in c.attrs if t in ("script", "img") and "src" in a]


def test_numbers_fixed_point(fixed_point):
    """照査の最大値・重量・荷重の合計が計算結果と同じ表記で載っている"""
    p, ctx, html, _ = fixed_point
    assert f"最大照査値 <b>{ctx['check'].max_ratio:.2f}</b>" in html
    assert f"{p.weights['地震用'][0] / 1e3:,.1f}" in html
    assert "-15,378.7" in html                              # D の鉛直合計
    for r in ctx["check"].governing(("item",)).values():
        assert f"{r.ratio:.2f}" in html
    assert "示方書の Ec の表" in html and "港湾基準（2018）の鋼管杭の照査式" in html   # 照合が必要な事項


def test_viewer_data(fixed_point):
    p, _, _, c = fixed_point
    data = json.loads(next(s for sid, s in c.scripts if sid == "model-data"))
    assert set(data["cases"]) == set(p.combinations)
    used = {n for e in p.model.elements.values() for n in (e.ni, e.nj)}
    assert set(map(int, data["nodes"])) == used


def test_level2_report(level2):
    """地盤ばね + レベル2: レベル2の章、判定、要求点、ヒンジ表示、入力データ"""
    p, ctx, html, c = level2
    assert "sec-level2" in c.ids and "v-hinge" in c.ids
    r = ctx["l2"]["+y"]
    assert f"kh = {r.demand['kh']:.3f}" in html
    assert ("すべて OK" in html) == r.ok
    for name, _, _ in r.criteria:
        assert name in html
    assert "地盤ばね" in html and "piles:" in html          # 杭のモデル・入力データ（YAML）
    data = json.loads(next(s for sid, s in c.scripts if sid == "model-data"))
    assert "レベル2 +y（最終）" in data["hinges"] and "レベル2 +y（最終）" in data["cases"]
    assert all(h[3] in ("c", "y") for h in data["hinges"]["レベル2 +y（最終）"])


def test_svg_finite(level2):
    _, _, html, _ = level2
    for m_ in re.finditer(r'\s(?:x1|y1|x2|y2|cx|cy|x|y|width|height|r)="([^"]+)"', html):
        assert math.isfinite(float(m_.group(1)))
    for m_ in re.finditer(r'points="([^"]+)"', html):
        for v in re.split(r"[ ,]", m_.group(1)):
            assert math.isfinite(float(v))


def test_xy_chart_scaling():
    """折れ線グラフ: 点の位置が軸の最大値に比例（x 最大 → 右端、y = 0 → 下端）"""
    s = svg.xy_chart([("a", [(0.0, 0.0), (2.0, 1.0)], "c0")], "x", "y")
    pts = re.search(r'<polyline class="curve c0" points="([^"]+)"', s).group(1).split()
    (x0, y0), (x1, y1) = [tuple(map(float, p.split(","))) for p in pts]
    W = 720 - 70 - 140
    assert x1 - x0 == pytest.approx(W / 1.05, rel=1e-3) and y1 < y0


def test_dispatch_and_cli(tmp_path):
    """build_report は Pier を桟橋の計算書に振り分け、python -m rcframe.report も桟橋の YAML を読める"""
    p = model_from_yaml((EXAMPLES / "pier_example.yaml").read_text(encoding="utf-8"))
    assert "sec-outline" in build_report(p)
    out = tmp_path / "pier.html"
    cp = subprocess.run([sys.executable, "-m", "rcframe.report", str(EXAMPLES / "pier_example.yaml"), "-o", str(out)],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert cp.returncode == 0, cp.stderr
    assert "RuntimeWarning" not in cp.stderr
    assert "直杭式横桟橋" in out.read_text(encoding="utf-8")
