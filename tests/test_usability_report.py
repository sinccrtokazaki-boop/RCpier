"""利用可否シナリオの検討報告書（HTML）のテスト"""
import re
from pathlib import Path

import pytest
import yaml

from rcframe.port_batch import Batch, main
from rcframe.port_usability import STATUS, final_rows, probability_summary
from rcframe.usability_report import SECTIONS, build_usability_report

from test_cards import _data, _well_formed

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def report():
    b = Batch(_data())
    rows = b.run(workers=1, log=lambda *a: None)
    return b, rows, build_usability_report(b, rows, "type: pier  # 入力の例")


def test_report_sections_and_html(report):
    """全ての章と目次があり、HTML のタグが対応し、外部参照がない。入力データを付録に載せる"""
    b, rows, html = report
    for sid, title, _ in SECTIONS:
        assert f'id="{sid}"' in html and title in html
    ok, c = _well_formed(html)
    assert ok, (c.errors, c.stack, c.external)
    assert "type: pier  # 入力の例" in html
    assert "&amp;#x27;" not in html and "&#x27;" not in html          # 二重エスケープがない


def test_report_numbers(report):
    """判定の件数・割合は集計と一致、全シナリオの表は最終結果の行数、代表シナリオは判定ごとに 1件"""
    b, rows, html = report
    fr = final_rows(rows)
    s = probability_summary(rows, b.pier.usability)
    for k, (p, _) in s["status"].items():
        n = sum(r["status"] == k for r in fr)
        assert re.search(rf">{k}</span></td><td class=n>{n}</td><td class=n>{p:.1%}</td>", html)
    assert html.count('<tr data-st="') == len(fr)
    reps = re.findall(r'<h3><span class="status st\d"[^>]*>([^<]+)</span> シナリオ', html)
    assert sorted(reps) == sorted({r["status"] for r in fr})


def test_report_pile_capacity_table(report):
    """杭の諸元の表: θu = (φu − φy)·Lp（表示の桁で一致）、式A の εcu = 1.14/(D/t) − 0.012（腐食後）"""
    b, rows, html = report
    m = re.search(r"<td>L</td><td>φ1000×14</td><td class=n>83</td><td class=n>[\d,]+</td><td class=n>A</td>"
                  r"<td class=n>([\d.]+)</td>(?:<td class=n>[^<]+</td>){4}<td class=n>([\d.]+)</td>"
                  r"<td class=n>([\d.]+)</td><td class=n>([\d.]+)</td>", html)
    assert m, "杭の諸元の表"
    ecu, py, pu, tu = (float(x) for x in m.groups())
    assert ecu == pytest.approx(1.14 / (996 / 12) - 0.012, abs=1e-5)
    assert tu == pytest.approx((pu - py) / 1e3 * 996, abs=2e-4)


def test_report_cli(tmp_path):
    """port_batch --report で報告書を書き出す（計算済みの結果を再利用）"""
    d = _data()
    src = tmp_path / "in.yaml"
    src.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "res"
    rep = tmp_path / "report.html"
    main([str(src), "-o", str(out), "-j", "1", "--report", str(rep)])
    html = rep.read_text(encoding="utf-8")
    assert "検討報告書" in html and (tmp_path / "res.csv").exists()
