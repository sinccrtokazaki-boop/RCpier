"""RC 桟橋の変状連鎖のデータ（rcframe/pier_chain.py）と作図（rcframe/chain_draw.py）のテスト"""
import re
import xml.etree.ElementTree as ET

import pytest

from rcframe import pier_chain as C
from rcframe.chain_draw import full_svg, layout, page_data, svg, write_page


def test_ids_unique_and_references_exist():
    ids = [n[0] for n in C.NODES]
    assert len(ids) == len(set(ids))
    for s, t, k in C.EDGES:
        assert s in C.NODE and t in C.NODE and k in ("p", "t", "c")
    assert len(C.EDGES) == len(set((s, t) for s, t, _ in C.EDGES))          # 同じ矢印の重複なし
    for sid, name, members, note in C.SCENARIOS:
        assert set(members) <= set(ids), sid


def test_node_fields():
    """段・帯・種類・印・点検の部位が定義の範囲にある。要因の列は作用・環境・設計、事故の列は重大事故"""
    for nid, c, lane, kind, label, codes, flags, desc in C.NODES:
        assert 0 <= c < len(C.COLS) and 0 <= lane < len(C.LANES) and kind in C.KINDS
        assert set(flags) <= set(C.FLAGS) and set(codes) <= set(C.CODES) and desc
        assert (kind in ("act", "env", "des")) == (c == 0), nid
        assert (kind == "acc") == (c == len(C.COLS) - 1), nid


def test_acyclic_and_forward():
    """循環がなく、矢印は同じ列か右の列へ向かう（図で左向きの矢印を描かない）"""
    order = C.topo_order()
    assert len(order) == len(C.NODES)
    for s, t, k in C.EDGES:
        assert C.NODE[t]["c"] >= C.NODE[s]["c"], (s, t)


def test_every_node_connected():
    """要因は出る矢印、重大事故は入る矢印、その他は両方を持つ"""
    outs = {s for s, _, _ in C.EDGES}
    ins = {t for _, t, _ in C.EDGES}
    for nid, c, lane, kind, *_ in C.NODES:
        if kind in ("act", "env", "des"):
            assert nid in outs, nid
        elif kind == "acc":
            assert nid in ins, nid
        else:
            assert nid in outs and nid in ins, nid


def test_svg_well_formed_and_no_overlap():
    """SVG は XML として読め、同じ帯・列のノードは重ならない。ノード・矢印の数が表と一致"""
    full, width, height = full_svg(C)
    root = ET.fromstring(full)
    assert root.tag.endswith("svg")
    body, nodes, w, h = svg(C, "page")
    assert body.count('class="node ') == len(C.NODES) and body.count('class="edge ') == len(C.EDGES)
    for a in nodes.values():
        for b in nodes.values():
            if a["id"] < b["id"] and a["lane"] == b["lane"] and a["c"] == b["c"]:
                assert a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"]
    assert re.search(r"計", full) and "rcframe" in full


def test_page(tmp_path):
    p = write_page(C, tmp_path / "page.html")
    text = p.read_text(encoding="utf-8")
    assert "<title>RC 桟橋の変状連鎖図</title>" in text
    data = page_data(C)
    assert len(data["nodes"]) == len(C.NODES) and len(data["scenarios"]) == len(C.SCENARIOS)
    assert all(text.count(f'data-sc="{s[0]}"') == 1 for s in C.SCENARIOS)
