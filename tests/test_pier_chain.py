"""RC 桟橋の変状連鎖のデータ（rcframe/pier_chain.py）と作図（rcframe/chain_draw.py）のテスト

データは加藤ほか（港湾空港技術研究所資料 No.1328, 2016）の図-3.4・図-4.1・表-4.1 による。
"""
import re
import xml.etree.ElementTree as ET

from rcframe import pier_chain as C
from rcframe.chain_draw import full_svg, page_data, svg, write_page


def test_ids_unique_and_references_exist():
    ids = [n[0] for n in C.NODES]
    assert len(ids) == len(set(ids))
    for s, t, k in C.EDGES:
        assert s in C.NODE and t in C.NODE and k in C.EDGE_KINDS
    assert len(C.EDGES) == len(set((s, t) for s, t, _ in C.EDGES))          # 同じ矢印の重複なし
    for sid, name, members, note in C.SCENARIOS:
        assert set(members) <= set(ids), sid


def test_node_fields():
    """段・帯・種類・印・点検の番号が定義の範囲にある。左端の列は作用・原因等、右端の列は性能への影響"""
    for nid, c, lane, kind, label, codes, flags, desc in C.NODES:
        assert 0 <= c < len(C.COLS) and 0 <= lane < len(C.LANES) and kind in C.KINDS
        assert set(flags) <= set(C.FLAGS) and set(codes) <= set(C.CODES) and desc
        assert (kind in C.FACTOR_KINDS) == (c == 0), nid
        assert (kind in C.END_KINDS) == (c == len(C.COLS) - 1), nid


def test_matches_reference_figure():
    """文献の図-3.4：作用・原因等 8、変状 20、性能への影響 2、矢印 41（突発型は地震・波浪・衝突から出る 6 本）。
    図-4.1 の着色（点検診断の項目の番号あり）と無着色（なし）が種類と一致する"""
    kinds = [n[3] for n in C.NODES]
    assert (kinds.count("act"), kinds.count("ins") + kinds.count("non"), kinds.count("out")) == (8, 20, 2)
    assert len(C.EDGES) == 41
    sudden = [(s, t) for s, t, k in C.EDGES if k == "s"]
    assert len(sudden) == 6 and {s for s, _ in sudden} == {"A_quake", "A_wave", "A_impact"}
    for nid, c, lane, kind, label, codes, *_ in C.NODES:
        if kind in ("ins", "non"):
            assert (kind == "ins") == bool(codes), nid
    assert {n[0] for n in C.NODES if n[3] == "non"} == {"D_defl", "D_retain", "D_embed", "D_mech"}
    assert set(C.CODES) == set(range(1, 13))
    assert C.NODE["D_pilecorr"]["codes"] == [5] and C.NODE["D_coat"]["codes"] == [6] and C.NODE["D_anode"]["codes"] == [7]


def test_acyclic_and_forward():
    """循環がなく、矢印は右の列へ向かう（図で左向き・同じ列の矢印を描かない）"""
    order = C.topo_order()
    assert len(order) == len(C.NODES)
    for s, t, k in C.EDGES:
        assert C.NODE[t]["c"] > C.NODE[s]["c"], (s, t)


def test_every_path_reaches_performance():
    """すべての作用・原因等と変状は、いずれかの性能への影響に行き着き、すべての変状にはどれかの作用・原因等から至る"""
    out = {}
    for s, t, _ in C.EDGES:
        out.setdefault(s, []).append(t)

    def reach(v):
        seen, st = set(), [v]
        while st:
            for w in out.get(st.pop(), []):
                if w not in seen:
                    seen.add(w)
                    st.append(w)
        return seen

    ends = {n[0] for n in C.NODES if n[3] in C.END_KINDS}
    from_factors = set().union(*(reach(n[0]) for n in C.NODES if n[3] in C.FACTOR_KINDS))
    for nid, c, lane, kind, *_ in C.NODES:
        if kind not in C.END_KINDS:
            assert reach(nid) & ends, nid
        if kind not in C.FACTOR_KINDS:
            assert nid in from_factors, nid


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
    assert re.search(r"計", full) and "rcframe" in full and "No.1328" in full
    assert "点検 (8)(9)" in full and "突発型の変状" in full


def test_page(tmp_path):
    p = write_page(C, tmp_path / "page.html")
    text = p.read_text(encoding="utf-8")
    assert "<title>RC 桟橋の変状連鎖図</title>" in text
    data = page_data(C)
    assert len(data["nodes"]) == len(C.NODES) and len(data["scenarios"]) == len(C.SCENARIOS)
    assert all(text.count(f'data-sc="{s[0]}"') == 1 for s in C.SCENARIOS)
    assert "表-3.1" in text and "荷役機械の基礎" in text and "表-4.1" in text
    assert data["nodes"]["D_cl"]["codes"] == ["(8)", "(9)"]
