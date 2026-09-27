"""通り芯・階からのモデル自動生成（フェーズ2）のテスト"""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from rcframe import Building, InputError, Result, load_building

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example():
    spec = importlib.util.spec_from_file_location("two_story_frame", EXAMPLES / "two_story_frame.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(nd):
    return (round(nd.x, 6), round(nd.y, 6), round(nd.z, 6))


def _close(a, b, scale):
    """相対誤差 1e-9（scale はケース全体の代表値。ほぼ 0 の成分の比較用）"""
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9 * scale)


def test_yaml_matches_example():
    """YAML から生成したモデルと two_story_frame.py の手組みモデルの結果が一致する"""
    ref, ref_masters = _load_example().build_model()
    bld = load_building(EXAMPLES / "two_story_frame.yaml")
    m = bld.model

    # 節点・要素を座標で対応付ける
    assert len(m.nodes) == len(ref.nodes)
    assert len(m.elements) == len(ref.elements)
    nmap = {_key(n): n.id for n in ref.nodes.values()}
    node_of = {n.id: nmap[_key(n)] for n in m.nodes.values()}
    emap = {(_key(ref.nodes[e.ni]), _key(ref.nodes[e.nj])): e for e in ref.elements.values()}
    elem_of = {}
    for e in m.elements.values():
        r = emap[(_key(m.nodes[e.ni]), _key(m.nodes[e.nj]))]
        assert (e.sec.name, e.sec.phi_y, e.mat.E) == (r.sec.name, r.sec.phi_y, r.mat.E)
        assert e.rigid_i == pytest.approx(r.rigid_i, abs=1e-9)
        assert e.rigid_j == pytest.approx(r.rigid_j, abs=1e-9)
        elem_of[e.id] = r.id

    # 床マスター節点は床の図心 (6000, 3500)
    for f, name in ((1, "2F"), (2, "RF")):
        assert node_of[bld.master(name)] == ref_masters[f]

    res = {c: m.analyze(c) for c in ("G", "Ex")}
    ref_res = {c: ref.analyze(c) for c in ("G", "Ex")}
    res["G+Ex"] = Result.combine("G+Ex", {res["G"]: 1.0, res["Ex"]: 1.0})
    ref_res["G+Ex"] = Result.combine("G+Ex", {ref_res["G"]: 1.0, ref_res["Ex"]: 1.0})

    for c in res:
        r, rr = res[c], ref_res[c]
        dscale = max(np.abs(d).max() for d in rr.disp.values())
        for nid, d in r.disp.items():
            _close(d, rr.disp[node_of[nid]], dscale)
        assert {node_of[n] for n in r.reactions} == set(rr.reactions)
        fscale = max(np.abs(v).max() for v in rr.reactions.values())
        for nid, v in r.reactions.items():
            _close(v, rr.reactions[node_of[nid]], fscale)
        for eid, f in r.forces.items():
            for at in ("face", "node"):
                _close(f[at], rr.forces[elem_of[eid]][at], fscale * 1e4)

    drifts = res["Ex"].story_drifts(bld.drift_levels(), 0)
    ref_drifts = ref_res["Ex"].story_drifts([("1F", ref_masters[1]), ("2F", ref_masters[2])], 0)
    assert [d[0] for d in drifts] == ["1F", "2F"]
    for d, rd in zip(drifts, ref_drifts):
        assert d[1:] == pytest.approx(rd[1:], rel=1e-9)


# ------------------------------------------------------------ 配置ルール
BASE = """
materials:
  Fc24: {Fc: 24}
  Fc30: {Fc: 30}
sections:
  C1: {b: 600, D: 600}
  C2: {b: 550, D: 550}
  G1: {b: 400, D: 700}
  G2: {b: 350, D: 600}
defaults: {material: Fc24}
grids:
  x: {X1: 0, X2: 6000, X3: 12000}
  y: {Y1: 0, Y2: 6000, Y3: 12000}
floors: {1F: 0, 2F: 4000, RF: 7500}
columns:
  - {at: "*", section: C1}
beams:
  - {at: "*", section: G1}
"""


def _data():
    return yaml.safe_load(BASE)


def test_section_per_floor():
    """階ごとの断面・材料の上書き（後のルールが優先）"""
    d = _data()
    d["columns"] += [{"floor": "2F", "at": "*", "section": "C2"},
                     {"floor": "1F", "at": "X2/*", "section": "C1", "material": "Fc30"}]
    d["beams"] += [{"floor": "RF", "at": "Y2/X1-X3", "section": "G2"}]
    b = Building(d)
    E = b.model.elements
    assert E[b.column("X1", "Y1", "1F")].sec.name == "C1"
    assert E[b.column("X1", "Y1", "2F")].sec.name == "C2"
    assert E[b.column("X2", "Y3", "1F")].mat.Fc == 30
    assert E[b.column("X1", "Y3", "1F")].mat.Fc == 24
    assert E[b.beam("Y2", "X1", "X2", "RF")].sec.name == "G2"
    assert E[b.beam("Y2", "X2", "X3", "RF")].sec.name == "G2"
    assert E[b.beam("Y2", "X1", "X2", "2F")].sec.name == "G1"
    assert E[b.beam("X2", "Y1", "Y2", "RF")].sec.name == "G1"     # Y方向梁は対象外
    # 柱 9本×2層, 梁 (2×3 + 3×2)×2層
    assert sum(e.kind == "column" for e in E.values()) == 18
    assert sum(e.kind == "beam" for e in E.values()) == 24


def test_missing_members():
    """section: none の柱・梁は生成されず、どこにも接続しない節点は作られない"""
    d = _data()
    d["columns"] += [{"floor": "2F", "at": "X2/Y2", "section": "none"},
                     {"floor": ["1F", "2F"], "at": "X3/Y3", "section": None}]
    d["beams"] += [{"floor": "RF", "at": ["X3/Y2-Y3", "Y3/X2-X3"], "section": "none"}]
    b = Building(d)
    with pytest.raises(KeyError):
        b.column("X2", "Y2", "2F")
    with pytest.raises(KeyError):
        b.beam("Y3", "X2", "X3", "RF")
    b.column("X2", "Y2", "1F")
    b.node("X2", "Y2", "RF")                   # 梁があるので節点は残る
    b.node("X3", "Y3", "2F")                   # 2F梁の交点
    with pytest.raises(KeyError):
        b.node("X3", "Y3", "RF")               # 柱も梁もない
    with pytest.raises(KeyError):
        b.node("X3", "Y3", "1F")
    assert len(b.model.elements) == 18 - 3 + 24 - 2
    r = b.model.analyze("none")                # 荷重なしでも特異にならない（安定）
    assert all(np.allclose(v, 0) for v in r.disp.values())


def test_opening_centroid_and_diaphragm():
    """吹抜けのある L 字形の床: 図心は手計算 (5000, 5000)、床に接しない節点は剛床から外れる

    区画 6000×6000 が 3 つ: 図心 (3000,3000), (9000,3000), (3000,9000) の平均
    """
    d = _data()
    d["openings"] = [{"floor": "2F", "area": "X2-X3/Y2-Y3"}]
    d["loads"] = {"E": {"floors": {"2F": [100e3, 0, 0], "RF": [200e3, 0, 0]}}}
    b = Building(d)
    m = b.model
    mst = m.nodes[b.master("2F")]
    assert (mst.x, mst.y, mst.z) == pytest.approx((5000, 5000, 4000))
    rf = m.nodes[b.master("RF")]
    assert (rf.x, rf.y) == pytest.approx((6000, 6000))

    slaves = dict(m.diaphragms)[b.master("2F")]
    assert b.node("X3", "Y3", "2F") not in slaves
    assert len(slaves) == 8
    assert len(dict(m.diaphragms)[b.master("RF")]) == 9

    # 剛床から外れた節点を含めても安定で、釣り合う
    r = m.analyze("E")
    assert sum(r.reactions.values())[0] == pytest.approx(-300e3, rel=1e-8)


def test_whole_floor_opening_no_diaphragm():
    """床が全面吹抜けの階は剛床なし（層間変形角は求められない）"""
    d = _data()
    d["openings"] = [{"floor": "2F", "area": "*"}]
    b = Building(d)
    assert b.master("2F") is None
    assert b.master("RF") is not None
    with pytest.raises(ValueError):
        b.drift_levels()


def test_beam_load_selection():
    """梁荷重は指定した階・位置の梁にだけ載り、反力合計 = w × 梁長の合計"""
    d = _data()
    d["loads"] = {"G": {"beams": [{"floor": "RF", "at": "Y1/*", "w": [0, 0, -20]}]}}
    b = Building(d)
    loaded = [eid for eid, _ in b.model.member_loads["G"]]
    assert sorted(loaded) == sorted([b.beam("Y1", "X1", "X2", "RF"), b.beam("Y1", "X2", "X3", "RF")])
    r = b.model.analyze("G")
    assert sum(r.reactions.values())[2] == pytest.approx(20 * 12000, rel=1e-8)


@pytest.mark.parametrize("patch, msg", [
    (lambda d: d["columns"].append({"at": "*", "section": "C9"}), "未定義の断面"),
    (lambda d: d["columns"].append({"floor": "RF", "section": "C1"}), "RF"),
    (lambda d: d["beams"].append({"floor": "1F", "section": "G1"}), "1F"),
    (lambda d: d["beams"].append({"at": "Y9/X1-X2", "section": "G1"}), "未定義の通り芯"),
    (lambda d: d["beams"].append({"at": "Y1/X1", "section": "G1"}), "範囲"),
    (lambda d: d["grids"]["x"].update({"X-4": 18000}), "使えません"),
    (lambda d: d["grids"]["y"].update({"X1": 18000}), "重複"),
    (lambda d: d["floors"].update({"3F": 5000}), "昇順"),
    (lambda d: d.update({"colums": []}), "未知の項目"),
    (lambda d: d["defaults"].clear(), "材料が指定されていません"),
    (lambda d: d.update({"loads": {"E": {"floors": {"2F": [1, 0]}}}}), "3 成分"),
])
def test_input_errors(patch, msg):
    d = copy.deepcopy(_data())
    patch(d)
    with pytest.raises(InputError, match=msg):
        Building(d)
