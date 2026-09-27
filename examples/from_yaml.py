"""計算例: YAML 入力ファイルからモデルを自動生成して解析する

使い方: python examples/from_yaml.py [入力ファイル.yaml]
        （省略時は examples/two_story_frame.yaml）
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rcframe import Result, load_building

path = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_name("two_story_frame.yaml")
bld = load_building(path)
m = bld.model
print(f"モデル {m.name}: 節点 {len(m.nodes)}, 要素 {len(m.elements)}, 剛床 {len(m.diaphragms)}")

print("\n=== 剛域（自動設定） ===")
for e in m.elements.values():
    print(f"要素{e.id:>3} {bld.label('elem', e.id):16s} 剛域 i={e.rigid_i:6.1f}  j={e.rigid_j:6.1f} mm")

rG = m.analyze("G")
rE = m.analyze("Ex")
rGE = Result.combine("G+Ex", {rG: 1.0, rE: 1.0})

print("\n=== 層間変形角（Ex） ===")
for name, du, h, ang in rE.story_drifts(bld.drift_levels(), 0):
    flag = "OK" if ang <= 1 / 200 else "NG"
    print(f"{name}: δ={du:6.2f} mm  h={h:.0f} mm  1/{1/ang:,.0f}  (≦1/200 : {flag})")

rE.print_reactions()
base = bld.floors[0][0]
cols_1F = [eid for (k, f, *_), eid in bld.elem_ids.items() if k == "c" and f == 0]
print(f"（{base}柱: " + ", ".join(f"{e}={bld.label('elem', e)}" for e in cols_1F) + "）")
rGE.print_section_forces("face", cols_1F)
