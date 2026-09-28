"""計算例: YAML 入力ファイルからモデルを自動生成して解析する

使い方: python examples/from_yaml.py [入力ファイル.yaml]
        （省略時は examples/two_story_frame.yaml）
    例: python examples/from_yaml.py examples/two_story_auto_loads.yaml
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rcframe import Result, load_building, run_checks, run_pushover

path = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_name("two_story_frame.yaml")
bld = load_building(path)
m = bld.model
cases = sorted(set(m.nodal_loads) | set(m.member_loads))
print(f"モデル {m.name}: 節点 {len(m.nodes)}, 要素 {len(m.elements)}, 剛床 {len(m.diaphragms)}")
print(f"荷重ケース: {', '.join(cases)}")

print("\n=== 剛域（自動設定） ===")
for e in m.elements.values():
    print(f"要素{e.id:>3} {bld.label('elem', e.id):16s} 剛域 i={e.rigid_i:6.1f}  j={e.rigid_j:6.1f} mm")

if bld.seismic:
    bld.print_seismic_table()
res = {c: m.analyze(c) for c in cases}

for c, d in (("Ex", 0), ("Ey", 1)):
    if c in res:
        print(f"\n=== 層間変形角（{c}） ===")
        for name, du, h, ang in res[c].story_drifts(bld.drift_levels(), d):
            flag = "OK" if abs(ang) <= 1 / 200 else "NG"
            print(f"{name}: δ={du:6.2f} mm  h={h:.0f} mm  1/{1 / abs(ang):,.0f}  (≦1/200 : {flag})")

long_term = [c for c in ("G", "P") if c in res]
for c in long_term:
    res[c].print_reactions()

base = bld.floors[0][0]
cols_1F = [eid for (k, f, *_), eid in bld.elem_ids.items() if k == "c" and f == 0]
print(f"\n（{base}柱: " + ", ".join(f"{e}={bld.label('elem', e)}" for e in cols_1F) + "）")
if "Ex" in res and long_term:
    name = "+".join(long_term + ["Ex"])
    comb = Result.combine(name, {res[c]: 1.0 for c in long_term + ["Ex"]})
    comb.print_section_forces("face", cols_1F)

# 配筋のある断面があれば断面検定（RC規準）
if any(s.rebar for s in bld.sections.values()):
    run_checks(bld, res).print_summary()

# nonlinear の指定があればプッシュオーバー
if bld.nonlinear["given"]:
    for d in bld.nonlinear["pushover"]["directions"]:
        po = run_pushover(bld, d, res)
        po.print_curve(every=20)
        po.print_hinges()
