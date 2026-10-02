"""計算例: 直杭式横桟橋のモデル生成と線形解析（桟橋編 P1）

使い方: python examples/pier_example.py [入力ファイル.yaml]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rcframe import load_model, run_flow, run_level2, run_pier_checks

path = sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).with_name("pier_example.yaml")
pier = load_model(path)
m = pier.model
print(f"モデル {m.name}: 節点 {len(m.nodes)}, 要素 {len(m.elements)}, 杭 {len(pier.piles)}"
      f"（杭のモデル: {'地盤ばね' if pier.pile_model == 'soil_spring' else '仮想固定点'}）")

print("\n=== 杭の仮想固定点（Chang の方法） ===")
print(f"{'杭':<8} {'海底面[m]':>9} {'kh[kN/m³]':>10} {'1/β[m]':>7} {'固定点[m]':>9} {'杭長[m]':>8} {'自由長[m]':>9}")
for (i, j), p in sorted(pier.piles.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if i != 0:
        continue                       # 列ごとに同じなので各列の先頭だけ表示
    print(f"{pier.gx[i][0]}/{pier.gy[j][0]:<5} {p['seabed'] / 1e3:9.2f} {p['kh'] * 1e6:10.0f} "
          f"{1 / p['beta'] / 1e3:7.2f} {p['z_fix'] / 1e3:9.2f} {p['length'] / 1e3:8.2f} "
          f"{p['free_length'] / 1e3:9.2f}")

if pier.weights:
    print("\n=== 重量と重心 ===")
    for k, (W, xg, yg) in pier.weights.items():
        print(f"{k:<4} W = {W / 1e3:9.1f} kN  重心 ({xg / 1e3:.2f}, {yg / 1e3:.2f}) m")
if pier.seismic:
    s = pier.seismic
    print(f"レベル1地震動: kh = {s['kh']:g}, kv = {s['kv']:g}, 慣性力 kh·W = {s['kh'] * s['W'] / 1e3:.1f} kN")

res = pier.analyze_cases()
if pier.combinations:
    print("\n=== 荷重の組合せ（部分係数は YAML の値。港湾基準で要確認） ===")
    print(f"{'組合せ':<12} {'状態':<3} {'ux[mm]':>7} {'uy[mm]':>7} {'杭の最大圧縮[kN]':>15} {'最大引張[kN]':>11} "
          f"{'杭頭の最大曲げ[kN·m]':>18}")
    for name, r in pier.analyze_combinations(res).items():
        d = r.disp[pier.master()]
        Ns, Ms = [], []
        for p in pier.piles.values():
            si, sj = r.section_forces(p["eid"], "face")
            Ns.append(sj[0])
            Ms.append(max(abs(sj[4]), abs(sj[5])))
        print(f"{name:<12} {pier.combinations[name]['state']:<4} {d[0]:7.2f} {d[1]:7.2f} "
              f"{-min(Ns) / 1e3:15.1f} {max(max(Ns), 0) / 1e3:11.1f} {max(Ms) / 1e6:18.1f}")

if pier.combinations and pier.checks["given"]:
    rep = run_pier_checks(pier, res)
    print("\n=== 照査項目ごとの最大照査値 ===")
    for (item,), r in sorted(rep.governing(("item",)).items()):
        print(f"{item:<10} {r.label:<14} {r.pos:<4} {r.combo:<12} {r.ratio:5.2f}  {'OK' if r.ok else 'NG'}  {r.note}")
    print(f"最大照査値 {rep.max_ratio:.2f}、NG {len(rep.ng)} 箇所（全部材は rep.print_summary()）")

if pier.level2["given"]:
    for d in pier.level2["directions"]:
        run_level2(pier, d, res).print()

for case in getattr(pier, "lateral_flow", []):          # 側方流動のシナリオ（U2）
    run_flow(pier, case, res).print()

for case in ("H", "W"):
    if case not in res:
        continue
    r = res[case]
    print(f"\n=== 荷重ケース {case} ===")
    if pier.master() is not None:
        d = r.disp[pier.master()]
        print(f"上部工の変位（図心）: ux={d[0]:.2f} uy={d[1]:.2f} mm, 回転 rz={d[5]:.2e} rad")
    print(f"{'杭':<8} {'N[kN]':>9} {'Qx[kN]':>8} {'Qy[kN]':>8} {'杭頭 Mx方向[kN·m]':>16} {'固定点[kN·m]':>12}")
    for (i, j), p in sorted(pier.piles.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if i not in (0, 3):
            continue
        si, sj = r.section_forces(p["eid"], "face")
        # 柱の局所 y = 全体 X、局所 z = 全体 Y。Y方向の曲げは My
        print(f"{pier.gx[i][0]}/{pier.gy[j][0]:<5} {sj[0] / 1e3:9.1f} {sj[1] / 1e3:8.1f} {sj[2] / 1e3:8.1f} "
              f"{sj[4] / 1e6:16.1f} {si[4] / 1e6:12.1f}")
