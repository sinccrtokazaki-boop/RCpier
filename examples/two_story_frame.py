"""計算例: RC造 2階建て 2スパン×1スパン 立体ラーメン

・柱 600x600, 梁 400x700（外端梁 φ=1.5, 中間梁 φ=2.0）, Fc24
・剛域は RC規準の慣用値で自動設定、各階剛床
・長期荷重 G（梁に等分布）と X方向地震力 E（各階重心に集中）
・組合せ G+E で層間変形角と断面力を確認
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rcframe import Model, Concrete, RectSection, Result

X = [0, 6000, 12000]
Y = [0, 7000]
Z = [0, 4000, 7500]           # 基礎, 2F, RF


def nid(i, j, k): return 1000 * k + 10 * i + j


def build_model():
    """モデルを生成して (Model, {階番号: 床マスター節点}) を返す"""
    fc = Concrete("Fc24", 24)
    col = RectSection("C1", 600, 600)
    g_out = RectSection("G1", 400, 700, phi_y=1.5)
    g_in = RectSection("G2", 400, 700, phi_y=2.0)

    m = Model("2F-RC")
    for k, z in enumerate(Z):
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                m.add_node(nid(i, j, k), x, y, z, fix=[1] * 6 if k == 0 else None)

    eid = 1
    for k in range(1, len(Z)):
        for i in range(len(X)):
            for j in range(len(Y)):
                m.add_element(eid, nid(i, j, k - 1), nid(i, j, k), fc, col, "column"); eid += 1
        for j in range(len(Y)):                       # X方向梁
            for i in range(len(X) - 1):
                m.add_element(eid, nid(i, j, k), nid(i + 1, j, k), fc, g_out); eid += 1
        for i in range(len(X)):                       # Y方向梁
            sec = g_in if 0 < i < len(X) - 1 else g_out
            m.add_element(eid, nid(i, 0, k), nid(i, 1, k), fc, sec); eid += 1

    m.auto_rigid_zones()

    masters = {}
    for k in range(1, len(Z)):
        mid = 9000 + k
        m.add_node(mid, 6000, 3500, Z[k])
        m.add_diaphragm(mid, [nid(i, j, k) for i in range(len(X)) for j in range(len(Y))])
        masters[k] = mid

    # 長期荷重（梁に等分布 30 N/mm = 30 kN/m）
    for e in m.elements.values():
        if e.kind == "beam":
            m.add_member_load("G", e.id, [0, 0, -30.0])
    # 地震力（X方向）: 2F 300 kN, RF 450 kN を重心に作用
    m.add_nodal_load("Ex", masters[1], [300e3, 0, 0, 0, 0, 0])
    m.add_nodal_load("Ex", masters[2], [450e3, 0, 0, 0, 0, 0])
    return m, masters


def main():
    m, masters = build_model()
    rG = m.analyze("G")
    rE = m.analyze("Ex")
    rGE = Result.combine("G+Ex", {rG: 1.0, rE: 1.0})

    print("=== 剛域（自動設定）の例 ===")
    for e in list(m.elements.values())[:10]:
        print(f"要素{e.id:>3} {e.kind:6s} 剛域 i={e.rigid_i:6.1f}  j={e.rigid_j:6.1f} mm")

    print("\n=== 層間変形角（Ex） ===")
    for name, du, h, ang in rE.story_drifts([("1F", masters[1]), ("2F", masters[2])], 0):
        flag = "OK" if ang <= 1 / 200 else "NG"
        print(f"{name}: δ={du:6.2f} mm  h={h:.0f} mm  1/{1/ang:,.0f}  (≦1/200 : {flag})")

    rE.print_reactions()
    cols_1F = [e.id for e in m.elements.values() if e.kind == "column" and m.nodes[e.ni].z == 0]
    rGE.print_section_forces("face", cols_1F)


if __name__ == "__main__":
    main()
