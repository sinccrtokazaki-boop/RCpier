"""荷重の自動生成に使う計算（床荷重の分配、地震力）

入力データの形式には依存しない純粋な計算のみを置く。
単位: mm, N（地震力の T は s, 建物高さは m）
"""
import math

# 地盤種別 → Tc [s]（昭和55年建設省告示第1793号 第2 表）
TC = {1: 0.4, 2: 0.6, 3: 0.8}


# ============================================================ 床荷重の分配
def tributary_profiles(lx, ly, edges):
    """長方形の床区画 lx × ly から周辺の梁への負担幅の分布を求める

    edges: 梁のある辺の集合。"x0"/"x1" は y 最小/最大側の X 方向の辺（長さ lx）、
           "y0"/"y1" は x 最小/最大側の Y 方向の辺（長さ ly）
    戻り値: {辺: [(s, 負担幅 mm)]}  s は辺の始点（座標の小さい側）からの距離
            床荷重 w [N/mm²] × 負担幅 = 梁の分布荷重 [N/mm]

    4辺に梁があれば亀甲分割（45°線）: 辺長 a, 直交方向の辺長 b として
        高さ h = min(a, b)/2 の台形（a ≦ b なら三角形）
    対向する 2辺だけに梁があれば一方向スラブ: 各辺に等分布（負担幅 = スパン/2）
    """
    edges = set(edges)
    if edges == {"x0", "x1", "y0", "y1"}:
        out = {}
        for e, a, b in (("x0", lx, ly), ("x1", lx, ly), ("y0", ly, lx), ("y1", ly, lx)):
            h = min(a, b) / 2
            out[e] = [(0.0, 0.0), (h, h), (a - h, h), (a, 0.0)]
        return out
    if edges == {"x0", "x1"}:
        return {e: [(0.0, ly / 2), (lx, ly / 2)] for e in ("x0", "x1")}
    if edges == {"y0", "y1"}:
        return {e: [(0.0, lx / 2), (ly, lx / 2)] for e in ("y0", "y1")}
    raise ValueError(f"梁のある辺 {sorted(edges)} では床荷重を分配できません"
                     "（4辺、または対向する 2辺に梁が必要）")


# ============================================================ 地震力（Ai 分布）
def natural_period(h_m, alpha=0.0):
    """設計用一次固有周期 T = h(0.02 + 0.01α) [s]
    （昭和55年建設省告示第1793号 第2。α: 鉄骨造部分の高さの比、RC造は 0）"""
    return h_m * (0.02 + 0.01 * alpha)


def rt_factor(T, soil):
    """振動特性係数 Rt（昭和55年建設省告示第1793号 第2）"""
    if soil not in TC:
        raise ValueError(f"地盤種別は 1, 2, 3 のいずれかです: {soil!r}")
    Tc = TC[soil]
    if T < Tc:
        return 1.0
    if T < 2 * Tc:
        return 1.0 - 0.2 * (T / Tc - 1.0) ** 2
    return 1.6 * Tc / T


def ai_factor(alpha_i, T):
    """層せん断力係数の高さ方向の分布係数
    Ai = 1 + (1/√αi − αi)·2T/(1 + 3T)（昭和55年建設省告示第1793号 第3）"""
    return 1.0 + (1.0 / math.sqrt(alpha_i) - alpha_i) * 2 * T / (1 + 3 * T)


def seismic_story_forces(W, h_m, Z=1.0, soil=2, C0=0.2, T=None):
    """Ai 分布による各階の地震層せん断力と床位置の外力

    W   : 下から順に、各床（最下階＝基礎を除く）の地震用重量 [N]
    h_m : 建物高さ [m]（T の算定用）
    T   : 固有周期 [s]。省略時は natural_period(h_m)
    Ci = Z·Rt·Ai·C0, Qi = Ci·ΣWi（建築基準法施行令第88条第1項）
    戻り値: (T, Rt, rows)
      rows[k] は k 層（床 k と k+1 の間。k=0 が最下層）:
        {"sumW": ΣWi, "alpha": αi, "Ai", "Ci", "Q": Qi, "P": 上の床に作用する外力}
    """
    if T is None:
        T = natural_period(h_m)
    Rt = rt_factor(T, soil)
    total = sum(W)
    rows = []
    for k in range(len(W)):
        sumW = sum(W[k:])
        alpha = sumW / total
        Ai = ai_factor(alpha, T)
        Ci = Z * Rt * Ai * C0
        rows.append({"sumW": sumW, "alpha": alpha, "Ai": Ai, "Ci": Ci, "Q": Ci * sumW})
    for k, r in enumerate(rows):
        r["P"] = r["Q"] - (rows[k + 1]["Q"] if k + 1 < len(rows) else 0.0)
    return T, Rt, rows
