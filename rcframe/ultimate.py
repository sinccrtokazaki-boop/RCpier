"""保有水平耐力計算に用いる計算（入力形式に依存しない）

・せん断終局強度（荒川式）
・部材種別 FA〜FD、部材群 A〜D、構造特性係数 Ds（昭和55年建設省告示第1792号 第4、RC造）
・剛性率・偏心率（建築基準法施行令第82条の6、昭和55年建設省告示第1791号）と
  形状係数 Fes = Fs·Fe（昭和55年建設省告示第1792号 第7）
単位: mm, N（応力は N/mm2）
"""
import math

ARAKAWA = {"mean": 0.068, "min": 0.053}

# RC造 柱及びはりの部材群と Ds（耐力壁のない純ラーメン、βu = 0）
DS_RC_FRAME = {"A": 0.30, "B": 0.35, "C": 0.40, "D": 0.45}

# 部材種別の境界（柱: h0/D の下限、σ0/Fc・pt・τu/Fc の上限、はり: τu/Fc の上限）
COLUMN_CLASS = (("FA", 2.5, 0.35, 0.008, 0.100),
                ("FB", 2.0, 0.45, 0.010, 0.125),
                ("FC", 0.0, 0.55, math.inf, 0.150))
BEAM_CLASS = (("FA", 0.15), ("FB", 0.20), ("FC", math.inf))
CLASS_ORDER = ("FA", "FB", "FC", "FD")


# ============================================================ せん断終局強度
def qsu_arakawa(b, d, pt, M_Qd, pw, sigma_wy, sigma0, Fc, kind="min"):
    """荒川式によるせん断終局強度
        Qsu = {k·pt^0.23·(Fc+18)/(M/(Q·d)+0.12) + 0.85·√(pw·σwy) + 0.1·σ0}·b·j
    k = 0.068（mean 式）/ 0.053（min 式）、pt は % 表示（引数は比 at/(b·d)）、
    1 ≦ M/(Q·d) ≦ 3、pw ≦ 1.2%、j = 7d/8、σ0: 平均軸応力度（圧縮正、はりは 0）
    """
    ptp = 100.0 * pt
    r = min(max(M_Qd, 1.0), 3.0)
    pw = min(pw, 0.012)
    tau = (ARAKAWA[kind] * ptp ** 0.23 * (Fc + 18) / (r + 0.12)
           + 0.85 * math.sqrt(pw * sigma_wy) + 0.1 * sigma0)
    return max(tau, 0.0) * b * 7 / 8 * d


# ============================================================ 部材種別・部材群・Ds
def column_class(h0_D, s0_Fc, pt, tau_Fc, shear_failure=False):
    """柱の部材種別（せん断破壊等が生じる場合は FD）"""
    if shear_failure:
        return "FD"
    for name, h0, s0, p, t in COLUMN_CLASS:
        if h0_D >= h0 and s0_Fc <= s0 and pt <= p and tau_Fc <= t:
            return name
    return "FD"


def beam_class(tau_Fc, shear_failure=False):
    """はりの部材種別"""
    if shear_failure:
        return "FD"
    for name, t in BEAM_CLASS:
        if tau_Fc <= t:
            return name
    return "FD"


def worse(*classes):
    """部材種別のうち最も低いもの"""
    return max(classes, key=CLASS_ORDER.index)


def member_group(gamma):
    """部材群の種別  gamma = {"FA": γA, ..., "FD": γD}（耐力の割合）
        D: FD の部材がある（γD > 0）
        A: γA ≧ 0.5 かつ γC ≦ 0.2
        B: γC < 0.5（A を除く）
        C: γC ≧ 0.5
    FD の部材を除いても崩壊形が保たれる場合に FD を除外して判定する規定は扱わず、
    FD があれば安全側に D とする
    """
    gA, gC, gD = gamma.get("FA", 0.0), gamma.get("FC", 0.0), gamma.get("FD", 0.0)
    if gD > 0:
        return "D"
    if gA >= 0.5 and gC <= 0.2:
        return "A"
    if gC < 0.5:
        return "B"
    return "C"


# ============================================================ 剛性率・偏心率・Fes
def stiffness_ratios(rs):
    """剛性率 Rs = rs / r̄s（rs = h/δ: 層間変形角の逆数、r̄s: 全層の相加平均）"""
    mean = sum(rs) / len(rs)
    return [r / mean for r in rs]


def fs_factor(Rs):
    """Fs = 1.0（Rs ≧ 0.6）、2.0 − Rs/0.6（Rs < 0.6）"""
    return 1.0 if Rs >= 0.6 else 2.0 - Rs / 0.6


def fe_factor(Re):
    """Fe = 1.0（Re ≦ 0.15）、1.0 + 0.5·(Re − 0.15)/0.15（0.15 < Re < 0.3）、1.5（Re ≧ 0.3）"""
    if Re <= 0.15:
        return 1.0
    if Re < 0.3:
        return 1.0 + 0.5 * (Re - 0.15) / 0.15
    return 1.5


def eccentricity(cols, xg, yg):
    """偏心率  cols: [(x, y, Dx, Dy)]（各柱の位置と水平剛性）、(xg, yg): 重心

    剛心 xk = ΣDy·x/ΣDy, yk = ΣDx·y/ΣDx
    ねじり剛性 KR = ΣDx·(y − yk)² + ΣDy·(x − xk)²
    弾力半径 rex = √(KR/ΣDx), rey = √(KR/ΣDy)
    偏心距離 ex = |xg − xk|, ey = |yg − yk|
    偏心率 Rex = ey/rex（X方向の加力）, Rey = ex/rey（Y方向の加力）
    """
    sDx = sum(c[2] for c in cols)
    sDy = sum(c[3] for c in cols)
    xk = sum(c[3] * c[0] for c in cols) / sDy
    yk = sum(c[2] * c[1] for c in cols) / sDx
    KR = sum(c[2] * (c[1] - yk) ** 2 + c[3] * (c[0] - xk) ** 2 for c in cols)
    rex, rey = math.sqrt(KR / sDx), math.sqrt(KR / sDy)
    ex, ey = abs(xg - xk), abs(yg - yk)
    return {"xk": xk, "yk": yk, "KR": KR, "rex": rex, "rey": rey, "ex": ex, "ey": ey,
            "Rex": ey / rex, "Rey": ex / rey}
