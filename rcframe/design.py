"""断面算定（許容応力度設計）: 日本建築学会「鉄筋コンクリート構造計算規準・同解説」2018年版

入力形式に依存しない計算のみを置く。単位: mm, N（応力は N/mm2）
term は "long"（長期）/ "short"（短期）
"""
import math

import numpy as np

N_RATIO = 15.0       # ヤング係数比 n（RC規準 6条）
PW_MAX = 0.012       # せん断補強筋比の算定上の上限（RC規準 15条）
PW_MIN = 0.002       # 同 下限（構造規定）


# ============================================================ コンクリートの許容応力度（RC規準 6条）
def concrete_fc(Fc, term):
    """許容圧縮応力度: 長期 Fc/3, 短期 2Fc/3"""
    return Fc / 3 if term == "long" else 2 * Fc / 3


def concrete_fs(Fc, term):
    """許容せん断応力度: 長期 min(Fc/30, 0.49 + Fc/100), 短期 = 長期 × 1.5"""
    fs = min(Fc / 30, 0.49 + Fc / 100)
    return fs if term == "long" else 1.5 * fs


# ============================================================ 曲げ・軸力（RC規準 13条・14条）
def allowable_moment(b, D, a_c, a_t, dt, N, fc, ft_c, ft_t, n=N_RATIO):
    """軸力 N を受ける矩形断面の許容曲げモーメント Ma（平面保持・弾性、コンクリートの引張無視）

    b, D : 幅, せい（曲げの方向）
    a_c  : 圧縮縁側の鉄筋断面積（圧縮縁から dt）
    a_t  : 引張縁側の鉄筋断面積（引張縁から dt）
    N    : 軸力（圧縮正）[N]
    fc   : コンクリートの許容圧縮応力度、ft_c, ft_t: 各鉄筋の許容応力度（圧縮・引張とも同値）
    鉄筋の応力は n 倍法（コンクリートの欠損は考慮しない）
    許容曲げモーメントは、コンクリート圧縮縁・各鉄筋のいずれかが許容応力度に達するときの値
    （梁は N = 0 として 13条、柱は 14条）。戻り値: Ma [N·mm]（N が許容範囲外なら 0）

    中立軸位置 xn（圧縮縁から）を −∞〜+∞ で動かすと、許容応力度に達する (N, M) は
    純引張から純圧縮まで単調に変化するので、N に一致する xn を二分法で求める。
    """
    d, dc = D - dt, dt
    ft_min = min(ft for ft, a in ((ft_c, a_c), (ft_t, a_t)) if a > 0) if (a_c or a_t) else 0.0
    N_max = min(fc, ft_min / n if ft_min else fc) * (b * D + n * (a_c + a_t))
    N_min = -ft_min * (a_c + a_t)
    if not (N_min < N < N_max):
        return 0.0

    def state(xn):
        """中立軸 xn での許容 (N, M)。M は断面中心まわり（圧縮縁側が圧縮となる向きを正）"""
        xc = min(max(xn, 0.0), D)                 # コンクリート圧縮域の深さ
        Nc = b * (xn * xc - xc ** 2 / 2)
        Mc = b * (xn * D / 2 * xc - (xn + D / 2) * xc ** 2 / 2 + xc ** 3 / 3)
        sc, st = n * (xn - dc), n * (xn - d)      # 単位応力場 s(y) = xn − y に対する鉄筋応力
        Nu = Nc + a_c * sc + a_t * st
        Mu = Mc + a_c * sc * (D / 2 - dc) + a_t * st * (D / 2 - d)
        lims = []
        if xn > 0:
            lims.append(fc / xn)
        if a_c and sc:
            lims.append(ft_c / abs(sc))
        if a_t and st:
            lims.append(ft_t / abs(st))
        lam = min(lims)
        return lam * Nu, lam * Mu

    xn_of = lambda th: D / 2 + D * math.tan(th)
    ths = np.linspace(-math.pi / 2 + 1e-9, math.pi / 2 - 1e-9, 401)
    Ns = [state(xn_of(t))[0] - N for t in ths]
    best = 0.0
    for k in range(len(ths) - 1):
        if Ns[k] == 0 or Ns[k] * Ns[k + 1] < 0:
            lo, hi, flo = ths[k], ths[k + 1], Ns[k]
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                fm = state(xn_of(mid))[0] - N
                if (fm < 0) == (flo < 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            best = max(best, state(xn_of(0.5 * (lo + hi)))[1])
    return best


# ============================================================ せん断（RC規準 15条）
def alpha_factor(M, Q, d):
    """せん断スパン比による割増し係数 α = 4 / (M/(Q·d) + 1)、1 ≦ α ≦ 2"""
    if Q == 0:
        return 1.0
    return min(2.0, max(1.0, 4.0 / (abs(M) / (abs(Q) * d) + 1.0)))


def _pw_term(pw):
    return max(0.0, min(pw, PW_MAX) - PW_MIN)


def beam_shear_long(b, d, fs, alpha):
    """梁の長期許容せん断力 QAL = b·j·α·fs（j = 7d/8）"""
    return b * 7 / 8 * d * alpha * fs


def beam_shear_short(b, d, fs, alpha, wft, pw):
    """梁の短期許容せん断力 QAS = b·j·{(2/3)·α·fs + 0.5·wft·(pw − 0.002)}（pw ≦ 1.2%）"""
    return b * 7 / 8 * d * (2 / 3 * alpha * fs + 0.5 * wft * _pw_term(pw))


def column_shear_long(b, d, fs, alpha):
    """柱の長期許容せん断力 QAL = b·j·α·fs"""
    return b * 7 / 8 * d * alpha * fs


def column_shear_short(b, d, fs, wft, pw):
    """柱の短期許容せん断力 QAS = b·j·{fs + 0.5·wft·(pw − 0.002)}（pw ≦ 1.2%）"""
    return b * 7 / 8 * d * (fs + 0.5 * wft * _pw_term(pw))
