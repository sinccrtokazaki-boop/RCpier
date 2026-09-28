"""材端ばねの骨格曲線（トリリニア）

出典: 国土交通省国土技術政策総合研究所ほか監修「2020年版 建築物の構造関係技術基準解説書」
      付録1-3（鉄筋コンクリート造部材の曲げひび割れ・曲げ降伏モーメント、降伏点剛性低下率の略算式）
単位: mm, N（応力は N/mm2）。軸力 N は圧縮正
"""
import math
from dataclasses import dataclass

ES = 2.05e5   # 鉄筋のヤング係数 [N/mm2]


def crack_moment(Fc, b, D, N=0.0):
    """曲げひび割れモーメント Mc = 0.56·√Fc·Ze + N·D/6（Ze = bD²/6、鉄筋を無視）"""
    return 0.56 * math.sqrt(Fc) * b * D ** 2 / 6 + N * D / 6


def beam_yield_moment(at, sy, d):
    """梁の曲げ降伏モーメント My = 0.9·at·σy·d"""
    return 0.9 * at * sy * d


def column_yield_moment(at, ag, sy, b, D, Fc, N):
    """柱の曲げ降伏モーメント（軸力 N による 3区分）
        Nmin ≦ N < 0          : My = 0.8·at·σy·D + 0.4·N·D
        0 ≦ N ≦ 0.4·b·D·Fc    : My = 0.8·at·σy·D + 0.5·N·D·(1 − N/(b·D·Fc))
        0.4·b·D·Fc < N ≦ Nmax : My = (0.8·at·σy·D + 0.12·b·D²·Fc)·(Nmax − N)/(Nmax − 0.4·b·D·Fc)
    Nmin = −ag·σy、Nmax = b·D·Fc + ag·σy。範囲外は 0
    """
    Nmax = b * D * Fc + ag * sy
    Nmin = -ag * sy
    Nb = 0.4 * b * D * Fc
    if N < Nmin or N > Nmax:
        return 0.0
    if N < 0:
        return max(0.0, 0.8 * at * sy * D + 0.4 * N * D)
    if N <= Nb:
        return 0.8 * at * sy * D + 0.5 * N * D * (1 - N / (b * D * Fc))
    return (0.8 * at * sy * D + 0.12 * b * D ** 2 * Fc) * (Nmax - N) / (Nmax - Nb)


def sugano_alpha_y(n, pt, a, D, d, eta0=0.0):
    """降伏点剛性低下率（菅野式） αy = (0.043 + 1.64·n·pt + 0.043·a/D + 0.33·η0)·(d/D)²
    n: ヤング係数比 Es/Ec、pt: 引張鉄筋比 at/(b·d)、a: せん断スパン、η0: 軸力比 N/(b·D·Fc)"""
    return (0.043 + 1.64 * n * pt + 0.043 * a / D + 0.33 * max(eta0, 0.0)) * (d / D) ** 2


def member_end_stiffness(EI, GA, Lf):
    """逆対称曲げを受ける部材（長さ Lf）の材端回転剛性 K0 = 1/(Lf/(6EI) + 2/(GA·Lf))"""
    return 1.0 / (Lf / (6 * EI) + (2.0 / (GA * Lf) if GA else 0.0))


@dataclass
class Skeleton:
    """ばね（片側、モーメント・回転角とも正の向きで表す）の骨格曲線

    剛（回転 0）で Mc まで → 勾配 ks2 で (θy, My) まで → 勾配 ks3
    theta_y = 0 のときはひび割れ域なし（Mc = My のバイリニア）
    """
    Mc: float
    My: float
    theta_y: float
    ks2: float
    ks3: float

    @classmethod
    def from_member(cls, Mc, My, alpha_y, K0, r):
        """部材の M–θ 関係（逆対称曲げ、初期剛性 K0）がトリリニア
        （Mc まで K0、降伏点 (My/(αy·K0), My)、降伏後 r·K0）となるように、
        弾性部材と直列のばねの骨格を定める"""
        if My <= 0:
            raise ValueError("My が 0 以下です")
        if not 0 < r < 1:
            raise ValueError("降伏後の剛性比は 0 < r < 1 としてください")
        Mc = min(Mc, My) if Mc > 0 else 1e-6 * My
        ks3 = r * K0 / (1 - r)                       # 1/(r·K0) = 1/K0 + 1/ks3
        if Mc >= My:
            return cls(My, My, 0.0, math.inf, ks3)
        ay = min(alpha_y, 0.999)
        theta_y = My * (1 / ay - 1) / K0              # 部材の降伏回転角 − 弾性分
        return cls(Mc, My, theta_y, (My - Mc) / theta_y, ks3)

    def M(self, th):
        """回転角 th ≧ 0 での骨格上のモーメント"""
        if th <= 0:
            return self.Mc
        if th < self.theta_y:
            return self.Mc + self.ks2 * th
        return self.My + self.ks3 * (th - self.theta_y)
