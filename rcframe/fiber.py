"""鋼管断面のファイバーモデルによる曲げモーメント－曲率（M–φ）関係（利用可否 U3）

    断面: 円環を周方向 n_circ × 半径方向 n_rad の扇形要素に分割（要素の面積・図心は厳密値）
    鋼材: バイリニア（弾性係数 E、降伏強度 fy、降伏後の剛性 H·E）、引張・圧縮で対称
    平面保持: ε(y) = ε0 − φ·y（y: 曲げの中立面からの座標、+y 側が圧縮）
    軸力 N（圧縮正）を一定として、各 φ で Σσ·A = −N となる ε0 を二分法で求め、M = −Σσ·A·y
    限界曲率 φu: 最外縁（y = Do/2）の圧縮ひずみが局部座屈ひずみ εcu に達する曲率

局部座屈ひずみ εcu（港湾の実務で準用される鋼製円形断面の式。いずれも下限 εy）:
    式A  εcu = 1.14/(D/t) − 0.012
    式B  εcu = 0.44·t/R − 0.0026（R: 板厚中心の半径）
    直接入力
    出典は道路橋示方書 V 耐震設計編・土木学会の鋼製円形断面の式とされる（照合が必要: VERIFY_ITEMS）
単位: mm, N
"""
import math
from dataclasses import dataclass, field

import numpy as np

VERIFY_ITEMS = [
    "局部座屈ひずみ εcu の式（式A 1.14/(D/t) − 0.012、式B 0.44·t/R − 0.0026）の出典と適用範囲。"
    "両式の値は大きく異なる（D/t = 71 で約 2.5 倍）ため、基準の原文で確認すること",
    "塑性ヒンジ長 Lp（既定 D）と θu = (φu − φp)·Lp の扱い",
]


def local_buckling_strain(D, t, fy, E, method="A", value=None):
    """局部座屈ひずみ εcu と注意のリスト。D, t は照査に用いる断面（腐食後）の外径・板厚

    method: "A"（1.14/(D/t) − 0.012）/ "B"（0.44·t/R − 0.0026）/ "value"（value を用いる）
    いずれも下限は降伏ひずみ εy = fy/E。D/t > 100 なら注意を返す
    """
    ey = fy / E
    warns = []
    if D / t > 100:
        warns.append(f"径厚比 D/t = {D / t:.0f} > 100（局部座屈ひずみの式の適用範囲外のおそれ）")
    if method == "A":
        ecu = 1.14 / (D / t) - 0.012
    elif method == "B":
        R = (D - t) / 2
        ecu = 0.44 * t / R - 0.0026
    elif method == "value":
        if value is None or value <= 0:
            raise ValueError("局部座屈ひずみを直接入力する場合は正の値を与えてください")
        ecu = float(value)
    else:
        raise ValueError(f"局部座屈ひずみの式は A, B, value のいずれかです: {method!r}")
    if ecu < ey:
        warns.append(f"εcu = {ecu:.5f} が降伏ひずみ εy = {ey:.5f} を下回るため εy とします")
        ecu = ey
    return ecu, warns


@dataclass
class SteelLaw:
    """鋼材のバイリニアの応力－ひずみ関係（引張正、対称）"""
    E: float
    fy: float
    H: float = 0.01               # 降伏後の剛性比（E に対する）

    @property
    def ey(self):
        return self.fy / self.E

    def stress(self, eps):
        """応力（eps はスカラーまたは配列）"""
        eps = np.asarray(eps, float)
        ey = self.ey
        a = np.abs(eps)
        return np.where(a <= ey, self.E * eps, np.sign(eps) * (self.fy + self.H * self.E * (a - ey)))


class PipeFibers:
    """円環断面のファイバー（外径 Do、内径 Di）"""

    def __init__(self, Do, Di, n_circ=72, n_rad=4):
        if not Do > Di >= 0:
            raise ValueError("鋼管の断面: Do > Di ≧ 0 としてください")
        self.Do, self.Di = Do, Di
        self.ymax = Do / 2
        dth = 2 * math.pi / n_circ
        sinc = math.sin(dth / 2) / (dth / 2)
        self.y, self.A = [], []
        for j in range(n_rad):
            r1 = Di / 2 + (Do - Di) / 2 * j / n_rad
            r2 = Di / 2 + (Do - Di) / 2 * (j + 1) / n_rad
            a = (r2 ** 2 - r1 ** 2) / 2 * dth                     # 扇形要素の面積
            rc = 2 / 3 * (r2 ** 3 - r1 ** 3) / (r2 ** 2 - r1 ** 2) * sinc   # 図心の半径
            for i in range(n_circ):
                th = (i + 0.5) * dth
                self.y.append(rc * math.cos(th))
                self.A.append(a)
        self.y, self.A = np.array(self.y), np.array(self.A)
        self.area = float(self.A.sum())

    @property
    def I(self):
        return math.pi * (self.Do ** 4 - self.Di ** 4) / 64

    @property
    def Z(self):
        return self.I / (self.Do / 2)

    @property
    def Zp(self):
        return (self.Do ** 3 - self.Di ** 3) / 6


def _bisect(f, lo, hi, tol=1e-14, it=200):
    """f(lo) ≦ 0 ≦ f(hi) の単調増加関数の根（区間幅が初期幅の tol 倍以下になるまで）"""
    flo = f(lo)
    w0 = hi - lo
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm <= 0) == (flo <= 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo <= tol * w0:
            break
    return 0.5 * (lo + hi)


@dataclass
class MPhi:
    """M–φ 関係（軸力一定）と特性点"""
    N: float
    ecu: float
    phi: list = field(default_factory=list)
    M: list = field(default_factory=list)
    phi_y: float = 0.0            # 最初の降伏（いずれかの縁のひずみが εy）
    My: float = 0.0
    phi_u: float = 0.0            # 限界曲率（最外縁の圧縮ひずみ = εcu）
    Mu: float = 0.0
    Mp: float = 0.0               # 全塑性モーメント（剛塑性の応力分布、軸力 N）
    warnings: list = field(default_factory=list)

    def moment_at(self, phi):
        """曲率 phi の M（表の線形補間、φu を超えれば Mu）"""
        if phi <= 0:
            return 0.0
        for k in range(1, len(self.phi)):
            if phi <= self.phi[k]:
                p0, p1 = self.phi[k - 1], self.phi[k]
                return self.M[k - 1] + (self.M[k] - self.M[k - 1]) * (phi - p0) / (p1 - p0)
        return self.M[-1]

    def phi_at(self, M):
        """M に達する最初の曲率（表の線形補間）。M が表の最大値を超えれば None"""
        if M <= 0:
            return 0.0
        for k in range(1, len(self.phi)):
            if M <= self.M[k]:
                m0, m1 = self.M[k - 1], self.M[k]
                return self.phi[k - 1] + (self.phi[k] - self.phi[k - 1]) * (M - m0) / (m1 - m0)
        return None


class PipeMPhi:
    """鋼管のファイバー断面の解析（軸力 N 一定）"""

    def __init__(self, fibers, law):
        self.f, self.law = fibers, law

    def _axial(self, e0, phi):
        return float(self.law.stress(e0 - phi * self.f.y) @ self.f.A)

    def state(self, phi, N):
        """曲率 phi・軸力 N（圧縮正）での (ε0, M)。Σσ·A = −N"""
        law = self.law
        ny = law.fy * self.f.area
        if abs(N) >= ny and law.H == 0:
            raise ValueError("軸力が降伏軸力以上です")
        span = abs(phi) * self.f.ymax + law.ey * (1 + abs(N) / ny) * 50 + 1e-3
        e0 = _bisect(lambda e: self._axial(e, phi) + N, -span, span)
        M = -float(law.stress(e0 - phi * self.f.y) @ (self.f.A * self.f.y))
        return e0, M

    def strain_edges(self, phi, N):
        """(圧縮縁のひずみ, 引張縁のひずみ)（縁 = y = ±Do/2）"""
        e0, _ = self.state(phi, N)
        r = self.f.ymax
        return e0 - phi * r, e0 + phi * r

    def full_plastic(self, N):
        """全塑性モーメント: 完全弾塑性（H = 0）の材料で曲率を十分大きくした（φ = 10⁴·εy/(Do/2)）ときの M。
        弾性域は中立軸の近傍の幅 2εy/φ だけで、剛塑性（σ = ±fy）の分布の M に一致する"""
        law = SteelLaw(self.law.E, self.law.fy, 0.0)
        return PipeMPhi(self.f, law).state(1e4 * law.ey / self.f.ymax, N)[1]

    def curve(self, N, ecu, n=60):
        """M–φ 関係（0 → φu を n 分割、φy と φu を含む）と特性点"""
        law, r = self.law, self.f.ymax
        out = MPhi(N=N, ecu=ecu)
        phi_e = law.ey / r                              # 目安の曲率（純曲げの初降伏）
        # 最初の降伏: max(|圧縮縁|, |引張縁|) = εy
        def g_y(phi):
            c, t = self.strain_edges(phi, N)
            return max(abs(c), abs(t)) - law.ey
        hi = phi_e
        while g_y(hi) < 0:
            hi *= 2
        out.phi_y = 0.0 if g_y(0.0) >= 0 else _bisect(g_y, 0.0, hi)
        out.My = self.state(out.phi_y, N)[1]
        # 限界曲率: 圧縮縁のひずみ = −εcu
        def g_u(phi):
            return -self.strain_edges(phi, N)[0] - ecu
        if g_u(0.0) >= 0:
            out.warnings.append("軸力による圧縮ひずみだけで εcu に達しています")
            out.phi_u = 0.0
        else:
            hi = phi_e
            while g_u(hi) < 0:
                hi *= 2
            out.phi_u = _bisect(g_u, 0.0, hi)
        out.Mu = self.state(out.phi_u, N)[1]
        out.Mp = self.full_plastic(N)
        pts = sorted({out.phi_u * k / n for k in range(n + 1)} | {out.phi_y})
        out.phi = pts
        out.M = [self.state(p, N)[1] if p > 0 else 0.0 for p in pts]
        return out


def pipe_mphi(sec, mat, N, ecu_method="A", ecu_value=None, hardening=0.01, n_circ=72, n_rad=4, n=60):
    """鋼管（腐食後の断面）の M–φ 関係。sec: PipeSection、mat: Steel、N: 軸力（圧縮正）"""
    Do, Di = sec.dims(corroded=True)
    t = (Do - Di) / 2
    ecu, warns = local_buckling_strain(Do, t, mat.fy, mat.E, ecu_method, ecu_value)
    calc = PipeMPhi(PipeFibers(Do, Di, n_circ, n_rad), SteelLaw(mat.E, mat.fy, hardening))
    c = calc.curve(N, ecu, n)
    c.warnings = warns + c.warnings
    return c
