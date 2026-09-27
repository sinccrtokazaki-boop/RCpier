"""材料と断面の定義

単位系: N, mm（応力は N/mm2）
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Concrete:
    """コンクリート材料

    ヤング係数は日本建築学会 RC規準式:
        E = 3.35e4 * (gamma/24)^2 * (Fc/60)^(1/3)   [N/mm2]
    """
    name: str
    Fc: float                 # 設計基準強度 [N/mm2]
    gamma: float = 24.0       # 単位体積重量 [kN/m3]
    nu: float = 0.2           # ポアソン比
    E: Optional[float] = None # 直接指定する場合

    def __post_init__(self):
        if self.E is None:
            self.E = 3.35e4 * (self.gamma / 24.0) ** 2 * (self.Fc / 60.0) ** (1.0 / 3.0)

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class RectSection:
    """矩形RC断面

    b : 部材局所 y 方向の幅 [mm]
    D : 部材局所 z 方向のせい [mm]
    phi_y, phi_z : 断面二次モーメントの剛性増大率（スラブ協力幅の考慮）
        梁で片側スラブ 1.5、両側スラブ 2.0 程度が慣用値
    kappa : せん断形状係数（矩形 1.2）

    Iy : 局所 y 軸まわり = b*D^3/12 （梁の鉛直面内曲げ＝強軸）
    Iz : 局所 z 軸まわり = D*b^3/12
    """
    name: str
    b: float
    D: float
    phi_y: float = 1.0
    phi_z: float = 1.0
    kappa: float = 1.2
    # 将来の断面算定・M-φ関係用（現段階の剛性計算では未使用）
    rebar: dict = field(default_factory=dict)

    @property
    def A(self) -> float:
        return self.b * self.D

    @property
    def Iy(self) -> float:
        return self.phi_y * self.b * self.D ** 3 / 12.0

    @property
    def Iz(self) -> float:
        return self.phi_z * self.D * self.b ** 3 / 12.0

    @property
    def J(self) -> float:
        """矩形断面のサンブナンねじり定数（近似式）"""
        a, c = max(self.b, self.D), min(self.b, self.D)
        return a * c ** 3 * (1.0 / 3.0 - 0.21 * (c / a) * (1.0 - c ** 4 / (12.0 * a ** 4)))

    @property
    def Asy(self) -> float:
        return self.A / self.kappa

    @property
    def Asz(self) -> float:
        return self.A / self.kappa
