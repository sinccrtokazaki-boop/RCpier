"""材料と断面の定義

単位系: N, mm（応力は N/mm2）
"""
import re
from dataclasses import dataclass
from typing import Optional, Union

# 異形棒鋼の公称断面積 [mm²]（JIS G 3112）
BAR_AREA = {10: 71.33, 13: 126.7, 16: 198.6, 19: 286.5, 22: 387.1, 25: 506.7,
            29: 642.4, 32: 794.2, 35: 956.6, 38: 1140.0, 41: 1340.0}


@dataclass
class SteelGrade:
    """鉄筋の鋼種と長期許容引張応力度（RC規準 6条 許容応力度）

    F       : 基準強度（短期許容応力度）[N/mm2]
    ftL     : 長期許容引張応力度（主筋, D25 以下）
    ftL_D29 : 長期許容引張応力度（主筋, D29 以上）
    せん断補強筋: 長期 min(F, 195), 短期 min(F, 390)
    """
    name: str
    F: float
    ftL: float
    ftL_D29: float

    def ft(self, dia, term):
        """主筋の許容引張（圧縮）応力度  term: "long" / "short" """
        if term == "short":
            return self.F
        return self.ftL if dia < 29 else self.ftL_D29

    def wft(self, term):
        """せん断補強筋の許容引張応力度"""
        return min(self.F, 195.0) if term == "long" else min(self.F, 390.0)


STEELS = {
    "SD295": SteelGrade("SD295", 295, 195, 195),
    "SD345": SteelGrade("SD345", 345, 215, 195),
    "SD390": SteelGrade("SD390", 390, 215, 195),
}


@dataclass
class Bars:
    """主筋 "4-D25" = 本数 n, 呼び径 dia"""
    n: int
    dia: int

    @property
    def area(self):
        return self.n * BAR_AREA[self.dia]

    @classmethod
    def parse(cls, s):
        mt = re.fullmatch(r"\s*(\d+)\s*-\s*D(\d+)\s*", str(s))
        if not mt or int(mt[2]) not in BAR_AREA:
            raise ValueError(f"主筋は \"本数-D径\"（例 4-D25, D10〜D41）で指定してください: {s!r}")
        return cls(int(mt[1]), int(mt[2]))

    def __str__(self):
        return f"{self.n}-D{self.dia}"


@dataclass
class Hoop:
    """せん断補強筋 "2-D10@200" = 組数（本数）legs, 呼び径 dia, ピッチ pitch [mm]"""
    legs: int
    dia: int
    pitch: float

    @property
    def aw(self):
        return self.legs * BAR_AREA[self.dia]

    def pw(self, width):
        return self.aw / (width * self.pitch)

    @classmethod
    def parse(cls, s):
        mt = re.fullmatch(r"\s*(\d+)\s*-\s*D(\d+)\s*@\s*(\d+(?:\.\d+)?)\s*", str(s))
        if not mt or int(mt[2]) not in BAR_AREA or float(mt[3]) <= 0:
            raise ValueError(f"せん断補強筋は \"本数-D径@ピッチ\"（例 2-D10@200）で指定してください: {s!r}")
        return cls(int(mt[1]), int(mt[2]), float(mt[3]))

    def __str__(self):
        return f"{self.legs}-D{self.dia}@{self.pitch:g}"


@dataclass
class BeamBars:
    """梁のある位置（端部・中央）の配筋"""
    top: Bars
    bottom: Bars
    stirrup: Hoop


@dataclass
class BeamRebar:
    """梁の配筋。dt: 主筋重心から上下縁までの距離 [mm]"""
    main: SteelGrade
    shear: SteelGrade
    dt: float
    end: BeamBars
    center: BeamBars


@dataclass
class ColumnRebar:
    """柱の配筋（柱頭・柱脚共通）

    x : X方向の曲げに対する 1面の主筋（X方向に直交する面、隅筋を含む）
    y : Y方向の曲げに対する 1面の主筋（隅筋を含む）
    hoop_x, hoop_y : X, Y方向のせん断に効く帯筋
    """
    main: SteelGrade
    shear: SteelGrade
    dt: float
    x: Bars
    y: Bars
    hoop_x: Hoop
    hoop_y: Hoop

    @property
    def total_area(self):
        """全主筋断面積（隅筋は x 側で数える）"""
        return 2 * self.x.area + 2 * (self.y.n - 2) * BAR_AREA[self.y.dia]


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
    # 断面算定用の配筋（剛性計算では未使用）。None なら検定しない
    rebar: Optional[Union[BeamRebar, ColumnRebar]] = None

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
