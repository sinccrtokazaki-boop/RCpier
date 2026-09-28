"""桟橋用の材料と断面: 鋼材・鋼管杭断面・コンクリート（コンクリート標準示方書）

単位: N, mm（応力は N/mm2）
Element.prepare が使う断面性能 A, Iy, Iz, J, Asy, Asz と材料の E, G を持つ（RectSection・Concrete と同じ形）
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# コンクリートのヤング係数 Ec [kN/mm²]（コンクリート標準示方書［設計編］の f'ck との対応表。
# 表の値はプログラム作成者の知識による。照合が必要）
JSCE_EC_TABLE = ((18, 22.0), (24, 25.0), (30, 28.0), (40, 31.0), (50, 33.0), (60, 35.0),
                 (70, 37.0), (80, 38.0))


@dataclass
class JsceConcrete:
    """上部工のコンクリート（コンクリート標準示方書 2022年版）

    fck : 設計基準強度 f'ck [N/mm2]
    E   : ヤング係数（省略時は JSCE_EC_TABLE を線形補間）
    gamma: 単位体積重量 [kN/m³]（鉄筋コンクリート）
    """
    name: str
    fck: float
    E: Optional[float] = None
    nu: float = 0.2
    gamma: float = 24.5

    def __post_init__(self):
        if self.E is None:
            f, e = zip(*JSCE_EC_TABLE)
            if not f[0] <= self.fck <= f[-1]:
                raise ValueError(f"f'ck = {self.fck} は Ec の表の範囲（{f[0]}〜{f[-1]}）外です。E を指定してください")
            self.E = float(np.interp(self.fck, f, e)) * 1e3

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def Fc(self):          # 建物編の表示・計算と共通の名前（= f'ck）
        return self.fck


@dataclass
class Steel:
    """鋼材（鋼管杭）  fy: 降伏強度 [N/mm2]"""
    name: str
    fy: float
    E: float = 2.0e5
    nu: float = 0.3
    gamma: float = 77.0

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))


@dataclass
class PipeSection:
    """鋼管断面（外径 D, 板厚 t [mm]）と腐食代

    corrosion_outer / corrosion_inner : 外面・内面の腐食代 [mm]
    stiffness : 解析の剛性に用いる断面 "intact"（腐食前）/ "corroded"（腐食後）
    せん断の有効断面積は A/2（薄肉円管のせん断形状係数 2）
    """
    name: str
    D: float
    t: float
    corrosion_outer: float = 0.0
    corrosion_inner: float = 0.0
    stiffness: str = "intact"
    rebar: object = None          # 建物編の検定と共通の属性（鋼管では常に None）
    phi_y: float = 1.0

    def __post_init__(self):
        if self.t <= 0 or self.D <= 2 * self.t:
            raise ValueError(f"鋼管 {self.name}: D > 2t, t > 0 としてください")
        if self.corrosion_outer + self.corrosion_inner >= self.t:
            raise ValueError(f"鋼管 {self.name}: 腐食代の合計が板厚以上です")
        if self.stiffness not in ("intact", "corroded"):
            raise ValueError(f"鋼管 {self.name}: stiffness は intact / corroded です")

    def dims(self, corroded):
        """(外径, 内径)"""
        if corroded:
            return self.D - 2 * self.corrosion_outer, self.D - 2 * self.t + 2 * self.corrosion_inner
        return self.D, self.D - 2 * self.t

    def props(self, corroded=False):
        """断面性能 {A, I, Z, J, As, D, t}（腐食前 / 腐食後）"""
        Do, Di = self.dims(corroded)
        A = math.pi * (Do ** 2 - Di ** 2) / 4
        I = math.pi * (Do ** 4 - Di ** 4) / 64
        return {"A": A, "I": I, "Z": I / (Do / 2), "J": 2 * I, "As": A / 2, "D": Do, "t": (Do - Di) / 2}

    # ---- Element.prepare 用（解析の剛性に用いる断面）
    def _p(self):
        return self.props(self.stiffness == "corroded")

    @property
    def A(self):
        return self._p()["A"]

    @property
    def Iy(self):
        return self._p()["I"]

    @property
    def Iz(self):
        return self._p()["I"]

    @property
    def J(self):
        return self._p()["J"]

    @property
    def Asy(self):
        return self._p()["As"]

    @property
    def Asz(self):
        return self._p()["As"]

    @property
    def b(self):                  # 図・表示用（外径）
        return self.D


def chang_beta(kh, D, EI):
    """Chang の β = (kh·D / 4EI)^(1/4)  （単位は一貫させる: 内部では N/mm³, mm, N·mm² → 1/mm）"""
    return (kh * D / (4 * EI)) ** 0.25
