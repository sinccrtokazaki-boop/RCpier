"""桟橋の荷重の生成に使う計算（入力形式に依存しない）

座標: X = 法線方向、Y = 陸側 → 海側（海側が +Y）、Z = 上向き。単位は呼び出し側で揃える（内部 mm, N）
"""
import math

import numpy as np


def lever_split(x, x0, x1):
    """位置 x の集中荷重を、梁の両端 x0, x1 の節点へ振り分ける係数（てこの原理 = 単純梁の反力）"""
    if not x0 <= x <= x1 or x1 <= x0:
        raise ValueError(f"位置 {x} が梁の範囲 [{x0}, {x1}] の外です")
    t = (x - x0) / (x1 - x0)
    return 1.0 - t, t


def point_to_nodes(P, F, Q0, Q1):
    """点 P（3D）に作用する力 F（3成分）を、X方向の梁 Q0–Q1 の両端節点へ移す

    1. P を梁の軸線上の点 Q（x は P と同じ）へ移し、偏心によるモーメント r × F（r = P − Q）を加える
    2. 力とモーメントを、てこの原理で両端節点に振り分ける（梁の面内・面外の曲げによる局所的な配分は
       剛床の上部工では小さいとして無視）
    戻り値: (節点 Q0 の [Fx, Fy, Fz, Mx, My, Mz], 節点 Q1 の同)
    """
    P, F, Q0, Q1 = (np.asarray(v, float) for v in (P, F, Q0, Q1))
    a0, a1 = lever_split(P[0], Q0[0], Q1[0])
    Q = np.array([P[0], Q0[1], Q0[2]])
    M = np.cross(P - Q, F)
    f = np.r_[F, M]
    return a0 * f, a1 * f


def mooring_force(P, angle_h, angle_v):
    """牽引力 P [N] の成分（海側 +Y 向き、水平角 angle_h: 法線直角方向から +X 側へ、仰角 angle_v）[°]"""
    h, v = math.radians(angle_h), math.radians(angle_v)
    return np.array([P * math.cos(v) * math.sin(h), P * math.cos(v) * math.cos(h), P * math.sin(v)])


class WeightSum:
    """重量と重心の集計"""

    def __init__(self):
        self.W = self.Wx = self.Wy = 0.0

    def add(self, w, x, y):
        self.W += w
        self.Wx += w * x
        self.Wy += w * y

    @property
    def center(self):
        return (self.Wx / self.W, self.Wy / self.W) if self.W else (math.nan, math.nan)
