"""材端ばねの履歴則（最大点指向型、除荷時は剛）

状態は (θ, M)。正側・負側それぞれに骨格曲線（skeleton.Skeleton）を持つ。
    ・骨格曲線上で最大点を更新しながら進む
    ・除荷はばね剛（回転角一定、部材は弾性的に戻る）
    ・M が 0 を越えて反対側へ向かうと、0 を越えた点（アンカー）から反対側の最大点
      （未経験なら反対側のひび割れ点）へ直線で向かい、到達後は骨格曲線に沿う
    ・直線の途中で除荷・再載荷した場合は、回転角一定で直線に戻る
折れ線なので、各分岐の接線柔性と次の折れ点（事象）のモーメントを返し、
pushover.py の事象追跡型増分解析で用いる。負側は符号を反転した座標で扱う。
"""
import math


class Hinge:
    def __init__(self, pos, neg):
        self.sk = {1: pos, -1: neg}
        self.theta = 0.0
        self.M = 0.0
        self.peak = {1: (0.0, pos.Mc), -1: (0.0, neg.Mc)}     # 各側の最大点（反転座標）
        self.anchor = {1: 0.0, -1: 0.0}                       # 各側へ向かう直線の始点 θ
        self.cracked = {1: False, -1: False}
        self.yielded = {1: False, -1: False}
        self.tolM = 1e-9 * max(pos.My, neg.My)
        self.tolT = 1e-12 + 1e-9 * max(pos.theta_y, neg.theta_y)

    def tangent(self, sigma):
        """M を sigma（+1/−1）方向に変化させるときの (接線柔性 dθ/dM, 次の事象の M, 分岐名)"""
        sk = self.sk[sigma]
        th, M = sigma * self.theta, sigma * self.M
        tp, Mp = self.peak[sigma]
        ta = self.anchor[sigma]
        if M < -self.tolM:                                  # 反対側からの除荷: 剛で M = 0 まで
            return 0.0, 0.0, "zero"
        if th >= tp - self.tolT:                            # 最大点以遠: 骨格曲線
            Ms = sk.M(max(th, 0.0))
            if M < Ms - self.tolM:
                return 0.0, sigma * Ms, "skeleton"
            if sk.theta_y > 0 and th < sk.theta_y - self.tolT:
                return 1.0 / sk.ks2, sigma * sk.My, "yield"
            return 1.0 / sk.ks3, sigma * math.inf, "post"
        Ml = Mp * (th - ta) / (tp - ta)                     # 最大点へ向かう直線
        if M < Ml - self.tolM:
            return 0.0, sigma * Ml, "line"
        return (tp - ta) / Mp, sigma * Mp, "peak"

    def advance(self, dM, flex, sigma, hit=False, limit=None, kind=None):
        """M を dM だけ変化させる（hit: 事象点に到達した）。発生した事象名のリストを返す"""
        self.theta += flex * dM
        self.M += dM
        if hit:
            self.M = limit
            if kind == "peak":
                self.theta = sigma * self.peak[sigma][0]
            elif kind == "yield":
                self.theta = sigma * self.sk[sigma].theta_y
        sk = self.sk[sigma]
        th, M = sigma * self.theta, sigma * self.M
        if hit and kind == "zero":
            self.anchor[sigma] = th
        events = []
        tp, Mp = self.peak[sigma]
        on_skeleton = th >= tp - self.tolT and abs(M - sk.M(max(th, 0.0))) <= self.tolM
        if on_skeleton and M >= Mp - self.tolM:
            self.peak[sigma] = (th, M)
        if not self.cracked[sigma] and M >= sk.Mc - self.tolM and M > 0:
            self.cracked[sigma] = True
            events.append("ひび割れ")
        if on_skeleton and not self.yielded[sigma] and M >= sk.My - self.tolM:
            self.yielded[sigma] = True
            events.append("降伏")
        return events

    @property
    def state(self):
        """表示用の状態: "弾性" / "ひび割れ" / "降伏"（正負いずれか）"""
        if any(self.yielded.values()):
            return "降伏"
        if any(self.cracked.values()):
            return "ひび割れ"
        return "弾性"
