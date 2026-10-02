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


class BilinearHinge:
    """鋼材の材端ばね: 剛塑性 + 線形硬化（移動硬化、バイリニア型の履歴）

    降伏線 M = ±My + g(θ)（skeleton の My）。g は硬化の関数で、|θ| ≦ θu では勾配 ks3、|θ| > θu では
    勾配 ks4（局部座屈後、θu・ks4 を与えたとき。省略時は全域 ks3）。降伏線の間（|M − g(θ)| < My）では
    ばね剛（部材は弾性的に応答）。除荷すると 2My の幅を剛のまま戻り、反対側の降伏線に達すると再降伏する。
    単調載荷では Hinge（バイリニア骨格 Mc = My）と同じ経路。Hinge と同じ使い方（tangent / advance）
    """

    def __init__(self, sk, theta_u=None, ks4=None):
        if sk.theta_y != 0:
            raise ValueError("BilinearHinge は Mc = My のバイリニア骨格（Skeleton.from_member(Mp, Mp, ...)）です")
        self.sk = {1: sk, -1: sk}
        self.My, self.ks3 = sk.My, sk.ks3
        self.theta_u = math.inf if theta_u is None else theta_u
        self.ks4 = sk.ks3 if ks4 is None else ks4
        self.theta = 0.0
        self.M = 0.0
        self.cracked = {1: False, -1: False}
        self.yielded = {1: False, -1: False}
        self.buckled = {1: False, -1: False}           # θu を越えた（局部座屈）
        self.tolM = 1e-9 * sk.My
        self.tolT = 1e-12 + 1e-9 * (0.0 if math.isinf(self.theta_u) else self.theta_u)

    def g(self, th):
        """降伏線の中心（移動硬化の背応力）"""
        tu = self.theta_u
        if abs(th) <= tu:
            return self.ks3 * th
        return math.copysign(self.ks3 * tu + self.ks4 * (abs(th) - tu), th)

    def tangent(self, sigma):
        back = self.g(self.theta)
        if sigma * (self.M - back) >= self.My - self.tolM:
            th, tu = self.theta, self.theta_u
            # θ は sigma の向きに進む。次の折れ点（|θ| = θu）までの勾配と、そこでのモーメント
            if math.isinf(tu):
                return 1.0 / self.ks3, sigma * math.inf, "post"
            if sigma * th >= tu - self.tolT:                    # 外側（局部座屈後）へ
                return 1.0 / self.ks4, sigma * math.inf, "post"
            if sigma * th < -tu - self.tolT:                    # 反対側の座屈域から戻る（勾配 ks4）
                return 1.0 / self.ks4, sigma * self.My + self.g(-sigma * tu), "post_b"
            return 1.0 / self.ks3, sigma * self.My + self.g(sigma * tu), "post_u"
        return 0.0, back + sigma * self.My, "elastic"

    def advance(self, dM, flex, sigma, hit=False, limit=None, kind=None):
        self.theta += flex * dM
        self.M += dM
        if hit:
            self.M = limit
            if kind == "post_u":
                self.theta = sigma * self.theta_u
            elif kind == "post_b":
                self.theta = -sigma * self.theta_u
        events = []
        if not self.yielded[sigma] and sigma * (self.M - self.g(self.theta)) >= self.My - self.tolM:
            self.yielded[sigma] = True
            self.cracked[sigma] = True
            events.append("降伏")
        if not self.buckled[sigma] and sigma * self.theta >= self.theta_u - self.tolT:
            self.buckled[sigma] = True
            events.append("局部座屈")
        return events

    @property
    def state(self):
        """表示用の状態（"弾性" / "降伏"）。局部座屈は buckled と事象「局部座屈」で示す"""
        return "降伏" if any(self.yielded.values()) else "弾性"


class EPSpring:
    """節点の並進ばね（地盤ばね）の弾完全塑性モデル（上限 cap、降伏後は r·k の微小剛性）

    力 F = ばねの抵抗力（変位と同じ符号を正）。除荷は弾性剛性 k。
    tangent / advance は Hinge と同じ使い方（事象追跡型の増分解析用）
    """

    def __init__(self, k, cap, r=1e-4):
        self.k, self.cap, self.r = k, cap, r
        self.F = 0.0
        self.yielded = False
        self.tol = 1e-9 * cap

    def tangent(self, sigma):
        """F を sigma 方向に変化させるときの (接線剛性, 次の事象の F, 分岐名)"""
        if sigma * self.F >= self.cap - self.tol:
            return self.r * self.k, sigma * math.inf, "plastic"
        return self.k, sigma * self.cap, "elastic"

    def advance(self, dF, hit=False, limit=None):
        self.F += dF
        if hit:
            self.F = limit
        if not self.yielded and abs(self.F) >= self.cap - self.tol:
            self.yielded = True
            return ["地盤降伏"]
        return []

    @property
    def state(self):
        return "降伏" if self.yielded else "弾性"
