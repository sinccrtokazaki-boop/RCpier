"""解析結果の保持・整形・荷重組合せ"""
import numpy as np

from .element import _profile_samples


class Result:
    def __init__(self, model, case, disp, forces, reactions, factors=None):
        self.model = model
        self.case = case
        self.disp = disp            # nid -> 6
        self.forces = forces        # eid -> {"face": 12, "node": 12}（部材に作用する局所端力）
        self.reactions = reactions  # nid -> 6
        self.factors = factors or {case: 1.0}   # 荷重ケース -> 係数（組合せ結果の部材荷重の復元用）
        self.spring_forces = {}                 # nid -> 節点ばねが節点に及ぼす力（Model.analyze が設定）

    # -------------------------------------------------- 断面力（設計用符号）
    def section_forces(self, eid, at="face"):
        """断面力 (N, Qy, Qz, Mx, My, Mz) を i端・j端で返す
        符号: N は引張正。材端力 f に対し i端 = -f_i, j端 = +f_j
        at: "face"（剛域端＝フェイス位置）/ "node"（節点位置）
        """
        f = self.forces[eid][at]
        return -f[:6], f[6:]

    def section_forces_at(self, eid, x):
        """i節点から距離 x [mm] の位置の断面力 (N, Qy, Qz, Mx, My, Mz)
        符号は section_forces と同じ（x = 0 で i端、x = L で j端の節点位置の値）

        i端側の部分 [0, x] の釣合いから求める（部材荷重は剛域上も含めて考慮）:
            F(x) = −F_i − ∫q dt
            My(x) = −My_i − x·Fz_i − ∫(x−t)·qz dt
            Mz(x) = −Mz_i + x·Fy_i + ∫(x−t)·qy dt
        水平梁（局所 z 上向き）では My > 0 が上端引張
        """
        e = self.model.elements[eid]
        f = self.forces[eid]["node"]
        Fi, Mi = f[0:3], f[3:6]
        q0 = np.zeros(3)    # ∫_0^x q dt
        q1 = np.zeros(3)    # ∫_0^x (x − t) q dt
        for case, c in self.factors.items():
            for le, w, profile in self.model.member_loads.get(case, []):
                if le != eid:
                    continue
                wl = c * (e.R @ w)
                if profile is None:
                    q0 += wl * x
                    q1 += wl * x ** 2 / 2
                else:
                    t, wt = _profile_samples(profile, 0.0, x)
                    q0 += wl * np.sum(wt)
                    q1 += wl * np.sum(wt * (x - t))
        s = np.zeros(6)
        s[0:3] = -Fi - q0
        s[3] = -Mi[0]
        s[4] = -Mi[1] - x * Fi[2] - q1[2]
        s[5] = -Mi[2] + x * Fi[1] + q1[1]
        return s

    # -------------------------------------------------- 荷重組合せ
    @staticmethod
    def combine(name, factors: dict):
        """線形重ね合わせ: factors = {Result: 係数}"""
        items = list(factors.items())
        r0 = items[0][0]
        disp = {n: sum(c * r.disp[n] for r, c in items) for n in r0.disp}
        forces = {e: {k: sum(c * r.forces[e][k] for r, c in items) for k in ("face", "node")}
                  for e in r0.forces}
        reac = {n: sum(c * r.reactions[n] for r, c in items) for n in r0.reactions}
        fac = {}
        for r, c in items:
            for case, k in r.factors.items():
                fac[case] = fac.get(case, 0.0) + c * k
        out = Result(r0.model, name, disp, forces, reac, fac)
        out.spring_forces = {n: sum(c * r.spring_forces[n] for r, c in items) for n in r0.spring_forces}
        return out

    # -------------------------------------------------- 層間変形角
    def story_drifts(self, masters_by_level, direction=0, base_z=None):
        """masters_by_level: 下階から順に [(階名, 床マスター節点 or None=基礎)]
        direction: 0=X, 1=Y
        base_z: 基礎（変位 0 とみなす面）の高さ [mm]
            省略時は支点（拘束のある節点）の最小 z、支点がなければ 0
        戻り値: [(階名, 層間変位[mm], 階高[mm], 層間変形角)]
        """
        if base_z is None:
            zs = [nd.z for nd in self.model.nodes.values() if any(nd.fix)]
            base_z = min(zs) if zs else 0.0
        out = []
        prev_u, prev_z = 0.0, base_z
        for name, nid in masters_by_level:
            if nid is None:
                continue
            nd = self.model.nodes[nid]
            u = self.disp[nid][direction]
            h = nd.z - prev_z
            du = u - prev_u
            out.append((name, du, h, du / h if h else np.nan))
            prev_u, prev_z = u, nd.z
        return out

    # -------------------------------------------------- 表示
    def print_displacements(self):
        print(f"\n■ 節点変位  [{self.case}]  (mm, rad)")
        print(f"{'節点':>5} {'ux':>10} {'uy':>10} {'uz':>10} {'rx':>11} {'ry':>11} {'rz':>11}")
        for nid in sorted(self.disp):
            d = self.disp[nid]
            print(f"{nid:>5} {d[0]:10.3f} {d[1]:10.3f} {d[2]:10.3f} "
                  f"{d[3]:11.3e} {d[4]:11.3e} {d[5]:11.3e}")

    def print_reactions(self):
        print(f"\n■ 支点反力  [{self.case}]  (kN, kN·m)")
        print(f"{'節点':>5} {'Rx':>9} {'Ry':>9} {'Rz':>9} {'Mx':>9} {'My':>9} {'Mz':>9}")
        tot = np.zeros(6)
        for nid in sorted(self.reactions):
            r = self.reactions[nid]
            tot += r
            print(f"{nid:>5} " + " ".join(f"{v:9.1f}" for v in _kn(r)))
        print(f"{'合計':>4} " + " ".join(f"{v:9.1f}" for v in _kn(tot)))

    def print_section_forces(self, at="face", eids=None):
        print(f"\n■ 断面力  [{self.case}]  位置={at}  (kN, kN·m, N:引張+)")
        print(f"{'要素':>5} {'端':>2} {'N':>9} {'Qy':>9} {'Qz':>9} {'Mx':>9} {'My':>9} {'Mz':>9}")
        for eid in (eids or sorted(self.forces)):
            si, sj = self.section_forces(eid, at)
            for tag, s in (("i", si), ("j", sj)):
                print(f"{eid:>5} {tag:>2} " + " ".join(f"{v:9.1f}" for v in _kn(s)))


def _kn(v):
    """[N, N, N, N·mm, N·mm, N·mm] -> [kN, kN, kN, kN·m, kN·m, kN·m]"""
    v = np.asarray(v, float)
    return np.r_[v[:3] / 1e3, v[3:] / 1e6]
