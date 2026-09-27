"""解析結果の保持・整形・荷重組合せ"""
import numpy as np


class Result:
    def __init__(self, model, case, disp, forces, reactions):
        self.model = model
        self.case = case
        self.disp = disp            # nid -> 6
        self.forces = forces        # eid -> {"face": 12, "node": 12}（部材に作用する局所端力）
        self.reactions = reactions  # nid -> 6

    # -------------------------------------------------- 断面力（設計用符号）
    def section_forces(self, eid, at="face"):
        """断面力 (N, Qy, Qz, Mx, My, Mz) を i端・j端で返す
        符号: N は引張正。材端力 f に対し i端 = -f_i, j端 = +f_j
        at: "face"（剛域端＝フェイス位置）/ "node"（節点位置）
        """
        f = self.forces[eid][at]
        return -f[:6], f[6:]

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
        return Result(r0.model, name, disp, forces, reac)

    # -------------------------------------------------- 層間変形角
    def story_drifts(self, masters_by_level, direction=0):
        """masters_by_level: 下階から順に [(階名, 床マスター節点 or None=基礎)]
        direction: 0=X, 1=Y
        戻り値: [(階名, 層間変位[mm], 階高[mm], 層間変形角)]
        """
        out = []
        prev_u, prev_z = 0.0, None
        for name, nid in masters_by_level:
            if nid is None:
                continue
            nd = self.model.nodes[nid]
            u = self.disp[nid][direction]
            if prev_z is None:
                prev_z = 0.0
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
