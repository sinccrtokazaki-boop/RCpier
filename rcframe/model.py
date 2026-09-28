"""解析モデル・剛性組立・求解

単位系: 長さ mm, 力 N（出力時に kN, kN·m へ換算可能）
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .element import (local_stiffness, local_stiffness_springs, local_axes, transformation_matrix,
                      rigid_zone_matrix, udl_equivalent_loads, distributed_equivalent_loads)
from .materials import Concrete, RectSection
from .results import Result

DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float
    fix: list = field(default_factory=lambda: [0] * 6)  # 1 = 拘束

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Element:
    id: int
    ni: int
    nj: int
    mat: Concrete
    sec: RectSection
    kind: str = "beam"          # "beam" / "column"
    beta: float = 0.0           # コードアングル [deg]
    rigid_i: float = 0.0        # i 端剛域長 [mm]
    rigid_j: float = 0.0        # j 端剛域長 [mm]
    shear_deform: bool = True
    # 材端の曲げばねの柔性 [面][端]（面 0 = xy 面, 面 1 = xz 面、0 = 剛）。None = ばねなし
    # 非線形解析（pushover.py）が各ステップの接線柔性を設定する
    springs: Optional[np.ndarray] = None

    # ---- 解析時に計算されるキャッシュ ----
    def prepare(self, xi, xj):
        # 形状・剛域・断面・ばねが前回と同じなら再計算しない（非線形解析の各ステップの高速化）
        key = (tuple(xi), tuple(xj), self.beta, self.rigid_i, self.rigid_j, id(self.sec), id(self.mat),
               self.shear_deform, None if self.springs is None else np.asarray(self.springs).tobytes())
        if getattr(self, "_prep_key", None) == key:
            return
        self._prep_key = key
        self.R, self.L = local_axes(xi, xj, self.beta)
        self.Lf = self.L - self.rigid_i - self.rigid_j
        if self.Lf <= 0:
            raise ValueError(f"要素 {self.id}: 剛域長が部材長以上です")
        s, m = self.sec, self.mat
        args = (m.E, m.G, s.A, s.Iy, s.Iz, s.J, self.Lf,
                s.Asy if self.shear_deform else None,
                s.Asz if self.shear_deform else None)
        if self.springs is None:
            self.k_flex = local_stiffness(*args)
        else:
            self.k_flex = local_stiffness_springs(*args, fs=self.springs)
        self.T = transformation_matrix(self.R)
        self.H = rigid_zone_matrix(self.rigid_i, self.rigid_j)
        k_node = self.H.T @ self.k_flex @ self.H
        self.k_global = self.T.T @ k_node @ self.T


class Model:
    def __init__(self, name="model"):
        self.name = name
        self.nodes: dict[int, Node] = {}
        self.elements: dict[int, Element] = {}
        self.nodal_loads = defaultdict(list)    # case -> [(node, F6)]
        self.member_loads = defaultdict(list)   # case -> [(elem, w_global, profile)]
        self.diaphragms = []                    # [(master, [slaves])]
        self.node_springs = {}                  # nid -> 6 成分の対角ばね剛性（全体座標、地盤ばね等）

    # ------------------------------------------------------------ 入力
    def add_node_spring(self, nid, k):
        """節点ばね k = (kx, ky, kz, krx, kry, krz) [N/mm, N·mm/rad]（全体座標、支点としての拘束なし）
        同じ節点に複数回与えると加算する"""
        k = np.asarray(k, float)
        self.node_springs[nid] = self.node_springs.get(nid, np.zeros(6)) + k
    def add_node(self, nid, x, y, z, fix=None):
        self.nodes[nid] = Node(nid, x, y, z, list(fix) if fix else [0] * 6)
        return self.nodes[nid]

    def fix(self, nid, dofs=(1, 1, 1, 1, 1, 1)):
        self.nodes[nid].fix = list(dofs)

    def add_element(self, eid, ni, nj, mat, sec, kind="beam", **kw):
        self.elements[eid] = Element(eid, ni, nj, mat, sec, kind, **kw)
        return self.elements[eid]

    def add_nodal_load(self, case, nid, F):
        """節点荷重 F = (Fx, Fy, Fz, Mx, My, Mz) [N, N·mm] 全体座標"""
        self.nodal_loads[case].append((nid, np.asarray(F, float)))

    def add_member_load(self, case, eid, w_global, profile=None):
        """部材分布荷重 w = (wx, wy, wz) [N/mm] 全体座標・部材長あたり

        profile=None なら全長に等分布。台形・三角形などは
        profile=[(x, f), ...]（x: i節点からの距離 [mm] 昇順, f: 係数）で
        区分線形の分布 w × f(x) を与える（範囲外は 0）
        """
        if profile is not None:
            profile = [(float(x), float(f)) for x, f in profile]
            if any(b[0] < a[0] for a, b in zip(profile, profile[1:])):
                raise ValueError(f"要素 {eid}: profile の x は昇順で指定してください")
        self.member_loads[case].append((eid, np.asarray(w_global, float), profile))

    def add_diaphragm(self, master, slaves):
        """剛床仮定: slaves の (ux, uy, rz) を master に従属させる
        master の (uz, rx, ry) は自動的に拘束される"""
        self.diaphragms.append((master, list(slaves)))

    # ------------------------------------------------------------ 剛域自動設定
    def auto_rigid_zones(self):
        """RC規準の慣用的な剛域設定
            梁端: 柱フェイス位置 − 梁せい/4
            柱端: 梁フェイス位置 − 柱せい/4
        （3次元では柱は各方向の梁ごとに求めた値の最小値を採用する簡略化）
        """
        conn = defaultdict(list)
        for e in self.elements.values():
            conn[e.ni].append(e)
            conn[e.nj].append(e)

        def axes(e):
            R, _ = local_axes(self.nodes[e.ni].xyz, self.nodes[e.nj].xyz, e.beta)
            return R

        for e in self.elements.values():
            R = axes(e)
            for end, nid in (("i", e.ni), ("j", e.nj)):
                others = [o for o in conn[nid] if o is not e and o.kind != e.kind]
                vals = []
                for o in others:
                    Ro = axes(o)
                    if e.kind == "beam":
                        # 柱の、梁軸方向の幅の半分
                        d = R[0]
                        half = 0.5 * (abs(d @ Ro[1]) * o.sec.b + abs(d @ Ro[2]) * o.sec.D)
                        vals.append(half - e.sec.D / 4)
                    else:
                        # 梁せいの半分 − 柱の梁軸方向の幅/4
                        d = Ro[0]
                        col_w = abs(d @ R[1]) * e.sec.b + abs(d @ R[2]) * e.sec.D
                        vals.append(o.sec.D / 2 - col_w / 4)
                if vals:
                    a = max(0.0, max(vals) if e.kind == "beam" else min(vals))
                    setattr(e, f"rigid_{end}", a)

    # ------------------------------------------------------------ 解析
    def _index(self):
        ids = sorted(self.nodes)
        return {nid: k for k, nid in enumerate(ids)}, 6 * len(ids)

    def assemble_K(self):
        idx, n = self._index()
        elems = list(self.elements.values())
        for e in elems:
            e.prepare(self.nodes[e.ni].xyz, self.nodes[e.nj].xyz)
        # 自由度番号の配列は、節点・要素・節点ばねの構成が同じなら使い回す（非線形解析の高速化）
        key = (tuple(idx), tuple((e.id, e.ni, e.nj) for e in elems), tuple(self.node_springs))
        if getattr(self, "_asm_key", None) != key:
            rows, cols = [], []
            for e in elems:
                d = np.r_[6 * idx[e.ni] + np.arange(6), 6 * idx[e.nj] + np.arange(6)]
                rows.append(np.repeat(d, 12)); cols.append(np.tile(d, 12))
            for nid in self.node_springs:
                d = 6 * idx[nid] + np.arange(6)
                rows.append(d); cols.append(d)
            self._asm_rc = (np.concatenate(rows), np.concatenate(cols))
            self._asm_key = key
        vals = [e.k_global.ravel() for e in elems] + [np.asarray(k, float) for k in self.node_springs.values()]
        K = sp.coo_matrix((np.concatenate(vals), self._asm_rc), shape=(n, n)).tocsr()
        return K, idx, n

    def _element_loads(self, case):
        """要素ごとの等価節点荷重（局所）を返す: eid -> (p_flex, p_rigid)"""
        out = {}
        for eid, w, profile in self.member_loads.get(case, []):
            e = self.elements[eid]
            wl = e.R @ w
            if profile is None and e.springs is None:
                pf, pr = udl_equivalent_loads(wl, e.Lf, e.rigid_i, e.rigid_j)
            else:
                if profile is None:
                    profile = [(0.0, 1.0), (e.L, 1.0)]
                if profile[0][0] < -1e-6 or profile[-1][0] > e.L + 1e-6:
                    raise ValueError(f"要素 {eid}: profile の範囲が部材長 {e.L:.1f} mm を超えています")
                s, m = e.sec, e.mat
                pf, pr = distributed_equivalent_loads(
                    wl, profile, e.L, e.rigid_i, e.rigid_j, m.E * s.Iy, m.E * s.Iz,
                    m.G * s.Asy if e.shear_deform else None,
                    m.G * s.Asz if e.shear_deform else None, fs=e.springs)
            if eid in out:
                out[eid] = (out[eid][0] + pf, out[eid][1] + pr)
            else:
                out[eid] = (pf, pr)
        return out

    def _constraint_matrix(self, idx, n):
        """拘束・剛床条件を表す変換 u_full = C @ u_free"""
        fixed = np.zeros(n, bool)
        for nid, nd in self.nodes.items():
            for k in range(6):
                if nd.fix[k]:
                    fixed[6 * idx[nid] + k] = True
        dep = {}  # 従属自由度 -> [(独立自由度, 係数)]
        for m, slaves in self.diaphragms:
            mm = self.nodes[m]
            for k in (2, 3, 4):
                fixed[6 * idx[m] + k] = True
            mux, muy, mrz = (6 * idx[m] + k for k in (0, 1, 5))
            for s in slaves:
                ns = self.nodes[s]
                dx, dy = ns.x - mm.x, ns.y - mm.y
                b = 6 * idx[s]
                dep[b + 0] = [(mux, 1.0), (mrz, -dy)]
                dep[b + 1] = [(muy, 1.0), (mrz, dx)]
                dep[b + 5] = [(mrz, 1.0)]
        free = [i for i in range(n) if not fixed[i] and i not in dep]
        col = {g: c for c, g in enumerate(free)}
        C = sp.lil_matrix((n, len(free)))
        for g in free:
            C[g, col[g]] = 1.0
        for g, terms in dep.items():
            for gi, coef in terms:
                if gi in col:
                    C[g, col[gi]] += coef
        return C.tocsr(), fixed

    def analyze(self, case) -> Result:
        K, idx, n = self.assemble_K()
        F = np.zeros(n)
        for nid, f in self.nodal_loads.get(case, []):
            F[6 * idx[nid]:6 * idx[nid] + 6] += f
        eloads = self._element_loads(case)
        for eid, (pf, pr) in eloads.items():
            e = self.elements[eid]
            pg = e.T.T @ (e.H.T @ pf + pr)
            F[6 * idx[e.ni]:6 * idx[e.ni] + 6] += pg[:6]
            F[6 * idx[e.nj]:6 * idx[e.nj] + 6] += pg[6:]

        C, fixed = self._constraint_matrix(idx, n)
        Kr = (C.T @ K @ C).tocsc()
        Fr = C.T @ F
        ur = spla.spsolve(Kr, Fr)
        if not np.all(np.isfinite(ur)):
            raise RuntimeError("剛性マトリクスが特異です（拘束不足・不安定構造の可能性）")
        u = C @ ur

        # 反力（支点のみ）
        R_all = K @ u - F
        reactions = {nid: np.where(np.array(self.nodes[nid].fix, bool),
                                   R_all[6 * k:6 * k + 6], 0.0)
                     for nid, k in idx.items() if any(self.nodes[nid].fix)}

        disp = {nid: u[6 * k:6 * k + 6] for nid, k in idx.items()}

        # 部材端力（局所座標, 部材に作用する力）
        forces = {}
        for eid, e in self.elements.items():
            dg = np.r_[disp[e.ni], disp[e.nj]]
            d_flex = e.H @ (e.T @ dg)
            pf, pr = eloads.get(eid, (np.zeros(12), np.zeros(12)))
            f_face = e.k_flex @ d_flex - pf
            f_node = e.H.T @ f_face - pr
            forces[eid] = {"face": f_face, "node": f_node}

        res = Result(self, case, disp, forces, reactions)
        # 節点ばねの反力（ばねが節点に及ぼす力 = −k·u）
        res.spring_forces = {nid: -np.asarray(k) * disp[nid] for nid, k in self.node_springs.items()}
        return res
