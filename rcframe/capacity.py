"""保有水平耐力の確認（建築基準法施行令第82条の3）: Qu ≧ Qun = Ds·Fes·Qud

各方向（±X, ±Y）のプッシュオーバーから
  Qu : 崩壊機構の形成・層間変形角の上限（既定 1/100）・柱のせん断破壊のうち最初の時点の層せん断力
  Ds : 部材種別 → 部材群（Qu 時の柱のせん断力の割合）→ Ds（RC 純ラーメン）
  Fes: 一次設計の線形解析（Ex, Ey）による剛性率・偏心率から Fs·Fe
  Qud: C0 = 1.0 の地震層せん断力（= Ai 分布の Qi / C0）
計算式は ultimate.py、非線形解析は pushover.py
"""
import math
from dataclasses import dataclass, field

from .pushover import PLANE_NAME, build_hinges, run_pushover
from .results import Result
from .ultimate import (DS_RC_FRAME, beam_class, column_class, eccentricity, fe_factor, fs_factor,
                       member_group, qsu_arakawa, stiffness_ratios, worse)


@dataclass
class MemberJudge:
    eid: int
    label: str
    kind: str            # "柱" / "梁"
    direction: str       # "X" / "Y"（梁は部材軸の方向）
    Qmu: float           # 曲げ降伏時のせん断力（梁は長期せん断力を含む）
    Qsu: float
    shear_failure: bool  # Qsu < 係数·Qmu（曲げ降伏より先にせん断破壊）
    h0_D: float = math.nan
    s0_Fc: float = math.nan
    pt: float = math.nan
    tau_Fc: float = math.nan
    cls: str = ""        # 部材種別

    @property
    def ratio(self):
        return self.Qsu / self.Qmu if self.Qmu else math.inf


@dataclass
class StoryJudge:
    story: str
    direction: str
    Qu: float
    Qud: float
    group: str
    gamma: dict
    Ds: float
    Rs: float
    Fs: float
    Re: float
    Fe: float
    reason: str          # Qu の決定理由
    step: int
    drift: float         # Qu 時の最大層間変形角
    mode: str            # 崩壊形

    @property
    def Fes(self):
        return self.Fs * self.Fe

    @property
    def Qun(self):
        return self.Ds * self.Fes * self.Qud

    @property
    def ratio(self):
        return self.Qu / self.Qun

    @property
    def ok(self):
        return self.ratio >= 1.0


@dataclass
class UltimateReport:
    stories: list = field(default_factory=list)       # StoryJudge
    members: dict = field(default_factory=dict)       # 方向 -> [MemberJudge]（部材種別は方向ごと）
    ecc: list = field(default_factory=list)           # 層ごとの剛性率・偏心率の計算表
    pushovers: dict = field(default_factory=dict)     # 方向 -> PushoverResult
    effective: dict = field(default_factory=dict)     # 方向 -> {柱 eid: 接続する梁を考慮した種別}
    settings: dict = field(default_factory=dict)

    @property
    def ok(self):
        return all(s.ok for s in self.stories)

    def print_summary(self):
        s = self.settings
        print(f"\n■ 保有水平耐力（令82条の3）  Qsu: 荒川 {s['qsu']} 式、曲げ先行の条件 Qsu ≧ "
              f"{s['qsu_factor']:g}·Qmu、Qu 判定の層間変形角 1/{1 / s['drift_limit']:.0f}")
        print(f"{'方向':<3} {'層':<3} {'Qu[kN]':>9} {'Qud[kN]':>9} {'群':>2} {'Ds':>5} {'Fs':>5} "
              f"{'Fe':>5} {'Fes':>5} {'Qun[kN]':>9} {'Qu/Qun':>7} 判定  Qu の決定（step, 層間変形角）/ 崩壊形")
        for r in self.stories:
            print(f"{r.direction:<4} {r.story:<3} {r.Qu / 1e3:9.1f} {r.Qud / 1e3:9.1f} {r.group:>2} "
                  f"{r.Ds:5.2f} {r.Fs:5.2f} {r.Fe:5.2f} {r.Fes:5.2f} {r.Qun / 1e3:9.1f} "
                  f"{r.ratio:7.2f} {'OK' if r.ok else '**NG**':<4} "
                  f"{r.reason}（{r.step}, 1/{1 / r.drift:,.0f}）/ {r.mode}")

    def print_members(self, direction=None):
        for dname, rows in self.members.items():
            if direction and dname != direction:
                continue
            eff = self.effective.get(dname, {})
            print(f"\n■ 部材の破壊形式と種別（{dname} 方向の加力）")
            print(f"{'部材':<18} {'Qmu[kN]':>8} {'Qsu[kN]':>8} {'Qsu/Qmu':>7} {'破壊形式':<6} "
                  f"{'h0/D':>5} {'σ0/Fc':>6} {'pt[%]':>6} {'τu/Fc':>6} 種別")
            for m in rows:
                mode = "せん断" if m.shear_failure else "曲げ"
                extra = f" → {eff[m.eid]}（梁を考慮）" if m.eid in eff and eff[m.eid] != m.cls else ""
                fmt = lambda v, f: format(v, f) if not math.isnan(v) else "-"
                print(f"{m.label:<18} {m.Qmu / 1e3:8.1f} {m.Qsu / 1e3:8.1f} {m.ratio:7.2f} {mode:<6} "
                      f"{fmt(m.h0_D, '5.2f'):>5} {fmt(m.s0_Fc, '6.3f'):>6} "
                      f"{fmt(m.pt * 100, '6.2f') if not math.isnan(m.pt) else '-':>6} "
                      f"{fmt(m.tau_Fc, '6.3f'):>6} {m.cls}{extra}")

    def print_eccentricity(self):
        print("\n■ 剛性率・偏心率（一次設計の線形解析 Ex, Ey）")
        print(f"{'層':<3} {'方向':<3} {'rs=h/δ':>8} {'Rs':>5} {'Fs':>5} {'重心':>16} {'剛心':>16} "
              f"{'弾力半径':>8} {'偏心距離':>8} {'Re':>6} {'Fe':>5}")
        for r in self.ecc:
            print(f"{r['story']:<3} {r['dir']:<4} {r['rs']:8.0f} {r['Rs']:5.2f} {r['Fs']:5.2f} "
                  f"({r['xg']:6.0f},{r['yg']:6.0f}) ({r['xk']:6.0f},{r['yk']:6.0f}) "
                  f"{r['re']:8.0f} {r['e']:8.0f} {r['Re']:6.3f} {r['Fe']:5.2f}")


def _story_drift(r, lower, upper, h, d):
    lo = r.disp[lower][d] if lower is not None else 0.0
    return (r.disp[upper][d] - lo) / h


def _levels(bld):
    """[(層名, 下の床のマスター or None, 上の床のマスター, 階高, 上の床の番号)]"""
    levels = []
    prev, z_prev = None, bld.floors[0][1]
    for f, (_, z) in enumerate(bld.floors[1:], 1):
        levels.append((bld.floors[f - 1][0], prev, bld.masters[f], z - z_prev, f))
        prev, z_prev = bld.masters[f], z
    return levels


def stiffness_eccentricity(bld, lin):
    """一次設計の線形解析結果 lin = {"Ex": Result, "Ey": Result} から剛性率・偏心率を求める

    各柱の水平剛性 D = 柱のせん断力 / 層間変位（X: Ex の Qy、Y: Ey の Qz）
    戻り値: (表の行のリスト, {(層名, "X"/"Y"): (Rs, Fs, Re, Fe)})
    """
    m = bld.model
    levels = _levels(bld)
    rows, fes = [], {}
    for dname, case, d in (("X", "Ex", 0), ("Y", "Ey", 1)):
        if case not in lin:
            continue
        drift = [abs(_story_drift(lin[case], lo, up, h, d)) for _, lo, up, h, _ in levels]
        Rs = stiffness_ratios([1 / x for x in drift])
        for k, (name, lo, up, h, f) in enumerate(levels):
            cols = []
            for (t, cf, i, j), eid in bld.elem_ids.items():
                if t != "c" or cf != k:
                    continue
                nd = m.nodes[m.elements[eid].ni]
                Dx = abs(lin["Ex"].forces[eid]["face"][1]) / abs(
                    _story_drift(lin["Ex"], lo, up, h, 0) * h) if "Ex" in lin else 0.0
                Dy = abs(lin["Ey"].forces[eid]["face"][2]) / abs(
                    _story_drift(lin["Ey"], lo, up, h, 1) * h) if "Ey" in lin else 0.0
                cols.append((nd.x, nd.y, Dx, Dy))
            _, xg, yg = bld.weights[f]
            ec = eccentricity(cols, xg, yg)
            Re = ec["Rex"] if dname == "X" else ec["Rey"]
            fes[(name, dname)] = (Rs[k], fs_factor(Rs[k]), Re, fe_factor(Re))
            rows.append(dict(story=name, dir=dname, rs=1 / drift[k], drift=drift[k], h=h,
                             Rs=Rs[k], Fs=fs_factor(Rs[k]),
                             xg=xg, yg=yg, xk=ec["xk"], yk=ec["yk"],
                             re=ec["rex"] if dname == "X" else ec["rey"],
                             e=ec["ey"] if dname == "X" else ec["ex"], Re=Re, Fe=fe_factor(Re)))
    return rows, fes


def run_ultimate(bld, directions=("+x", "-x", "+y", "-y")) -> UltimateReport:
    """保有水平耐力の確認（Qu, Ds, Fes, Qun）"""
    m = bld.model
    uo = bld.ultimate
    if bld.seismic is None:
        raise ValueError("地震力（seismic）が設定されていません")
    cases = set(m.nodal_loads) | set(m.member_loads)
    long = bld.design["long"] or [c for c in ("G", "P") if c in cases]
    lin = {c: m.analyze(c) for c in set(long) | {"Ex", "Ey"} if c in cases}
    rL = Result.combine("+".join(long), {lin[c]: 1.0 for c in long})
    hinges, _, _ = build_hinges(bld, rL, bld.nonlinear)     # My（骨格曲線）を Qmu に用いる
    rows = bld.seismic["rows"]
    C0 = bld.seismic["C0"]
    rep = UltimateReport(settings=dict(uo))

    # ---- 部材の Qmu, Qsu（方向ごと）
    mem = {"X": [], "Y": []}
    shear_caps, shear_labels = {}, {}
    col_info = {}                                     # (eid, 面) -> 分類用の値
    for key, eid in sorted(bld.elem_ids.items(), key=lambda kv: kv[1]):
        e = m.elements[eid]
        rb = e.sec.rebar
        if rb is None:
            continue
        s, mat, Lf = e.sec, e.mat, e.Lf
        label = bld.label("elem", eid)
        f_face = rL.forces[eid]["face"]
        if key[0] in ("x", "y"):
            d = s.D - rb.dt
            My_b = hinges[(eid, 1, 0)].sk[1].My         # 下端引張
            My_t = hinges[(eid, 1, 0)].sk[-1].My        # 上端引張
            Q0 = max(abs(f_face[2]), abs(f_face[8]))    # 長期荷重によるせん断力
            Qmu = Q0 + (My_b + My_t) / Lf
            pw = rb.end.stirrup.pw(s.b)
            Qsu = min(qsu_arakawa(s.b, d, bars.area / (s.b * d), Lf / (2 * d), pw, rb.shear.F, 0.0,
                                  mat.Fc, uo["qsu"]) for bars in (rb.end.top, rb.end.bottom))
            jm = MemberJudge(eid, label, "梁", key[0].upper(), Qmu, Qsu,
                             Qsu < uo["qsu_factor"] * Qmu, tau_Fc=Qmu / (s.b * s.D) / mat.Fc)
            mem[key[0].upper()].append(jm)
            shear_caps[(eid, 1)] = Qsu
            shear_labels[(eid, 1)] = label
        else:
            N = max(f_face[0], -f_face[6])              # 圧縮正（大きい方）
            Nm = 0.5 * (f_face[0] - f_face[6])
            for plane, w, dep, bars, hoop in ((0, s.D, s.b, rb.x, rb.hoop_x),
                                              (1, s.b, s.D, rb.y, rb.hoop_y)):
                d = dep - rb.dt
                Qmu = (hinges[(eid, plane, 0)].sk[1].My + hinges[(eid, plane, 1)].sk[1].My) / Lf
                Qsu = qsu_arakawa(w, d, bars.area / (w * d), Lf / (2 * d), hoop.pw(w), rb.shear.F,
                                  Nm / (s.b * s.D), mat.Fc, uo["qsu"])
                jm = MemberJudge(eid, f"{label} {PLANE_NAME[plane]}", "柱", PLANE_NAME[plane], Qmu,
                                 Qsu, Qsu < uo["qsu_factor"] * Qmu, h0_D=Lf / dep,
                                 s0_Fc=N / (s.b * s.D) / mat.Fc, pt=bars.area / (w * d),
                                 tau_Fc=Qmu / (s.b * s.D) / mat.Fc)
                mem[PLANE_NAME[plane]].append(jm)
                shear_caps[(eid, plane)] = Qsu
                shear_labels[(eid, plane)] = f"{label} {PLANE_NAME[plane]}"
                col_info[(eid, plane)] = jm

    # ---- 剛性率・偏心率（一次設計）
    rep.ecc, fes = stiffness_eccentricity(bld, lin)
    levels = _levels(bld)

    # ---- 方向ごとのプッシュオーバーと判定
    lim = uo["drift_limit"]
    for direction in directions:
        dname = direction.lstrip("+-").upper()
        plane = 0 if dname == "X" else 1
        po = run_pushover(bld, direction, lin, max_drift=max(bld.nonlinear["pushover"]["max_drift"], lim),
                          shear_caps=shear_caps, shear_labels=shear_labels, mechanism_check=True,
                          drift_marks=(lim,), keep_states=True)
        rep.pushovers[direction] = po
        lat = po.lateral
        cand = []
        if po.mechanism_step is not None:
            cand.append((po.mechanism_step, "崩壊機構"))
        s_lim = next((s["step"] for s in lat if max(abs(x) for x in s["drifts"]) >= lim * (1 - 1e-9)), None)
        if s_lim is not None:
            cand.append((s_lim, f"層間変形角 1/{1 / lim:.0f}"))
        if uo["stop_at_column_shear"]:
            s_sh = next((e["step"] for e in po.events if e["event"] == "せん断破壊"
                         and e["key"] in col_info and e["key"][1] == plane), None)
            if s_sh is not None:
                cand.append((s_sh, "柱のせん断破壊"))
        qu_step, reason = min(cand)
        st = next(s for s in lat if s["step"] == qu_step)
        _, face = po.states[qu_step]
        sheared = {e["key"] for e in po.events if e["event"] == "せん断破壊" and e["step"] <= qu_step}

        # 部材種別（解析でせん断破壊した部材は FD）
        cls = {}
        for jm in mem[dname] if dname in mem else []:
            key = (jm.eid, plane if jm.kind == "柱" else 1)
            sf = jm.shear_failure or key in sheared
            jm.cls = (column_class(jm.h0_D, jm.s0_Fc, jm.pt, jm.tau_Fc, sf) if jm.kind == "柱"
                      else beam_class(jm.tau_Fc, sf))
            cls[jm.eid] = jm.cls
        # 柱の種別: 柱頭・柱脚の節点に接続する加力方向の梁の種別のうち低い方も考慮
        eff = {}
        beams_at = {}
        for (t, *_), eid in bld.elem_ids.items():
            if t == dname.lower() and eid in cls:
                e = m.elements[eid]
                for nid in (e.ni, e.nj):
                    beams_at.setdefault(nid, []).append(cls[eid])
        for (t, *_), eid in bld.elem_ids.items():
            if t == "c" and eid in cls:
                e = m.elements[eid]
                eff[eid] = worse(cls[eid], *beams_at.get(e.ni, []), *beams_at.get(e.nj, []))
        rep.effective[dname] = eff
        mode = ("機構未形成" if po.mechanism_step is None or po.mechanism_step > qu_step else
                ("全体崩壊形" if len(po.mechanism_stories) == len(levels)
                 else f"部分崩壊形（{', '.join(po.mechanism_stories)}）"))

        for k, (name, *_rest) in enumerate(levels):
            shares = {}
            for (t, cf, *_), eid in bld.elem_ids.items():
                if t == "c" and cf == k and eid in eff:
                    q = abs(face[eid][1 if plane == 0 else 2])
                    shares[eff[eid]] = shares.get(eff[eid], 0.0) + q
            tot = sum(shares.values())
            gamma = {c: v / tot for c, v in shares.items()} if tot else {}
            group = member_group(gamma) if gamma else "D"
            Ds = uo["ds_override"].get(name, DS_RC_FRAME[group])
            Rs, Fs, Re, Fe = fes.get((name, dname), (1.0, 1.0, 0.0, 1.0))
            rep.stories.append(StoryJudge(
                name, direction, st["lam"] * rows[k]["Q"], rows[k]["Q"] / C0, group, gamma, Ds,
                Rs, Fs, Re, Fe, reason, qu_step, max(abs(x) for x in st["drifts"]), mode))
    rep.members = mem
    return rep
