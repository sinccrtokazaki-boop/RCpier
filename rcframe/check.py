"""断面検定: 荷重組合せ（建築基準法施行令第82条）と RC規準による許容応力度検定

長期: G + P
短期: G + P ± E（地震の各荷重ケース、偶発偏心ケースを含む）
      せん断は設計用せん断力 QD = QL + n·QE（RC規準 15条、n = design.shear_factor）
検定位置: 梁は両端フェイス・中央、柱は柱脚・柱頭のフェイス。柱は X, Y 方向を独立に検定
"""
import math
from dataclasses import dataclass

from .design import (PW_MIN, allowable_moment, alpha_factor, beam_shear_long, beam_shear_short,
                     column_shear_long, column_shear_short, concrete_fc, concrete_fs)
from .materials import BeamRebar, ColumnRebar
from .results import Result


@dataclass
class CheckRow:
    eid: int
    label: str
    pos: str          # 梁 "i端" / "中央" / "j端"、柱 "柱脚" / "柱頭"
    direction: str    # 梁 ""、柱 "X" / "Y"
    item: str         # "曲げ" / "せん断"
    term: str         # "長期" / "短期"
    combo: str        # 例 "G+P", "G+P+Ex", "G+P-Ex"
    N: float          # 軸力（圧縮正）[N]
    demand: float     # |M| [N·mm] または QD [N]
    capacity: float   # Ma [N·mm] または QA [N]
    note: str = ""    # 梁の曲げ: "上端引張" / "下端引張"

    @property
    def ratio(self):
        if self.capacity <= 0:
            return math.inf if self.demand > 0 else 0.0
        return self.demand / self.capacity

    @property
    def ok(self):
        return self.ratio <= 1.0


class CheckReport:
    def __init__(self, rows, warnings, long, seismic, shear_factor):
        self.rows = rows
        self.warnings = warnings
        self.long, self.seismic, self.shear_factor = long, seismic, shear_factor

    def governing(self, keys=("eid", "item")):
        """keys ごとの検定比最大の行"""
        out = {}
        for r in self.rows:
            k = tuple(getattr(r, a) for a in keys)
            if k not in out or r.ratio > out[k].ratio:
                out[k] = r
        return out

    @property
    def ng(self):
        return [r for r in self.governing(("eid", "pos", "direction", "item")).values() if not r.ok]

    @property
    def max_ratio(self):
        return max((r.ratio for r in self.rows), default=0.0)

    # -------------------------------------------------- 表示
    def _header(self):
        L = "+".join(self.long)
        seis = ", ".join(f"{L}±{c}" for c in self.seismic) or "なし"
        print(f"\n■ 断面検定（RC規準 2018）  長期: {L}   短期: {seis}")
        print(f"  短期せん断: QD = QL + {self.shear_factor:g}·QE")
        print(f"{'要素':>4} {'部材':<14} {'位置':<4} {'方向':<2} {'検定':<4} {'組合せ':<12} "
              f"{'N[kN]':>8} {'応力':>9} {'許容':>9} {'検定比':>6}  判定")

    @staticmethod
    def _line(r):
        k = 1e6 if r.item == "曲げ" else 1e3
        unit = "kN·m" if r.item == "曲げ" else "kN"
        judge = "OK" if r.ok else "**NG**"
        note = f"  {r.note}" if r.note else ""
        print(f"{r.eid:>4} {r.label:<14} {r.pos:<4} {r.direction or '-':<2} {r.item:<4} "
              f"{r.combo:<12} {r.N / 1e3:8.1f} {r.demand / k:9.1f} {r.capacity / k:9.1f} "
              f"{r.ratio:6.2f}  {judge} [{unit}]{note}")

    def print_summary(self):
        """部材ごとに曲げ・せん断の最大検定比の行を表示"""
        self._header()
        for _, r in sorted(self.governing().items(), key=lambda kv: (kv[0][0], kv[0][1] != "曲げ")):
            self._line(r)
        self._footer()

    def print_detail(self, eids=None):
        """位置・方向ごとに曲げ・せん断の最大検定比の行を表示"""
        self._header()
        g = self.governing(("eid", "pos", "direction", "item"))
        for _, r in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] != "曲げ")):
            if eids is None or r.eid in eids:
                self._line(r)
        self._footer()

    def _footer(self):
        ng = self.ng
        print(f"  最大検定比 {self.max_ratio:.2f}、NG {len(ng)} 箇所")
        for w in self.warnings:
            print(f"  [構造規定] {w}")


def run_checks(bld, results=None) -> CheckReport:
    """Building の全部材（配筋のある断面）を検定する

    results: {荷重ケース: Result}（省略時・不足分はここで解析する）
    """
    m = bld.model
    cases = set(m.nodal_loads) | set(m.member_loads)
    opt = bld.design
    long = opt["long"] or [c for c in ("G", "P") if c in cases]
    seis = opt["seismic"] or sorted(c for c in cases if c.startswith("E"))
    if not long:
        raise ValueError("長期荷重のケース（G, P）がありません（design.long で指定できます）")
    unknown = [c for c in long + seis if c not in cases]
    if unknown:
        raise ValueError(f"design: 荷重ケース {unknown} がありません")
    res = dict(results or {})
    for c in long + seis:
        if c not in res:
            res[c] = m.analyze(c)
    Lname = "+".join(long)
    rL = Result.combine(Lname, {res[c]: 1.0 for c in long})
    rE = {c: res[c] for c in seis}
    ctx = dict(rL=rL, rE=rE, Lname=Lname, nsh=opt["shear_factor"], nr=opt["young_ratio"])

    rows, warns = [], []
    for key, eid in sorted(bld.elem_ids.items(), key=lambda kv: kv[1]):
        e = m.elements[eid]
        rb = e.sec.rebar
        if rb is None:
            continue
        label = bld.label("elem", eid)
        if key[0] == "c":
            if not isinstance(rb, ColumnRebar):
                raise ValueError(f"{label}: 断面 {e.sec.name} の配筋は梁用です")
            _check_column(e, label, rb, ctx, rows, warns)
        else:
            if not isinstance(rb, BeamRebar):
                raise ValueError(f"{label}: 断面 {e.sec.name} の配筋は柱用です")
            _check_beam(e, label, rb, ctx, rows, warns)
    return CheckReport(rows, warns, long, seis, opt["shear_factor"])


def _combos(ctx, eid, x):
    """[(長期/短期, 組合せ名, term, 長期の断面力, 地震の断面力×符号 or None)]"""
    sL = ctx["rL"].section_forces_at(eid, x)
    out = [("長期", ctx["Lname"], "long", sL, None)]
    for c, r in ctx["rE"].items():
        sE = r.section_forces_at(eid, x)
        for sg, k in (("+", 1.0), ("-", -1.0)):
            out.append(("短期", f"{ctx['Lname']}{sg}{c}", "short", sL, k * sE))
    return out


def _check_beam(e, label, rb, ctx, rows, warns):
    b, D, Fc = e.sec.b, e.sec.D, e.mat.Fc
    d = D - rb.dt
    for name, bars in (("端部", rb.end), ("中央", rb.center)):
        pw = bars.stirrup.pw(b)
        if pw < PW_MIN:
            warns.append(f"{label} {name}: あばら筋比 pw = {pw:.2%} < 0.2%（{bars.stirrup}）")
    for pos, x, bars in (("i端", e.rigid_i, rb.end), ("中央", e.L / 2, rb.center),
                         ("j端", e.L - e.rigid_j, rb.end)):
        pw = bars.stirrup.pw(b)
        for tj, combo, term, sL, sE in _combos(ctx, e.id, x):
            s = sL if sE is None else sL + sE
            M = -s[4]                                  # 下端引張を正
            if M >= 0:
                ac, at, note = bars.top, bars.bottom, "下端引張"
            else:
                ac, at, note = bars.bottom, bars.top, "上端引張"
            Ma = allowable_moment(b, D, ac.area, at.area, rb.dt, 0.0, concrete_fc(Fc, term),
                                  rb.main.ft(ac.dia, term), rb.main.ft(at.dia, term), ctx["nr"])
            rows.append(CheckRow(e.id, label, pos, "", "曲げ", tj, combo, 0.0, abs(M), Ma, note))

            QD = abs(sL[2] if sE is None else sL[2] + ctx["nsh"] * sE[2])
            alpha = alpha_factor(M, QD, d)
            fs = concrete_fs(Fc, term)
            QA = (beam_shear_long(b, d, fs, alpha) if term == "long"
                  else beam_shear_short(b, d, fs, alpha, rb.shear.wft(term), pw))
            rows.append(CheckRow(e.id, label, pos, "", "せん断", tj, combo, 0.0, QD, QA))


def _check_column(e, label, rb, ctx, rows, warns):
    b, D, Fc = e.sec.b, e.sec.D, e.mat.Fc
    pg = rb.total_area / (b * D)
    if pg < 0.008:
        warns.append(f"{label}: 主筋比 pg = {pg:.2%} < 0.8%")
    # 柱の局所 y = 全体 X（幅 b）、局所 z = 全体 Y（せい D）
    #   X方向: 曲げ Mz, せん断 Qy, 曲げ方向の寸法 b, 幅 D
    #   Y方向: 曲げ My, せん断 Qz, 曲げ方向の寸法 D, 幅 b
    dirs = (("X", 5, 1, b, D, rb.x, rb.hoop_x), ("Y", 4, 2, D, b, rb.y, rb.hoop_y))
    for dn, _, _, _, width, _, hoop in dirs:
        pw = hoop.pw(width)
        if pw < PW_MIN:
            warns.append(f"{label} {dn}方向: 帯筋比 pw = {pw:.2%} < 0.2%（{hoop}）")
    for pos, x in (("柱脚", e.rigid_i), ("柱頭", e.L - e.rigid_j)):
        for tj, combo, term, sL, sE in _combos(ctx, e.id, x):
            s = sL if sE is None else sL + sE
            N = -s[0]
            for dn, im, iq, depth, width, bars, hoop in dirs:
                d = depth - rb.dt
                M = abs(s[im])
                ft = rb.main.ft(bars.dia, term)
                Ma = allowable_moment(width, depth, bars.area, bars.area, rb.dt, N,
                                      concrete_fc(Fc, term), ft, ft, ctx["nr"])
                rows.append(CheckRow(e.id, label, pos, dn, "曲げ", tj, combo, N, M, Ma))

                QD = abs(sL[iq] if sE is None else sL[iq] + ctx["nsh"] * sE[iq])
                fs = concrete_fs(Fc, term)
                QA = (column_shear_long(width, d, fs, alpha_factor(M, QD, d)) if term == "long"
                      else column_shear_short(width, d, fs, rb.shear.wft(term), hoop.pw(width)))
                rows.append(CheckRow(e.id, label, pos, dn, "せん断", tj, combo, N, QD, QA))
