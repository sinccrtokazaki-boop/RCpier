"""桟橋の照査（桟橋編 P3）

    上部工の RC 梁: コンクリート標準示方書（2022年版）
        断面破壊（曲げ）γi·Md/Mud、断面破壊（せん断）γi·Vd/Vyd と γi·Vd/Vwcd、使用性（ひび割れ幅）w/wa
    鋼管杭（腐食後の断面）: γi·(N/(A·fyd) + M/(Z·fyd))、せん断 γi·τ/(fyd/√3)
    杭の軸方向支持力: γi·N/Rcd（押込み）、γi·N/Rtd（引抜き）（Rcd, Rtd は入力値）

式の形・係数のうちプログラム作成者が確定できないものは VERIFY_ITEMS に列挙する。
組合せの荷重係数は Pier.combinations（入力）で考慮済みとし、ここでは構造物係数 γi を乗じる。
構造解析係数 γa は 1.0 とする。単位: mm, N
"""
import math
from dataclasses import dataclass, field

from .check import CheckRow

ES = 2.0e5

VERIFY_ITEMS = [
    "示方書（2022年版）の等価応力ブロック k1 = 0.85, β = 0.8, ε'cu = 0.0035（f'ck ≦ 50 の値）",
    "示方書（2022年版）の Vcd: fvcd = 0.20·f'cd^(1/3)（≦ 0.72）、βd = (1000/d)^(1/4)（≦ 1.5）、"
    "βp = (100pw)^(1/3)（≦ 1.5）、Vsd の z = d/1.15、Vwcd = 1.25√f'cd（≦ 7.8）·bw·d/γb",
    "示方書（2022年版）のひび割れ幅の式 w = 1.1·k1·k2·k3·{4c + 0.7(cs − φ)}·(σse/Es + ε'csd) と "
    "k2 = 15/(f'c + 20) + 0.7（f'c = f'ck とした）、k3 = 5(n + 2)/(7n + 8)、ε'csd の値",
    "港湾の腐食環境でのひび割れ幅の限界値 wa（既定 0.0035c）",
    "示方書の Ec の表（f'ck との対応）",
    "港湾基準（2018）の鋼管杭の照査式の形（線形の相関式とした、座屈は考慮しない）と材料係数 γm",
    "梁端の検定断面を杭の前面（節点から杭の半径）とした扱い",
    "鋼管杭のせん断形状係数 2（有効断面積 A/2）",
]


# ============================================================ RC（示方書）
def jsce_mu(b, D, as_c, as_t, dc, dt, fcd, fyd_c, fyd_t, Es=ES, k1=0.85, beta=0.8, ecu=0.0035):
    """曲げ耐力 Mu（等価応力ブロック、ひずみの適合条件、軸力 0）[N·mm]

    圧縮縁から: 圧縮鉄筋 as_c は dc、引張鉄筋 as_t は d = D − dt の位置
    中立軸 x を二分法で求める: k1·f'cd·b·β·x + σ's·A's − σs·As = 0
    鉄筋の応力は Es·ε（±fyd で頭打ち）。圧縮鉄筋位置のコンクリートの欠損は考慮しない
    """
    d = D - dt

    def forces(x):
        a = min(beta * x, D)
        Cc = k1 * fcd * b * a
        sc = max(-fyd_c, min(fyd_c, Es * ecu * (x - dc) / x))
        st = max(-fyd_t, min(fyd_t, Es * ecu * (d - x) / x))
        return Cc + as_c * sc - as_t * st, Cc, a, sc, st

    lo, hi = 1e-6 * D, 2.0 * D / beta
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if forces(mid)[0] < 0:
            lo = mid
        else:
            hi = mid
    _, Cc, a, sc, st = forces(0.5 * (lo + hi))
    return Cc * (d - a / 2) + as_c * sc * (d - dc)


def jsce_vcd(bw, d, pw, fcd, gamma_b):
    """せん断耐力（コンクリート分） Vcd = βd·βp·βn·fvcd·bw·d / γb（βn = 1）"""
    fvcd = min(0.20 * fcd ** (1 / 3), 0.72)
    bd = min((1000.0 / d) ** 0.25, 1.5)
    bp = min((100.0 * pw) ** (1 / 3), 1.5)
    return bd * bp * fvcd * bw * d / gamma_b


def jsce_vsd(aw, fwyd, s, d, gamma_b):
    """せん断耐力（鉛直スターラップ分） Vsd = Aw·fwyd·z / (s·γb)、z = d/1.15"""
    return aw * fwyd * (d / 1.15) / (s * gamma_b)


def jsce_vwcd(bw, d, fcd, gamma_b):
    """腹部コンクリートの斜め圧縮破壊耐力 Vwcd = fwcd·bw·d / γb、fwcd = 1.25√f'cd（≦ 7.8）"""
    return min(1.25 * math.sqrt(fcd), 7.8) * bw * d / gamma_b


def cracked_steel_stress(M, b, D, as_c, as_t, dc, dt, n):
    """ひび割れ断面（弾性、コンクリートの引張無視）の引張鉄筋の応力 σse [N/mm²]
    中立軸: (b/2)x² + n(A's + As)x − n(A's·dc + As·d) = 0、Icr = bx³/3 + n·As(d−x)² + n·A's(x−dc)²"""
    d = D - dt
    a2, a1, a0 = b / 2, n * (as_c + as_t), -n * (as_c * dc + as_t * d)
    x = (-a1 + math.sqrt(a1 ** 2 - 4 * a2 * a0)) / (2 * a2)
    Icr = b * x ** 3 / 3 + n * as_t * (d - x) ** 2 + n * as_c * (x - dc) ** 2
    return n * abs(M) * (d - x) / Icr


def crack_width(sigma_se, c, cs, phi, fck, eps_csd, layers=1, k1=1.0, Es=ES):
    """ひび割れ幅 w = 1.1·k1·k2·k3·{4c + 0.7(cs − φ)}·(σse/Es + ε'csd)
    k2 = 15/(f'c + 20) + 0.7、k3 = 5(n + 2)/(7n + 8)（n: 引張鉄筋の段数）"""
    k2 = 15 / (fck + 20) + 0.7
    k3 = 5 * (layers + 2) / (7 * layers + 8)
    return 1.1 * k1 * k2 * k3 * (4 * c + 0.7 * (cs - phi)) * (sigma_se / Es + eps_csd)


# ============================================================ 鋼管杭
def pile_ratio(N, M, A, Z, fyd):
    """軸力と曲げの相関 |N|/(A·fyd) + M/(Z·fyd)"""
    return abs(N) / (A * fyd) + abs(M) / (Z * fyd)


def pile_shear_ratio(Q, A, fyd):
    """せん断 τ/(fyd/√3)、τ = 2Q/A（薄肉円管の最大せん断応力度）"""
    return (2 * abs(Q) / A) / (fyd / math.sqrt(3))


# ============================================================ 照査の実行
@dataclass
class PierCheckReport:
    rows: list = field(default_factory=list)          # CheckRow（term = 限界状態）
    settings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def governing(self, keys=("eid", "item")):
        out = {}
        for r in self.rows:
            k = tuple(getattr(r, a) for a in keys)
            if k not in out or r.ratio > out[k].ratio:
                out[k] = r
        return out

    @property
    def ng(self):
        return [r for r in self.governing(("eid", "pos", "item")).values() if not r.ok]

    @property
    def max_ratio(self):
        return max((r.ratio for r in self.rows), default=0.0)

    def print_summary(self):
        s = self.settings
        print(f"\n■ 桟橋の照査（示方書 2022・港湾基準 2018）  γi = {s['gamma_i']:g}  "
              f"※係数は入力値。規準で要確認")
        print(f"{'要素':>4} {'部材':<18} {'位置':<5} {'照査':<10} {'組合せ':<12} {'応答':>9} {'限界':>9} "
              f"{'照査値':>6}  判定")
        unit = {"曲げ": 1e6, "せん断": 1e3, "せん断(腹部)": 1e3, "ひび割れ幅": 1.0, "軸力+曲げ": 1.0,
                "杭せん断": 1.0, "押込み": 1e3, "引抜き": 1e3}
        for _, r in sorted(self.governing(("eid", "item")).items(), key=lambda kv: kv[0]):
            k = unit.get(r.item, 1.0)
            print(f"{r.eid:>4} {r.label:<18} {r.pos:<5} {r.item:<10} {r.combo:<12} {r.demand / k:9.2f} "
                  f"{r.capacity / k:9.2f} {r.ratio:6.2f}  {'OK' if r.ok else '**NG**'}")
        print(f"  最大照査値 {self.max_ratio:.2f}、NG {len(self.ng)} 箇所")
        for w in self.warnings:
            print(f"  [注意] {w}")


def run_pier_checks(pier, results=None):
    """桟橋の照査（上部工 RC 梁・鋼管杭・杭の支持力）"""
    opt = pier.checks
    gi = opt["gamma_i"]
    cf = opt["concrete"]
    res = results or pier.analyze_cases()
    combos = pier.analyze_combinations(res)
    if opt["strength_combinations"] == "all":            # 永続・変動のすべて（偶発は P4）
        strength = [c for c in combos if pier.combinations[c]["state"] in ("永続", "変動")]
    else:
        strength = [c for c in opt["strength_combinations"] if c in combos]
    crack = [c for c in opt["crack"]["combinations"] if c in combos]
    rep = PierCheckReport(settings=dict(opt))
    m = pier.model
    pile_at = {pier.node_ids[ij]: sec for ij, (sec, _) in pier.pile_sec.items()}

    # ---- 上部工の梁
    for key, eid in sorted(pier.elem_ids.items(), key=lambda kv: kv[1]):
        if key[0] == "p":
            continue
        e = m.elements[eid]
        rb, s, mat = e.sec.rebar, e.sec, e.mat
        label = pier.label("elem", eid)
        if rb is None:
            rep.warnings.append(f"{label}: 配筋がないため照査しません")
            continue
        fcd = mat.fck / cf["gamma_c"]
        fyd = rb.main.F / cf["gamma_s"]
        fwyd = rb.shear.F / cf["gamma_s"]
        d = s.D - rb.dt
        ri = pile_at[e.ni].D / 2 if e.ni in pile_at else 0.0     # 梁端の検定断面 = 杭の前面
        rj = pile_at[e.nj].D / 2 if e.nj in pile_at else 0.0
        for pos, x, bars in (("i端", ri, rb.end), ("中央", e.L / 2, rb.center), ("j端", e.L - rj, rb.end)):
            for c in strength:
                sf = combos[c].section_forces_at(eid, x)
                M, V = -sf[4], abs(sf[2])                      # 下端引張を正
                comp, tens = (bars.top, bars.bottom) if M >= 0 else (bars.bottom, bars.top)
                side = "下端引張" if M >= 0 else "上端引張"
                Mu = jsce_mu(s.b, s.D, comp.area, tens.area, rb.dt, rb.dt, fcd, fyd, fyd)
                Mud = Mu / cf["gamma_b_flexure"]
                rep.rows.append(CheckRow(eid, label, pos, "", "曲げ", "断面破壊", c, 0.0, gi * abs(M), Mud, side))
                Vyd = (jsce_vcd(s.b, d, tens.area / (s.b * d), fcd, cf["gamma_b_vc"])
                       + jsce_vsd(bars.stirrup.aw, fwyd, bars.stirrup.pitch, d, cf["gamma_b_vs"]))
                rep.rows.append(CheckRow(eid, label, pos, "", "せん断", "断面破壊", c, 0.0, gi * V, Vyd))
                Vw = jsce_vwcd(s.b, d, fcd, cf["gamma_b_vw"])
                rep.rows.append(CheckRow(eid, label, pos, "", "せん断(腹部)", "断面破壊", c, 0.0, gi * V, Vw))
            for c in crack:
                sf = combos[c].section_forces_at(eid, x)
                M = -sf[4]
                comp, tens = (bars.top, bars.bottom) if M >= 0 else (bars.bottom, bars.top)
                sig = cracked_steel_stress(M, s.b, s.D, comp.area, tens.area, rb.dt, rb.dt, ES / mat.E)
                cover = s.cover if getattr(s, "cover", None) else rb.dt - tens.dia / 2   # 主鉄筋の表面まで
                cs =(s.b - 2 * cover - tens.dia) / (tens.n - 1) if tens.n > 1 else s.b
                w = crack_width(sig, cover, cs, tens.dia, mat.fck, opt["crack"]["eps_csd"])
                wa = opt["crack"]["wa_ratio"] * cover
                rep.rows.append(CheckRow(eid, label, pos, "", "ひび割れ幅", "使用性", c, 0.0, w, wa,
                                         f"σse = {sig:.0f} N/mm²"))

    # ---- 鋼管杭（腐食後の断面）と支持力
    for (i, j), p in sorted(pier.piles.items(), key=lambda kv: kv[1]["eid"]):
        e = m.elements[p["eid"]]
        sec, mat = e.sec, e.mat
        cp = sec.props(corroded=True)
        fyd = mat.fy / opt["pile"]["gamma_m"]
        label = pier.label("elem", p["eid"])
        Rcd, Rtd = pier.pile_capacity.get(j, (None, None))
        for c in strength:
            for pos, x in (("固定点", 0.0), ("杭頭", e.L - e.rigid_j)):
                sf = combos[c].section_forces_at(p["eid"], x)
                N = -sf[0]                                   # 圧縮正
                M = math.hypot(sf[4], sf[5])
                Q = math.hypot(sf[1], sf[2])
                rep.rows.append(CheckRow(p["eid"], label, pos, "", "軸力+曲げ", "断面破壊", c, N,
                                         gi * pile_ratio(N, M, cp["A"], cp["Z"], fyd), 1.0,
                                         f"N = {N / 1e3:.0f} kN, M = {M / 1e6:.0f} kN·m"))
                rep.rows.append(CheckRow(p["eid"], label, pos, "", "杭せん断", "断面破壊", c, N,
                                         gi * pile_shear_ratio(Q, cp["A"], fyd), 1.0))
            N = -combos[c].section_forces_at(p["eid"], e.L - e.rigid_j)[0]
            if N >= 0 and Rcd:
                rep.rows.append(CheckRow(p["eid"], label, "杭頭", "", "押込み", "支持力", c, N, gi * N, Rcd))
            if N < 0 and Rtd:
                rep.rows.append(CheckRow(p["eid"], label, "杭頭", "", "引抜き", "支持力", c, N, gi * -N, Rtd))
    if not pier.pile_capacity:
        rep.warnings.append("杭の設計支持力（piles.rows の Rcd, Rtd）がないため支持力の照査をしていません")
    return rep
