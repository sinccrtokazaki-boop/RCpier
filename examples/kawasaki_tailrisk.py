# -*- coding: utf-8 -*-
"""川崎港 東扇島 −12m 岸壁 桟橋部: 利用可否の計算結果（G4）からテイルリスクの図を作る

  python examples/kawasaki_tailrisk.py            # matplotlib が要る（.venv には入っていないので、入っている python で実行）

入力は examples/kawasaki_guideline/results/<ケース>.csv（kawasaki_guideline.py の結果。再計算はしない）。
出力は examples/kawasaki_guideline/ の テイルリスク_超過確率・テイルリスク_内訳・テイルリスク_確率密度（.png・.svg）、tailrisk.json。

損失は「復旧費（この 1 区画の再調達費に対する %）」で、抽出ごとの損傷の項目に仮定の費用を割り当てて足す（上限 100%）。
経路の内訳は、同じ地震の抽出（地震・土留部の抽出値は年数によらず共通）の 0 年の結果と比べて、部材ごとに
0 年にもある損傷を S3（地震）、劣化で増えた分を梁 → S1（上部工の塩害）、杭 → S2（杭の腐食。S4 洗掘を含む）に、
土留部・渡版の項目を S5 に振り分ける。常時（地震なし）は 0 年の損失を 0 とみなす。
地震の年発生率（L1 1/75 年、L2 1/1000 年）と費用は仮定。値は COST・RATE を直して作り直す。
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "kawasaki_guideline"
RES = OUT / "results"

# ---- 仮定
REQ = {"surcharge": 20.0, "berthing": 500.0, "tractive": 350.0, "aftershock": 0.15}   # 3-C と余震（kawasaki_guideline.py と同じ）
WALL = (0.30, 0.50)                    # 土留部の残留水平変位 [m]（limited, unusable）
STEP, OPEN = 0.10, 0.20                # 段差・渡版の開き [m]（超えれば緊急復旧）
COST = {                               # 損傷の項目ごとの復旧費 [再調達費の %]（仮定）
    "buckle_head": 20,                 # 杭の局部座屈（杭頭だけ）: 杭頭の補強
    "buckle_ground": 40,               # 杭の局部座屈（地中部を含む）: 増し杭・取り替え
    "plastic": 10,                     # 杭の塑性化（局部座屈なし）: 杭の補修
    "deck": 15,                        # 上部工の梁が Mu に達した: 上部工の補修
    "strengthen": 10,                  # 損傷はないが耐力が足りない（利用時の荷重・余震）: 補強
    "wall_limited": 10, "wall_unusable": 25,   # 土留部の復旧
    "emergency": 1,                    # 段差の解消・渡版の仮設
}
RATE = {"L1_all": 1 / 75, "L2_all": 1 / 1000}   # 地震の年発生率 [1/年]（L1 は再現期間 75 年、L2 は仮定）
TAIL = 0.10                            # テイル = 損失の大きい側の 10%（CVaR90）
COST_BETA = 0.4                        # なめらかな確率密度: 部材ごとの復旧費のばらつき（対数正規の β、平均は COST のまま。仮定）

CASES = {"L1_all": "L1 地震後", "L2_all": "L2 地震後", "normal_all": "常時（地震なし）"}
AGES = (0.0, 40.0, 60.0)
PATHS = [("S3", "S3 地震（劣化がなくても起きる損傷）", "#7B8794"),
         ("S5", "S5 土留部・渡版", "#56B4E9"),
         ("S1", "S1 上部工の塩害", "#E69F00"),
         ("S2", "S2 杭の腐食（S4 洗掘を含む）", "#C2410C")]
AGE_STYLE = {0.0: ("#8FB3D1", (0, (5, 2.5)), "0 年（劣化なし）"), 40.0: ("#2F6E9E", (0, (8, 2, 1.5, 2)), "40 年"),
             60.0: ("#0B2545", "-", "60 年")}


def load(case):
    """2段目があれば 2段目の行（rcframe.port_usability.final_rows と同じ）"""
    by = {}
    with open(RES / f"{case}.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["id"] not in by or r["tier"] == "refine":
                by[r["id"]] = r
    return list(by.values())


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x):
    v = _f(x)
    return int(v) if v is not None else 0


def costs(r):
    """部材ごとの復旧費 {pile, beam, wall}"""
    nb, nbh = _i(r["n_buckled"]), _i(r["n_buckled_head"])
    pile = COST["buckle_ground"] if nb > nbh else COST["buckle_head"] if nb else (COST["plastic"] if _i(r["n_plastic"]) else 0)
    beam = COST["deck"] if (_f(r["deck_mu_max"]) or 0) >= 1.0 else 0
    if not pile and not beam:
        short = [k for k in REQ if _f(r["allow_" + k]) is not None and _f(r["allow_" + k]) < REQ[k] - 1e-9]
        if short:
            if any("梁" in r[f"allow_{k}_by"] for k in short):
                beam = COST["strengthen"]
            else:
                pile = COST["strengthen"]
    wm = _f(r["wall_move"]) or 0
    wall = COST["wall_unusable"] if wm >= WALL[1] else COST["wall_limited"] if wm >= WALL[0] else 0
    if (_f(r["apron_step"]) or 0) > STEP or (_f(r["bridge_rel"]) or 0) > OPEN:
        wall += COST["emergency"]
    return {"pile": pile, "beam": beam, "wall": wall}


def split(r, base):
    """経路ごとの損失 {S1, S2, S3, S5}。base は同じ地震の 0 年の行（常時は None）"""
    c = costs(r)
    b = costs(base) if base is not None else {"pile": 0, "beam": 0, "wall": 0}
    out = {"S3": min(c["pile"], b["pile"]) + min(c["beam"], b["beam"]),
           "S2": max(c["pile"] - b["pile"], 0), "S1": max(c["beam"] - b["beam"], 0), "S5": c["wall"]}
    tot = sum(out.values())
    return {k: v * 100 / tot for k, v in out.items()} if tot > 100 else out


def _key(r):
    return tuple(r[k] for k in ("surface", "kh", "kh_factor", "pu_factor", "wall_ratio"))


def study():
    """{ケース: {年数: [経路ごとの損失 dict]}}"""
    out = {}
    for case in CASES:
        rows = load(case)
        base = {_key(r): r for r in rows if _f(r["age"]) == 0.0} if case in RATE else {}
        out[case] = {}
        for a in sorted({_f(r["age"]) for r in rows}):
            g = [r for r in rows if _f(r["age"]) == a]
            out[case][a] = [split(r, base[_key(r)] if base else None) for r in g]
    return out


def stats(comp):
    L = [sum(c.values()) for c in comp]
    n = len(L)
    order = sorted(range(n), key=lambda k: -L[k])
    top = order[:max(1, round(n * TAIL))]
    mean = {p: sum(c[p] for c in comp) / n for p, _, _ in PATHS}
    tail = {p: sum(comp[k][p] for k in top) / len(top) for p, _, _ in PATHS}
    return dict(n=n, mean=sum(L) / n, cvar=sum(L[k] for k in top) / len(top), var=L[top[-1]],
                p_loss=sum(x > 0 for x in L) / n, mean_by=mean, tail_by=tail, losses=L)


def exceed(L):
    """P(L > x) の階段（x = 0 と損失の値で折れる）"""
    xs = sorted({0.0, *L, 100.0})
    n = len(L)
    return xs, [sum(v > x for v in L) / n for x in xs]


def annual(S, age):
    xs = sorted({0.0, 100.0, *(v for c in RATE for v in S[c][age]["losses"])})
    lam = [sum(RATE[c] * sum(v > x for v in S[c][age]["losses"]) / S[c][age]["n"] for c in RATE) for x in xs]
    return xs, lam


def _font(names, default):
    """候補のうち matplotlib が見つけた最初のフォント（Windows: BIZ UD、Linux: Noto CJK（fonts-noto-cjk）など）"""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), default)


def _style():
    plt.rcParams.update({"font.family": _font(("BIZ UDPGothic", "Noto Sans CJK JP", "IPAexGothic"), "sans-serif"), "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
                         "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
                         "svg.fonttype": "path", "axes.titlelocation": "left", "legend.frameon": False})


NOTE = ("損失＝復旧費（この 1 区画の再調達費に対する %、仮定）：杭の局部座屈 杭頭だけ 20%・地中部を含む 40%、杭の塑性化 10%、"
        "上部工の梁の Mu 到達 15%、損傷はないが耐力不足の補強 10%、土留部 0.30 m 以上 10%・0.50 m 以上 25%、段差・渡版の緊急復旧 1%（上限 100%）。"
        "各 200 回の抽出（川崎港 東扇島 桟橋部の 1 区画、ガイドライン第2版の判定、劣化・地震の分布と判定基準は仮定を含む）。")


def _table(ax, x, y, head, rows, va):
    """図の中の小さな表（等幅の BIZ UDGothic で桁をそろえる）"""
    ax.text(x, y, "\n".join([head] + rows), transform=ax.transAxes, ha="left", va=va, fontsize=9.5,
            family=_font(("BIZ UDGothic", "Noto Sans Mono CJK JP", "IPAGothic"), "monospace"), linespacing=1.45, color="#222222",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#CCCCCC", linewidth=0.8))


def fig_exceed(S, path):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, case, ttl, pos in ((axs[0], "L1_all", "(a) L1 地震が起きたとき", (0.52, 0.96, "top")),
                               (axs[1], "L2_all", "(b) L2 地震が起きたとき", (0.03, 0.03, "bottom"))):
        for a in AGES:
            col, ls, _ = AGE_STYLE[a]
            xs, ps = exceed(S[case][a]["losses"])
            ax.plot(xs, ps, drawstyle="steps-post", color=col, ls=ls, lw=2.2)
        _table(ax, pos[0], pos[1], "        平均  上位10%の平均",
               [f"{a:>2.0f} 年  {S[case][a]['mean']:5.1f}%  {S[case][a]['cvar']:9.0f}%" for a in AGES], pos[2])
        ax.set_title(ttl)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("復旧費 x（再調達費に対する %）")
        ax.set_ylabel("P（復旧費 > x）")
        ax.grid(alpha=0.25)
    ax = axs[2]
    for a in AGES:
        col, ls, _ = AGE_STYLE[a]
        xs, lam = annual(S, a)
        k = next(j for j, y in enumerate(lam) if y == 0)      # 頻度が 0 になる x（最大の損失）で下へ落とす（対数軸）
        ax.plot(xs[:k] + [xs[k]], lam[:k] + [1e-6], drawstyle="steps-post", color=col, ls=ls, lw=2.2)
    _table(ax, 0.03, 0.03, "      年期待損失",
           [f"{a:>2.0f} 年  {sum(RATE[c] * S[c][a]['mean'] for c in RATE):.3f}%/年" for a in AGES], "bottom")
    for y, s in ((1 / 100, "年 1/100"), (1 / 1000, "年 1/1000")):
        ax.axhline(y, color="#999999", lw=0.8, ls=":")
        ax.text(99, y * 1.12, s, ha="right", va="bottom", fontsize=9, color="#666666")
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(1e-5, 3e-2)
    ax.set_title("(c) 年超過頻度（L1・L2 の発生を合わせる）")
    ax.set_xlabel("復旧費 x（再調達費に対する %）")
    ax.set_ylabel("復旧費 > x となる地震の頻度 [1/年]")
    ax.grid(alpha=0.25, which="both")
    hs = [Line2D([], [], color=AGE_STYLE[a][0], ls=AGE_STYLE[a][1], lw=2.2, label=AGE_STYLE[a][2]) for a in AGES]
    fig.legend(handles=hs, loc="upper center", ncol=3, fontsize=10.5, handlelength=3.2, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.015, NOTE + " (c) は L1 を年 1/75、L2 を年 1/1000 で起きるとした（仮定）。", fontsize=8.3, color="#444444",
             wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, path)


def _bars(ax, S, case, ages, key, ylabel, title, fmt="{:.0f}%", mean=True):
    xs = range(len(ages))
    for i, a in enumerate(ages):
        st = S[case][a]
        y0 = 0
        for p, _, col in PATHS:
            v = st[key][p]
            if v > 0:
                ax.bar(i, v, bottom=y0, color=col, width=0.62, edgecolor="#FFFFFF", linewidth=0.8)
            y0 += v
        ax.text(i, y0, fmt.format(y0), ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#222222")
        if mean:
            ax.plot(i, st["mean"], marker="D", color="#111111", ms=7, zorder=5)
    ax.set_xticks(list(xs), [f"{a:.0f} 年" for a in ages])
    ax.set_xlim(-0.6, len(ages) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def fig_split(S, path):
    fig, axs = plt.subplots(1, 4, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [3, 3, 2, 3]})
    _bars(axs[0], S, "L1_all", AGES, "tail_by", "上位 10% の平均（CVaR90）[%]", "(a) L1 地震後")
    _bars(axs[1], S, "L2_all", AGES, "tail_by", "上位 10% の平均（CVaR90）[%]", "(b) L2 地震後")
    _bars(axs[2], S, "normal_all", (40.0, 60.0), "tail_by", "上位 10% の平均（CVaR90）[%]", "(c) 常時（地震なし）")
    for ax in axs[:3]:
        ax.set_ylim(0, 100)
    E = {a: {p: sum(RATE[c] * S[c][a]["mean_by"][p] for c in RATE) for p, _, _ in PATHS} for a in AGES}
    S2 = {"annual": {a: {"eal_by": E[a], "mean": 0} for a in AGES}}
    _bars(axs[3], S2, "annual", AGES, "eal_by", "年期待損失 [再調達費の %/年]", "(d) 年期待損失（地震）", fmt="{:.3f}%",
          mean=False)
    hs = [Patch(color=col, label=lab) for _, lab, col in PATHS]
    hs.append(Line2D([], [], marker="D", color="#111111", ls="none", ms=7, label="全抽出の平均（期待値）"))
    fig.legend(handles=hs, loc="upper center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.015, "棒の内訳：同じ地震の抽出の 0 年の結果と比べ、部材ごとに 0 年にもある損傷を S3、劣化で増えた分を梁 → S1・杭 → S2、"
             "土留部・渡版を S5 とした（常時は 0 年の損失を 0 とした）。(d) は L1 年 1/75・L2 年 1/1000（仮定）。" + NOTE,
             fontsize=8.3, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    _save(fig, path)


BIN = 5.0                              # 確率密度の棒の幅 [%]


def fig_density(S, path):
    """損失の確率密度（棒の幅 BIN）。損失 0 は点の質量なので密度に入れず、題名に確率を書く"""
    cols = [("L1_all", "L1 地震後"), ("L2_all", "L2 地震後"), ("normal_all", "常時（地震なし）")]
    fig, axs = plt.subplots(3, 3, figsize=(15.5, 10.8), sharex=True)
    edges = [BIN * k for k in range(int(100 / BIN) + 1)]
    for j, (case, cname) in enumerate(cols):
        top = 0
        for i, a in enumerate(AGES):
            ax = axs[i, j]
            if a not in S[case]:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{cname} の {a:.0f} 年は損失なし（新設）", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10.5, color="#666666")
                continue
            st = S[case][a]
            L, n = st["losses"], st["n"]
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = sum(lo <= v < hi or (hi == 100 and v == 100) for v in L if v > 0)
                if k:
                    d = k / n / BIN
                    ax.bar(lo, d, width=BIN, align="edge", color="#C2410C" if lo > st["var"] else "#5B8DB8",
                           edgecolor="#FFFFFF", linewidth=0.8)
                    top = max(top, d)
            ax.axvline(st["mean"], color="#111111", ls="--", lw=1.4)
            ax.axvline(st["var"], color="#777777", ls=":", lw=1.6)
            ax.axvline(st["cvar"], color="#C2410C", ls="-", lw=1.8)
            ax.set_title(f"{cname}・{a:.0f} 年\n損失なし {1 - st['p_loss']:.1%}・平均 {st['mean']:.1f}%・"
                         f"VaR90 {st['var']:.0f}%・CVaR90 {st['cvar']:.0f}%", fontsize=10.5, linespacing=1.35)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlim(0, 100)
            if j == 0 or (j == 2 and i == 1):
                ax.set_ylabel("確率密度 [1/%]")
            if i == 2:
                ax.set_xlabel("復旧費（再調達費に対する %）")
        for i in range(3):
            if axs[i, j].axison:
                axs[i, j].set_ylim(0, top * 1.12)
    hs = [Patch(color="#5B8DB8", label="損失の確率密度"), Patch(color="#C2410C", label="VaR90 を超える部分（テイル）"),
          Line2D([], [], color="#111111", ls="--", lw=1.4, label="平均（期待値）"),
          Line2D([], [], color="#777777", ls=":", lw=1.6, label="VaR90（90% 点）"),
          Line2D([], [], color="#C2410C", lw=1.8, label="上位 10% の平均（CVaR90）")]
    fig.legend(handles=hs, loc="upper center", ncol=5, fontsize=10.5, bbox_to_anchor=(0.5, 0.997))
    fig.text(0.01, 0.012, f"棒の幅 {BIN:.0f}%、高さ＝その幅に入る確率 ÷ 幅（棒の面積の合計＝損失がある確率）。損失 0 の確率は密度に含めず題名に示した。"
             "損失は損傷の項目ごとの費用の和なので、とびとびの値になる。" + NOTE, fontsize=8.3, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    _save(fig, path)


DX = 0.05                              # なめらかな確率密度の刻み [%]


def raw_costs():
    """{ケース: {年数: [抽出ごとの部材別の復旧費（0 を除く）]}}（上限 100% で切る前）"""
    out = {}
    for case in CASES:
        rows = load(case)
        out[case] = {}
        for a in sorted({_f(r["age"]) for r in rows}):
            out[case][a] = [[v for v in costs(r).values() if v > 0] for r in rows if _f(r["age"]) == a]
    return out


def smooth(items, beta=COST_BETA):
    """部材ごとの復旧費に独立な対数正規のばらつき（平均は元の値）を掛けた損失の分布。
    抽出ごとの和は Fenton–Wilkinson 近似（平均・分散の等しい対数正規）で、全抽出の混合。上限 100% を超える分は 100% に集める。
    戻り値: x, 密度 f（0 < x < 100）, 0 の確率, 100% の確率, 平均, VaR, CVaR"""
    import math
    import numpy as np
    x = np.arange(DX / 2, 100, DX)
    f = np.zeros_like(x)
    n, p0, p100 = len(items), 0.0, 0.0
    lx = np.log(x)
    for c in items:
        if not c:
            p0 += 1 / n
            continue
        m = sum(c)
        v = sum(ci * ci for ci in c) * (math.exp(beta ** 2) - 1)
        s2 = math.log(1 + v / m ** 2)
        mu, s = math.log(m) - s2 / 2, math.sqrt(s2)
        f += np.exp(-(lx - mu) ** 2 / (2 * s2)) / (x * s * math.sqrt(2 * math.pi)) / n
        p100 += 0.5 * math.erfc((math.log(100) - mu) / (s * math.sqrt(2))) / n
    F = p0 + np.cumsum(f) * DX
    q = 1 - TAIL
    var = 0.0 if p0 >= q else (float(x[np.argmax(F >= q)]) if F[-1] >= q else 100.0)
    mean = float((x * f).sum() * DX + 100 * p100)
    cvar = var + (float((np.clip(x - var, 0, None) * f).sum() * DX) + (100 - var) * p100) / TAIL
    return x, f, p0, p100, mean, var, cvar


def fig_density_smooth(R, path):
    """なめらかな確率密度（部材ごとの復旧費のばらつきを入れる）。0 と 100% の確率は点に集まるので題名に書く"""
    cols = [("L1_all", "L1 地震後"), ("L2_all", "L2 地震後"), ("normal_all", "常時（地震なし）")]
    fig, axs = plt.subplots(3, 3, figsize=(15.5, 11.8), sharex=True)
    out = {}
    for j, (case, cname) in enumerate(cols):
        top = 0
        for i, a in enumerate(AGES):
            ax = axs[i, j]
            if a not in R[case]:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{cname} の {a:.0f} 年は損失なし（新設）", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10.5, color="#666666")
                continue
            x, f, p0, p100, mean, var, cvar = smooth(R[case][a])
            out.setdefault(case, {})[f"{a:.0f}"] = dict(p0=round(p0, 4), p100=round(p100, 4), mean=round(mean, 3),
                                                       var=round(var, 2), cvar=round(cvar, 2))
            ax.fill_between(x, f, color="#5B8DB8", alpha=0.35, lw=0)
            ax.plot(x, f, color="#2F6E9E", lw=1.8)
            t = x >= var
            ax.fill_between(x[t], f[t], color="#C2410C", alpha=0.55, lw=0)
            ax.axvline(mean, color="#111111", ls="--", lw=1.4)
            ax.axvline(var, color="#777777", ls=":", lw=1.6)
            ax.axvline(cvar, color="#C2410C", ls="-", lw=1.8)
            top = max(top, float(f.max()))
            ax.set_title(f"{cname}・{a:.0f} 年\n損失なし {p0:.1%}・上限 100% に達する {p100:.1%}\n"
                         f"平均 {mean:.1f}%・VaR90 {var:.0f}%・CVaR90 {cvar:.0f}%", fontsize=10.5, linespacing=1.35)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlim(0, 100)
            if j == 0 or (j == 2 and i == 1):
                ax.set_ylabel("確率密度 [1/%]")
            if i == 2:
                ax.set_xlabel("復旧費（再調達費に対する %）")
        for i in range(3):
            if axs[i, j].axison:
                axs[i, j].set_ylim(0, top * 1.1)
    hs = [Patch(color="#8FB0CF", label="損失の確率密度"), Patch(color="#D98A5F", label="VaR90 を超える部分（テイル）"),
          Line2D([], [], color="#111111", ls="--", lw=1.4, label="平均（期待値）"),
          Line2D([], [], color="#777777", ls=":", lw=1.6, label="VaR90（90% 点）"),
          Line2D([], [], color="#C2410C", lw=1.8, label="上位 10% の平均（CVaR90）")]
    fig.legend(handles=hs, loc="upper center", ncol=5, fontsize=10.5, bbox_to_anchor=(0.5, 0.997))
    fig.text(0.01, 0.012, f"部材（杭・上部工・土留部）ごとの復旧費に、平均が下の値で β = {COST_BETA} の対数正規のばらつき（独立）を掛けた分布"
             "（抽出ごとの和は Fenton–Wilkinson 近似、全抽出の混合、上限 100%）。損失 0 と上限 100% の確率は点に集まるので密度に含めず題名に示した。"
             + NOTE, fontsize=8.3, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    _save(fig, path)
    return out


def _save(fig, path):
    for ext in ("png", "svg"):                       # 画像ビューアーで開いていても上書きできるよう、別名に書いて置き換える
        tmp = path.with_name(f"_{path.name}.tmp.{ext}")
        fig.savefig(tmp, dpi=200, facecolor="#FFFFFF", format=ext)
        os.replace(tmp, path.with_suffix("." + ext))
    plt.close(fig)


def main():
    _style()
    comp = study()
    S = {c: {a: stats(v) for a, v in d.items()} for c, d in comp.items()}
    fig_exceed(S, OUT / "テイルリスク_超過確率")
    fig_split(S, OUT / "テイルリスク_内訳")
    fig_density(S, OUT / "テイルリスク_確率密度_ヒストグラム")
    dens = fig_density_smooth(raw_costs(), OUT / "テイルリスク_確率密度")
    js = {"assumptions": {"cost_percent": COST, "rate_per_year": RATE, "required": REQ, "wall": WALL,
                          "step": STEP, "opening": OPEN, "tail": TAIL},
          "cases": {c: {f"{a:.0f}": {k: (round(v, 4) if isinstance(v, float) else
                                         {p: round(x, 4) for p, x in v.items()} if isinstance(v, dict) else v)
                                     for k, v in st.items() if k != "losses"} for a, st in d.items()} for c, d in S.items()},
          "annual": {f"{a:.0f}": {"eal": round(sum(RATE[c] * S[c][a]["mean"] for c in RATE), 5),
                                  "eal_by": {p: round(sum(RATE[c] * S[c][a]["mean_by"][p] for c in RATE), 5)
                                             for p, _, _ in PATHS}} for a in AGES},
          "density_smooth": {"cost_beta": COST_BETA, **dens}}
    (OUT / "tailrisk.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    for c, d in S.items():
        for a, st in d.items():
            print(f"{CASES[c]} {a:.0f} 年: 平均 {st['mean']:.2f}%  CVaR90 {st['cvar']:.1f}%  "
                  + "  ".join(f"{p} {st['tail_by'][p]:.1f}" for p, _, _ in PATHS))
    for a, v in js["annual"].items():
        print(f"年期待損失 {a} 年: {v['eal']:.4f}%/年 ", v["eal_by"])
    for c, d in dens.items():
        for a, v in d.items():
            print(f"なめらか {CASES[c]} {a} 年:", v)


if __name__ == "__main__":
    main()
