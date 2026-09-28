"""桟橋の計算書（HTML）: 建物編の report.py と同じ仕組み（1ファイル完結、外部参照なし）で章立てを桟橋用にする

    1 概要（平面図・断面図・杭の諸元） 2 荷重と組合せ 3 解析結果（3D 表示・曲げモーメント図）
    4 照査（P3） 5 レベル2（P4） 6 付録（照合が必要な事項・簡略化・警告・入力データ）
計算は既存の関数（analyze_cases / analyze_combinations / run_pier_checks / run_level2）を呼ぶだけ
"""
import datetime
import math
from html import escape

import numpy as np

from . import svg
from .port_check import VERIFY_ITEMS as CHECK_VERIFY, run_pier_checks
from .port_level2 import VERIFY_ITEMS as L2_VERIFY, run_level2
from .report import Raw, assemble_html, figure, inv, judge, kN, table, viewer_html

KNOWN = [
    "杭は仮想固定点モデル（海底面 − 1/β で固定）または地盤ばねモデル（kh·D·負担長の水平ばね）",
    "上部工は 1つの剛床、梁端の剛域なし、床版はモデル化せず荷重の分配のみ",
    "張出し部の荷重は端の梁への線荷重（ねじりは無視）。防舷材・係船柱の力はてこの原理で節点へ",
    "杭の慣性力・付加質量、車両・荷役機械の集中荷重の自動生成は扱わない",
    "照査の荷重係数は組合せ（入力）で考慮し、照査では構造物係数 γi のみ。構造解析係数 γa = 1.0",
    "鋼管杭の照査は線形の相関式（座屈は考慮しない）",
    "レベル2: 杭の軸力は長期荷重時の値で固定、載荷方向の曲げ面のみ、上部工の梁は弾性（M ≦ Mu を確認）",
    "地盤反力度の上限 pu は深さ方向に一定",
]
EC_VERIFY = ["示方書の Ec の表（f'ck との対応）", "鋼管杭のせん断形状係数 2（有効断面積 A/2）"]


def _m(v):
    return f"{v / 1e3:+.2f}"


# ================================================================ 計算
def _compute(pier):
    lin = pier.analyze_cases()
    ctx = {"lin": lin, "comb": pier.analyze_combinations(lin) if pier.combinations else {}, "warnings": []}
    has_rebar = any(getattr(s, "rebar", None) for s in pier.sections.values())
    if ctx["comb"] and (pier.checks["given"] or has_rebar):
        ctx["check"] = run_pier_checks(pier, lin)
        ctx["warnings"] += [f"照査: {w}" for w in ctx["check"].warnings]
    if pier.level2["given"]:
        ctx["l2"] = {d: run_level2(pier, d, lin) for d in pier.level2["directions"]}
        for d, r in ctx["l2"].items():
            ctx["warnings"] += [f"レベル2 {d}: {w}" for w in dict.fromkeys(r.warnings)]
    return ctx


# ================================================================ 章
def _cover(pier):
    rp = pier.report
    title = rp.get("title") or f"構造計算書 {pier.name}"
    rows = [("モデル名", pier.name), ("作成者", rp.get("author", "")),
            ("日付", rp.get("date", datetime.date.today().isoformat())), ("備考", rp.get("notes", ""))]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows if v)
    return (f'<section class="cover"><h1>{escape(title)}</h1>'
            f'<p class="mute">rcframe による直杭式横桟橋の計算書（港湾基準 2018・コンクリート標準示方書 2022、'
            f'単位: kN, kN·m, m）</p><dl>{dl}</dl></section>')


def sec_outline(pier, ctx):
    m = pier.model
    out = [table(["項目", "値"], [
        ["杭のモデル", "地盤ばね" if pier.pile_model == "soil_spring" else "仮想固定点（Chang の方法）"],
        ["杭の本数", len(pier.piles)], ["梁の本数", sum(k[0] in ("x", "y") for k in pier.elem_ids)],
        ["節点数・要素数", f"{len(m.nodes)}・{len(m.elements)}"],
        ["上部工の天端・骨組・下面", f"{_m(pier.deck_top)}・{_m(pier.z_model)}・{_m(pier.z_soffit)} m"],
        ["床版厚・張出し（海側・陸側）",
         f"{pier.slab_t / 1e3:.2f} m・{pier.overhang['sea'] / 1e3:.2f} m・{pier.overhang['land'] / 1e3:.2f} m"],
        ["荷重ケース", ", ".join(pier.cases)]])]
    out.append("<h3>通り芯</h3>")
    out.append(table(["法線方向（X）", "位置 [m]"], [[n, f"{p / 1e3:g}"] for n, p in pier.gx], num=(1,)))
    out.append(table(["陸側 → 海側（Y）", "位置 [m]"], [[n, f"{p / 1e3:g}"] for n, p in pier.gy], num=(1,)))
    out.append("<h3>材料</h3>")
    rows = []
    for k, mt in pier.materials.items():
        if hasattr(mt, "fy"):
            rows.append([k, "鋼材", f"fy = {mt.fy:g}", f"{mt.E:,.0f}"])
        else:
            rows.append([k, "コンクリート", f"f'ck = {mt.fck:g}", f"{mt.E:,.0f}"])
    out.append(table(["名前", "種類", "強度 [N/mm²]", "E [N/mm²]"], rows, num=(3,)))
    out.append("<h3>断面</h3>")
    rows = []
    for n, s in pier.sections.items():
        if hasattr(s, "t"):
            c0, c1 = s.props(False), s.props(True)
            rows.append([n, f"鋼管 φ{s.D:g} × t{s.t:g}", f"外 {s.corrosion_outer:g}・内 {s.corrosion_inner:g}",
                         f"A = {c0['A']:,.0f} / {c1['A']:,.0f} mm²、Z = {c0['Z'] / 1e6:,.2f} / {c1['Z'] / 1e6:,.2f} ×10⁶ mm³"])
        else:
            rb = s.rebar
            reb = ("-" if rb is None else
                   f"端部 上{rb.end.top} 下{rb.end.bottom} {rb.end.stirrup} ／ 中央 上{rb.center.top} "
                   f"下{rb.center.bottom} {rb.center.stirrup}（{rb.main.name}, dt={rb.dt:g}, "
                   f"かぶり {getattr(s, 'cover', None) or '-'}）")
            rows.append([n, f"梁 {s.b:g} × {s.D:g}", "-", reb])
    out.append(table(["断面", "寸法 [mm]", "腐食代 [mm]", "性能（腐食前 / 腐食後）・配筋"], rows))
    out.append("<h3>杭の諸元（列ごと）</h3>")
    rows = []
    for j, (name, _) in enumerate(pier.gy):
        info = next((v for (i, jj), v in sorted(pier.piles.items()) if jj == j), None)
        if info is None:
            continue
        cap = pier.pile_capacity.get(j, (None, None))
        lower = (f"固定点 {_m(info['z_fix'])}" if pier.pile_model == "fixed_point" else f"先端 {_m(info['tip'])}")
        rows.append([name, _m(info["seabed"]), f"{info['kh'] * 1e6:,.0f}", f"{1 / info['beta'] / 1e3:.2f}",
                     lower, f"{info['free_length'] / 1e3:.2f}",
                     f"{cap[0] / 1e3:,.0f}" if cap[0] else "-", f"{cap[1] / 1e3:,.0f}" if cap[1] else "-"])
    out.append(table(["列", "海底面 [m]", "kh [kN/m³]", "1/β [m]", "下端 [m]", "自由長 [m]", "Rcd [kN]", "Rtd [kN]"],
                     rows, num=range(1, 8)))
    out.append(figure(svg.pier_plan(pier), "平面図（● 杭、橙 防舷材、紫 係船柱、灰色は上部工の外形）"))
    out.append(figure(svg.pier_frame(pier, pier.gx[0][0]), f"{pier.gx[0][0]} 通りの断面図（茶色は海底面）"))
    return "".join(out)


def sec_loads(pier, ctx):
    lin = ctx["lin"]
    out = ["<h3>荷重ケースの合計（支点反力・地盤ばね反力の逆符号）</h3>"]
    rows = []
    for c in pier.cases:
        tot = np.zeros(6)
        for v in list(lin[c].reactions.values()) + list(lin[c].spring_forces.values()):
            tot += v
        rows.append([c, kN(-tot[0]), kN(-tot[1]), kN(-tot[2])])
    out.append(table(["ケース", "ΣFx [kN]", "ΣFy [kN]", "ΣFz [kN]"], rows, num=(1, 2, 3)))
    if pier.weights:
        out.append("<h3>重量と重心</h3>")
        out.append(table(["区分", "W [kN]", "xg [m]", "yg [m]"],
                         [[k, kN(W), f"{x / 1e3:.2f}", f"{y / 1e3:.2f}"] for k, (W, x, y) in pier.weights.items()],
                         num=(1, 2, 3)))
    if pier.seismic:
        s = pier.seismic
        out.append(f"<p>レベル1地震動: kh = {s['kh']:g}、kv = {s['kv']:g}、慣性力 kh·W = {kN(s['kh'] * s['W'])} kN"
                   f"（上部工の重心 ({s['xg'] / 1e3:.2f}, {s['yg'] / 1e3:.2f}) m に作用）</p>")
    if pier.combinations:
        out.append("<h3>荷重の組合せ（部分係数は入力値。港湾基準で要確認）</h3>")
        cases = sorted({c for v in pier.combinations.values() for c in v["factors"]})
        out.append(table(["組合せ", "状態"] + cases,
                         [[n, v["state"]] + [f"{v['factors'][c]:g}" if c in v["factors"] else "" for c in cases]
                          for n, v in pier.combinations.items()], num=range(2, 2 + len(cases))))
    return "".join(out)


def sec_analysis(pier, ctx):
    m = pier.model
    comb = ctx["comb"]
    results = dict(comb) if comb else dict(ctx["lin"])
    pos = {f"レベル2 {d}（最終）": r.po for d, r in (ctx.get("l2") or {}).items()}
    for name, po in pos.items():
        results[name] = po.final if po.final is not None else None
    results = {k: v for k, v in results.items() if v is not None}
    out = ["<h3>3次元モデルと変形図</h3>", Raw(viewer_html(m, results, pos, column_label="杭"))]
    if comb:
        out.append("<h3>組合せごとの結果</h3>")
        rows = []
        for name, r in comb.items():
            d = r.disp[pier.master()]
            Ns, Ms = [], []
            for info in pier.piles.values():
                e = m.elements[info["eid"]]
                sj = r.section_forces_at(info["eid"], e.L - e.rigid_j)
                Ns.append(sj[0])
                Ms.append(math.hypot(sj[4], sj[5]))
            rows.append([name, pier.combinations[name]["state"], f"{d[0]:.2f}", f"{d[1]:.2f}",
                         kN(-min(Ns)), kN(max(max(Ns), 0)), f"{max(Ms) / 1e6:,.1f}"])
        out.append(table(["組合せ", "状態", "ux [mm]", "uy [mm]", "杭の最大圧縮 [kN]", "最大引張 [kN]",
                          "杭頭の最大曲げ [kN·m]"], rows, num=range(2, 7)))
        show = [n for n, v in pier.combinations.items() if v["state"] == "永続"][:1]
        show += [n for n in comb if "EQy" in pier.combinations[n]["factors"]][:1]
        show += [n for n in comb if "B" in pier.combinations[n]["factors"]][:1]
        for name in show:
            r = comb[name]
            mmax = max(abs(r.section_forces_at(e.id, x)[4]) for e in m.elements.values()
                       for x in (0.0, e.L / 2, e.L))
            scale = 1500.0 / mmax if mmax else 0.0
            out.append(f"<h3>曲げモーメント図（Y方向の断面）: {escape(name)}</h3>")
            out.append('<p class="note">引張側に描画、数値は梁端・梁中央・杭頭・杭の地中部の最大 [kN·m]</p>')
            for xn, _ in pier.gx:
                out.append(figure(svg.pier_frame(pier, xn, r, scale), f"{xn} 通り {name}"))
    return "".join(out)


def sec_checks(pier, ctx):
    rep = ctx.get("check")
    if rep is None:
        return None
    ng = rep.ng
    badge = '<span class="badge ok">OK</span>' if not ng else '<span class="badge ng">NG あり</span>'
    out = [f"<p>最大照査値 <b>{rep.max_ratio:.2f}</b>、NG {len(ng)} 箇所 {badge}　構造物係数 γi = "
           f"{rep.settings['gamma_i']:g}（係数は入力値。示方書・港湾基準で要確認）</p>"]
    unit = {"曲げ": (1e6, "kN·m"), "せん断": (1e3, "kN"), "せん断(腹部)": (1e3, "kN"), "ひび割れ幅": (1.0, "mm"),
            "押込み": (1e3, "kN"), "引抜き": (1e3, "kN")}
    out.append("<h3>照査項目ごとの最大</h3>")
    g = rep.governing(("item",))
    out.append(table(["照査", "限界状態", "部材", "位置", "組合せ", "照査値", "判定", "備考"],
                     [[k[0], r.term, r.label, r.pos, r.combo, f"{r.ratio:.2f}", judge(r.ok), r.note]
                      for k, r in sorted(g.items())], num=(5,)))
    out.append("<h3>部材・照査項目ごとの最大</h3>")
    rows, cls = [], []
    for _, r in sorted(rep.governing(("eid", "item")).items(), key=lambda kv: kv[0]):
        k, u = unit.get(r.item, (1.0, "-"))
        rows.append([r.label, r.pos, r.item, r.combo, f"{r.demand / k:,.2f}", f"{r.capacity / k:,.2f}", u,
                     f"{r.ratio:.2f}", judge(r.ok), r.note])
        cls.append("" if r.ok else "ng")
    out.append(table(["部材", "位置", "照査", "組合せ", "応答", "限界", "単位", "照査値", "判定", "備考"], rows,
                     num=(4, 5, 7), row_cls=cls))
    return "".join(out)


def sec_level2(pier, ctx):
    l2 = ctx.get("l2")
    if not l2:
        return None
    o = pier.level2
    dem = (f"照査用震度 kh = {o['kh']:g}" if o["kh"] is not None
           else f"要求変位 {o['target_displacement'] / 1e3:g} m")
    ok = all(r.ok for r in l2.values())
    out = [f'<p>判定 <span class="badge {"ok" if ok else "ng"}">{"すべて OK" if ok else "NG あり"}</span>　'
           f'要求側: {dem}（入力値。港湾基準で要確認）。長期荷重 {escape("+".join(o["gravity"]))} の後、'
           f'上部工の重心に水平力（換算震度 = 水平力 / W）を比例増加。杭のヒンジは全塑性モーメント Mp(N)。</p>']
    for d, r in l2.items():
        out.append(f"<h3>{escape(d)} 方向</h3>")
        marks = [(r.demand["disp"] / 1e3, "要求点")] if r.demand.get("reached") else []
        hmarks = [(o["kh"], f"kh = {o['kh']:g}")] if o["kh"] is not None else []
        out.append(figure(svg.xy_chart([("上部工", [(x / 1e3, y) for x, y in r.curve], "c0")],
                                       "上部工の変位 [m]", "換算震度 kh", marks, hmarks),
                          f"{d} 方向 荷重–変位曲線"))
        rows = [[n, judge(okk), note] for n, okk, note in r.criteria]
        if r.deck:
            rows.append(["上部工の梁が弾性（M ≦ Mu）", judge(r.deck["ratio"] <= 1),
                         f"max M/Mu = {r.deck['ratio']:.2f}（{r.deck['label']}）"])
        out.append(table(["判定項目", "判定", "内容"], rows))
        if r.demand.get("reached"):
            out.append(f"<p>要求点: kh = {r.demand['kh']:.3f}、上部工の変位 {r.demand['disp'] / 1e3:.3f} m</p>")
        ev = [e for e in r.po.events if e["event"] in ("降伏", "地盤降伏") and e["phase"] == "水平"]
        if ev:
            out.append(table(["順", "step", "位置", "事象", "換算震度", "変位 [m]"],
                             [[k + 1, e["step"], e["label"], "全塑性" if e["event"] == "降伏" else e["event"],
                               f"{e['lam']:.3f}", f"{e['drift'] / 1e3:.3f}"] for k, e in enumerate(ev[:40])],
                             num=(0, 1, 4, 5)))
            if len(ev) > 40:
                out.append(f'<p class="note">ほか {len(ev) - 40} 件</p>')
        if r.po.mechanism_step:
            out.append(f"<p>崩壊機構の形成: step {r.po.mechanism_step}</p>")
        if d.lstrip("+-").lower() == "y":
            for xn in (pier.gx[0][0], pier.gx[len(pier.gx) // 2][0]):
                out.append(figure(svg.pier_frame(pier, xn, po=r.po),
                                  f"{xn} 通り {d} 方向 最終状態の塑性ヒンジ（数字は全塑性の順番）"))
    return "".join(out)


def sec_appendix(pier, ctx):
    items = EC_VERIFY + CHECK_VERIFY + (L2_VERIFY if ctx.get("l2") else [])
    out = ["<h3>規準書・告示での照合が必要な事項</h3><ul class='list'>" +
           "".join(f"<li>{escape(x)}</li>" for x in dict.fromkeys(items)) + "</ul>",
           "<h3>既知の簡略化</h3><ul class='list'>" + "".join(f"<li>{escape(x)}</li>" for x in KNOWN) + "</ul>"]
    if ctx["warnings"]:
        out.append("<h3>警告</h3><ul class='list'>" +
                   "".join(f"<li>{escape(w)}</li>" for w in ctx["warnings"]) + "</ul>")
    if pier.source_text:
        out.append(f'<h3>入力データ</h3><pre class="src">{escape(pier.source_text)}</pre>')
    return "".join(out)


SECTIONS = (
    ("sec-outline", "概要", sec_outline),
    ("sec-loads", "荷重と組合せ", sec_loads),
    ("sec-analysis", "解析結果", sec_analysis),
    ("sec-checks", "照査（示方書 2022・港湾基準 2018）", sec_checks),
    ("sec-level2", "レベル2地震動（プッシュオーバー）", sec_level2),
    ("sec-appendix", "付録", sec_appendix),
)


def build_pier_report(pier, ctx=None):
    ctx = ctx or _compute(pier)
    return assemble_html(pier, SECTIONS, ctx, _cover(pier))
