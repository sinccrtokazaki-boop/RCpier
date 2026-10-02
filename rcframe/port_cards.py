"""利用可否の判定カードと一覧（HTML、利用可否 U6）

判定カード（シナリオごと、1ファイル完結）:
    判定（色分け）と理由、主な数値、許容荷重と必要な荷重の比較、杭の損傷図（残留変形を拡大し、
    ヒンジを損傷の程度で色分け、地盤変位の分布を破線で重ねる）、計算の条件
一覧（index.html）: 判定ごとの件数と割合（標準誤差）、許容荷重が不足する割合、フラジリティ曲線、
    シナリオの表（列の見出しで並べ替え、判定で絞り込み、カードへのリンク）
カードの杭の損傷図には部材ごとの状態が必要なので、カードを作るシナリオは解析し直す（port_batch.analyze）。
結果の行と同じ解析なので、数値は一括実行の結果と一致する

計算書（report.py）の CSS と作図（svg.py）を用いる。単位: 表示は m・rad・kN/m²・kN
"""
import json
import math
from concurrent.futures import ProcessPoolExecutor
from html import escape
from pathlib import Path

from .report import CSS, Raw, figure, table
from .svg import View, circle, line, poly, svg, text, xy_chart
from .port_usability import LOAD_NAME, LOAD_UNIT, STATUS, final_rows, probability_summary

M = 1e3
STATUS_CLS = {STATUS[0]: "st0", STATUS[1]: "st1", STATUS[2]: "st2"}
LEVEL_CLS = {"塑性化": "lv-pl", "局部座屈": "lv-lb"}
CARD_CSS = """
.st0{background:#1b7f3b}.st1{background:#b7791f}.st2{background:#c0392b}
.status{display:inline-block;padding:4px 16px;border-radius:14px;color:#fff;font-weight:700;font-size:18px}
.head{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin:6px 0 10px}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px;margin:10px 0}
.kv div{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 10px}
.kv b{display:block;font-size:18px}.kv span{color:var(--mute);font-size:12px}
.lv-pl{fill:#e69f00;stroke:#fff;stroke-width:1}.lv-lb{fill:#c0392b;stroke:#fff;stroke-width:1.5}
.defm{stroke:#c0392b;stroke-width:2;fill:none}.orig{stroke:#c9ced6;stroke-width:1.5}
.gdisp{stroke:#8d6e3f;stroke-width:1.5;stroke-dasharray:5 3;fill:none}
.bar-ok{fill:#1b7f3b}.bar-ng{fill:#c0392b}.bar-bg{fill:#eef1f5}.req{stroke:#1d2430;stroke-width:1.5}
.ctl{display:flex;gap:12px;align-items:center;margin:8px 0;font-size:13px}
.wrap td{white-space:normal;min-width:7em}
.pt0{fill:#1b7f3b;fill-opacity:.85}.pt1{fill:#e69f00;fill-opacity:.9}.pt2{fill:#c0392b;fill-opacity:.85}
th.sort{cursor:pointer}th.sort:hover{background:#e2e7ee}
td a{color:var(--acc)}
"""
INDEX_JS = r"""
(function(){
 var tb=document.getElementById('rows'); if(!tb) return;
 var sel=document.getElementById('flt');
 sel.addEventListener('input',function(){
  Array.prototype.forEach.call(tb.rows,function(r){
   r.style.display=(!sel.value||r.getAttribute('data-st')===sel.value)?'':'none';});});
 Array.prototype.forEach.call(document.querySelectorAll('th.sort'),function(th,k){
  var asc=true;
  th.addEventListener('click',function(){
   var rs=Array.prototype.slice.call(tb.rows);
   rs.sort(function(a,b){var x=a.cells[k].getAttribute('data-v')||a.cells[k].textContent,
    y=b.cells[k].getAttribute('data-v')||b.cells[k].textContent,fx=parseFloat(x),fy=parseFloat(y);
    var c=(!isNaN(fx)&&!isNaN(fy))?fx-fy:String(x).localeCompare(String(y),'ja');return asc?c:-c;});
   asc=!asc; rs.forEach(function(r){tb.appendChild(r);});});});
})();
"""


def _page(title, body, js=""):
    return ("<!DOCTYPE html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{CSS}{CARD_CSS}</style></head><body><main>{body}</main>"
            + (f"<script>{js}</script>" if js else "") + "</body></html>")


def _fmt(v, f="{:.3f}"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)


# ============================================================ 図
def damage_levels(fr):
    """ヒンジ -> 損傷の程度（弾性 / 塑性化 / 局部座屈）。ファイバーの諸元がなければ降伏 = 塑性化"""
    if fr.caps:
        return {r["key"]: r["level"] for r in fr.damage(residual=False)}
    return {k: ("塑性化" if h.state == "降伏" else "弾性") for k, h in fr.inc.hinges.items()}


def damage_frame(pier, fr, levels, width=720):
    """杭の損傷図: 載荷方向の鉛直断面（損傷の最も多い通り）。元の形（灰）、残留変形（赤、拡大）、
    ヒンジ（塑性化 橙・局部座屈 赤）、地盤変位の分布（茶の破線、変形と同じ倍率）"""
    m = pier.model
    d = 1 if fr.case.direction.lstrip("+-").lower() == "y" else 0
    other = 0 if d == 1 else 1                              # 断面を切る通りの軸
    lines = pier.gx if d == 1 else pier.gy
    count = {k: 0 for k in range(len(lines))}
    for (i, j), info in pier.piles.items():
        k = i if d == 1 else j
        count[k] += sum(levels.get((eid, pl, end), "弾性") != "弾性"
                        for eid in info["elements"] for pl in (0, 1) for end in (0, 1))
    kbest = max(count, key=lambda k: (count[k], -k))
    piles = [(ij, info) for ij, info in pier.piles.items() if (ij[0] if d == 1 else ij[1]) == kbest]
    beams = [e for key, e in pier.elem_ids.items() if key[0] == ("y" if d == 1 else "x") and key[1] == kbest]
    coord = (lambda nd: nd.y) if d == 1 else (lambda nd: nd.x)
    idx = fr.inc.idx
    u0 = fr.inc.res.states[fr.phases[0][1]][0]               # 長期荷重時
    du = fr.inc.u - u0
    zb = min([info.get("tip", info["z_fix"]) for _, info in piles] + [pier.z_soffit])
    span = [g for _, g in (pier.gy if d == 1 else pier.gx)]
    u_lo, u_hi = min(span) - 3000, max(span) + 3000
    umax = max([abs(du[6 * idx[n] + d]) for n in m.nodes] + [abs(fr.inc.ug[6 * idx[n] + d]) for n in m.nodes]
               + [1e-9])
    scale = 0.15 * (u_hi - u_lo) / umax                     # 最大の変位を全幅の 15% で描く
    ext = [scale * v for v in list(du[d::6]) + list(fr.inc.ug[d::6])]
    V = View(u_lo + min(ext + [0.0]), u_hi + max(ext + [0.0]), zb, pier.deck_top, width=width,
             margin=(30, 40, 34, 60), max_height=620)
    body = [line(V.p(u_lo, pier.deck_top), V.p(u_hi, pier.deck_top), "deckline")]
    for _, info in piles:
        g = coord(m.nodes[m.elements[info["eid"]].nj])
        body.append(line(V.p(g - 2500, info["seabed"]), V.p(g + 2500, info["seabed"]), "seabed"))
    members = beams + [eid for _, info in piles for eid in info["elements"]]
    for eid in members:
        e = m.elements[eid]
        a, b = m.nodes[e.ni], m.nodes[e.nj]
        body.append(line(V.p(coord(a), a.z), V.p(coord(b), b.z), "orig"))
    for eid in members:
        e = m.elements[eid]
        a, b = m.nodes[e.ni], m.nodes[e.nj]
        pa = V.p(coord(a) + scale * du[6 * idx[a.id] + d], a.z)
        pb = V.p(coord(b) + scale * du[6 * idx[b.id] + d], b.z)
        body.append(line(pa, pb, "defm"))
    for _, info in piles:                                    # 地盤変位の分布
        g = coord(m.nodes[m.elements[info["eid"]].nj])
        pts = [V.p(g + scale * fr.inc.ug[6 * idx[n] + d], z) for n, z, _, _ in info["ground"]]
        if len(pts) > 1 and any(abs(fr.inc.ug[6 * idx[n] + d]) > 0 for n, _, _, _ in info["ground"]):
            body.append(poly(pts, "gdisp", closed=False))
    marks = []
    for eid in members:
        e = m.elements[eid]
        a, b = m.nodes[e.ni], m.nodes[e.nj]
        for end in (0, 1):
            for pl in (0, 1):
                lv = levels.get((eid, pl, end))
                if lv not in LEVEL_CLS:
                    continue
                nd = a if end == 0 else b
                q = V.p(coord(nd) + scale * du[6 * idx[nd.id] + d], nd.z)
                marks.append(circle(q, 6 if lv == "局部座屈" else 5, LEVEL_CLS[lv], f"{fr.labels[(eid, pl, end)]}: {lv}"))
    name = (pier.gx if d == 1 else pier.gy)[kbest][0]
    body.append(text((V.w - 8, 14), f"{name} 通り（変形・地盤変位は {scale:.0f} 倍）", "axlbl", anchor="end"))
    return svg(V.w, V.h, body + marks), name, scale


def allowable_bars(row, opts, width=720):
    """許容荷重 / 必要な荷重 の棒グラフ（1 = 必要な荷重、2 倍以上は 2 で頭打ち）"""
    items = [(k, row.get(f"allow_{k}"), req) for k, req in opts["required"].items() if row.get(f"allow_{k}") is not None]
    if not items:
        return None
    ml, mr, h = 120, 230, 30
    W = width - ml - mr
    body = []
    for k, (kind, a, req) in enumerate(items):
        y = 10 + k * (h + 12)
        r = min(a / req, 2.0) if req > 0 else 2.0
        body.append(f'<rect class="bar-bg" x="{ml}" y="{y}" width="{W}" height="{h}"/>')
        body.append(f'<rect class="{"bar-ok" if a >= req else "bar-ng"}" x="{ml}" y="{y}" '
                    f'width="{W * r / 2:.1f}" height="{h}"/>')
        body.append(text((ml - 8, y + h / 2), LOAD_NAME[kind], "axlbl", anchor="end", dy=4))
        body.append(text((ml + W + 8, y + h / 2), f"{a:,.1f} / {req:g} {LOAD_UNIT[kind]}", "axlbl", anchor="start", dy=4))
    H = 10 + len(items) * (h + 12)
    body.append(line((ml + W / 2, 4), (ml + W / 2, H), "req"))
    body.append(text((ml + W / 2, H + 14), "必要な荷重", "axlbl"))
    return svg(width, H + 22, body)


# ============================================================ カード
def card_html(pier, row, fr, opts, params):
    """1シナリオの判定カード（HTML 文字列）"""
    st = row.get("status", "")
    reasons = [x for x in (row.get("reasons") or "").split(" / ") if x]
    levels = damage_levels(fr)
    parts = [f'<h1>利用可否の判定カード</h1><p class="mute">{escape(pier.name)} ・ シナリオ {escape(row["id"])}'
             f'（{"詳細化（ファイバー）" if row["tier"] == "refine" else "ふるい分け"}）</p>']
    parts.append(f'<div class="head"><span class="status {STATUS_CLS.get(st, "")}">{escape(st or "判定なし")}</span></div>')
    if reasons:
        parts.append("<ul class=\"list\">" + "".join(f"<li>{escape(x)}</li>" for x in reasons) + "</ul>")
    kv = [("上部工の残留変位", f"{row['deck_residual']:.3f} m"), ("上部工の最大変位", f"{row['deck_peak']:.3f} m"),
          ("平面の残留回転", f"{0.0 if abs(row['deck_rot']) < 5e-6 else row['deck_rot']:.5f} rad"),
          ("杭頭の塑性回転角（残留）", f"{row['head_rot_res']:.4f} rad"),
          ("地中部の塑性回転角（残留）", f"{row['inground_rot_res']:.4f} rad"),
          ("降伏したヒンジ", f"{row['n_yield']} 箇所"),
          ("局部座屈", "-" if row.get("n_buckled") is None else f"{row['n_buckled']} 箇所"),
          ("最大ひずみ（推定）", _fmt(row.get("max_strain"), "{:.5f}")),
          ("降伏した地盤ばね", f"{row['soil_yield']} 箇所")]
    parts.append('<div class="kv">' + "".join(f"<div><span>{escape(a)}</span><b>{escape(b)}</b></div>" for a, b in kv)
                 + "</div>")
    if opts:
        bars = allowable_bars(row, opts)
        if bars:
            parts.append("<h3>被災後の許容荷重</h3>")
            parts.append(figure(bars, "許容荷重（限界に達する荷重）と必要な荷重。緑: 足りる、赤: 不足"))
            parts.append(table(["荷重", "許容値", "必要な値", "決めた限界"],
                               [[LOAD_NAME[k], f"{row[f'allow_{k}']:,.1f} {LOAD_UNIT[k]}", f"{req:g} {LOAD_UNIT[k]}",
                                 row.get(f"allow_{k}_by", "")]
                                for k, req in opts["required"].items() if row.get(f"allow_{k}") is not None]))
    fig, name, scale = damage_frame(pier, fr, levels)
    parts.append("<h3>杭の損傷と残留変形</h3>")
    parts.append(figure(fig, f"{name} 通りの断面。灰: 元の形、赤線: 残留変形（{scale:.0f} 倍）、茶の破線: 地盤変位、"
                             "橙の丸: 塑性化、赤の丸: 局部座屈"))
    cnt = {lv: sum(v == lv for v in levels.values()) for lv in ("局部座屈", "塑性化")}
    parts.append(f'<p class="note">全体で 局部座屈 {cnt["局部座屈"]} 箇所、塑性化 {cnt["塑性化"]} 箇所</p>')
    c = fr.case
    surf = ", ".join(f"{pier.gy[j][0]}: {v / M:.3f} m" for j, v in c.surface.items())  # 杭の列（Y通り）ごと
    cond = [["方向", c.direction], ["海底面の地盤変位", surf], ["流動層", c.layer_text],
            ["分布形", c.shape], ["地盤ばねの低減（剛性・上限）", f"{c.kh_factor:g}・{c.pu_factor:g}"],
            ["慣性力の換算震度", f"{c.kh:.3g}"], ["載荷順序", c.order], ["慣性力の除荷", "する" if c.unload else "しない"],
            ["局部座屈ひずみの式", str(params.get("ecu", "A"))]]
    parts.append("<h3>計算の条件</h3>")
    parts.append(table(["項目", "値"], cond))
    parts.append('<p class="note">判定基準・必要な荷重は入力値です（ガイドライン・利用計画で要確認）。</p>')
    return _page(f"判定カード {row['id']}", "".join(parts))


def sorted_final(rows):
    """最終結果の行（判定の厳しい順、同じ判定では残留変位の大きい順）"""
    return sorted(final_rows(rows), key=lambda r: (-STATUS.index(r["status"]) if r.get("status") in STATUS else 0,
                                                   -abs(r["deck_residual"])))


def summary_parts(rows, opts, h="h3"):
    """判定の割合・許容荷重の不足・フラジリティ曲線（HTML 部品のリスト）"""
    fr = final_rows(rows)
    s = probability_summary(rows, opts)
    parts = [f"<{h}>判定の割合</{h}>",
             table(["判定", "件数", "割合", "標準誤差"],
                   [[Raw(f'<span class="status {STATUS_CLS[k]}" style="font-size:13px">{k}</span>'),
                     sum(r.get("status") == k for r in fr), f"{p:.1%}", f"{se:.1%}"]
                    for k, (p, se) in s["status"].items()], num=(1, 2, 3))]
    if s["allow"]:
        parts.append(table(["荷重", "必要な値", "許容値が不足する割合", "標準誤差"],
                           [[LOAD_NAME[k], f"{opts['required'][k]:g} {LOAD_UNIT[k]}", f"{p:.1%}", f"{se:.1%}"]
                            for k, (p, se) in s["allow"].items()], num=(2, 3)))
    series, notes, fits = [], [], []
    xs = [float(r["surface"]) for r in fr if not isinstance(r["surface"], str)]
    xmax = max(xs + [0.1]) * 1.2
    for lab, cls in (("制限付き利用以上", "c1"), ("利用不可", "c4")):
        f = s["fragility"][lab]
        if f is None:
            notes.append(f"{lab}: 推定できません（全て同じ判定）")
            continue
        th, b = f
        from scipy.stats import norm
        pts = [(x, float(norm.cdf(math.log(x / th) / b))) for x in (xmax * (i + 1) / 80 for i in range(80))]
        series.append((lab, pts, cls))
        fits.append([lab, f"{th:.3f} m", f"{b:.3f}"])
    if series:
        parts.append(f"<{h}>フラジリティ曲線</{h}>")
        parts.append(figure(xy_chart(series, "海底面の地盤変位 [m]", "確率", xfmt="{:.2f}", yfmt="{:.2f}"),
                            "海底面の地盤変位（surface）に対する判定の確率 P = Φ(ln(x/θ)/β)（最尤法）"))
        parts.append(table(["判定", "中央値 θ", "対数標準偏差 β"], fits, num=(1, 2)))
        if any(float(f[2]) < 0.01 for f in fits):
            notes.append("β がほぼ 0 の曲線は、判定が地盤変位だけで完全に分かれている（他の変数のばらつきの影響が"
                         "小さい、またはシナリオが少ない）ため、最尤推定が階段状になったものです")
    for n in notes:
        parts.append(f'<p class="note">{escape(n)}</p>')
    parts.append('<p class="note">確率として読めるのは scenarios.distributions で抽出した場合です'
                 '（grid の場合は組合せの中の割合）。</p>')
    return parts


def scenario_table(rows, cards=None, interactive=True):
    """シナリオの表（HTML）。cards: {id: カードのファイル名}（リンク）、interactive: 並べ替え・絞り込み"""
    cards = cards or {}
    head = ["id", "判定", "地盤変位 [m]", "層厚・下端 [m]", "kh", "εcu", "残留変位 [m]", "杭の降伏", "局部座屈",
            "上載 [kN/m²]", "接岸 [kN]", "牽引 [kN]", "段階"]
    trs = []
    for r in sorted_final(rows):
        st = r.get("status", "")
        link = f'<a href="{escape(cards[r["id"]])}">{escape(r["id"])}</a>' if r["id"] in cards else escape(r["id"])
        cells = [link, f'<span class="status {STATUS_CLS.get(st, "")}" style="font-size:12px">{escape(st)}</span>',
                 escape(_fmt(r["surface"]) if not isinstance(r["surface"], str) else r["surface"]),
                 _fmt(r["depth"], "{:g}") if r.get("bottom") is None else f"下端 {r['bottom']:+g}",
                 _fmt(r["kh"], "{:.3g}"), escape(str(r["ecu"])),
                 _fmt(r["deck_residual"]), str(r["n_yield"]), _fmt(r.get("n_buckled"), "{}"),
                 _fmt(r.get("allow_surcharge"), "{:,.1f}"), _fmt(r.get("allow_berthing"), "{:,.0f}"),
                 _fmt(r.get("allow_tractive"), "{:,.0f}"), "詳細化" if r["tier"] == "refine" else "ふるい分け"]
        vals = ["", str(STATUS.index(st)) if st in STATUS else "", "", "", "", "", "", "", "", "", "", "", ""]
        tds = "".join(f'<td data-v="{escape(v)}">{c}</td>' if v else f"<td>{c}</td>" for c, v in zip(cells, vals))
        trs.append(f'<tr data-st="{escape(st)}">{tds}</tr>')
    ctl = ""
    if interactive:
        opts_html = "".join(f'<option value="{s_}">{s_}</option>' for s_ in STATUS)
        ctl = (f'<div class="ctl screen-only">判定で絞り込み <select id="flt"><option value="">すべて</option>{opts_html}'
               '</select><span class="mute">見出しをクリックで並べ替え</span></div>')
    return (ctl + '<div class="tw"><table><thead><tr>'
            + "".join(f'<th class="sort">{escape(x)}</th>' for x in head)
            + f'</tr></thead><tbody id="rows">{"".join(trs)}</tbody></table></div>')


def index_html(name, rows, opts, cards):
    """一覧ページ（HTML 文字列）。cards: {id: カードのファイル名}"""
    parts = [f"<h1>利用可否シナリオの一覧</h1><p class=\"mute\">{escape(name)} ・ {len(final_rows(rows))} シナリオ</p>"]
    if opts:
        parts += summary_parts(rows, opts)
    parts.append("<h3>シナリオ</h3>")
    parts.append(scenario_table(rows, cards))
    return _page(f"利用可否シナリオの一覧 {name}", "".join(parts), INDEX_JS)


# ============================================================ 書き出し
def card_one(data, sid, params, tier, segments, row):
    """シナリオを解析し直してカードの HTML を返す（子プロセスでも動く）。戻り値: (HTML, 解析し直した行)"""
    from .port_batch import _summary, analyze
    from .port_usability import judge, usability_row
    pier, fr = analyze(data, sid, params, tier, segments)
    again = _summary(fr, sid, tier, params, 0.0)
    if pier.usability:
        again.update(usability_row(pier, fr, pier.usability))
        again["status"], rs = judge(again, pier.usability)
        again["reasons"] = " / ".join(rs)
    return card_html(pier, {**row, **again}, fr, pier.usability, params), again


def write_cards(batch, rows, outdir, which=(STATUS[1], STATUS[2]), workers=1, log=print):
    """判定カード（which の判定のシナリオ、"all" なら全て）と一覧 index.html を outdir に書く。戻り値: index のパス"""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    params = dict(batch.scenarios)
    fr = final_rows(rows)
    todo = [r for r in fr if which == "all" or r.get("status") in which]
    log(f"判定カード: {len(todo)} 件")
    args = [(batch.data, r["id"], params[r["id"]], r["tier"], batch.segments, r) for r in todo]
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(card_one, *zip(*args)))
    else:
        results = [card_one(*a) for a in args]
    cards = {}
    for r, (html, _) in zip(todo, results):
        fn = f"card_{r['id']}.html"
        (out / fn).write_text(html, encoding="utf-8")
        cards[r["id"]] = fn
    idx = out / "index.html"
    idx.write_text(index_html(batch.pier.name, rows, batch.pier.usability, cards), encoding="utf-8")
    return idx
