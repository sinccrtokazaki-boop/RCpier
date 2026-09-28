"""計算書の HTML 出力（1ファイルで完結: CSS・JavaScript・図をすべて埋め込み、外部参照なし）

    write_report(bld, "report.html")
    python -m rcframe.report input.yaml -o report.html

入力に含まれる内容に応じて章を出力する（配筋がなければ断面検定以降の章は省略）。
計算はすべて既存の関数（解析・検定・プッシュオーバー・保有水平耐力）を呼び出し、ここでは整形のみ行う。
"""
import argparse
import datetime
import json
import math
from html import escape
from pathlib import Path

from . import svg
from .capacity import run_ultimate, stiffness_eccentricity
from .check import run_checks
from .pushover import run_pushover
from .results import Result

KNOWN_SIMPLIFICATIONS = [
    "柱の自動剛域は方向ごとの値の最小値を採用",
    "柱梁接合部の変形、耐力壁・ブレースは未対応",
    "剛床は 1 つの階に 1 つ（吹抜けで床が分断されても 1 つの剛床）",
    "自動生成は直交する通り芯のみ（斜め通り芯、片持ち梁、小梁、階ごとのレベル差は未対応）",
    "床荷重は通り芯で囲まれた区画単位（小梁による区画の分割、片持ちスラブは未対応）",
    "梁の自重は b×D で算定し、スラブとの重複を差し引かない（安全側）",
    "地震力は RC造として T = 0.02h。剛床のない階がある建物には Ai 分布の地震力を作用させられない",
    "断面算定: 柱の 2軸曲げの同時考慮、付着、柱梁接合部、曲げ降伏時せん断力による設計は未対応。"
    "柱の曲げは各方向の 1面の主筋のみ考慮。梁の中央はスパン中央の値で検定",
    "非線形解析: 曲げばねのみ（せん断ばねなし）、柱の変動軸力による My の変化と 2方向曲げの相互作用は未考慮、"
    "P-Δ 効果なし、梁の My はスラブ筋の寄与を無視",
    "保有水平耐力: 耐力壁のある建物は未対応。FD の柱があれば部材群 D（安全側）。局部崩壊形の判定はしない",
    "変形図（3D 表示）は節点の並進変位を直線で結んだもの（部材の曲がりは表示しない）",
]

VERIFY_ITEMS = [
    "RC規準（2018年版）の式番号（プログラムのコメントは条番号まで）",
    "骨格曲線の略算式の形と係数（2020年版 建築物の構造関係技術基準解説書 付録1-3）",
    "菅野式の pt の定義（at/(b·d) としている）",
    "部材種別の表の境界値、τu の定義（Qmu/(b·D) としている）、柱の pt の定義（at/(b·d) としている）",
    "部材群 D の条件（FD の部材を除いても崩壊形が保たれる場合の扱い）",
]

CSS = """
:root{--ink:#1d2430;--mute:#5d6673;--line:#d5dae1;--bg:#fbfbfa;--head:#eef1f5;--acc:#1f4e79;
--ok:#1b7f3b;--ng:#c0392b;--warn:#b7791f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 "Hiragino Sans","Yu Gothic UI",
"Yu Gothic","Meiryo",system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:44px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--acc)}
h3{font-size:16px;margin:26px 0 8px;color:var(--acc)}
p.note,.mute{color:var(--mute);font-size:13px}
nav.toc{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 18px;margin:18px 0}
nav.toc ol{margin:6px 0;padding-left:22px}
nav.toc a{color:var(--acc);text-decoration:none}
.cover{background:#fff;border:1px solid var(--line);border-radius:10px;padding:28px 28px 20px;margin-bottom:10px}
.cover dl{display:grid;grid-template-columns:8em 1fr;gap:4px 12px;margin:16px 0 0}
.cover dt{color:var(--mute)}
.tw{overflow-x:auto;margin:8px 0 14px}
table{border-collapse:collapse;background:#fff;font-size:13px;min-width:50%}
th,td{border:1px solid var(--line);padding:3px 8px;white-space:nowrap}
th{background:var(--head);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tr.ng td{background:#fdecea}
.ok{color:var(--ok);font-weight:600}.ng{color:var(--ng);font-weight:700}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-weight:700;color:#fff}
.badge.ok{background:var(--ok);color:#fff}.badge.ng{background:var(--ng);color:#fff}
figure{margin:10px 0 18px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px;
overflow-x:auto}
figcaption{font-size:13px;color:var(--mute);margin-top:2px}
svg.fig{max-width:100%;height:auto;display:block}
.gridline{stroke:#b9c0ca;stroke-width:1;stroke-dasharray:6 4}
.gridlbl,.axlbl{font-size:12px;fill:var(--mute)}
.mem-c{stroke:#1f4e79;stroke-width:3}.mem-b{stroke:#5b8c3a;stroke-width:3}
.mdia{fill:rgba(192,57,43,.14);stroke:#c0392b;stroke-width:1.2}
.mval{font-size:10.5px;fill:#8e2a1f;text-anchor:middle}
.mem-r{stroke-width:5}.r-lo{stroke:#9cc3e6}.r-mid{stroke:#3b82c4}.r-hi{stroke:#e69f00}
.r-ng{stroke:#c0392b}.r-none{stroke:#ccc}
.rval{font-size:11px;fill:#333;text-anchor:middle}
.h-y{fill:#c0392b;stroke:#fff;stroke-width:1.5}.h-c{fill:#e69f00;stroke:#fff;stroke-width:1}
.hnum{font-size:10px;fill:#8e2a1f;font-weight:700;text-anchor:middle}
.opening{fill:#eceff3}.col{fill:#1f4e79}.plbl{font-size:10.5px;fill:#3d6b25}
.axis{stroke:#555;stroke-width:1.2}.tick{stroke:#e3e6ea;stroke-width:1}
.mark{stroke:#c0392b;stroke-width:1;stroke-dasharray:4 3}.marklbl{font-size:11px;fill:#c0392b}
.curve{fill:none;stroke-width:2}.c0{stroke:#1f4e79}.c1{stroke:#e69f00}.c2{stroke:#1b7f3b}
.c3{stroke:#8e44ad}.c4{stroke:#c0392b}.c5{stroke:#16a085}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--mute);margin:4px 0}
.sw{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.viewer{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px}
.viewer canvas{width:100%;height:460px;display:block;cursor:grab;touch-action:none}
.viewer .ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:13px;margin-bottom:6px}
pre.src{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;
font-size:12px;line-height:1.5}
ul.list li{margin:2px 0}
.print-only{display:none}
@media print{
 @page{size:A4;margin:14mm}
 body{background:#fff;font-size:11px}
 main{max-width:none;padding:0}
 nav.toc,.screen-only{display:none}
 .print-only{display:block}
 h2{page-break-before:always}
 figure,table{page-break-inside:avoid}
 .tw{overflow:visible}
}
"""

JS = r"""
(function(){
 var el=document.getElementById('model-data'); if(!el) return;
 var D=JSON.parse(el.textContent);
 var cv=document.getElementById('viewer'), ctx=cv.getContext('2d');
 var selC=document.getElementById('v-case'), selH=document.getElementById('v-hinge');
 var sc=document.getElementById('v-scale'), info=document.getElementById('v-info');
 var yaw=-0.6, pitch=0.5, zoom=1, drag=null;
 var lo=[1e30,1e30,1e30], hi=[-1e30,-1e30,-1e30];
 Object.keys(D.nodes).forEach(function(k){var p=D.nodes[k];
  for(var a=0;a<3;a++){lo[a]=Math.min(lo[a],p[a]);hi[a]=Math.max(hi[a],p[a]);}});
 var c=[0,1,2].map(function(a){return (lo[a]+hi[a])/2;});
 var R=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])||1;
 function proj(p,W,H){
  var x=p[0]-c[0],y=p[1]-c[1],z=p[2]-c[2];
  var cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  var x1=x*cy-y*sy,y1=x*sy+y*cy,k=zoom*0.8*Math.min(W,H)/R;
  return [W/2+x1*k,H/2-(z*cp-y1*sp)*k];
 }
 function draw(){
  var dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
  cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
  var cs=selC.value,disp=cs?D.cases[cs]:null,f=disp?parseFloat(sc.value)*D.auto[cs]:0;
  function P(n,def){var p=D.nodes[n];
   if(def&&disp&&disp[n]){var d=disp[n];p=[p[0]+f*d[0],p[1]+f*d[1],p[2]+f*d[2]];}
   return proj(p,W,H);}
  function seg(a,b,col,w){ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();
   ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
  ctx.lineCap='round';
  D.elems.forEach(function(e){seg(P(e[0],false),P(e[1],false),
   disp?'#c9ced6':(e[2]==='c'?'#1f4e79':'#5b8c3a'),disp?1.2:2.2);});
  if(disp){D.elems.forEach(function(e){seg(P(e[0],true),P(e[1],true),
   e[2]==='c'?'#1f4e79':'#c0392b',2);});}
  var hs=(selH&&selH.value)?D.hinges[selH.value]:null;
  if(hs){hs.forEach(function(h){var q=proj(h,W,H);ctx.fillStyle=h[3]==='y'?'#c0392b':'#e69f00';
   ctx.beginPath();ctx.arc(q[0],q[1],h[3]==='y'?4.5:3.2,0,6.2832);ctx.fill();});}
  info.textContent=disp?('変形の表示倍率 '+f.toFixed(0)+' 倍'):'ドラッグで回転、ホイールで拡大縮小';
 }
 cv.addEventListener('pointerdown',function(ev){drag=[ev.clientX,ev.clientY,yaw,pitch];
  cv.setPointerCapture(ev.pointerId);});
 cv.addEventListener('pointermove',function(ev){if(!drag)return;
  yaw=drag[2]+(ev.clientX-drag[0])*0.01;
  pitch=Math.max(-1.5,Math.min(1.5,drag[3]+(ev.clientY-drag[1])*0.01));draw();});
 cv.addEventListener('pointerup',function(){drag=null;});
 cv.addEventListener('wheel',function(ev){ev.preventDefault();zoom*=Math.exp(-ev.deltaY*0.001);
  zoom=Math.max(0.2,Math.min(8,zoom));draw();},{passive:false});
 [selC,selH,sc].forEach(function(x){if(x)x.addEventListener('input',draw);});
 window.addEventListener('resize',draw);draw();
})();
"""


# ================================================================ 小道具
class Raw(str):
    """エスケープ済みの HTML"""


def _e(v):
    return v if isinstance(v, Raw) else escape(str(v))


def table(head, rows, num=(), row_cls=None):
    """head: 見出し、rows: 行のリスト、num: 右寄せ（数値）の列番号"""
    h = "".join(f"<th>{_e(x)}</th>" for x in head)
    body = []
    for k, r in enumerate(rows):
        cls = f' class="{row_cls[k]}"' if row_cls and row_cls[k] else ""
        tds = "".join(f'<td{" class=n" if i in num else ""}>{_e(x)}</td>' for i, x in enumerate(r))
        body.append(f"<tr{cls}>{tds}</tr>")
    return Raw(f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}'
               f"</tbody></table></div>")


def figure(svg_html, caption):
    return Raw(f"<figure>{svg_html}<figcaption>{escape(caption)}</figcaption></figure>")


def kN(v):
    return f"{v / 1e3:,.1f}"


def kNm(v):
    return f"{v / 1e6:,.1f}"


def inv(r):
    return f"1/{1 / abs(r):,.0f}" if r else "0"


def judge(ok):
    return Raw('<span class="ok">OK</span>' if ok else '<span class="ng">NG</span>')


# ================================================================ 計算
def _compute(bld):
    """計算書に必要な解析をまとめて実行する"""
    m = bld.model
    cases = sorted(set(m.nodal_loads) | set(m.member_loads))
    lin = {c: m.analyze(c) for c in cases}
    long = [c for c in (bld.design["long"] or ["G", "P"]) if c in lin]
    rL = Result.combine("+".join(long), {lin[c]: 1.0 for c in long}) if long else None
    ctx = {"cases": cases, "lin": lin, "long": long, "rL": rL, "warnings": []}
    has_rebar = any(s.rebar for s in bld.sections.values())
    if has_rebar and rL is not None:
        ctx["check"] = run_checks(bld, lin)
        ctx["warnings"] += [f"断面検定: {w}" for w in ctx["check"].warnings]
    if has_rebar and bld.seismic and bld.ultimate["given"]:
        ctx["ult"] = run_ultimate(bld)
        ctx["pushovers"] = ctx["ult"].pushovers
    elif has_rebar and bld.seismic and bld.nonlinear["given"]:
        ctx["pushovers"] = {d: run_pushover(bld, d, lin) for d in bld.nonlinear["pushover"]["directions"]}
    for d, po in (ctx.get("pushovers") or {}).items():
        ctx["warnings"] += [f"プッシュオーバー {d}: {w}" for w in dict.fromkeys(po.warnings)]
    if "Ex" in lin and "Ey" in lin and bld.weights and len(bld.masters) == len(bld.floors) - 1:
        ctx["ecc"], _ = stiffness_eccentricity(bld, lin)
    return ctx


# ================================================================ 章
def _cover(bld):
    rp = bld.report
    title = rp.get("title") or f"構造計算書 {bld.name}"
    rows = [("モデル名", bld.name), ("作成者", rp.get("author", "")),
            ("日付", rp.get("date", datetime.date.today().isoformat())), ("備考", rp.get("notes", ""))]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows if v)
    return (f'<section class="cover"><h1>{escape(title)}</h1>'
            f'<p class="mute">rcframe による RC造 3次元フレーム解析の計算書（単位: kN, kN·m, mm）</p>'
            f"<dl>{dl}</dl></section>")


def sec_model(bld, ctx):
    m = bld.model
    out = []
    n_used = len({n for e in m.elements.values() for n in (e.ni, e.nj)})
    out.append(table(["項目", "値"], [
        ["節点数（剛床のマスター節点を除く）", n_used], ["柱", sum(k[0] == "c" for k in bld.elem_ids)],
        ["梁", sum(k[0] != "c" for k in bld.elem_ids)], ["剛床", len(m.diaphragms)],
        ["荷重ケース", ", ".join(ctx["cases"]) or "なし"]], num=(1,)))
    out.append("<h3>通り芯</h3>")
    for axis, grid in (("X", bld.gx), ("Y", bld.gy)):
        rows = [[n, f"{p:,.0f}", f"{grid[k + 1][1] - p:,.0f}" if k + 1 < len(grid) else "-"]
                for k, (n, p) in enumerate(grid)]
        out.append(table([f"{axis}通り", "位置 [mm]", "次の通りまで [mm]"], rows, num=(1, 2)))
    out.append("<h3>階</h3>")
    rows = [[n, f"{z:,.0f}", f"{bld.floors[k + 1][1] - z:,.0f}" if k + 1 < len(bld.floors) else "-"]
            for k, (n, z) in enumerate(bld.floors)]
    out.append(table(["階", "床レベル [mm]", "階高 [mm]"], rows, num=(1, 2)))
    out.append("<h3>材料</h3>")
    out.append(table(["名前", "Fc [N/mm²]", "E [N/mm²]", "γ [kN/m³]", "ν"],
                     [[k, f"{c.Fc:g}", f"{c.E:,.0f}", f"{c.gamma:g}", f"{c.nu:g}"]
                      for k, c in bld.materials.items()], num=(1, 2, 3, 4)))
    out.append("<h3>断面</h3>")
    rows = []
    for name, s in bld.sections.items():
        rb = s.rebar
        if rb is None:
            reb = "-"
        elif hasattr(rb, "end"):
            reb = (f"端部 上{rb.end.top} 下{rb.end.bottom} {rb.end.stirrup} ／ "
                   f"中央 上{rb.center.top} 下{rb.center.bottom} {rb.center.stirrup}（{rb.main.name}, "
                   f"{rb.shear.name}, dt={rb.dt:g}）")
        else:
            reb = (f"X面 {rb.x}, Y面 {rb.y}, 帯筋 X {rb.hoop_x} Y {rb.hoop_y}（{rb.main.name}, "
                   f"{rb.shear.name}, dt={rb.dt:g}）")
        rows.append([name, f"{s.b:g} × {s.D:g}", f"{s.phi_y:g}", reb])
    out.append(table(["断面", "b × D [mm]", "φy", "配筋"], rows))
    out.append("<h3>伏図</h3>")
    for f in range(1, len(bld.floors)):
        out.append(figure(svg.floor_plan(bld, f), f"{bld.floors[f][0]} 伏図（梁と断面名、下階の柱、網掛けは吹抜け）"))
    out.append(figure(svg.isometric(bld), "骨組の全体図"))
    return "".join(out)


def sec_loads(bld, ctx):
    out = []
    fl = getattr(bld, "floor_loads", None)
    if fl:
        out.append("<h3>床荷重 [kN/m²]</h3>")
        out.append(table(["用途", "DL", "LL（架構用）", "LL（地震用）"],
                         [[k, f"{v['DL'] * 1e3:g}", f"{v['LL_frame'] * 1e3:g}", f"{v['LL_seismic'] * 1e3:g}"]
                          for k, v in fl.items()], num=(1, 2, 3)))
        use = {}
        for (f, i, j), u in bld.cell_use.items():
            if (f, i, j) in bld.openings:
                continue
            a = (bld.gx[i + 1][1] - bld.gx[i][1]) * (bld.gy[j + 1][1] - bld.gy[j][1]) / 1e6
            use[(f, u)] = use.get((f, u), 0.0) + a
        out.append(table(["階", "用途", "面積 [m²]"],
                         [[bld.floors[f][0], u, f"{a:,.1f}"] for (f, u), a in sorted(use.items())], num=(2,)))
    out.append("<h3>荷重ケースの合計（支点反力の逆符号）</h3>")
    rows = []
    for c in ctx["cases"]:
        s = -sum(ctx["lin"][c].reactions.values())
        rows.append([c, kN(s[0]), kN(s[1]), kN(s[2])])
    out.append(table(["ケース", "ΣFx [kN]", "ΣFy [kN]", "ΣFz [kN]"], rows, num=(1, 2, 3)))
    if bld.weights:
        out.append("<h3>地震用重量と重心</h3>")
        out.append(table(["階", "W [kN]", "xg [mm]", "yg [mm]"],
                         [[bld.floors[f][0], kN(w), f"{x:,.0f}", f"{y:,.0f}"]
                          for f, (w, x, y) in sorted(bld.weights.items())], num=(1, 2, 3)))
    if bld.seismic:
        s = bld.seismic
        out.append("<h3>地震力（Ai 分布）</h3>")
        out.append(f'<p>Z = {s["Z"]:g}、地盤種別 {s["soil"]}（Tc = {s["Tc"]:g} s）、h = {s["h"]:.2f} m、'
                   f'T = {s["T"]:.3f} s、Rt = {s["Rt"]:.3f}、C0 = {s["C0"]:g}'
                   f'{"、偶発偏心 " + format(s["eccentricity"], "g") if s["eccentricity"] else ""}</p>')
        out.append(table(["層", "Wi [kN]", "ΣWi [kN]", "αi", "Ai", "Ci", "Qi [kN]", "Pi [kN]"],
                         [[r["story"], kN(r["W"]), kN(r["sumW"]), f"{r['alpha']:.3f}", f"{r['Ai']:.3f}",
                           f"{r['Ci']:.3f}", kN(r["Q"]), kN(r["P"])] for r in reversed(s["rows"])],
                         num=range(1, 8)))
    return "".join(out)


def _viewer(bld, ctx):
    m = bld.model
    used = sorted({n for e in m.elements.values() for n in (e.ni, e.nj)})
    data = {"nodes": {str(n): [round(float(v), 1) for v in m.nodes[n].xyz] for n in used},
            "elems": [[str(e.ni), str(e.nj), "c" if e.kind == "column" else "b"] for e in m.elements.values()],
            "cases": {}, "auto": {}, "hinges": {}}
    size = max(float(max(m.nodes[n].xyz[a] for n in used) - min(m.nodes[n].xyz[a] for n in used))
               for a in range(3)) or 1.0
    results = dict(ctx["lin"])
    if ctx["rL"] is not None and len(ctx["long"]) > 1:
        results[ctx["rL"].case] = ctx["rL"]
    for d, po in (ctx.get("pushovers") or {}).items():
        results[f"プッシュオーバー {d}（最終）"] = po.final
    for name, r in results.items():
        disp = {str(n): [float(r.disp[n][k]) for k in range(3)] for n in used}
        mx = max(max(abs(v) for v in d) for d in disp.values())
        data["cases"][name] = {k: [round(v, 4) for v in d] for k, d in disp.items()}
        data["auto"][name] = 0.08 * size / mx if mx > 0 else 1.0
    for d, po in (ctx.get("pushovers") or {}).items():
        pts = []
        for (eid, plane, end), h in po.hinges.items():
            if h.state == "弾性":
                continue
            e = m.elements[eid]
            x = e.rigid_i + 0.06 * e.Lf if end == 0 else e.L - e.rigid_j - 0.06 * e.Lf
            a, b = m.nodes[e.ni].xyz, m.nodes[e.nj].xyz
            p = a + (b - a) * (x / e.L)
            pts.append([round(float(v), 1) for v in p] + ["y" if h.state == "降伏" else "c"])
        data["hinges"][d] = pts
    opts = "".join(f'<option value="{escape(k)}">{escape(k)}</option>' for k in data["cases"])
    hopt = "".join(f'<option value="{escape(k)}">{escape(k)}</option>' for k in data["hinges"])
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    hsel = (f'<label>ヒンジ <select id="v-hinge"><option value="">表示しない</option>{hopt}</select></label>'
            if data["hinges"] else "")
    return (f'<div class="viewer screen-only"><div class="ctl">'
            f'<label>変形 <select id="v-case"><option value="">表示しない</option>{opts}</select></label>'
            f'<label>倍率 <input id="v-scale" type="range" min="0.2" max="5" step="0.1" value="1"></label>'
            f'{hsel}<span id="v-info" class="mute"></span></div>'
            f'<canvas id="viewer"></canvas>'
            f'<div class="legend"><span><i class="sw" style="background:#1f4e79"></i>柱</span>'
            f'<span><i class="sw" style="background:#5b8c3a"></i>梁</span>'
            f'<span><i class="sw" style="background:#e69f00"></i>ひび割れ</span>'
            f'<span><i class="sw" style="background:#c0392b"></i>降伏</span></div></div>'
            f'<script type="application/json" id="model-data">{payload}</script>')


def _frames(bld, axis):
    """加力方向 axis（0 = X, 1 = Y）の架構の通り名"""
    return [n for n, _ in (bld.gy if axis == 0 else bld.gx)]


def sec_analysis(bld, ctx):
    m = bld.model
    out = ["<h3>3次元モデルと変形図</h3>", Raw(_viewer(bld, ctx)),
           '<p class="note print-only">画面表示では 3次元モデルを回転して変形図を確認できます。</p>']
    diagrams = []
    if ctx["rL"] is not None:
        diagrams.append((ctx["rL"], "長期（" + "+".join(ctx["long"]) + "）", (0, 1)))
    for c, axis in (("Ex", 0), ("Ey", 1)):
        if c in ctx["lin"]:
            diagrams.append((ctx["lin"][c], f"地震 {c}", (axis,)))
    for r, title, axes in diagrams:
        mmax, hmin = 0.0, min(b[1] - a[1] for a, b in zip(bld.floors, bld.floors[1:]))
        for e in m.elements.values():
            for x in (0.0, e.rigid_i, e.L / 2, e.L - e.rigid_j, e.L):
                s = r.section_forces_at(e.id, x)
                mmax = max(mmax, abs(s[4]), abs(s[5]))
        scale = 0.3 * hmin / mmax if mmax else 0.0
        out.append(f"<h3>曲げモーメント図: {escape(title)}</h3>")
        out.append('<p class="note">引張側に描画、数値はフェイス位置（梁は中央も）の曲げモーメント [kN·m]</p>')
        for axis in axes:
            for ln in _frames(bld, axis):
                out.append(figure(svg.moment_elevation(bld, ln, r, scale),
                                  f"{ln} 通り（{'X' if axis == 0 else 'Y'} 方向の架構） {title}"))
    rows = []
    sup = sorted({n for r in ctx["lin"].values() for n in r.reactions})
    cols = [(c, k) for c in ([ctx["rL"].case] if ctx["rL"] is not None else []) for k in (2,)]
    cols += [(c, k) for c, k in (("Ex", 0), ("Ex", 2), ("Ey", 1), ("Ey", 2)) if c in ctx["lin"]]
    res = dict(ctx["lin"])
    if ctx["rL"] is not None:
        res[ctx["rL"].case] = ctx["rL"]
    for n in sup:
        rows.append([bld.label("node", n)] + [kN(res[c].reactions[n][k]) for c, k in cols])
    out.append("<h3>支点反力 [kN]</h3>")
    out.append(table(["支点"] + [f"{c} {'RxRyRz'[2 * k:2 * k + 2]}" for c, k in cols], rows,
                     num=range(1, len(cols) + 1)))
    return "".join(out)


def sec_drift(bld, ctx):
    lin = ctx["lin"]
    rows = []
    for c, d in (("Ex", 0), ("Ey", 1)):
        if c not in lin or len(bld.masters) != len(bld.floors) - 1:
            continue
        for name, du, h, ang in lin[c].story_drifts(bld.drift_levels(), d):
            rows.append([c, name, f"{du:.2f}", f"{h:,.0f}", inv(ang), judge(abs(ang) <= 1 / 200)])
    if not rows:
        return None
    out = ["<h3>層間変形角（≦ 1/200）</h3>",
           table(["ケース", "層", "層間変位 [mm]", "階高 [mm]", "層間変形角", "判定"], rows, num=(2, 3))]
    if ctx.get("ecc"):
        out.append("<h3>剛性率・偏心率（令82条の6）</h3>")
        out.append(table(["層", "方向", "rs = h/δ", "Rs", "Fs", "重心 (x, y)", "剛心 (x, y)", "弾力半径",
                          "偏心距離", "Re", "Fe", "判定"],
                         [[r["story"], r["dir"], f"{r['rs']:,.0f}", f"{r['Rs']:.2f}", f"{r['Fs']:.2f}",
                           f"({r['xg']:,.0f}, {r['yg']:,.0f})", f"({r['xk']:,.0f}, {r['yk']:,.0f})",
                           f"{r['re']:,.0f}", f"{r['e']:,.0f}", f"{r['Re']:.3f}", f"{r['Fe']:.2f}",
                           judge(r["Rs"] >= 0.6 and r["Re"] <= 0.15)] for r in ctx["ecc"]],
                         num=(2, 3, 4, 7, 8, 9, 10)))
        out.append('<p class="note">判定は剛性率 Rs ≧ 0.6 かつ偏心率 Re ≦ 0.15（満たさない場合は Fes で割増し）</p>')
    return "".join(out)


def sec_check(bld, ctx):
    rep = ctx.get("check")
    if rep is None:
        return None
    ng = rep.ng
    long = "+".join(rep.long)
    short = ", ".join(long + "±" + c for c in rep.seismic) or "なし"
    badge = '<span class="badge ok">OK</span>' if not ng else '<span class="badge ng">NG あり</span>'
    out = [f"<p>最大検定比 <b>{rep.max_ratio:.2f}</b>、NG {len(ng)} 箇所 {badge}　"
           f"長期: {escape(long)}、短期: {escape(short)}、"
           f"短期せん断 QD = QL + {rep.shear_factor:g}·QE</p>"]
    g = rep.governing(("eid", "pos", "direction", "item"))
    rows, cls = [], []
    for _, r in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] != "曲げ")):
        k = kNm if r.item == "曲げ" else kN
        rows.append([r.eid, r.label, r.pos, r.direction or "-", r.item, r.term, r.combo, kN(r.N),
                     k(r.demand), k(r.capacity), f"{r.ratio:.2f}", judge(r.ok), r.note])
        cls.append("" if r.ok else "ng")
    out.append("<h3>部材・位置ごとの最大検定比</h3>")
    out.append('<p class="note">曲げは M・Ma [kN·m]、せん断は QD・QA [kN]、N は圧縮正 [kN]</p>')
    out.append(table(["要素", "部材", "位置", "方向", "検定", "長短", "組合せ", "N", "応力", "許容", "検定比",
                      "判定", "備考"], rows, num=(0, 7, 8, 9, 10), row_cls=cls))
    ratios = {}
    for r in rep.rows:
        ratios[r.eid] = max(ratios.get(r.eid, 0.0), r.ratio)
    out.append("<h3>軸組図: 部材ごとの最大検定比</h3>")
    out.append('<div class="legend"><span><i class="sw" style="background:#9cc3e6"></i>≦ 0.5</span>'
               '<span><i class="sw" style="background:#3b82c4"></i>≦ 0.8</span>'
               '<span><i class="sw" style="background:#e69f00"></i>≦ 1.0</span>'
               '<span><i class="sw" style="background:#c0392b"></i>&gt; 1.0</span></div>')
    for axis in (0, 1):
        for ln in _frames(bld, axis):
            out.append(figure(svg.ratio_elevation(bld, ln, ratios), f"{ln} 通り"))
    if rep.warnings:
        out.append("<h3>構造規定</h3><ul class='list'>" +
                   "".join(f"<li>{escape(w)}</li>" for w in rep.warnings) + "</ul>")
    return "".join(out)


def sec_pushover(bld, ctx):
    pos = ctx.get("pushovers")
    if not pos:
        return None
    nl = bld.nonlinear
    out = [f'<p>材端ばねモデル（トリリニア・最大点指向型）、事象追跡型の荷重増分法。σy = '
           f'{nl["sigma_y_factor"]:g}F、降伏後の剛性比 {nl["post_yield_ratio"]:g}。長期荷重を載荷後、'
           f'Ai 分布の水平力を比例的に増加。</p>']
    ult = ctx.get("ult")
    rows = bld.seismic["rows"]
    for d, po in pos.items():
        axis = 0 if d.lstrip("+-").lower() == "x" else 1
        out.append(f"<h3>{escape(d)} 方向</h3>")
        series = []
        for k, lv in enumerate(po.levels):
            pts = [(0.0, 0.0)] + [(s["drifts"][k], s["lam"] * rows[k]["Q"]) for s in po.lateral]
            series.append((f"{lv.name}", pts, f"c{k % 6}"))
        marks = []
        if ult:
            st = next((s for s in ult.stories if s.direction == d), None)
            if st:
                marks.append((st.drift, f"Qu（{st.reason}）"))
        out.append(figure(svg.curve_chart(series, marks), f"{d} 方向 層せん断力–層間変形角"))
        ev = [e for e in po.events if e["event"] in ("降伏", "せん断破壊") and e["phase"] == "水平"]
        if ev:
            out.append(table(["順", "step", "部材端", "事象", "ベースシア [kN]", "最大層間変形角"],
                             [[k + 1, e["step"], e["label"], e["event"], kN(e["Q"]), inv(e["drift"])]
                              for k, e in enumerate(ev)], num=(0, 1, 4)))
        if po.mechanism_step:
            out.append(f"<p>崩壊機構の形成: step {po.mechanism_step}（{escape(', '.join(po.mechanism_stories))}）</p>")
        for ln in _frames(bld, axis):
            out.append(figure(svg.hinge_elevation(bld, ln, po, axis),
                              f"{ln} 通り {d} 方向 最終状態（● 降伏（数字は降伏の順番）、● ひび割れ）"))
    return "".join(out)


def sec_ultimate(bld, ctx):
    ult = ctx.get("ult")
    if ult is None:
        return None
    s = ult.settings
    out = [f'<p>判定 <span class="badge {"ok" if ult.ok else "ng"}">{"すべて OK" if ult.ok else "NG あり"}</span>　'
           f'Qsu: 荒川 {escape(s["qsu"])} 式、曲げ破壊先行の条件 Qsu ≧ {s["qsu_factor"]:g}·Qmu、'
           f'Qu 判定の層間変形角 1/{1 / s["drift_limit"]:.0f}'
           f'{"、柱のせん断破壊で Qu を決定" if s["stop_at_column_shear"] else ""}</p>']
    rows, cls = [], []
    for r in ult.stories:
        rows.append([r.direction, r.story, kN(r.Qu), kN(r.Qud), r.group, f"{r.Ds:.2f}", f"{r.Fs:.2f}",
                     f"{r.Fe:.2f}", f"{r.Fes:.2f}", kN(r.Qun), f"{r.ratio:.2f}", judge(r.ok),
                     f"{r.reason}（step {r.step}, {inv(r.drift)}）", r.mode])
        cls.append("" if r.ok else "ng")
    out.append("<h3>保有水平耐力と必要保有水平耐力</h3>")
    out.append(table(["方向", "層", "Qu [kN]", "Qud [kN]", "部材群", "Ds", "Fs", "Fe", "Fes", "Qun [kN]",
                      "Qu/Qun", "判定", "Qu の決定", "崩壊形"], rows, num=range(2, 11), row_cls=cls))
    for dname, mems in ult.members.items():
        eff = ult.effective.get(dname, {})
        out.append(f"<h3>部材の破壊形式と種別（{dname} 方向）</h3>")
        f = lambda v, fmt: format(v, fmt) if not math.isnan(v) else "-"
        out.append(table(["部材", "Qmu [kN]", "Qsu [kN]", "Qsu/Qmu", "破壊形式", "h0/D", "σ0/Fc", "pt [%]",
                          "τu/Fc", "種別", "梁を考慮"],
                         [[mj.label, kN(mj.Qmu), kN(mj.Qsu), f"{mj.ratio:.2f}",
                           "せん断" if mj.shear_failure else "曲げ", f(mj.h0_D, ".2f"), f(mj.s0_Fc, ".3f"),
                           f(mj.pt * 100, ".2f") if not math.isnan(mj.pt) else "-", f(mj.tau_Fc, ".3f"),
                           mj.cls, eff.get(mj.eid, "-") if mj.kind == "柱" else "-"] for mj in mems],
                         num=range(1, 9)))
    return "".join(out)


def sec_appendix(bld, ctx):
    out = ["<h3>規準書・告示での照合が必要な事項</h3><ul class='list'>" +
           "".join(f"<li>{escape(x)}</li>" for x in VERIFY_ITEMS) + "</ul>",
           "<h3>既知の簡略化</h3><ul class='list'>" +
           "".join(f"<li>{escape(x)}</li>" for x in KNOWN_SIMPLIFICATIONS) + "</ul>"]
    if ctx["warnings"]:
        out.append("<h3>警告</h3><ul class='list'>" +
                   "".join(f"<li>{escape(w)}</li>" for w in ctx["warnings"]) + "</ul>")
    if bld.source_text:
        out.append(f'<h3>入力データ</h3><pre class="src">{escape(bld.source_text)}</pre>')
    return "".join(out)


SECTIONS = (
    ("sec-model", "建物概要", sec_model),
    ("sec-loads", "荷重", sec_loads),
    ("sec-analysis", "応力解析", sec_analysis),
    ("sec-drift", "層間変形角・剛性率・偏心率", sec_drift),
    ("sec-check", "断面検定（RC規準 2018）", sec_check),
    ("sec-pushover", "静的非線形解析（プッシュオーバー）", sec_pushover),
    ("sec-ultimate", "保有水平耐力（令82条の3）", sec_ultimate),
    ("sec-appendix", "付録", sec_appendix),
)


def build_report(bld, ctx=None):
    """計算書の HTML（文字列）を返す"""
    ctx = ctx or _compute(bld)
    parts, toc = [], []
    for k, (sid, title, fn) in enumerate(SECTIONS, 1):
        body = fn(bld, ctx)
        if body is None:
            continue
        n = len(toc) + 1
        toc.append(f'<li><a href="#{sid}">{escape(title)}</a></li>')
        parts.append(f'<section id="{sid}"><h2>{n}. {escape(title)}</h2>{body}</section>')
    title = bld.report.get("title") or f"構造計算書 {bld.name}"
    return ("<!DOCTYPE html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{CSS}</style></head><body><main>"
            f"{_cover(bld)}<nav class=\"toc\"><b>目次</b><ol>{''.join(toc)}</ol></nav>"
            f"{''.join(parts)}</main><script>{JS}</script></body></html>")


def write_report(bld, path):
    """計算書を HTML ファイルに書き出してパスを返す"""
    path = Path(path)
    path.write_text(build_report(bld), encoding="utf-8")
    return path


def main(argv=None):
    from .io_yaml import load_building
    ap = argparse.ArgumentParser(description="YAML 入力から計算書（HTML）を作成する")
    ap.add_argument("input", help="入力 YAML ファイル")
    ap.add_argument("-o", "--output", help="出力 HTML（省略時は入力名_report.html）")
    a = ap.parse_args(argv)
    out = a.output or str(Path(a.input).with_name(Path(a.input).stem + "_report.html"))
    p = write_report(load_building(a.input), out)
    print(f"計算書を出力しました: {p}")


if __name__ == "__main__":
    main()
