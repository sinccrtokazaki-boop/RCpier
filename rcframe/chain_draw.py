# -*- coding: utf-8 -*-
"""変状連鎖図の描画（調書用の SVG/PNG と、クリックで経路をたどる公開ページ）.

道路橋の変状連鎖図（rc_fukugen/tailrisk/chain_draw.py）と同じ描き方で、データ（spec）を引数で受け取る。
spec はモジュール（例: rcframe.pier_chain）で、次を持つ:
    NODES, EDGES, LANES, COLS, COL_SUB, KINDS, SCENARIOS, flat,
    TITLE, STEM, CODE_PREFIX, CODES（点検の番号 → 内容）, CODE_NOTE, FLAGS（印 → (記号, 説明)）, EXTRA_FACTORS, PAGE
次は省略できる（括弧内は省略時の値）:
    EDGE_KINDS（p：進展, t：引き金となる外力, c：連成）, FACTOR_KINDS（act, env, des）, END_KINDS（acc）,
    ROUND_KINDS（= FACTOR_KINDS）, BOLD_KINDS・BOLD_LABEL_KINDS（fail, acc）, COL_GROUPS（なし）,
    CODE_FMT（2 桁の数字）, CODE_SEP（・）, CODE_HEAD（部位, 主な点検の内容）, SOURCE（なし）
配置: 帯 = 部位（LANES）、列 = 段階（COLS）。
"""
import html
import json
import os
import subprocess
import tempfile

# ------------------------------------------------------------------ 配置
W, PADX = 172, 9          # ノードの幅・左右の余白
G = 46                    # 段の間隔
GUT = 104                 # 左の帯の名前の幅
HEAD = 62                 # 上の段の名前の高さ
GROUP_H = 24              # 列のまとまりの見出しの高さ（COL_GROUPS があるとき）
LH, FS, FS2 = 16.5, 12.5, 10.5
LANE_PAD, NODE_GAP = 16, 11

EDGE_KINDS = {"p": "進展", "t": "引き金となる外力", "c": "連成（互いに進め合う）"}
# 矢印の描き方：色の名前（FIXED と CSS の変数）, 破線, 線の太さ
EDGE_STYLE = {"p": ("edge", "", 1.3), "t": ("trig", "5 4", 1.3), "s": ("trig", "4 3", 1.4),
              "c": ("cpl", "9 3 2 3", 1.8)}


def _opt(spec):
    """spec の省略できる設定."""
    factor = tuple(getattr(spec, "FACTOR_KINDS", ("act", "env", "des")))
    return dict(edge_kinds=getattr(spec, "EDGE_KINDS", EDGE_KINDS), factor=factor,
                end=tuple(getattr(spec, "END_KINDS", ("acc",))),
                round=tuple(getattr(spec, "ROUND_KINDS", factor)),
                bold=tuple(getattr(spec, "BOLD_KINDS", ("fail", "acc"))),
                bold_label=tuple(getattr(spec, "BOLD_LABEL_KINDS", ("fail", "acc"))),
                groups=list(getattr(spec, "COL_GROUPS", [])))


def head_h(spec):
    return HEAD + (GROUP_H if getattr(spec, "COL_GROUPS", None) else 0)


def wrap(text, limit=12.2):
    """ラベルの改行（\\n）を守り、さらに全角 1, 半角 0.55 として limit を超える行だけ折り返す
    （行頭禁則：）、。・、行末禁則：（）."""
    out = []
    for part in text.split("\n"):
        cur, w = "", 0.0
        for ch in part:
            cw = 0.55 if ord(ch) < 0x2000 else 1.0
            if w + cw > limit and cur:
                if ch in "）、。・」":
                    out.append(cur + ch)
                    cur, w = "", 0.0
                    continue
                if cur.endswith("（"):
                    out.append(cur[:-1])
                    cur, w = "（", 1.0
                else:
                    out.append(cur)
                    cur, w = "", 0.0
            cur += ch
            w += cw
        if cur:
            out.append(cur)
    return out


def code_str(spec, c):
    """点検の番号の表示（数値は CODE_FMT、文字はそのまま）"""
    return getattr(spec, "CODE_FMT", "{:02d}").format(c) if isinstance(c, int) else str(c)


def code_text(spec, codes):
    """ノードの下に書く点検の表示"""
    return spec.CODE_PREFIX + " " + getattr(spec, "CODE_SEP", "・").join(code_str(spec, c) for c in codes)


def layout(spec):
    nodes = {}
    for nid, c, lane, kind, label, codes, flags, desc in spec.NODES:
        lines = wrap(label)
        h = 9 + len(lines) * LH + (15 if codes else 0) + 8
        nodes[nid] = dict(id=nid, c=c, lane=lane, kind=kind, label=label, lines=lines, codes=codes, flags=flags,
                          desc=desc, h=h, w=W)
    lane_top, y = [], head_h(spec)
    for L in range(len(spec.LANES)):
        stacks = []
        for c in range(len(spec.COLS)):
            col = [n for n in nodes.values() if n["lane"] == L and n["c"] == c]
            stacks.append((col, sum(n["h"] for n in col) + NODE_GAP * max(len(col) - 1, 0)))
        lh = max(s for _, s in stacks) + 2 * LANE_PAD
        for c, (col, s) in enumerate(stacks):
            yy = y + (lh - s) / 2
            for n in col:
                n["x"] = GUT + 14 + c * (W + G)
                n["y"] = yy
                yy += n["h"] + NODE_GAP
        lane_top.append((y, lh))
        y += lh
    width = GUT + 14 + len(spec.COLS) * W + (len(spec.COLS) - 1) * G + 30
    return nodes, lane_top, width, y + 18


def ports(spec, nodes):
    """入る矢印・出る矢印の端を、相手の高さの順にノードの辺に並べる."""
    out_l, in_l = {}, {}
    for s, t, k in spec.EDGES:
        out_l.setdefault(s, []).append(t)
        in_l.setdefault(t, []).append(s)
    pos = {}
    for nid, lst in out_l.items():
        n = nodes[nid]
        lst = sorted(lst, key=lambda t: nodes[t]["y"] + nodes[t]["h"] / 2)
        for k, t in enumerate(lst):
            pos[(nid, t, "out")] = n["y"] + n["h"] * (k + 1) / (len(lst) + 1)
    for nid, lst in in_l.items():
        n = nodes[nid]
        lst = sorted(lst, key=lambda s: nodes[s]["y"] + nodes[s]["h"] / 2)
        for k, s in enumerate(lst):
            pos[(s, nid, "in")] = n["y"] + n["h"] * (k + 1) / (len(lst) + 1)
    return pos


# 固定色（調書・スライド用 SVG）。ページは CSS の変数で同じ役割の色を使う
FIXED = dict(bg="#ffffff", laneA="#f4f6f3", laneB="#ffffff", ink="#1e2528", ink2="#5d676c", rule="#cfd4cf",
             edge="#7d878c", trig="#c0582e", cpl="#a8327a",
             act=("#fbe3d6", "#c0582e", "#1e2528"), env=("#dcefe9", "#2f7d6b", "#1e2528"),
             des=("#e7e4f4", "#5b4fa0", "#1e2528"), dmg=("#ffffff", "#8a9499", "#1e2528"),
             perf=("#fff3d1", "#a87b12", "#1e2528"), fail=("#f8dcd9", "#a33a2e", "#1e2528"),
             acc=("#2b3034", "#2b3034", "#ffffff"),
             ins=("#dde2e4", "#6b767b", "#1e2528"), non=("#ffffff", "#1e2528", "#1e2528"),
             out=("#2b3034", "#2b3034", "#ffffff"),
             fk="#2c5a78", lc="#8a4b14", rc="#2e6b3f")


ROUTE_M, ROUTE_STEP = 7, 5        # 列を飛ばす矢印：箱からの離れ, 同じすき間を通る矢印どうしの間隔


def _free_bands(nodes):
    """列ごとの、箱が占める高さの区間（離れを含めて重なりをまとめたもの）"""
    bands = {}
    for n in nodes.values():
        bands.setdefault(n["c"], []).append([n["y"] - ROUTE_M, n["y"] + n["h"] + ROUTE_M])
    for c, lst in bands.items():
        lst.sort()
        merged = [lst[0]]
        for a, b in lst[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        bands[c] = merged
    return bands


def _edge_path(S, T, sy, ty, bands, used):
    """矢印の経路。隣の列へはなめらかな曲線。列を飛ばすときは、途中の列で箱のすき間を水平に通る
    （直線で結んだ高さが箱にかかれば、近い方の箱の上か下へずらす。同じすき間は ROUTE_STEP ずつ離す）."""
    sx, tx = S["x"] + S["w"], T["x"]
    if S["c"] == T["c"]:
        tx = T["x"] + T["w"]
        bend = sx + 20 + 6 * abs(T["lane"] - S["lane"])
        return f"M{sx:.1f},{sy:.1f} C{bend:.1f},{sy:.1f} {bend:.1f},{ty:.1f} {tx + 2:.1f},{ty:.1f}"
    pts = [(sx, sy)]
    for c in range(S["c"] + 1, T["c"]):
        xl = GUT + 14 + c * (W + G)
        xm = xl + W / 2
        y = sy + (ty - sy) * (xm - sx) / (tx - sx)
        for a, b in bands.get(c, []):
            if a < y < b:
                edge_y, sign = (a, -1) if y - a <= b - y else (b, 1)
                k = used.get((c, edge_y), 0)
                used[(c, edge_y)] = k + 1
                y = edge_y + sign * ROUTE_STEP * k
                break
        pts += [(xl - 6, y), (xl + W + 6, y)]
    pts.append((tx - 1, ty))
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if abs(y1 - y0) < 0.05:
            d.append(f"L{x1:.1f},{y1:.1f}")
        else:
            h = (x1 - x0) / 2
            d.append(f"C{x0 + h:.1f},{y0:.1f} {x1 - h:.1f},{y1:.1f} {x1:.1f},{y1:.1f}")
    return " ".join(d)


def _edge_attrs(k, mode):
    colname, dash, sw = EDGE_STYLE[k]
    paint = f'stroke="{FIXED[colname]}" fill="none"' if mode == "file" else f'class="edge e-{k}" fill="none"'
    return paint + f' stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "")


def svg(spec, mode="file"):
    """mode="file"：色を属性で固定。"page"：class を付け、色はページの CSS に任せる."""
    nodes, lane_top, width, height = layout(spec)
    pos = ports(spec, nodes)
    F = FIXED
    O = _opt(spec)
    off = head_h(spec) - HEAD
    fam = "Meiryo, 'Yu Gothic', 'Hiragino Sans', sans-serif" if mode == "file" else "var(--font-body)"
    ink = (f'fill="{F["ink"]}"' if mode == "file" else 'class="t-ink"')
    ink2 = (f'fill="{F["ink2"]}"' if mode == "file" else 'class="t-ink2"')
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}" '
         f'font-family="{fam}" role="img" aria-label="{html.escape(spec.TITLE)}" id="chain">']
    o.append("<defs>")
    for nm in sorted(set(O["edge_kinds"]) | {"p"}):
        col = F[EDGE_STYLE[nm][0]]
        fill = f'fill="{col}"' if mode == "file" else f'class="ah-{nm}"'
        o.append(f'<marker id="ah-{nm}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                 f'<path d="M0,0 L10,5 L0,10 z" {fill}/></marker>')
    o.append("</defs>")
    if mode == "file":
        o.append(f'<rect width="100%" height="100%" fill="{F["bg"]}"/>')
    # 帯
    for L, (y, lh) in enumerate(lane_top):
        fill = (F["laneA"] if L % 2 == 0 else F["laneB"])
        o.append(f'<rect x="0" y="{y:.1f}" width="{width:.0f}" height="{lh:.1f}" '
                 + (f'fill="{fill}"' if mode == "file" else f'class="lane lane-{L % 2}"') + "/>")
        o.append(f'<line x1="0" x2="{width:.0f}" y1="{y:.1f}" y2="{y:.1f}" '
                 + (f'stroke="{F["rule"]}"' if mode == "file" else 'class="rule"') + ' stroke-width="1"/>')
        lines = wrap(spec.LANES[L], 7.5)
        y0 = y + lh / 2 - (len(lines) - 1) * 9
        for k, s in enumerate(lines):
            o.append(f'<text x="12" y="{y0 + k * 18:.1f}" font-size="13.5" font-weight="700" dominant-baseline="middle" '
                     f'{ink}>{html.escape(s)}</text>')
    # 列のまとまりの見出し（例：文献の図の「変状」）
    for a, b, name in O["groups"]:
        x1, x2 = GUT + 14 + a * (W + G), GUT + 14 + b * (W + G) + W
        xm, half = (x1 + x2) / 2, 9 * len(name) + 14
        o.append(f'<path d="M{x1:.1f},{27} L{x1:.1f},{19} L{xm - half:.1f},{19} M{xm + half:.1f},{19} L{x2:.1f},{19} '
                 f'L{x2:.1f},{27}" fill="none" stroke-width="1.2" '
                 + (f'stroke="{F["ink2"]}"' if mode == "file" else 'class="s-ink2"') + "/>")
        o.append(f'<text x="{xm:.1f}" y="{19.5:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="15" '
                 f'font-weight="700" {ink}>{html.escape(name)}</text>')
    # 段の名前（改行があれば 2 行）
    for c, name in enumerate(spec.COLS):
        x = GUT + 14 + c * (W + G) + W / 2
        parts = name.split("\n")
        for k, s in enumerate(parts):
            yy = off + 26 + (k - (len(parts) - 1) / 2) * 17
            o.append(f'<text x="{x:.1f}" y="{yy:.1f}" text-anchor="middle" font-size="14.5" font-weight="700" '
                     f'{ink}>{html.escape(s)}</text>')
        if spec.COL_SUB[c]:
            o.append(f'<text x="{x:.1f}" y="{off + 47}" text-anchor="middle" font-size="10.5" '
                     f'{ink2}>{html.escape(spec.COL_SUB[c])}</text>')
        if c < len(spec.COLS) - 1:
            xa = GUT + 14 + c * (W + G) + W + 8
            o.append(f'<path d="M{xa:.1f},{off + 22} l{G - 16},0" stroke-width="1.4" '
                     + (f'stroke="{F["ink2"]}"' if mode == "file" else 'class="s-ink2"') + ' marker-end="url(#ah-p)"/>')
    # 矢印
    o.append('<g id="edges">')
    bands, used = _free_bands(nodes), {}
    for s, t, k in spec.EDGES:
        S, T = nodes[s], nodes[t]
        d = _edge_path(S, T, pos[(s, t, "out")], pos[(s, t, "in")], bands, used)
        o.append(f'<path d="{d}" {_edge_attrs(k, mode)} marker-end="url(#ah-{k})" data-s="{s}" data-t="{t}"/>')
    o.append("</g>")
    # ノード
    o.append('<g id="nodes">')
    for n in nodes.values():
        fill, stroke, txt = F[n["kind"]]
        x, y, h = n["x"], n["y"], n["h"]
        rx = 9 if n["kind"] in O["round"] else 3
        sw = 2.2 if n["kind"] in O["bold"] else 1.2
        o.append(f'<g class="node k-{n["kind"]}" data-id="{n["id"]}" tabindex="0" role="button" '
                 f'aria-label="{html.escape(spec.flat(n["label"]))}">')
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{W}" height="{h:.1f}" rx="{rx}" stroke-width="{sw}" '
                 + (f'fill="{fill}" stroke="{stroke}"' if mode == "file" else 'class="box"') + "/>")
        for k, s in enumerate(n["lines"]):
            o.append(f'<text x="{x + PADX:.1f}" y="{y + 9 + (k + 0.5) * LH:.1f}" font-size="{FS}" dominant-baseline="middle" '
                     + (f'fill="{txt}"' if mode == "file" else 'class="lbl"')
                     + (' font-weight="700"' if n["kind"] in O["bold_label"] else "") + f'>{html.escape(s)}</text>')
        if n["codes"]:
            o.append(f'<text x="{x + PADX:.1f}" y="{y + h - 12:.1f}" font-size="{FS2}" dominant-baseline="middle" '
                     + (f'fill="{F["ink2"]}"' if mode == "file" else 'class="code"')
                     + f'>{html.escape(code_text(spec, n["codes"]))}</text>')
        bx = x + W - 11
        for f_ in n["flags"]:
            mark = spec.FLAGS[f_][0]
            o.append(f'<circle cx="{bx:.1f}" cy="{y + 2:.1f}" r="8.5" '
                     + (f'fill="{F[f_]}"' if mode == "file" else f'class="flag f-{f_}"') + "/>")
            o.append(f'<text x="{bx:.1f}" y="{y + 2.5:.1f}" font-size="10" font-weight="700" text-anchor="middle" '
                     f'dominant-baseline="middle" fill="#ffffff">{mark}</text>')
            bx -= 20
        o.append("</g>")
    o.append("</g></svg>")
    return "\n".join(o), nodes, width, height


def legend_svg(spec):
    """調書用の凡例（SVG の下に付ける）。1 行目：種類と矢印、2 行目：印、3 行目以降：点検の注記と出典."""
    F = FIXED
    O = _opt(spec)
    o, x = [], 14
    for k, name in spec.KINDS.items():
        fill, stroke, _ = F[k]
        rx = 6 if k in O["round"] else 3
        sw = 2.0 if k in O["bold"] else 1.2
        o.append(f'<rect x="{x}" y="8" width="22" height="14" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
                 f'<text x="{x + 28}" y="15.5" font-size="12" dominant-baseline="middle" fill="{F["ink"]}">{html.escape(name)}</text>')
        x += 28 + 12.5 * len(name) + 18
    for k, name in O["edge_kinds"].items():
        o.append(f'<path d="M{x},15 l36,0" {_edge_attrs(k, "file")} marker-end="url(#ah-{k})"/>'
                 f'<text x="{x + 44}" y="15.5" font-size="12" dominant-baseline="middle" fill="{F["ink"]}">{html.escape(name)}</text>')
        x += 44 + 12.5 * len(name) + 22
    x_end = x
    x = 14
    for f_, (mark, name) in spec.FLAGS.items():
        o.append(f'<circle cx="{x + 8}" cy="42" r="8.5" fill="{F[f_]}"/><text x="{x + 8}" y="42.5" font-size="10" '
                 f'font-weight="700" text-anchor="middle" dominant-baseline="middle" fill="#fff">{mark}</text>'
                 f'<text x="{x + 22}" y="42.5" font-size="12" dominant-baseline="middle" fill="{F["ink"]}">{html.escape(name)}</text>')
        x += 22 + 12.5 * len(name) + 22
    notes = [spec.CODE_NOTE] + ([f"出典：{spec.SOURCE}"] if getattr(spec, "SOURCE", "") else [])
    for i, t in enumerate(notes):
        o.append(f'<text x="14" y="{66.5 + 18 * i}" font-size="11" dominant-baseline="middle" fill="{F["ink2"]}">'
                 f'{html.escape(t)}</text>')
    height = 66.5 + 18 * (len(notes) - 1) + 14
    return "\n".join(o), max([x_end, x] + [14 + 11 * len(t) for t in notes]), height


LEGEND_H = 80                     # 凡例の高さ（3 行。出典があれば 1 行ふえる）


def full_svg(spec):
    """凡例を下に付けた 1 枚の SVG（文字列）と大きさ."""
    body, nodes, width, height = svg(spec, "file")
    leg, lw, lh = legend_svg(spec)
    leg_h = max(LEGEND_H, lh)
    total_h = height + leg_h
    head, rest = body.split(">", 1)
    head = head.replace(f'height="{height:.0f}"', f'height="{total_h:.0f}"').replace(
        f'viewBox="0 0 {width:.0f} {height:.0f}"', f'viewBox="0 0 {width:.0f} {total_h:.0f}"')
    rest = rest.replace("</svg>", f'<rect x="0" y="{height:.0f}" width="{width:.0f}" height="{leg_h:.0f}" fill="#ffffff"/>'
                                  f'<g transform="translate(0,{height:.0f})">{leg}</g></svg>')
    return head + ">" + rest, width, total_h


def write_file_outputs(spec, out_dir):
    """out_dir に <STEM>.svg と、Edge で描いた <STEM>.png（2 倍解像度）を書く."""
    full, width, total_h = full_svg(spec)
    p_svg = os.path.join(out_dir, spec.STEM + ".svg")
    with open(p_svg, "w", encoding="utf-8") as f:
        f.write(full)
    edge = next((p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"] if os.path.exists(p)), None)
    p_png = os.path.join(out_dir, spec.STEM + ".png")
    if edge:
        tmp = os.path.join(tempfile.gettempdir(), "henjo_rensa_render.html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"<!doctype html><meta charset='utf-8'><style>html,body{{margin:0;background:#fff}}</style>{full}")
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={int(width)},{int(total_h)}", f"--screenshot={p_png}", "file:///" + tmp.replace("\\", "/")],
                       capture_output=True, timeout=60)
    return p_svg, p_png if os.path.exists(p_png) else None


def page_data(spec):
    O = _opt(spec)
    nodes = {n[0]: dict(label=spec.flat(n[4]), c=spec.flat(spec.COLS[n[1]]), lane=spec.flat(spec.LANES[n[2]]),
                        kind=spec.KINDS[n[3]], k=n[3], codes=[code_str(spec, c) for c in n[5]],
                        flags=[spec.FLAGS[f][1] for f in n[6]], desc=n[7]) for n in spec.NODES}
    paths = spec.PAGE.get("paths") or [("この変状に至る要因", "up", list(O["factor"]) + ["*extra"]),
                                       ("行き着く破壊", "down", ["fail"]),
                                       ("行き着く重大事故・利用への影響", "down", list(O["end"]))]
    return dict(nodes=nodes, edges=[[s, t, k] for s, t, k in spec.EDGES], prefix=spec.CODE_PREFIX,
                sep=getattr(spec, "CODE_SEP", "・"), extra=list(spec.EXTRA_FACTORS),
                paths=[[lab, d, list(ks)] for lab, d, ks in paths],
                hint=spec.PAGE.get("hint", "ノード（四角）を押すと、その変状に至る要因と、その先への経路を強調する。"
                                           "上のボタンで重点シナリオの経路だけを見られる。"),
                scenarios=[dict(id=i, name=nm, members=m, note=nt) for i, nm, m, nt in spec.SCENARIOS])


PAGE_CSS = """
/* 図が主役の 1 枚もの：上に操作と凡例、中央に横スクロールの図、下に選んだノードの説明と表 */
:root {
  --paper: #f3f4f1; --sheet: #ffffff; --ink: #1e2528; --ink2: #5d676c; --rule: #cfd4cf;
  --laneA: #f4f6f3; --laneB: #ffffff; --edge: #7d878c; --trig: #c0582e; --cpl: #a8327a; --accent: #2c5a78;
  --act-f: #fbe3d6; --act-s: #c0582e; --env-f: #dcefe9; --env-s: #2f7d6b; --des-f: #e7e4f4; --des-s: #5b4fa0;
  --dmg-f: #ffffff; --dmg-s: #8a9499; --perf-f: #fff3d1; --perf-s: #a87b12; --fail-f: #f8dcd9; --fail-s: #a33a2e;
  --acc-f: #2b3034; --acc-s: #2b3034; --acc-t: #ffffff;
  --ins-f: #dde2e4; --ins-s: #6b767b; --non-f: #ffffff; --non-s: #1e2528; --out-f: #2b3034; --out-s: #2b3034; --out-t: #ffffff;
  --fk: #2c5a78; --lc: #8a4b14; --rc: #2e6b3f; --sel: #1f6fb2;
  --font-head: "Shippori Mincho B1", "Hiragino Mincho ProN", "Yu Mincho", serif;
  --font-body: "BIZ UDPGothic", "Hiragino Sans", "Yu Gothic", "Meiryo", sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #121619; --sheet: #191e22; --ink: #e5e8e6; --ink2: #9aa6ab; --rule: #34404a;
  --laneA: #1d2327; --laneB: #191e22; --edge: #8c979c; --trig: #e58a5f; --cpl: #e27bb8; --accent: #86b5d4;
  --act-f: #4a2e22; --act-s: #e58a5f; --env-f: #1f3a33; --env-s: #6cc0a8; --des-f: #2e2a45; --des-s: #a99ce6;
  --dmg-f: #232a2f; --dmg-s: #7d898f; --perf-f: #3d3318; --perf-s: #d9ad46; --fail-f: #45231f; --fail-s: #e88a7c;
  --acc-f: #e5e8e6; --acc-s: #e5e8e6; --acc-t: #15191c;
  --ins-f: #343e44; --ins-s: #93a0a6; --non-f: #191e22; --non-s: #e5e8e6; --out-f: #e5e8e6; --out-s: #e5e8e6; --out-t: #15191c;
  --fk: #4f8fbd; --lc: #b8742f; --rc: #4f9a63; --sel: #7fb8e6; color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #121619; --sheet: #191e22; --ink: #e5e8e6; --ink2: #9aa6ab; --rule: #34404a;
  --laneA: #1d2327; --laneB: #191e22; --edge: #8c979c; --trig: #e58a5f; --cpl: #e27bb8; --accent: #86b5d4;
  --act-f: #4a2e22; --act-s: #e58a5f; --env-f: #1f3a33; --env-s: #6cc0a8; --des-f: #2e2a45; --des-s: #a99ce6;
  --dmg-f: #232a2f; --dmg-s: #7d898f; --perf-f: #3d3318; --perf-s: #d9ad46; --fail-f: #45231f; --fail-s: #e88a7c;
  --acc-f: #e5e8e6; --acc-s: #e5e8e6; --acc-t: #15191c;
  --ins-f: #343e44; --ins-s: #93a0a6; --non-f: #191e22; --non-s: #e5e8e6; --out-f: #e5e8e6; --out-s: #e5e8e6; --out-t: #15191c;
  --fk: #4f8fbd; --lc: #b8742f; --rc: #4f9a63; --sel: #7fb8e6; color-scheme: dark; }
body { background: var(--paper); color: var(--ink); font-family: var(--font-body); font-size: 15px; line-height: 1.75; }
.wrap { max-width: 1480px; margin: 0 auto; padding-inline: 16px; padding-block: 22px 56px; display: grid; gap: 16px; }
header { display: grid; gap: 4px; }
.kicker { font-size: 12px; letter-spacing: .12em; color: var(--ink2); }
h1 { font-family: var(--font-head); font-weight: 800; font-size: clamp(22px, 3.2vw, 30px); margin: 0; text-wrap: balance; }
h2 { font-family: var(--font-head); font-weight: 800; font-size: 19px; margin: 8px 0 6px; text-wrap: balance; }
.lead { max-width: 52em; margin: 2px 0 0; }
.panel { background: var(--sheet); border: 1px solid var(--rule); padding: 14px 16px; min-width: 0; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.toolbar .lab { font-size: 13px; color: var(--ink2); margin-right: 4px; }
button.chip { font: inherit; font-size: 13.5px; padding: 5px 12px; border: 1px solid var(--rule); background: var(--sheet); color: var(--ink);
  border-radius: 999px; cursor: pointer; }
button.chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--sheet); }
button.chip:focus-visible { outline: 2px solid var(--sel); outline-offset: 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink2); margin-top: 10px; align-items: center; }
.legend span.sw { display: inline-block; width: 22px; height: 13px; border-radius: 3px; border: 1.3px solid; vertical-align: -2px; margin-right: 5px; }
.legend span.sw.rd { border-radius: 6px; } .legend span.sw.bd2 { border-width: 2.2px; }
.legend span.ln { display: inline-block; width: 30px; border-top: 1.6px solid var(--edge); vertical-align: 4px; margin-right: 5px; }
.legend span.ln.t, .legend span.ln.s { border-top: 1.6px dashed var(--trig); }
.legend span.ln.c { border-top: 2.2px dashed var(--cpl); }
.legend span.bd { display: inline-grid; place-items: center; width: 17px; height: 17px; border-radius: 50%; color: #fff; font-size: 10px; font-weight: 700; margin-right: 5px; }
.scroll { overflow-x: auto; }
#chain { display: block; max-width: none; height: auto; }
#chain .lane-0 { fill: var(--laneA); } #chain .lane-1 { fill: var(--laneB); } #chain .rule { stroke: var(--rule); }
#chain .t-ink { fill: var(--ink); } #chain .t-ink2 { fill: var(--ink2); } #chain .s-ink2 { stroke: var(--ink2); }
#chain .edge.e-p { stroke: var(--edge); } #chain .edge.e-t, #chain .edge.e-s { stroke: var(--trig); } #chain .edge.e-c { stroke: var(--cpl); }
#chain .ah-p { fill: var(--edge); } #chain .ah-t, #chain .ah-s { fill: var(--trig); } #chain .ah-c { fill: var(--cpl); }
#chain .lbl { fill: var(--ink); } #chain .code { fill: var(--ink2); }
#chain .k-act .box { fill: var(--act-f); stroke: var(--act-s); } #chain .k-env .box { fill: var(--env-f); stroke: var(--env-s); }
#chain .k-des .box { fill: var(--des-f); stroke: var(--des-s); } #chain .k-dmg .box { fill: var(--dmg-f); stroke: var(--dmg-s); }
#chain .k-perf .box { fill: var(--perf-f); stroke: var(--perf-s); } #chain .k-fail .box { fill: var(--fail-f); stroke: var(--fail-s); }
#chain .k-acc .box { fill: var(--acc-f); stroke: var(--acc-s); } #chain .k-acc .lbl { fill: var(--acc-t); }
#chain .k-ins .box { fill: var(--ins-f); stroke: var(--ins-s); } #chain .k-non .box { fill: var(--non-f); stroke: var(--non-s); }
#chain .k-out .box { fill: var(--out-f); stroke: var(--out-s); } #chain .k-out .lbl { fill: var(--out-t); }
#chain .f-fk { fill: var(--fk); } #chain .f-lc { fill: var(--lc); } #chain .f-rc { fill: var(--rc); }
#chain .node { cursor: pointer; transition: opacity .15s; } #chain .edge { transition: opacity .15s; }
#chain .node:focus { outline: none; } #chain .node:focus-visible .box, #chain .node.sel .box { stroke: var(--sel); stroke-width: 3.2; }
#chain.focus .node:not(.on) { opacity: .16; } #chain.focus .edge:not(.on) { opacity: .06; }
#chain.focus .edge.on { stroke-width: 2.2; }
@media (prefers-reduced-motion: reduce) { #chain .node, #chain .edge { transition: none; } }
.detail h3 { font-family: var(--font-head); font-size: 18px; margin: 0 0 4px; }
.detail .tags { display: flex; flex-wrap: wrap; gap: 6px; font-size: 12.5px; margin: 4px 0 8px; }
.detail .tags span { border: 1px solid var(--rule); padding: 0 8px; border-radius: 4px; color: var(--ink2); }
.detail p { margin: 4px 0; max-width: 60em; }
.detail .paths { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px 20px; margin-top: 8px; font-size: 14px; }
.detail .paths b { font-size: 13px; color: var(--ink2); font-weight: 700; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap: 16px; }
.sc { display: grid; gap: 10px; }
.sc div { border-top: 1px solid var(--rule); padding-top: 8px; }
.sc b { font-family: var(--font-head); }
table { border-collapse: collapse; font-size: 13.5px; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 8px; text-align: left; vertical-align: top; }
th { color: var(--ink2); font-weight: 700; font-size: 12.5px; }
td.no { white-space: nowrap; }
td.mk { text-align: center; }
.note { font-size: 13px; color: var(--ink2); }
ul.note { padding-left: 1.2em; margin: 4px 0; }
"""

PAGE_JS = """
(function () {
  const D = JSON.parse(document.getElementById('chain-data').textContent);
  const svg = document.getElementById('chain');
  const nodes = Array.from(svg.querySelectorAll('.node'));
  const edges = Array.from(svg.querySelectorAll('.edge'));
  const out = {}, inn = {};
  D.edges.forEach(([s, t]) => { (out[s] = out[s] || []).push(t); (inn[t] = inn[t] || []).push(s); });
  const walk = (start, adj) => { const seen = new Set(); const st = [start];
    while (st.length) { const v = st.pop(); (adj[v] || []).forEach(w => { if (!seen.has(w)) { seen.add(w); st.push(w); } }); } return seen; };
  const detail = document.getElementById('detail');
  const chips = Array.from(document.querySelectorAll('button.chip[data-sc]'));
  const esc = s => s.replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const name = i => esc(D.nodes[i].label) + (D.nodes[i].codes.length ? ' ' + esc(D.nodes[i].codes.join(D.sep)) : '');
  const names = ids => ids.map(name).join('、') || '—';
  function clear() {
    svg.classList.remove('focus'); nodes.forEach(n => n.classList.remove('on', 'sel')); edges.forEach(e => e.classList.remove('on'));
  }
  function press(id) { chips.forEach(c => c.setAttribute('aria-pressed', String(c.dataset.sc === id))); }
  function showAll() { clear(); press('all'); detail.innerHTML = '<p class="note">' + esc(D.hint) + '</p>'; }
  function selectNode(id) {
    clear(); press('');
    const up = walk(id, inn), down = walk(id, out);
    svg.classList.add('focus');
    nodes.forEach(n => { const k = n.dataset.id; if (k === id || up.has(k) || down.has(k)) n.classList.add('on'); if (k === id) n.classList.add('sel'); });
    edges.forEach(e => { const s = e.dataset.s, t = e.dataset.t;
      const upEdge = (up.has(s) || s === id) && (up.has(t) || t === id) && t !== s;
      const downEdge = (down.has(s) || s === id) && down.has(t);
      if (upEdge || downEdge) e.classList.add('on'); });
    const n = D.nodes[id];
    const order = Object.keys(D.nodes);
    const lists = D.paths.map(([lab, dir, kinds]) => {
      const pool = dir === 'up' ? up : down;
      const ids = order.filter(k => pool.has(k) && (kinds.includes(D.nodes[k].k) || (kinds.includes('*extra') && D.extra.includes(k))));
      return '<div><b>' + esc(lab) + '</b><br>' + names(ids) + '</div>';
    });
    detail.innerHTML = '<h3>' + esc(n.label) + '</h3><div class="tags"><span>' + esc(n.kind) + '</span><span>' + esc(n.lane) +
      '</span><span>' + esc(n.c) + '</span>' + n.codes.map(c => '<span>' + esc(D.prefix) + ' ' + esc(c) + '</span>').join('') +
      n.flags.map(f => '<span>' + esc(f) + '</span>').join('') + '</div><p>' + esc(n.desc) + '</p>' +
      '<div class="paths">' + lists.join('') + '</div>';
  }
  function selectScenario(sc) {
    clear(); press(sc.id); svg.classList.add('focus');
    const set = new Set(sc.members);
    nodes.forEach(n => { if (set.has(n.dataset.id)) n.classList.add('on'); });
    edges.forEach(e => { if (set.has(e.dataset.s) && set.has(e.dataset.t)) e.classList.add('on'); });
    detail.innerHTML = '<h3>' + esc(sc.name) + '</h3><p>' + esc(sc.note) + '</p>';
  }
  nodes.forEach(n => {
    n.addEventListener('click', () => selectNode(n.dataset.id));
    n.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectNode(n.dataset.id); } });
  });
  chips.forEach(c => c.addEventListener('click', () => {
    if (c.dataset.sc === 'all') showAll(); else selectScenario(D.scenarios.find(s => s.id === c.dataset.sc));
  }));
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape') showAll(); });
  showAll();
})();
"""


def _table(head, rows, classes=None):
    """表の HTML（rows はセルの文字列のリスト。classes は列ごとの td の class）"""
    classes = classes or []
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(
        f'<td{f" class={chr(34)}{classes[i]}{chr(34)}" if i < len(classes) and classes[i] else ""}>{html.escape(str(v))}</td>'
        for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{th}</tr>{body}</table></div>'


def write_page(spec, path):
    """クリックで経路をたどれる公開ページ用の HTML（本文だけ, Artifact で公開する）."""
    body, nodes, width, height = svg(spec, "page")
    data = page_data(spec)
    O = _opt(spec)
    P = spec.PAGE
    chips = '<button class="chip" type="button" data-sc="all" aria-pressed="true">すべて</button>' + "".join(
        f'<button class="chip" type="button" data-sc="{s["id"]}" aria-pressed="false">{html.escape(s["name"])}</button>'
        for s in data["scenarios"])
    sw = "".join(f'<span><span class="sw{" rd" if k in O["round"] else ""}{" bd2" if k in O["bold"] else ""}" '
                 f'style="background:var(--{k}-f);border-color:var(--{k}-s)"></span>{html.escape(nm)}</span>'
                 for k, nm in spec.KINDS.items())
    lns = "".join(f'<span><span class="ln {k}"></span>{html.escape(nm)}</span>' for k, nm in O["edge_kinds"].items())
    flags = "".join(f'<span><span class="bd" style="background:var(--{f})">{m}</span>{html.escape(nm)}</span>'
                    for f, (m, nm) in spec.FLAGS.items())
    legend = sw + lns + flags
    scen = "".join(f'<div><b>{html.escape(s["name"])}</b><br>{html.escape(s["note"])}</div>' for s in data["scenarios"])
    rows = []
    for code, what in spec.CODES.items():
        hits = [spec.flat(n[4]) for n in spec.NODES if code in n[5]]
        cells = list(what) if isinstance(what, (tuple, list)) else [what]
        rows.append([code_str(spec, code)] + cells + ["、".join(hits) or "―"])
    head = list(getattr(spec, "CODE_HEAD", ["部位", "主な点検の内容"])) + ["連鎖図の変状"]
    codes_table = _table(head, rows, ["no"])
    extra = "".join(
        f'<section class="panel"><h2>{html.escape(t["title"])}</h2>'
        f'{_table(t["head"], t["rows"], [""] + ["mk"] * (len(t["head"]) - 1))}'
        + (f'<p class="note">{html.escape(t["note"])}</p>' if t.get("note") else "") + "</section>"
        for t in P.get("tables", []))
    notes = "".join(f"<li>{html.escape(t)}</li>" for t in P["notes"])
    page = f"""<title>{html.escape(spec.TITLE)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Shippori+Mincho+B1:wght@800&family=BIZ+UDPGothic:wght@400;700&display=swap">
<style>{PAGE_CSS}</style>
<div class="wrap">
<header>
  <div class="kicker">{html.escape(P["kicker"])}</div>
  <h1>{html.escape(spec.TITLE)}</h1>
  <p class="lead">{html.escape(P["lead"])}</p>
</header>
<section class="panel">
  <div class="toolbar"><span class="lab">重点シナリオ</span>{chips}</div>
  <div class="legend">{legend}</div>
</section>
<section class="panel scroll" aria-label="{html.escape(spec.TITLE)}">{body}</section>
<section class="panel detail" id="detail" aria-live="polite"></section>
<section class="panel"><h2>{html.escape(P["scenario_title"])}</h2><div class="sc">{scen}</div></section>
<section class="panel"><h2>{html.escape(P["code_title"])}</h2>
  {codes_table}
  <p class="note">{html.escape(spec.CODE_NOTE)}</p>
</section>
{extra}
<section class="panel"><h2>作り方と、これから詰めること</h2>
<ul class="note">{notes}</ul>
</section>
</div>
<script type="application/json" id="chain-data">{json.dumps(data, ensure_ascii=False).replace("</", "<\\/")}</script>
<script>{PAGE_JS}</script>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
