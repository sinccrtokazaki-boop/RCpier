"""計算書用の SVG 作図（軸組図・伏図・曲げモーメント図・荷重変形曲線・静止 3D 図）

色や線種は計算書の CSS のクラス（report.py）で指定する。単位: モデル座標 mm
曲げモーメント図は引張側に描く:
    xz 面（My）は +My が局所 +z 側の引張、xy 面（Mz）は +Mz が局所 −y 側の引張
"""
import math
from html import escape

import numpy as np


class View:
    """モデル座標 (u, v) [mm] → SVG 座標 [px]（v は上向き）"""

    def __init__(self, umin, umax, vmin, vmax, width=720, margin=(46, 46, 46, 46), max_height=560):
        self.mt, self.mr, self.mb, self.ml = margin
        su = max(umax - umin, 1.0)
        sv = max(vmax - vmin, 1.0)
        self.s = min((width - self.ml - self.mr) / su, (max_height - self.mt - self.mb) / sv)
        self.umin, self.vmax = umin, vmax
        self.w = width
        self.h = sv * self.s + self.mt + self.mb

    def p(self, u, v):
        return (self.ml + (u - self.umin) * self.s, self.mt + (self.vmax - v) * self.s)


def _f(v):
    return f"{v:.1f}"


def svg(w, h, body, cls="fig"):
    return (f'<svg class="{cls}" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
            f'role="img">{"".join(body)}</svg>')      # HTML に直接埋め込むので xmlns は不要


def line(p, q, cls):
    return f'<line class="{cls}" x1="{_f(p[0])}" y1="{_f(p[1])}" x2="{_f(q[0])}" y2="{_f(q[1])}"/>'


def poly(points, cls, closed=True):
    tag = "polygon" if closed else "polyline"
    pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
    return f'<{tag} class="{cls}" points="{pts}"/>'


def text(p, s, cls="lbl", anchor="middle", dy=0.0):
    return (f'<text class="{cls}" x="{_f(p[0])}" y="{_f(p[1] + dy)}" text-anchor="{anchor}">'
            f'{escape(str(s))}</text>')


def circle(p, r, cls, title=None):
    t = f"<title>{escape(title)}</title>" if title else ""
    return f'<circle class="{cls}" cx="{_f(p[0])}" cy="{_f(p[1])}" r="{r:.1f}">{t}</circle>'


# ============================================================ 曲げモーメント図
def moment_diagram_points(p0, p1, ts, offsets):
    """材軸 p0→p1（2D モデル座標）上の位置 ts（0〜1）と、各位置の図の偏位（2D、モデル座標）
    から、モーメント図の多角形の頂点（モデル座標）を返す（材軸の両端を含む閉多角形）"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    base = [p0 + t * (p1 - p0) for t in ts]
    return [tuple(base[0])] + [tuple(b + o) for b, o in zip(base, offsets)] + [tuple(base[-1])]


def member_moment_offsets(res, e, plane, xs):
    """部材 e の位置 xs [mm] の曲げモーメントと、引張側への偏位ベクトル（3D、単位モーメントあたり）
    戻り値: (M のリスト [N·mm], 偏位方向の単位ベクトル × 符号のリスト)"""
    out_m, out_v = [], []
    for x in xs:
        s = res.section_forces_at(e.id, x)
        if plane == 1:
            M, v = s[4], e.R[2]           # +My → 局所 +z 側が引張
        else:
            M, v = s[5], -e.R[1]          # +Mz → 局所 −y 側が引張
        out_m.append(M)
        out_v.append(M * v)
    return out_m, out_v


# ============================================================ 軸組図
def frame_members(bld, line_name):
    """通り line_name 上の柱・梁 [(key, eid)] と、横軸の全体座標軸（0 = X, 1 = Y）"""
    if line_name in bld.iy:                       # Y通り → X方向の架構
        j = bld.iy[line_name]
        keys = [k for k in bld.elem_ids if (k[0] == "c" and k[3] == j) or (k[0] == "x" and k[2] == j)]
        return [(k, bld.elem_ids[k]) for k in keys], 0
    i = bld.ix[line_name]
    keys = [k for k in bld.elem_ids if (k[0] == "c" and k[2] == i) or (k[0] == "y" and k[2] == i)]
    return [(k, bld.elem_ids[k]) for k in keys], 1


def _frame_view(bld, axis, width=720):
    grid = bld.gx if axis == 0 else bld.gy
    zs = [z for _, z in bld.floors]
    return View(grid[0][1], grid[-1][1], zs[0], zs[-1], width=width, margin=(34, 60, 40, 60))


def _frame_base(bld, axis, V, body):
    """通り芯・階の補助線と記号"""
    grid = bld.gx if axis == 0 else bld.gy
    zs = [z for _, z in bld.floors]
    for name, u in grid:
        body.append(line(V.p(u, zs[0]), V.p(u, zs[-1]), "gridline"))
        body.append(text(V.p(u, zs[0]), name, "gridlbl", dy=24))
    for name, z in bld.floors:
        body.append(line(V.p(grid[0][1], z), V.p(grid[-1][1], z), "gridline"))
        x, y = V.p(grid[0][1], z)
        body.append(text((x - 30, y), name, "gridlbl", anchor="end", dy=4))


def _end_xy(m, e, axis):
    a, b = m.nodes[e.ni], m.nodes[e.nj]
    return (a.xyz[axis], a.z), (b.xyz[axis], b.z)


def moment_elevation(bld, line_name, res, scale, n=13, width=720):
    """通りの軸組図に曲げモーメント図（引張側）を描く。scale: モデル mm / (N·mm)"""
    m = bld.model
    members, axis = frame_members(bld, line_name)
    V = _frame_view(bld, axis, width)
    body = []
    _frame_base(bld, axis, V, body)
    labels = []
    for key, eid in members:
        e = m.elements[eid]
        plane = 1 if key[0] != "c" else (0 if axis == 0 else 1)
        p0, p1 = _end_xy(m, e, axis)
        xs = list(np.linspace(0.0, e.L, n))
        xs = sorted(set(xs) | {e.rigid_i, e.L - e.rigid_j, e.L / 2})
        Ms, vs = member_moment_offsets(res, e, plane, xs)
        offs = [(v[axis] * scale, v[2] * scale) for v in vs]
        pts = moment_diagram_points(p0, p1, [x / e.L for x in xs], offs)
        body.append(poly([V.p(*q) for q in pts], "mdia"))
        body.append(line(V.p(*p0), V.p(*p1), "mem-c" if key[0] == "c" else "mem-b"))
        # 数値はフェイス位置（梁は中央も）。接合部で隣の部材と重ならないよう材の内側へ寄せる
        show = [(e.rigid_i, 1), (e.L - e.rigid_j, -1)] + ([(e.L / 2, 0)] if key[0] != "c" else [])
        a, b = V.p(*p0), V.p(*p1)
        L2 = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        t = ((b[0] - a[0]) / L2, (b[1] - a[1]) / L2)          # 材軸方向（画面）
        for x, inward in show:
            k = xs.index(x)
            if abs(Ms[k]) < 0.5e6:                           # 0.5 kN·m 未満は省略
                continue
            q = V.p(*pts[k + 1])
            q = (q[0] + 14 * inward * t[0], q[1] + 14 * inward * t[1])
            labels.append(text(q, f"{abs(Ms[k]) / 1e6:.0f}", "mval", dy=4 if key[0] == "c" else -3))
    return svg(V.w, V.h, body + labels)


def ratio_elevation(bld, line_name, ratios, width=720):
    """通りの軸組図に部材ごとの最大検定比を色分けで描く  ratios: {eid: 検定比}"""
    m = bld.model
    members, axis = frame_members(bld, line_name)
    V = _frame_view(bld, axis, width)
    body = []
    _frame_base(bld, axis, V, body)
    for key, eid in members:
        e = m.elements[eid]
        p0, p1 = _end_xy(m, e, axis)
        r = ratios.get(eid)
        cls = ("r-none" if r is None else "r-ng" if r > 1 else "r-hi" if r > 0.8
               else "r-mid" if r > 0.5 else "r-lo")
        body.append(line(V.p(*p0), V.p(*p1), f"mem-r {cls}"))
        if r is not None:
            mid = V.p((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            body.append(text(mid, f"{r:.2f}", "rval", dy=-5 if key[0] != "c" else 4))
    return svg(V.w, V.h, body)


def hinge_elevation(bld, line_name, po, plane_for_columns, width=720):
    """通りの軸組図にプッシュオーバー最終状態の材端ばね（ひび割れ・降伏、降伏の順番）を描く"""
    m = bld.model
    members, axis = frame_members(bld, line_name)
    V = _frame_view(bld, axis, width)
    body = []
    _frame_base(bld, axis, V, body)
    order = {}
    for ev in po.events:
        if ev["event"] == "降伏" and ev["key"] not in order:
            order[ev["key"]] = ev["step"]
    ranks = {k: i + 1 for i, k in enumerate(sorted(order, key=lambda k: order[k]))}
    marks = []
    for key, eid in members:
        e = m.elements[eid]
        p0, p1 = _end_xy(m, e, axis)
        body.append(line(V.p(*p0), V.p(*p1), "mem-c" if key[0] == "c" else "mem-b"))
        plane = plane_for_columns if key[0] == "c" else 1
        for end in (0, 1):
            hk = (eid, plane, end)
            h = po.hinges.get(hk)
            if h is None or h.state == "弾性":
                continue
            x = e.rigid_i + 0.06 * e.Lf if end == 0 else e.L - e.rigid_j - 0.06 * e.Lf
            t = x / e.L
            q = V.p(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
            cls = "h-y" if h.state == "降伏" else "h-c"
            marks.append(circle(q, 5.5 if cls == "h-y" else 4, cls, f"{po.labels[hk]}: {h.state}"))
            if hk in ranks:
                marks.append(text(q, str(ranks[hk]), "hnum", dy=-8))
    return svg(V.w, V.h, body + marks)


# ============================================================ 伏図
def floor_plan(bld, f, width=720):
    """階 f の伏図: 梁（断面名）、下階の柱（実寸）、吹抜け（網掛け）"""
    m = bld.model
    V = View(bld.gx[0][1], bld.gx[-1][1], bld.gy[0][1], bld.gy[-1][1], width=width,
             margin=(40, 60, 40, 60), max_height=480)
    body = []
    for (ff, i, j) in sorted(bld.openings):
        if ff == f:
            a = V.p(bld.gx[i][1], bld.gy[j + 1][1])
            b = V.p(bld.gx[i + 1][1], bld.gy[j][1])
            body.append(f'<rect class="opening" x="{_f(a[0])}" y="{_f(a[1])}" '
                        f'width="{_f(b[0] - a[0])}" height="{_f(b[1] - a[1])}"/>')
    for name, x in bld.gx:
        body.append(line(V.p(x, bld.gy[0][1]), V.p(x, bld.gy[-1][1]), "gridline"))
        body.append(text(V.p(x, bld.gy[0][1]), name, "gridlbl", dy=26))
    for name, y in bld.gy:
        body.append(line(V.p(bld.gx[0][1], y), V.p(bld.gx[-1][1], y), "gridline"))
        px, py = V.p(bld.gx[0][1], y)
        body.append(text((px - 26, py), name, "gridlbl", anchor="end", dy=4))
    for key, eid in bld.elem_ids.items():
        e = m.elements[eid]
        if key[0] in ("x", "y") and key[1] == f:
            a, b = m.nodes[e.ni], m.nodes[e.nj]
            body.append(line(V.p(a.x, a.y), V.p(b.x, b.y), "mem-b"))
            mid = V.p((a.x + b.x) / 2, (a.y + b.y) / 2)
            body.append(text(mid, e.sec.name, "plbl", dy=-5 if key[0] == "x" else 0,
                             anchor="middle" if key[0] == "x" else "start"))
        elif key[0] == "c" and key[1] == f - 1:
            a = m.nodes[e.nj]
            s = e.sec          # 柱の局所 y = 全体 X（幅 b）、局所 z = 全体 Y（せい D）
            p = V.p(a.x - s.b / 2, a.y + s.D / 2)
            body.append(f'<rect class="col" x="{_f(p[0])}" y="{_f(p[1])}" '
                        f'width="{_f(s.b * V.s)}" height="{_f(s.D * V.s)}"><title>{escape(s.name)}</title></rect>')
    return svg(V.w, V.h, body)


# ============================================================ 荷重変形曲線
def _nice_ticks(vmax, n=5):
    if vmax <= 0:
        return [0.0]
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((c * mag for c in (1, 2, 2.5, 5, 10) if c * mag >= raw), default=raw)
    return [k * step for k in range(int(vmax / step) + 1)]


def curve_chart(series, marks=(), width=720, height=380, xmax=None):
    """層せん断力–層間変形角の曲線  series: [(名前, [(層間変形角, Q [N])], クラス)]
    marks: [(層間変形角, 説明)] の縦線"""
    ml, mr, mt, mb = 70, 130, 20, 50
    xm = xmax or max((abs(x) for _, pts, _ in series for x, _ in pts), default=0.01) * 1.05
    ym = max((y for _, pts, _ in series for _, y in pts), default=1.0) * 1.1
    W, H = width - ml - mr, height - mt - mb
    P = lambda x, y: (ml + x / xm * W, mt + H - y / ym * H)
    body = [line(P(0, 0), P(xm, 0), "axis"), line(P(0, 0), P(0, ym), "axis")]
    for t in _nice_ticks(ym / 1e3):
        body.append(line(P(0, t * 1e3), P(xm, t * 1e3), "tick"))
        body.append(text(P(0, t * 1e3), f"{t:,.0f}", "axlbl", anchor="end", dy=4))
    for den in (400, 200, 150, 100, 75, 50, 30):
        x = 1 / den
        if x <= xm:
            body.append(line(P(x, 0), P(x, ym), "tick"))
            body.append(text(P(x, 0), f"1/{den}", "axlbl", dy=16))
    for x, label in marks:
        if x <= xm:
            body.append(line(P(x, 0), P(x, ym), "mark"))
            body.append(text(P(x, ym), label, "marklbl", anchor="start", dy=12))
    for k, (name, pts, cls) in enumerate(series):
        body.append(poly([P(abs(x), y) for x, y in pts], f"curve {cls}", closed=False))
        body.append(line((width - mr + 12, mt + 16 + 18 * k), (width - mr + 36, mt + 16 + 18 * k),
                         f"curve {cls}"))
        body.append(text((width - mr + 42, mt + 20 + 18 * k), name, "axlbl", anchor="start"))
    body.append(text((ml + W / 2, height - 8), "層間変形角 [rad]", "axlbl"))
    body.append(f'<text class="axlbl" transform="translate(16,{mt + H / 2:.0f}) rotate(-90)" '
                f'text-anchor="middle">層せん断力 [kN]</text>')
    return svg(width, height, body)


# ============================================================ 汎用の折れ線グラフ
def xy_chart(series, xlabel, ylabel, marks=(), hmarks=(), width=720, height=380, xfmt="{:g}", yfmt="{:g}"):
    """series: [(名前, [(x, y)], クラス)]、marks: [(x, 説明)] の縦線、hmarks: [(y, 説明)] の横線"""
    ml, mr, mt, mb = 70, 140, 20, 50
    xm = max([abs(x) for _, pts, _ in series for x, _ in pts] + [m_[0] for m_ in marks] + [1e-12]) * 1.05
    ym = max([y for _, pts, _ in series for _, y in pts] + [m_[0] for m_ in hmarks] + [1e-12]) * 1.1
    W, H = width - ml - mr, height - mt - mb
    P = lambda x, y: (ml + x / xm * W, mt + H - y / ym * H)
    body = [line(P(0, 0), P(xm, 0), "axis"), line(P(0, 0), P(0, ym), "axis")]
    for t in _nice_ticks(ym):
        body.append(line(P(0, t), P(xm, t), "tick"))
        body.append(text(P(0, t), yfmt.format(t), "axlbl", anchor="end", dy=4))
    for t in _nice_ticks(xm):
        body.append(line(P(t, 0), P(t, ym), "tick"))
        body.append(text(P(t, 0), xfmt.format(t), "axlbl", dy=16))
    for x, label in marks:
        body.append(line(P(x, 0), P(x, ym), "mark"))
        body.append(text(P(x, ym), label, "marklbl", anchor="start", dy=12))
    for y, label in hmarks:
        body.append(line(P(0, y), P(xm, y), "mark"))
        body.append(text(P(xm, y), label, "marklbl", anchor="end", dy=-4))
    for k, (name, pts, cls) in enumerate(series):
        body.append(poly([P(abs(x), y) for x, y in pts], f"curve {cls}", closed=False))
        body.append(line((width - mr + 12, mt + 16 + 18 * k), (width - mr + 36, mt + 16 + 18 * k),
                         f"curve {cls}"))
        body.append(text((width - mr + 42, mt + 20 + 18 * k), name, "axlbl", anchor="start"))
    body.append(text((ml + W / 2, height - 8), xlabel, "axlbl"))
    body.append(f'<text class="axlbl" transform="translate(16,{mt + H / 2:.0f}) rotate(-90)" '
                f'text-anchor="middle">{escape(ylabel)}</text>')
    return svg(width, height, body)


def scatter_chart(points, xlabel, ylabel, legend=(), width=720, height=380, xfmt="{:g}", yfmt="{:g}"):
    """散布図  points: [(x, y, クラス, 説明)]（x, y ≧ 0）、legend: [(名前, クラス)]"""
    ml, mr, mt, mb = 70, 140, 20, 50
    xm = max([x for x, _, _, _ in points] + [1e-12]) * 1.08
    ym = max([y for _, y, _, _ in points] + [1e-12]) * 1.1
    W, H = width - ml - mr, height - mt - mb
    P = lambda x, y: (ml + x / xm * W, mt + H - y / ym * H)
    body = [line(P(0, 0), P(xm, 0), "axis"), line(P(0, 0), P(0, ym), "axis")]
    for t in _nice_ticks(ym):
        body.append(line(P(0, t), P(xm, t), "tick"))
        body.append(text(P(0, t), yfmt.format(t), "axlbl", anchor="end", dy=4))
    for t in _nice_ticks(xm):
        body.append(line(P(t, 0), P(t, ym), "tick"))
        body.append(text(P(t, 0), xfmt.format(t), "axlbl", dy=16))
    for x, y, cls, title in points:
        body.append(circle(P(x, y), 4.5, cls, title))
    for k, (name, cls) in enumerate(legend):
        body.append(circle((width - mr + 22, mt + 16 + 18 * k), 4.5, cls))
        body.append(text((width - mr + 34, mt + 20 + 18 * k), name, "axlbl", anchor="start"))
    body.append(text((ml + W / 2, height - 8), xlabel, "axlbl"))
    body.append(f'<text class="axlbl" transform="translate(16,{mt + H / 2:.0f}) rotate(-90)" '
                f'text-anchor="middle">{escape(ylabel)}</text>')
    return svg(width, height, body)


# ============================================================ 桟橋
def pier_plan(pier, width=720):
    """桟橋の平面図: 上部工の外形（張出しを含む）、梁、杭（外径の実寸）、防舷材・係船柱"""
    m = pier.model
    y0 = pier.gy[0][1] - pier.overhang["land"]
    y1 = pier.gy[-1][1] + pier.overhang["sea"]
    x0, x1 = pier.gx[0][1], pier.gx[-1][1]
    V = View(x0, x1, y0, y1, width=width, margin=(40, 60, 40, 60), max_height=420)
    a, b = V.p(x0, y1), V.p(x1, y0)
    body = [f'<rect class="deck" x="{_f(a[0])}" y="{_f(a[1])}" width="{_f(b[0] - a[0])}" '
            f'height="{_f(b[1] - a[1])}"/>']
    for name, x in pier.gx:
        body.append(line(V.p(x, y0), V.p(x, y1), "gridline"))
        body.append(text(V.p(x, y0), name, "gridlbl", dy=26))
    for name, y in pier.gy:
        body.append(line(V.p(x0, y), V.p(x1, y), "gridline"))
        px, py = V.p(x0, y)
        body.append(text((px - 26, py), name, "gridlbl", anchor="end", dy=4))
    for key, eid in pier.elem_ids.items():
        if key[0] in ("x", "y"):
            e = m.elements[eid]
            p0, p1 = m.nodes[e.ni], m.nodes[e.nj]
            body.append(line(V.p(p0.x, p0.y), V.p(p1.x, p1.y), "mem-b"))
    for (i, j), (sec, _) in pier.pile_sec.items():
        c = V.p(pier.gx[i][1], pier.gy[j][1])
        body.append(circle(c, max(3.0, sec.D / 2 * V.s), "pile", f"杭 {pier.gx[i][0]}/{pier.gy[j][0]} φ{sec.D:g}"))
    for name, pts, cls in (("防舷材", pier.fenders, "fender"), ("係船柱", pier.bollards, "bollard")):
        for x in pts:
            body.append(circle(V.p(x, y1), 5, cls, f"{name} x = {x / 1e3:g} m"))
    body.append(text((V.w - 8, 14), "↑ 海側（+Y）", "axlbl", anchor="end"))
    return svg(V.w, V.h, body)


def pier_frame(pier, x_name, res=None, scale=0.0, po=None, width=720):
    """X通り x_name の法線直角方向の断面（Y方向の梁・杭・海底面）
    res を与えると曲げモーメント図（引張側）、po を与えるとレベル2の塑性ヒンジを描く"""
    m = pier.model
    i = pier.ix[x_name]
    piles = [(ij, info) for ij, info in pier.piles.items() if ij[0] == i]
    beams = [(k, e) for k, e in pier.elem_ids.items() if k[0] == "y" and k[1] == i]
    zb = min([info.get("tip", info["z_fix"]) for _, info in piles] + [pier.z_soffit])
    y0 = pier.gy[0][1] - pier.overhang["land"]
    y1 = pier.gy[-1][1] + pier.overhang["sea"]
    V = View(y0, y1, zb, pier.deck_top, width=width, margin=(30, 70, 34, 70), max_height=620)
    body = [line(V.p(y0, pier.deck_top), V.p(y1, pier.deck_top), "deckline"),
            line(V.p(y0, pier.z_soffit), V.p(y1, pier.z_soffit), "deckline")]
    for name, y in pier.gy:
        body.append(text(V.p(y, zb), name, "gridlbl", dy=24))
    labels = []
    members = [(e, 1) for _, e in beams] + [(eid, 1) for _, info in piles for eid in info["elements"]]
    for (i_, j), info in piles:                      # 海底面（杭の位置の前後）と固定点
        y = pier.gy[j][1]
        body.append(line(V.p(y - 2500, info["seabed"]), V.p(y + 2500, info["seabed"]), "seabed"))
        body.append(text(V.p(y + 2500, info["seabed"]), f"{info['seabed'] / 1e3:+.1f}", "axlbl",
                         anchor="start", dy=4))
        if pier.pile_model == "fixed_point":
            body.append(poly([V.p(y, info["z_fix"]), (V.p(y, info["z_fix"])[0] - 7, V.p(y, info["z_fix"])[1] + 10),
                              (V.p(y, info["z_fix"])[0] + 7, V.p(y, info["z_fix"])[1] + 10)], "fixmark"))
    for eid, plane in members:
        e = m.elements[eid]
        a, b = m.nodes[e.ni], m.nodes[e.nj]
        p0, p1 = (a.y, a.z), (b.y, b.z)
        if res is not None and scale:
            L = math.dist(a.xyz, b.xyz)
            xs = sorted(set(np.linspace(0.0, L, 7)) | {e.rigid_i, L - e.rigid_j})
            Ms, vs = member_moment_offsets(res, e, plane, xs)
            offs = [(v[1] * scale, v[2] * scale) for v in vs]
            pts = moment_diagram_points(p0, p1, [x / L for x in xs], offs)
            body.append(poly([V.p(*q) for q in pts], "mdia"))
        body.append(line(V.p(*p0), V.p(*p1), "mem-c" if e.kind == "column" else "mem-b"))
    if res is not None and scale:                    # 数値: 杭頭・梁端・梁中央・杭の地中部の最大
        for _, e_id in beams:
            e = m.elements[e_id]
            L = math.dist(m.nodes[e.ni].xyz, m.nodes[e.nj].xyz)
            for x, anchor, dx in ((0.0, "start", 6), (L / 2, "middle", 0), (L, "end", -6)):   # 端は材の内側へ
                M = abs(res.section_forces_at(e_id, x)[4])
                q = V.p(m.nodes[e.ni].y + (m.nodes[e.nj].y - m.nodes[e.ni].y) * x / L, pier.z_model)
                labels.append(text((q[0] + dx, q[1]), f"{M / 1e6:.0f}", "mval", anchor=anchor, dy=-8))
        for _, info in piles:
            e = m.elements[info["eid"]]
            L = math.dist(m.nodes[e.ni].xyz, m.nodes[e.nj].xyz)
            Mh = abs(res.section_forces_at(info["eid"], L - e.rigid_j)[4])
            labels.append(text(V.p(m.nodes[e.nj].y, pier.z_soffit), f"{Mh / 1e6:.0f}", "mval", dy=14))
            gm = [(abs(res.section_forces_at(k, 0.0)[4]), m.nodes[m.elements[k].ni].z)
                  for k in info["elements"]]
            Mg, zg = max(gm)
            labels.append(text(V.p(m.nodes[e.nj].y, zg), f"{Mg / 1e6:.0f}", "mval", anchor="start", dy=4))
    if po is not None:
        order = {}
        for ev in po.events:
            if ev["event"] == "降伏" and ev["key"] not in order:
                order[ev["key"]] = ev["step"]
        ranks = {k: n + 1 for n, k in enumerate(sorted(order, key=order.get))}
        for eid, plane in members:
            for end in (0, 1):
                hk = (eid, 1, end)
                h = po.hinges.get(hk)
                if h is None or h.state == "弾性":
                    continue
                e = m.elements[eid]
                a, b = m.nodes[e.ni], m.nodes[e.nj]
                L = math.dist(a.xyz, b.xyz)
                x = e.rigid_i if end == 0 else L - e.rigid_j
                q = V.p(a.y + (b.y - a.y) * x / L, a.z + (b.z - a.z) * x / L)
                labels.append(circle(q, 5.5 if h.state == "降伏" else 4, "h-y" if h.state == "降伏" else "h-c",
                                     f"{po.labels[hk]}: {h.state}"))
                if hk in ranks:
                    labels.append(text(q, str(ranks[hk]), "hnum", anchor="start", dy=-6))
    return svg(V.w, V.h, body + labels)


# ============================================================ 静止 3D 図（印刷用）
def isometric(bld, width=720, yaw=-0.6, pitch=0.5):
    """骨組の軸測投影（画面の 3D 表示と同じ視点の静止図）"""
    m = bld.model
    used = {n for e in m.elements.values() for n in (e.ni, e.nj)}
    pts = np.array([m.nodes[n].xyz for n in used])
    c = (pts.min(0) + pts.max(0)) / 2

    def proj(p):
        x, y, z = np.asarray(p) - c
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        x1, y1 = x * cy - y * sy, x * sy + y * cy
        return x1, z * cp - y1 * sp

    P = [proj(p) for p in pts]
    us, vs = [p[0] for p in P], [p[1] for p in P]
    V = View(min(us), max(us), min(vs), max(vs), width=width, max_height=440)
    body = []
    for e in m.elements.values():
        a, b = proj(m.nodes[e.ni].xyz), proj(m.nodes[e.nj].xyz)
        body.append(line(V.p(*a), V.p(*b), "mem-c" if e.kind == "column" else "mem-b"))
    return svg(V.w, V.h, body)
