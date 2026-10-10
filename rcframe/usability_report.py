"""桟橋の利用可否シナリオの検討報告書（HTML、1ファイル完結）

一括実行（port_batch）の結果の行から、構造の技術者・研究者向けの報告書を組み立てる。計算書（report.py）と
同じ仕組み（assemble_html、CSS、svg）で、章は
    1 概要  2 解析モデルと手法（杭の M–φ・諸元を含む）  3 シナリオ  4 判定基準と必要な荷重
    5 結果（判定の割合・フラジリティ・散布図・変数ごとの影響・全シナリオの表）  6 代表シナリオ（損傷図）
    7 理論解・手計算との照合（テスト）  8 留意事項と照合が必要な事項  9 付録（列の定義・入力データ）
代表シナリオは部材ごとの状態が必要なので解析し直す（port_batch.analyze、結果の行と同じ解析）。
使い方: python -m rcframe.port_batch 入力.yaml -o 結果 -j 4 --report 報告書.html
"""
import datetime
import math
from html import escape
from pathlib import Path

from . import svg
from .fiber import VERIFY_ITEMS as FIBER_VERIFY
from .port_batch import COLUMNS, analyze
from .port_cards import CARD_CSS, INDEX_JS, STATUS_CLS, allowable_bars, damage_frame, damage_levels, \
    scenario_table, sorted_final, summary_parts
from .port_level2 import VERIFY_ITEMS as L2_VERIFY, pile_capacity, pipe_mp
from .port_usability import LOAD_NAME, LOAD_UNIT, STATUS, VERIFY_ITEMS as USE_VERIFY, final_rows
from .report import Raw, assemble_html, figure, table
from .results import Result

M = 1e3
KNOWN = [
    "杭は地盤ばねモデル（地中部を分割し、各節点に水平地盤ばね k = kh·D·負担長、上限 pu·D·負担長）。"
    "杭先端は鉛直・ねじりを拘束。上部工は 1つの剛床",
    "地盤変位は杭の列ごとに海底面からの深さの関数（流動方向の 1成分）。分布形・大きさ・層厚は入力（FLIP 等の結果は未使用）",
    "流動層の地盤ばねの低減は、海底面からの深さ ζ < H の節点に一律に適用",
    "杭の軸力は長期荷重時の値で固定、載荷方向の曲げ面のみ。上部工の梁は弾性（許容荷重で M ≦ Mu を確認）",
    "鋼管杭のヒンジ: ふるい分けは Mp(N) の cos 式のバイリニア、詳細化はファイバーの M–φ から My〜Mu の直線"
    "（θu = (φu − φy)·Lp）、局部座屈後の耐力低下は扱わない",
    "許容荷重の算定では、利用時の地盤ばねを地震時と同じ（流動層の低減を含む）とし、地震時の上載荷重 Ls を取り除く",
    "超過確率は抽出したシナリオの件数の比（grid の場合は組合せの中の割合で、確率ではない）",
]
CHECKS = [
    ("除荷・状態の引き継ぎ（U1）", "片持ち柱・門形ラーメンの残留変位 = δmax − Pmax/K（弾性剛性）、再載荷で元の最大点に戻り骨格曲線に復帰、"
     "変位制御の停止点", "tests/test_unload.py"),
    ("地盤変位（U2）", "杭頭自由の杭は一様な地盤変位で剛体移動（曲げ 0）、杭頭固定の杭は Chang の解 H = 4EIβ³·ug、"
     "M0 = H/2β（1% 以内）、両端固定の杭に地盤反力の上限が作用する場合の端モーメント = ΣPab²/L²（≒ qL²/12）、"
     "地盤変位 0 で P4 のプッシュオーバーと一致、弾性範囲で載荷順序によらない、上部工の水平方向の釣合い", "tests/test_flow.py"),
    ("ファイバー M–φ（U3）", "弾性 M = EIφ と分割の収束、最初の降伏 My = (fy − N/A)·Z、N = 0 で Mp = Zp·fy、"
     "薄肉の極限で Mp(N) = Zp·fy·cos(πN/2Ny)、N = 0 で φu = εcu/(Do/2)、θ = θu の縁ひずみ = εcu、"
     "片持ち柱の降伏荷重 My/L・局部座屈荷重 Mu/L と変位 PL³/3EI + θu·L、εcu の式の手計算", "tests/test_fiber.py"),
    ("一括実行（U4）", "1段目の行 = run_flow の直接実行、慣性力のみ = P4（仮想固定点）の要求点、再開（計算済みを飛ばす）、"
     "並列 = 逐次、要素の剛性のキャッシュ = 新しく計算した剛性", "tests/test_batch.py"),
    ("許容荷重・判定・確率（U5）", "弾性範囲の許容荷重 = 線形解析の重ね合わせ（梁の Mu、杭の Rcd、杭の降伏）、算定後に状態が元に戻る、"
     "判定の境界、フラジリティの最尤推定が既知の θ・β を再現、分布の抽出の統計量、確率と標準誤差", "tests/test_usability.py"),
    ("判定カード（U6）", "解析し直した値 = 一括実行の行、HTML のタグの対応と外部参照なし、一覧の件数・割合 = 集計",
     "tests/test_cards.py"),
]
COLUMN_DOC = {
    "id": "シナリオの id（変数の値から決まるハッシュ）", "tier": "screen（ふるい分け）/ refine（詳細化）",
    "surface": "海底面の地盤変位 [m]（surface_ratio があれば海側の列の値）", "depth": "流動層の厚さ H [m]", "bottom": "流動層の下端の標高 [m]",
    "kh": "慣性力の換算震度", "ecu": "局部座屈ひずみの式",
    "rebar_loss_bottom": "梁の下端の主筋の断面欠損率", "rebar_loss_top": "梁の上端の主筋の断面欠損率",
    "corrosion_head": "杭頭側の外面の腐食量 [mm]", "corrosion_sea": "海中部の外面の腐食量 [mm]",
    "corrosion_soil": "地中部の外面の腐食量 [mm]", "scour": "洗掘による海底面の低下 [m]",
    "perforation": "腐食量が板厚に達したヒンジがある（開孔）",
    "reached": "打ち切り変位に達せず全段階を載荷できた",
    "deck_peak": "上部工の最大変位 [m]", "deck_residual": "上部工の残留変位 [m]（流動方向を正）",
    "deck_rot": "上部工の平面の残留回転 [rad]", "head_rot_max": "杭頭の塑性回転角の最大 [rad]",
    "head_rot_res": "杭頭の塑性回転角の残留 [rad]", "inground_rot_max": "地中部の塑性回転角の最大 [rad]",
    "inground_rot_res": "地中部の塑性回転角の残留 [rad]", "n_yield": "降伏したヒンジの数",
    "n_buckled": "局部座屈したヒンジの数（詳細化のみ）", "n_plastic": "塑性化したヒンジの数（詳細化のみ）",
    "max_strain": "杭の圧縮縁のひずみの最大（推定、詳細化のみ）", "soil_yield": "上限に達した地盤ばねの数",
    "allow_surcharge": "許容上載荷重 [kN/m²]", "allow_berthing": "許容接岸力 [kN]", "allow_tractive": "許容牽引力 [kN]",
    "status": "判定", "reasons": "判定の理由", "refine_reason": "2段目に回した理由", "seconds": "計算時間 [s]",
}


def _fmt(v, f="{:.3f}"):
    return "-" if v is None else f.format(v)


def _show(v):
    """入力値の表示（dict・list を読みやすく。エスケープは table が行う）"""
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_show(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_show(x) for x in v) + "]"
    return str(v)


def _when(w):
    """2段目に回す条件の説明"""
    if w in ("all", "none"):
        return {"all": "全シナリオ", "none": "なし"}[w]
    out = []
    if w.get("pile_yield"):
        out.append("杭が降伏した")
    if "residual" in w:
        out.append(f"上部工の残留変位が {w['residual'][0]:g}〜{w['residual'][1]:g} m")
    out.append("打ち切り変位に達した")
    return "、".join(out) + " シナリオ"


def _wrap(html):
    """長い文の表は折り返す"""
    return Raw(html.replace('<div class="tw">', '<div class="tw wrap">', 1))


# ================================================================ 計算（代表シナリオ・杭の諸元）
def _compute(batch, rows, source_text, representatives):
    pier = batch.pier
    ctx = {"batch": batch, "rows": rows, "final": sorted_final(rows), "source": source_text, "reps": []}
    # 杭の列ごとの長期荷重時の軸力（杭頭）と M–φ
    gc = list(pier.level2["gravity"])
    rG = Result.combine("+".join(gc), {pier.model.analyze(c): 1.0 for c in gc})
    ecus = sorted({str(r["ecu"]) for r in ctx["final"]}) or ["A"]
    caps = []
    for j in sorted({j for (_, j) in pier.piles}):
        (i, _), info = min(((ij, p) for ij, p in pier.piles.items() if ij[1] == j), key=lambda x: x[0][0])
        sec, mat = pier.pile_sec[(i, j)]
        N = -rG.forces[info["eid"]]["face"][6]                 # 杭頭の軸力（圧縮正）
        for e in ecus:
            opt = dict(pier.level2["pile_hinge"])
            opt["ecu"] = (e, None) if e in ("A", "B") else ("value", float(e))
            cap, warns = pile_capacity(sec, mat, N, opt)
            caps.append(dict(row=pier.gy[j][0], sec=sec, mat=mat, N=N, ecu=e, cap=cap, warns=warns,
                             Mp=pipe_mp(sec, mat.fy, N)))
    ctx["caps"] = caps
    # 代表シナリオ: 各判定で残留変位の最も大きいもの
    if representatives:
        params = dict(batch.scenarios)
        seen = set()
        for r in ctx["final"]:
            st = r.get("status")
            if st in seen:
                continue
            seen.add(st)
            p_, fr = analyze(batch.data, r["id"], params[r["id"]], r["tier"], batch.segments)
            ctx["reps"].append((r, p_, fr))
    return ctx


# ================================================================ 章
def _cover(pier, ctx):
    rp = pier.report
    title = f"桟橋の利用可否シナリオ検討報告書 {pier.name}"
    rows = [("対象", pier.name), ("作成者", rp.get("author", "")),
            ("日付", rp.get("date", datetime.date.today().isoformat())),
            ("シナリオ数", f"{len(ctx['final'])}（詳細化 {sum(r['tier'] == 'refine' for r in ctx['final'])}）"),
            ("備考", rp.get("notes", ""))]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in rows if v)
    return (f'<section class="cover"><h1>{escape(title)}</h1>'
            f'<p class="mute">rcframe による直杭式横桟橋（鋼管杭）の地震後の利用可否の検討。側方流動（地盤変位）と慣性力の'
            f'シナリオを事象追跡型の増分解析で計算し、被災状態からの許容荷重と判定を求める。単位: m, kN, kN·m, rad</p>'
            f'<dl>{dl}</dl></section>')


def sec_outline(pier, ctx):
    fr = ctx["final"]
    opts = pier.usability
    cnt = {s: sum(r.get("status") == s for r in fr) for s in STATUS}
    out = []
    if opts:
        out.append("<p>" + "、".join(
            f'<span class="status {STATUS_CLS[s]}" style="font-size:13px">{s}</span> {cnt[s]} 件（{cnt[s] / len(fr):.0%}）'
            for s in STATUS) + f"（全 {len(fr)} シナリオ）。</p>")
    worst = fr[0] if fr else None
    if worst:
        out.append(f'<p>上部工の残留変位の最大は {abs(worst["deck_residual"]):.3f} m（地盤変位 '
                   f'{_fmt(worst["surface"]) if not isinstance(worst["surface"], str) else worst["surface"]} m、'
                   + (f'層厚 {worst["depth"]:g} m' if worst.get("bottom") is None else f'流動層の下端 {worst["bottom"]:+g} m')
                   + f'、kh = {worst["kh"]:.3g}）。</p>')
    out.append('<p class="note">判定基準・必要な荷重は入力値（計算例は仮の値）。結果は入力した地盤変位の分布形と大きさに'
               '強く依存する（8 章）。</p>')
    m = pier.model
    out.append(table(["項目", "値"], [
        ["杭の本数・列", f"{len(pier.piles)} 本・{len(pier.gy)} 列（{', '.join(n for n, _ in pier.gy)}）"],
        ["上部工の天端・下面", f"{pier.deck_top / M:+.2f} m・{pier.z_soffit / M:+.2f} m"],
        ["節点数・要素数（詳細化のモデル）", f"{len(m.nodes)}・{len(m.elements)}"],
        ["地震用重量 W", f"{pier.weights['地震用'][0] / 1e3:,.0f} kN" if "地震用" in pier.weights else "-"],
    ]))
    out.append(figure(svg.pier_plan(pier), "平面図（杭は外径の実寸）"))
    rows = []
    for j, (name, y) in enumerate(pier.gy):
        if j in pier.pile_rows:
            sb, kh = pier.pile_rows[j]
            soil = pier.pile_soil.get(j, {})
            cap = pier.pile_capacity.get(j, (None, None))
            rows.append([name, f"{sb / M:+.2f}", f"{kh * 1e6:,.0f}",
                         "-" if not soil.get("pu") else f"{soil['pu'] * 1e3:,.0f}",
                         "-" if soil.get("tip") is None else f"{soil['tip'] / M:+.2f}",
                         "-" if cap[0] is None else f"{cap[0] / 1e3:,.0f}", "-" if cap[1] is None else f"{cap[1] / 1e3:,.0f}"])
    out.append(table(["列", "海底面 [m]", "kh [kN/m³]", "pu [kN/m²]", "杭先端 [m]", "Rcd [kN]", "Rtd [kN]"], rows,
                     num=(1, 2, 3, 4, 5, 6)))
    return "".join(out)


def sec_method(pier, ctx):
    b = ctx["batch"]
    ph = pier.level2["pile_hinge"]
    seg = b.segments
    out = ["<h3>解析の流れ</h3><ol class='list'>"
           "<li>長期荷重（level2.gravity）を載荷する</li>"
           "<li>地盤ばねの地盤側の端に地盤変位 ug(ζ) を与える（ばねの力 = k·(u杭 − ug)、剛性方程式には等価節点力 k·Δug）。"
           "慣性力 kh·W（上部工の重心）と、指定の順序（同時・地盤変位が先・慣性力が先）で載荷する</li>"
           "<li>慣性力を除荷し、残留変位・残留の塑性回転角を求める（材端ばね・地盤ばねは除荷剛）</li>"
           "<li>被災状態から、地震時の上載荷重 Ls を取り除き、利用時の荷重（上載・接岸力・牽引力）の単位荷重を増やして"
           "許容荷重を求める</li><li>判定基準から利用可否を判定する</li></ol>",
           "<p>載荷はすべて事象追跡型の荷重増分解析（ばねの折れ点ごとにステップを区切る）。事象の間は応答が荷重係数に比例するので、"
           "許容荷重の限界の値はステップ内の線形補間で厳密に求まる。</p>",
           "<h3>地盤変位の分布</h3>",
           "<p>ζ: 海底面からの深さ、H: 流動層の厚さ、ug0: 海底面の地盤変位（列ごと）。"
           "cos: ug = ug0·cos(πζ/2H)、linear: ug = ug0·(1 − ζ/H)、uniform: ug = ug0（ζ ≦ H）、ζ > H では 0。"
           "流動層（ζ < H）の地盤ばねの剛性・上限に低減係数を掛けられる。</p>",
           "<h3>2段階の計算</h3>",
           table(["段階", "地中部の分割", "杭のヒンジ", "対象"], [
               ["1 ふるい分け", f"{seg['screen']:g} m", "Mp(N) = Zp·fy·cos(π|N|/2Ny) のバイリニア",
                "全シナリオ（地盤変位 0 は仮想固定点モデル）"],
               ["2 詳細化", f"{seg['refine'] if seg['refine'] else 0.5:g} m", "ファイバーの M–φ（My〜Mu の直線、θu で局部座屈）",
                _when(b.cfg["refine"]["when"])]]),
           "<h3>鋼管杭のファイバー M–φ</h3>",
           f"<p>腐食後の円環断面を周方向 72 × 半径方向 4 に分割、鋼材はバイリニア（降伏後の剛性比 {ph['hardening']:g}）。"
           "軸力（長期荷重時）一定で、最初の降伏 (φy, My)、圧縮縁が局部座屈ひずみ εcu に達する限界曲率 (φu, Mu) を求める。"
           f"塑性ヒンジ長 Lp = {ph['hinge_length']:g}D、局部座屈に至る回転角 θu = (φu − φy)·Lp。"
           "εcu: 式A 1.14/(D/t) − 0.012、式B 0.44·t/R − 0.0026（R: 板厚中心の半径）、下限 εy。</p>"]
    rows, series = [], []
    cls = ["c0", "c1", "c2", "c3", "c4", "c5"]
    for k, c in enumerate(ctx["caps"]):
        cap = c["cap"]
        Do, Di = c["sec"].dims(corroded=True)
        rows.append([c["row"], f"φ{c['sec'].D:g}×{c['sec'].t:g}", f"{Do / ((Do - Di) / 2):.0f}", f"{c['N'] / 1e3:,.0f}",
                     c["ecu"], f"{cap.curve.ecu:.5f}", f"{cap.My / 1e6:,.0f}", f"{cap.Mu / 1e6:,.0f}",
                     f"{cap.curve.Mp / 1e6:,.0f}", f"{c['Mp'] / 1e6:,.0f}", f"{cap.curve.phi_y * 1e3:.5f}",
                     f"{cap.curve.phi_u * 1e3:.5f}", f"{cap.theta_u:.4f}"])
    out.append(table(["列", "断面", "D/t（腐食後）", "N [kN]", "εcu の式", "εcu", "My", "Mu", "Mp（ファイバー）",
                      "Mp（cos 式）", "φy [1/m]", "φu [1/m]", "θu [rad]"], rows, num=tuple(range(2, 13))))
    out.append('<p class="note">モーメントの単位は kN·m。杭頭の軸力（各列の先頭の杭）で算定。</p>')
    nmax = max(c["N"] for c in ctx["caps"])
    shown = [c for c in ctx["caps"] if c["N"] == nmax]                     # 軸力の最も大きい列
    for k, c in enumerate(shown):
        cv = c["cap"].curve
        series.append((f"εcu 式{c['ecu']}（{c['row']} 列）", [(p * 1e3, Mv / 1e6) for p, Mv in zip(cv.phi, cv.M)],
                       cls[k % len(cls)]))
    marks = [(c["cap"].curve.phi_u * 1e3, f"φu 式{c['ecu']}") for c in shown]
    out.append(figure(svg.xy_chart(series[::-1], "曲率 φ [1/m]", "M [kN·m]", marks=marks, xfmt="{:.4f}",
                                   yfmt="{:,.0f}"),
                      f"鋼管杭の M–φ 関係（{shown[0]['row']} 列、限界曲率 φu まで）。曲線は φu までで、"
                      "εcu の式で限界曲率が大きく異なる"))
    out.append("<h3>許容荷重と判定</h3><p>許容荷重は、限界（杭の塑性回転の増加 または 局部座屈、上部工の梁の |M| = Mu、"
               "杭の軸力 = Rcd・Rtd）に最初に達する単位荷重の係数。接岸力は防舷材ごと、牽引力は係船柱ごとの最小値。"
               "判定は、残留変位・杭の残留の塑性回転角・損傷の程度・許容荷重 < 必要な荷重 の各項目のうち最も厳しいもの。"
               "フラジリティ曲線 P = Φ(ln(x/θ)/β) は判定（0/1）の尤度を最大にする θ・β。</p>")
    return "".join(out)


def sec_scenarios(pier, ctx):
    cfg = ctx["batch"].cfg
    out = [table(["変数", "値"], [[k, _show(v)] for k, v in cfg["base"].items()])]
    if cfg.get("dists"):
        out.append("<h3>確率分布（抽出）</h3>")
        out.append(table(["変数", "分布"], [[k, _show(v)] for k, v in cfg["dists"].items()]))
        out.append(f"<p>抽出数 {cfg['sample']}、乱数の種 {cfg['seed']}。</p>")
    else:
        out.append("<h3>組合せ（grid）</h3>")
        out.append(table(["変数", "候補"], [[k, ", ".join(str(x) for x in v)] for k, v in cfg["grid"].items()]))
        n = math.prod(len(v) for v in cfg["grid"].values()) if cfg["grid"] else 1
        out.append(f"<p>組合せ {n} 通り" + (f"から {cfg['sample']} 件を抽出" if cfg["sample"] else "") + "。</p>")
    fr = ctx["final"]
    out.append(f"<p>計算したシナリオ {len(fr)} 件のうち、2段目（詳細化）に回したもの "
               f"{sum(r['tier'] == 'refine' for r in fr)} 件。</p>")
    return "".join(out)


def sec_criteria(pier, ctx):
    opts = pier.usability
    if not opts:
        return None
    lim = opts["limits"]
    rows = []
    if "residual_displacement" in lim:
        lo, hi = lim["residual_displacement"]
        rows.append(["上部工の残留水平変位", f"{lo / M:g} m 以上", f"{hi / M:g} m 以上"])
    if "pile_rotation" in lim:
        lo, hi = lim["pile_rotation"]
        rows.append(["杭の残留の塑性回転角", f"{lo:g} rad 以上", f"{hi:g} rad 以上"])
    pd = lim["pile_damage"]
    for lv in ("塑性化", "局部座屈"):
        rows.append([f"杭の{lv}", "あり" if pd.get(lv) == "limited" else "-", "あり" if pd.get(lv) == "unusable" else "-"])
    rows.append(["許容荷重", "必要な荷重を下回る", "0（載せられない）"])
    out = [table(["項目", "制限付き利用", "利用不可"], rows),
           table(["荷重", "必要な値"], [[LOAD_NAME[k], f"{v:g} {LOAD_UNIT[k]}"] for k, v in opts["required"].items()]),
           table(["許容荷重の限界", "設定"], [["杭", {"plastic": "塑性回転が増えない", "buckling": "局部座屈しない",
                                                   "none": "考慮しない"}[opts["pile"]]],
                                         ["上部工の梁", "|M| ≦ Mu" if opts["deck_mu"] else "考慮しない"],
                                         ["杭の軸力", "≦ Rcd・Rtd" if opts["pile_axial"] else "考慮しない"],
                                         ["取り除く地震時の荷重", ", ".join(opts["remove"]) or "なし"]]),
           '<p class="note">判定基準・必要な荷重は入力値。計算例は仮の値で、ガイドライン・利用計画で定めること。</p>']
    return "".join(out)


def sec_results(pier, ctx):
    rows, fr = ctx["rows"], ctx["final"]
    opts = pier.usability
    out = summary_parts(rows, opts) if opts else []
    pts = []
    for r in fr:
        if isinstance(r["surface"], str):
            continue
        k = STATUS.index(r["status"]) if r.get("status") in STATUS else 0
        pts.append((float(r["surface"]), abs(r["deck_residual"]), f"pt{k}",
                    f"{r['id']}: {r.get('status', '')}、残留 {abs(r['deck_residual']):.3f} m"))
    if pts:
        out.append("<h3>地盤変位と残留変位</h3>")
        out.append(figure(svg.scatter_chart(pts, "海底面の地盤変位 [m]", "上部工の残留変位 [m]",
                                            [(s, f"pt{k}") for k, s in enumerate(STATUS)], xfmt="{:.2f}",
                                            yfmt="{:.2f}"),
                          "シナリオごとの上部工の残留変位（色は判定）"))
    # 変数ごとの影響（候補が離散的な変数）
    cfg = ctx["batch"].cfg
    keys = [k for k, v in cfg["grid"].items() if len(v) > 1 and k != "surface"]
    keys += [k for k, v in (cfg.get("dists") or {}).items() if "choice" in v]
    for key in keys:
        vals = sorted({str(r.get(key, cfg["base"].get(key))) for r in fr})
        trs = []
        for v in vals:
            g = [r for r in fr if str(r.get(key, cfg["base"].get(key))) == v]
            nb = [r["n_buckled"] for r in g if r.get("n_buckled") is not None]
            trs.append([v, len(g)] + [f"{sum(r.get('status') == s for r in g) / len(g):.0%}" for s in STATUS]
                       + [f"{sum(abs(r['deck_residual']) for r in g) / len(g):.3f}",
                          "-" if not nb else f"{sum(nb) / len(nb):.1f}"])
        out.append(f"<h3>{escape(key)} の影響</h3>")
        out.append(table([key, "件数"] + list(STATUS) + ["残留変位の平均 [m]", "局部座屈の平均（詳細化）"], trs,
                         num=tuple(range(1, 7))))
    out.append("<h3>全シナリオ</h3>")
    out.append(scenario_table(rows))
    return "".join(out)


def sec_representatives(pier, ctx):
    if not ctx["reps"]:
        return None
    out = ["<p>各判定で上部工の残留変位の最も大きいシナリオ。図は損傷の最も多い通りの断面で、灰: 元の形、赤線: 残留変形"
           "（拡大）、茶の破線: 地盤変位、橙の丸: 塑性化、赤の丸: 局部座屈。</p>"]
    for r, p_, fr in ctx["reps"]:
        st = r.get("status", "")
        out.append(f'<h3><span class="status {STATUS_CLS.get(st, "")}" style="font-size:14px">{escape(st)}</span> '
                   f'シナリオ {escape(r["id"])}</h3>')
        c = fr.case
        out.append(f"<p>地盤変位 {', '.join(f'{p_.gy[j][0]} {v / M:.3f} m' for j, v in c.surface.items())}、"
                   f"{c.layer_text}、{c.shape}、kh = {c.kh:.3g}、εcu 式{escape(str(r['ecu']))}"
                   f"（{'詳細化' if r['tier'] == 'refine' else 'ふるい分け'}）。残留変位 {abs(r['deck_residual']):.3f} m。</p>")
        if r.get("reasons"):
            out.append("<ul class='list'>" + "".join(f"<li>{escape(x)}</li>" for x in r["reasons"].split(" / ")) + "</ul>")
        fig, name, scale = damage_frame(p_, fr, damage_levels(fr))
        out.append(figure(fig, f"{name} 通りの断面（変形・地盤変位は {scale:.0f} 倍）"))
        if pier.usability:
            bars = allowable_bars(r, pier.usability)
            if bars:
                out.append(figure(bars, "許容荷重と必要な荷重（緑: 足りる、赤: 不足）"))
    return "".join(out)


def sec_checks(pier, ctx):
    return (_wrap(table(["項目", "照合した理論値・手計算", "テスト"], CHECKS))
            + '<p class="note">いずれも pytest で自動的に確認している（全テストが合格した状態で報告書を作成すること）。</p>')


def sec_notes(pier, ctx):
    items = list(dict.fromkeys(FIBER_VERIFY + USE_VERIFY + L2_VERIFY))
    warns = sorted({w for c in ctx["caps"] for w in c["warns"]})
    out = ["<h3>規準書・ガイドラインでの照合が必要な事項</h3><ul class='list'>"
           + "".join(f"<li>{escape(x)}</li>" for x in items) + "</ul>",
           "<h3>モデル化の仮定・簡略化</h3><ul class='list'>" + "".join(f"<li>{escape(x)}</li>" for x in KNOWN) + "</ul>"]
    if warns:
        out.append("<h3>注意</h3><ul class='list'>" + "".join(f"<li>{escape(w)}</li>" for w in warns) + "</ul>")
    return "".join(out)


def sec_appendix(pier, ctx):
    out = ["<h3>結果の表（CSV）の列</h3>",
           table(["列", "内容"], [[c, COLUMN_DOC.get(c, "")] for c in COLUMNS if c in COLUMN_DOC])]
    if ctx["source"]:
        out.append(f'<h3>入力データ</h3><pre class="src">{escape(ctx["source"])}</pre>')
    return "".join(out)


SECTIONS = (
    ("sec-outline", "概要", sec_outline),
    ("sec-method", "解析モデルと手法", sec_method),
    ("sec-scenarios", "シナリオ", sec_scenarios),
    ("sec-criteria", "判定基準と必要な荷重", sec_criteria),
    ("sec-results", "結果", sec_results),
    ("sec-reps", "代表シナリオ", sec_representatives),
    ("sec-checks", "理論解・手計算との照合", sec_checks),
    ("sec-notes", "留意事項と照合が必要な事項", sec_notes),
    ("sec-appendix", "付録", sec_appendix),
)


def build_usability_report(batch, rows, source_text=None, representatives=True):
    """報告書の HTML（文字列）"""
    ctx = _compute(batch, rows, source_text, representatives)
    pier = batch.pier
    return assemble_html(pier, SECTIONS, ctx, _cover(pier, ctx),
                         extra_css=CARD_CSS, extra_js=INDEX_JS, title=f"利用可否シナリオ検討報告書 {pier.name}")


def write_usability_report(batch, rows, path, source_text=None, representatives=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_usability_report(batch, rows, source_text, representatives), encoding="utf-8")
    return path
