"""変状連鎖の重点シナリオの確率評価の報告書（HTML、1ファイル完結、D5）

examples/kawasaki_deterioration.py の結果のフォルダ（inputs/*.yaml、results/*.jsonl）から、構造の技術者・研究者向けの
報告書を組み立てる。計算書と同じ仕組み（report.assemble_html、CSS、svg）で、数値はすべて結果の行から求める。章は
    1 概要  2 変状連鎖図と重点シナリオ  3 確率分布と劣化の量  4 解析と判定の方法  5 常時の結果
    6 地震後の結果  7 シナリオごとの考察  8 留意事項  9 付録
ケースの種類: 入力の scenarios.distributions に surface があれば地震後、なければ常時。
地震後のケースは年数の間で地震の抽出値が共通（examples/kawasaki_deterioration.py）なので、同じ地震で年数を比べる。
使い方: python -m rcframe.deterioration_report 結果のフォルダ [-o 報告書.html]
"""
import argparse
import datetime
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from . import pier_chain, svg
from .chain_draw import code_str, full_svg
from .port_cards import CARD_CSS, STATUS_CLS
from .port_usability import (LOAD_NAME, METHOD_SEP, RETAIN_LIMITS, STATUS, final_rows, fit_fragility, judge, judge_detail,
                             load_text, read_usability, required_values)
from .usability_report import criteria_html
from .report import Raw, assemble_html, figure, table

BLOCK_SCENARIO = {"chloride": "S1", "chloride_top": "S1", "pile": "S2", "scour": "S4"}
SC_NAME = {s[0]: s[1] for s in pier_chain.SCENARIOS}
# 判定の理由の分類（判定の理由の文字列に含まれる語 → 表示名）
CATS = [("上部工の残留変位", "上部工の残留変位"), ("杭の残留の塑性回転角", "杭の回転角"), ("杭の局部座屈", "杭の局部座屈"),
        ("杭の塑性化", "杭の塑性化"), ("渡版の開き", "渡版の開き"), ("土留部が上部工", "土留部が寄る"),
        ("段差", "背後の段差"), ("上載荷重", "上載荷重の不足"), ("接岸力", "接岸力の不足"), ("牽引力", "牽引力の不足"),
        ("上部工の梁が Mu", "上部工の梁の Mu"), ("杭の軸力が設計支持力", "杭の軸力"), ("余震", "余震に対する余裕"),
        ("限界曲率の超過は未確認", "限界曲率が未確認"), ("土留部の残留水平変位", "土留部の変位")]
DET_KEYS = [("rebar_loss_bottom", "梁の下端筋の断面欠損率", "{:.1%}"), ("corrosion_head", "杭頭側の腐食量 [mm]", "{:.2f}"),
            ("corrosion_sea", "海中部の腐食量 [mm]", "{:.2f}"), ("corrosion_soil", "地中部の腐食量 [mm]", "{:.2f}"),
            ("scour", "洗掘深さ（海側の列）[m]", "{:.2f}")]
HEAD_BINS = (0, 4, 7, 10, 13, math.inf)
MIN_WORSE = 10                    # 悪化した抽出がこれより少なければ劣化の量を比べない
SURF_BINS = (0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, math.inf)
SHORT_ST = ("利用可", "注意事項あり", "利用不可")                 # 図の凡例（STATUS の順）
LEGEND_ST = [(s, f"pt{k}") for k, s in enumerate(SHORT_ST)]
EXTRA_CSS = """
figure svg#chain{max-width:100%;height:auto;display:block}
.bar0{fill:#1b7f3b}.bar1{fill:#e69f00}.bar2{fill:#c0392b}.barlbl{font-size:11px;fill:#fff;font-weight:700}
.keyfig{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:12px 0}
.keyfig div{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.keyfig b{display:block;font-size:22px}.keyfig span{color:var(--mute);font-size:12.5px}
.pa0{fill:#1f4e79;fill-opacity:.8}.pa1{fill:#8e44ad;fill-opacity:.8}.pa2{fill:#d35400;fill-opacity:.85}
details{margin:6px 0}summary{cursor:pointer;color:var(--acc)}
"""
KNOWN = [
    "劣化の確率分布は基準類の標準値と工学的な仮定による（この施設の点検・調査の値は使っていない）。特に杭頭の腐食"
    "（モルタルライニングの寿命の後に一様に減る）の仮定が地震後の結果を大きく左右する",
    "鉄筋の腐食は一様腐食（孔食・腐食ひび割れによる加速なし）。断面欠損率は全ての梁の下端筋に一律で、梁の Mu だけに効く"
    "（剛性・付着は変えない）",
    "杭の腐食は杭頭側・海中部・地中部の 3 部位ごとに一律で、各ヒンジの位置の部位の断面で Mp・M–φ を求める。剛性は腐食前の断面。"
    "腐食量が板厚に達した場合は残りの板厚を 1 mm とする（開孔）",
    "腐食後の径厚比が大きい範囲（板厚の残りが数 mm）は局部座屈ひずみ εcu の式の適用範囲外で、下限の εy を用いる"
    "（局部座屈の判定は安全側）",
    "土留部はモデル化していない。土留部の移動 = 比 × 陸側の列の地盤変位、背後の沈下 = 比 × 移動 の後処理で、"
    "土留部が上部工を押す相互作用は考えない",
    "洗掘は海側の列（隣の列は半分）の海底面の低下で、平衡の深さに時定数 10 年で近づくとした",
    "判定基準・必要な荷重は仮の値（常時は常時の上載荷重、地震後は緊急時の上載荷重を必要な値とした）",
    "地震後は、1段目の残留変位が skip_above 以上のシナリオを詳細化していない（利用不可に決まるため）。"
    "これらの杭の損傷の数は 1段目（cos 式のヒンジ）の値",
    "確率は抽出したシナリオの件数の比（抽出 200 回の標準誤差は最大 3.5%）",
]
NEEDS = [
    ("鋼管杭の肉厚（点検 (5)、超音波厚み計）", "杭頭側・海中部の腐食量の分布を実測で置き換える。地震後の結果に最も効く"),
    ("被覆防食・電気防食の状態（点検 (6)(7)、電位・陽極の消耗量）", "防食の寿命の分布と、すでに腐食が始まっているかを決める"),
    ("上部工の塩化物イオン量・かぶり（点検 (8)、詳細調査）", "腐食の開始時期 t_i の分布を絞る"),
    ("海底地盤の深浅測量（点検 (1)）", "洗掘深さの分布を実測で置き換える"),
    ("土留部の構造形式（点検 (4)）と地盤の調査", "土留部の移動・沈下を比ではなく解析で求める"),
]


# ================================================================ 集計
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_study(root):
    """結果のフォルダを読む。戻り値: [case]（常時 → 地震後、名前の順）。case: dict(name, data, rows, eq, level, label,
    scenarios, opts, ages, text)。判定（status・reasons・methods_ok）は入力の usability の設定で判定し直す
    （判定の規則・閾値を変えたときに計算し直さずに報告書を作れる。解析の値・許容荷重は結果の行のまま）"""
    root = Path(root)
    cases = []
    for p in sorted((root / "inputs").glob("*.yaml")):
        jl = root / "results" / f"{p.stem}.jsonl"
        if not jl.exists():
            continue
        text = p.read_text(encoding="utf-8")
        d = yaml.safe_load(text)
        sc = d.get("scenarios") or {}
        eq = "surface" in (sc.get("distributions") or {})
        rows = final_rows([json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines() if x.strip()])
        det = sc.get("deterioration") or {}
        ids = sorted({BLOCK_SCENARIO[b] for b in det if b in BLOCK_SCENARIO})
        opts = read_usability(None, d["usability"])
        rows = [{**r, **_judged(r, opts)} for r in rows]
        if eq:
            ids = sorted(set(ids) | {"S3"} | ({"S5"} if any(k in opts["limits"] for k in RETAIN_LIMITS) else set()))
        level = p.stem.split("_")[0] if eq else "常時"
        sc_txt = "S1〜S5" if ids == ["S1", "S2", "S3", "S4", "S5"] else "＋".join(ids)
        label = f"{level if not eq else level + ' 地震後'}・{sc_txt}"
        cases.append(dict(name=p.stem, data=d, rows=rows, eq=eq, level=level, label=label, scenarios=ids, opts=opts,
                          ages=sorted({r["age"] for r in rows}), text=text))
    cases.sort(key=lambda c: (c["eq"], c["name"]))
    return cases


def _judged(r, opts):
    st, reasons, ok = judge_detail(r, opts)
    return {"status": st, "reasons": " / ".join(reasons), "methods_ok": None if ok is None else METHOD_SEP.join(ok)}


def probs(rows):
    n = len(rows)
    c = Counter(r["status"] for r in rows)
    return {s: (c[s] / n if n else math.nan, math.sqrt(c[s] / n * (1 - c[s] / n) / n) if n else math.nan) for s in STATUS}


def reason_lines(r, level):
    """判定 level を決めた理由の行。利用方法がなければ「level: …」の行、あれば「… で不可: …」の行
    （注意事項ありでは「注意事項: …」も）"""
    lines = [x for x in (r.get("reasons") or "").split(" / ") if x]
    if any(x.startswith("使える利用方法:") for x in lines):
        return [x for x in lines if "で不可:" in x or (level == STATUS[1] and x.startswith("注意事項:"))]
    return [x for x in lines if x.startswith(level + ":")]


def reason_counts(rows, level):
    """判定 level の行の、その判定の理由の分類ごとの件数"""
    k = Counter()
    for r in rows:
        if r["status"] != level:
            continue
        hit = set()
        for x in reason_lines(r, level):
            hit |= {name for key, name in CATS if key in x}
        k.update(hit)
    return k


def at(rows, age):
    return [r for r in rows if r["age"] == age]


def quant(v, q):
    return float(np.quantile(v, q)) if len(v) else math.nan


def paired(rows):
    """同じ地震の抽出値（surface, kh）ごとの {年数: 行}"""
    by = defaultdict(dict)
    for r in rows:
        by[(r["surface"], r["kh"])][r["age"]] = r
    return [v for v in by.values() if len(v) > 1]


def rejudge(rows, opts, drop):
    """閾値 drop を除いた判定"""
    o = {**opts, "limits": {k: v for k, v in opts["limits"].items() if k not in drop}}
    return [judge(r, o)[0] for r in rows]


def safe_fit(x, y):
    try:
        return fit_fragility(x, y)
    except (OverflowError, ValueError, FloatingPointError):
        return None


def _compute(root):
    cases = load_study(root)
    if not cases:
        raise ValueError(f"{root}: inputs/*.yaml と results/*.jsonl の組がありません")
    return dict(root=Path(root), cases=cases, normal=[c for c in cases if not c["eq"]], eq=[c for c in cases if c["eq"]])


# ================================================================ 図
def wrapt(html):
    """セルを折り返す表"""
    return Raw(f'<div class="wrap">{html}</div>')


DIST_NAME = {"surface": "陸側の列の海底面の地盤変位 [m]", "surface_ratio": "列ごとの地盤変位の比（L・C2・C1・S）",
             "kh": "同時に作用する慣性力の換算震度", "kh_factor": "流動層の地盤ばねの剛性の低減係数",
             "pu_factor": "流動層の地盤反力の上限の低減係数", "wall_ratio": "土留部の移動 / 陸側の列の地盤変位",
             "settlement_ratio": "背後の沈下 / 土留部の移動", "settlement": "背後の沈下 [m]"}


def stack_chart(items, width=720):
    """判定の確率の積み上げ横棒  items: [(ラベル, {判定: (p, se)})]"""
    ml, mr, bh, gap, mt = 210, 20, 18, 8, 26
    W = width - ml - mr
    h = mt + len(items) * (bh + gap) + 24
    body = []
    for k, s in enumerate(SHORT_ST):
        x = ml + k * 130
        body.append(f'<rect class="bar{k}" x="{x}" y="4" width="14" height="10"/>')
        body.append(svg.text((x + 20, 13), s, "axlbl", anchor="start"))
    for i, (lab, p) in enumerate(items):
        y = mt + i * (bh + gap)
        body.append(svg.text((ml - 8, y + bh - 5), lab, "axlbl", anchor="end"))
        x = ml
        for k, s in enumerate(STATUS):
            w = p[s][0] * W
            if w > 0:
                body.append(f'<rect class="bar{k}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bh}"><title>{escape(lab)}: '
                            f'{escape(s)} {p[s][0]:.1%}</title></rect>')
                if w > 34:
                    body.append(svg.text((x + w / 2, y + bh - 5), f"{p[s][0]:.0%}", "barlbl"))
            x += w
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(svg.text((ml + t * W, h - 6), f"{t:.0%}", "axlbl"))
    return svg.svg(width, h, body)


# ================================================================ 章
def _cover(study, ctx):
    c0 = ctx["cases"][0]
    rows = [("対象", c0["data"].get("name", "").split("（")[0]), ("日付", datetime.date.today().isoformat()),
            ("ケース", "、".join(c["label"] for c in ctx["cases"])),
            ("シナリオ数", f"{sum(len(c['rows']) for c in ctx['cases']):,}"),
            ("出典（変状連鎖）", pier_chain.SOURCE)]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in rows)
    return (f'<section class="cover"><h1>{escape(study.title)}</h1>'
            '<p class="mute">変状連鎖図（港湾空港技術研究所資料 No.1328 に基づく）の重点シナリオ S1〜S5 に確率分布を与え、'
            'rcframe の事象追跡型の増分解析で、劣化した直杭式横桟橋の常時と地震後の利用可否を求めた。'
            + ('判定は港湾施設の利用可否判断に係るガイドライン 第2版（令和8年5月）の表 D-5 に合わせ、利用方法ごとに行った。'
               if any(_methods(c) for c in ctx["cases"]) else "")
            + '分布・判定基準・必要な荷重は仮の値で、数値は仮定に対する目安である。単位: m, mm, kN, 年</p>'
            f"<dl>{dl}</dl></section>")


def _key_numbers(ctx):
    """概要の数値（常時の全部・地震後の各ケースの、最後と最初の年数の利用不可・注意事項あり以上の確率）"""
    out = []
    for c in ctx["normal"]:
        if len(c["scenarios"]) > 1 or c is ctx["normal"][-1]:
            a = c["ages"][-1]
            p = probs(at(c["rows"], a))
            out.append((f"{p[STATUS[1]][0] + p[STATUS[2]][0]:.0%}", f"{c['label']}（{a:g} 年）の注意事項あり以上"))
            break
    for c in ctx["eq"]:
        p0, p1 = probs(at(c["rows"], c["ages"][0])), probs(at(c["rows"], c["ages"][-1]))
        out.append((f"{p0[STATUS[2]][0]:.1%} → {p1[STATUS[2]][0]:.1%}",
                    f"{c['level']} 地震後の利用不可（{c['ages'][0]:g} 年 → {c['ages'][-1]:g} 年）"))
        ms = _methods(c)
        if ms:
            m3 = list(ms)[-1]
            g0, g1 = at(c["rows"], c["ages"][0]), at(c["rows"], c["ages"][-1])
            out.append((f"{sum(_usable(r, m3) for r in g0) / len(g0):.0%} → {sum(_usable(r, m3) for r in g1) / len(g1):.0%}",
                        f"{c['level']} 地震後に {m3}（フェーズ {ms[m3]['phase']}）が使える割合"))
    return out


def _driver(c):
    """地震後のケースで、最初の年数より判定が悪くなった抽出と、それ以外の劣化の量の平均（年数は最後）"""
    a0, a1 = c["ages"][0], c["ages"][-1]
    worse, other = [], []
    for v in paired(c["rows"]):
        if a0 in v and a1 in v:
            (worse if STATUS.index(v[a1]["status"]) > STATUS.index(v[a0]["status"]) else other).append(v[a1])
    stats = []
    for k, name, fmt in DET_KEYS:
        w = [r[k] for r in worse if _num(r.get(k))]
        o = [r[k] for r in other if _num(r.get(k))]
        if w and o:
            sd = np.std(w + o) or 1.0
            stats.append((k, name, fmt, float(np.mean(w)), float(np.mean(o)), (np.mean(w) - np.mean(o)) / sd))
    return worse, other, stats


def sec_outline(study, ctx):
    out = ['<div class="keyfig">' + "".join(f"<div><b>{escape(v)}</b><span>{escape(t)}</span></div>"
                                           for v, t in _key_numbers(ctx)) + "</div>"]
    items = []
    for c in ctx["cases"]:
        for a in c["ages"]:
            items.append((f"{c['label']} {a:g} 年", probs(at(c["rows"], a))))
    out.append(figure(Raw(stack_chart(items)), "ケース・供用年数ごとの判定の確率（各 200 回の抽出）"))
    li = []
    if ctx["normal"]:
        worst = max(ctx["normal"], key=lambda c: 1 - probs(at(c["rows"], c["ages"][-1]))[STATUS[0]][0])
        p = probs(at(worst["rows"], worst["ages"][-1]))
        li.append(f"常時: 最も厳しいのは {worst['label']} の {worst['ages'][-1]:g} 年で、利用可 {p[STATUS[0]][0]:.1%}・"
                  f"注意事項あり {p[STATUS[1]][0]:.1%}・利用不可 {p[STATUS[2]][0]:.1%}。")
    for c in ctx["eq"]:
        worse, other, stats = _driver(c)
        p0, p1 = probs(at(c["rows"], c["ages"][0])), probs(at(c["rows"], c["ages"][-1]))
        s = (f"{c['level']} 地震後: 利用不可の確率は {c['ages'][0]:g} 年（劣化なし）{p0[STATUS[2]][0]:.1%} → "
             f"{c['ages'][-1]:g} 年 {p1[STATUS[2]][0]:.1%}。同じ地震で判定が悪くなった抽出は {len(worse)}/{len(worse) + len(other)} 件")
        if stats and len(worse) >= MIN_WORSE:
            top = max(stats, key=lambda x: x[5])
            s += f"で、悪くなった抽出は{top[1].split(' [')[0]}が特に大きい（平均 {top[2].format(top[3])}、それ以外 {top[2].format(top[4])}）"
        li.append(s + "。")
    out.append("<h3>主な結果</h3><ul class='list'>" + "".join(f"<li>{escape(x)}</li>" for x in li) + "</ul>")
    out.append('<p class="note">分布・判定基準・必要な荷重は仮の値。杭頭の腐食の仮定が地震後の結果を大きく左右する（8 章）。'
               '各シナリオの読み方は 7 章。</p>')
    return "".join(out)


def sec_chain(study, ctx):
    out = [figure(Raw(full_svg(pier_chain)[0]), "RC 桟橋の変状連鎖図（港空研資料 No.1328 の図-3.4 に基づく。灰色: 現状の方法で"
                  "点検診断できる変状、白の太枠: 点検手法がない変状、計: rcframe で計算している箇所）")]
    used = defaultdict(list)
    for c in ctx["cases"]:
        for s in c["scenarios"]:
            used[s].append(c["label"])
    model = {"S1": "梁の下端筋の断面欠損率 → 許容荷重の Mu", "S2": "杭の部位ごとの腐食量 → ヒンジの Mp・M–φ・εcu",
             "S3": "海底面の地盤変位・慣性力（側方流動）→ 増分解析", "S4": "海底面の低下 → 杭の突出長・地盤ばね",
             "S5": "土留部の移動・背後の沈下 → 渡版の開き・段差（後処理）"}
    rows = []
    for sid, name, members, note in pier_chain.SCENARIOS:
        codes = sorted({c for m in members for c in pier_chain.NODE[m]["codes"]})
        rows.append([sid, name, "".join(code_str(pier_chain, c) for c in codes) or "-", model.get(sid, ""),
                     "、".join(used.get(sid, [])) or "-"])
    out.append(wrapt(table(["", "重点シナリオ", "点検診断の項目", "解析での扱い", "ケース"], rows)))
    out.append('<p class="note">点検診断の項目の番号は港空研資料 No.1328 の表-4.1（港湾の施設の点検診断ガイドラインによる）。</p>')
    return "".join(out)


def _show(v):
    if isinstance(v, dict) and len(v) == 1:
        k, a = next(iter(v.items()))
        if k == "lognormal":
            return f"対数正規 中央値 {a['median']:g}、β {a['beta']:g}"
        if k == "normal":
            return f"正規 平均 {a['mean']:g}、標準偏差 {a['sd']:g}"
        if k == "uniform":
            return f"一様 [{a[0]:g}, {a[1]:g}]"
        if k == "choice":
            return "選択 " + " ／ ".join(_show_dict(x) if isinstance(x, dict) else f"{x:g}" for x in a)
    if isinstance(v, dict):
        return _show_dict(v)
    if isinstance(v, float):
        return f"{v:g}"
    return json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)


def _show_dict(d):
    return "・".join(f"{k} {x:g}" for k, x in d.items())


DET_DOC = {
    ("chloride", "C0"): ("表面塩化物イオン濃度 [kg/m³]", "示方書の飛沫帯 13.0"),
    ("chloride", "D"): ("見かけの拡散係数 [cm²/年]", "示方書式 log D = 3.0·W/C − 1.8 で W/C 0.55 → 約 0.7、ひび割れを見込む"),
    ("chloride", "cover"): ("かぶり（主筋の表面まで）[mm]", "入力の値と施工のばらつき"),
    ("chloride", "Clim"): ("腐食発生限界濃度 [kg/m³]", "示方書式 −3.0·W/C + 3.4 で約 1.75、旧来の 1.2"),
    ("chloride", "icorr"): ("腐食電流密度 [μA/cm²]", "1 μA/cm² で半径 0.0116 mm/年"),
    ("chloride", "bar"): ("下端筋の径 [mm]", "梁の配筋（中央 D22・端部 D25 の小さい方）"),
    ("pile", "head"): ("杭頭側（飛沫帯・干満帯）", "モルタルライニングの寿命の後、港湾基準の標準値 0.1〜0.3 mm/年"),
    ("pile", "sea"): ("海中部〜海底面", "電気防食（防食率 90%）の寿命の後、標準値 0.1〜0.2 mm/年"),
    ("pile", "soil"): ("地中部", "標準値 0.03 mm/年"),
    ("scour", "depth"): ("平衡の洗掘深さ（海側の列）[m]", "仮定（データなし）"),
    ("scour", "tau"): ("洗掘の時定数 [年]", "仮定"),
}


def sec_dists(study, ctx):
    det = {}
    dist, ret = {}, {}
    for c in ctx["cases"]:
        for b, v in ((c["data"]["scenarios"].get("deterioration") or {}).items()):
            if b != "age":
                det.setdefault(b, v)
        for k, v in (c["data"]["scenarios"].get("distributions") or {}).items():
            (ret if k in ("wall_ratio", "settlement_ratio", "settlement") else dist).setdefault((c["level"], k), v)
    rows = []
    for b in ("chloride", "chloride_top", "pile", "scour"):
        if b not in det:
            continue
        sid = BLOCK_SCENARIO[b]
        for k, v in det[b].items():
            if b == "pile" and k in ("head", "sea", "soil"):
                name, basis = DET_DOC[(b, k)]
                parts = [f"{ {'life': '寿命 [年]', 'rate': '腐食速度 [mm/年]', 'factor': '寿命までの速度の比'}[n]} {_show(x)}"
                         for n, x in v.items()]
                rows.append([sid, name, "；".join(parts), basis])
            elif b == "pile" and k == "boundary":
                rows.append([sid, "杭頭側の下端の標高 [m]", f"{v:+g}（L.W.L. − 1 m、L.W.L. ±0 と仮定）", "港湾基準の腐食の区分"])
            elif b == "scour" and k == "ratio":
                rows.append([sid, "洗掘の列ごとの比", _show(v), "仮定"])
            else:
                name, basis = DET_DOC.get((b if b != "chloride_top" else "chloride", k), (k, ""))
                rows.append([sid, name, _show(v), basis])
    out = ["<h3>劣化（S1・S2・S4）</h3>", wrapt(table(["", "変数", "分布", "根拠"], rows))]
    if dist:
        out.append("<h3>地震（S3）</h3>")
        out.append(wrapt(table(["地震", "変数", "分布"], [[lv, DIST_NAME.get(k, k), _show(v)]
                                                       for (lv, k), v in sorted(dist.items())])))
    if ret:
        out.append("<h3>土留部（S5）</h3>")
        out.append(table(["地震", "変数", "分布"], [[lv, DIST_NAME.get(k, k), _show(v)] for (lv, k), v in sorted(ret.items())]))
    # 劣化の量（抽出した値から作った変数）: 劣化の全部を含むケースで年数ごと
    ref = max(ctx["cases"], key=lambda c: (len(c["scenarios"]), not c["eq"]))
    trs = []
    for a in ref["ages"]:
        g = at(ref["rows"], a)
        for k, name, fmt in DET_KEYS:
            v = [r[k] for r in g if _num(r.get(k))]
            if v:
                trs.append([f"{a:g}", name, fmt.format(np.mean(v)), fmt.format(np.median(v)), fmt.format(quant(v, 0.9)),
                            fmt.format(quant(v, 0.99))])
        ti = [r.get("t_init") for r in g]
        if any(r.get("rebar_loss_bottom") is not None for r in g):
            trs.append([f"{a:g}", "鉄筋の腐食が始まっている割合", f"{sum(t is not None and t < a for t in ti) / len(g):.0%}",
                        "", "", ""])
        trs.append([f"{a:g}", "開孔した杭のあるシナリオ", f"{sum(bool(r.get('perforation')) for r in g)} 件", "", "", ""])
    out.append(f"<h3>劣化の量（{escape(ref['label'])} の抽出）</h3>")
    out.append(table(["年数", "変数", "平均", "中央値", "90%", "99%"], trs, num=(2, 3, 4, 5)))
    secs = {k: v for k, v in ref["data"]["sections"].items() if "pipe" in v}
    out.append('<p class="note">杭の板厚: ' + "、".join(f"{k} φ{v['pipe']['D']:g}×t{v['pipe']['t']:g}" for k, v in secs.items())
               + "。杭の断面の腐食代は 0 とし、腐食は劣化のモデルだけで与えた（0 年 = 建設時）。</p>")
    return "".join(out)


def sec_method(study, ctx):
    c = ctx["cases"][0]
    sc = c["data"]["scenarios"]
    out = ["<h3>劣化のモデル</h3><ul class='list'>",
           "<li>塩化物イオン（Fick の第 2 法則、表面濃度一定）: C(x, t) = Ci + (C0 − Ci)·erfc(x/(2√(D·t)))。"
           "腐食の開始 t_i = x²/(4·D·z²)、z = erfc⁻¹((Clim − Ci)/(C0 − Ci))</li>",
           "<li>鉄筋の一様腐食: 半径の減少 Δr = 0.0116·icorr·(t − t_i) [mm]、断面欠損率 1 − ((φ − 2Δr)/φ)²</li>",
           "<li>杭の腐食: 腐食量 = rate·(factor·min(t, life) + max(t − life, 0))（寿命までは防食、その後は腐食速度 rate）</li>",
           "<li>洗掘: 深さ = depth·(1 − exp(−t/τ))</li></ul>",
           "<h3>抽出と計算</h3>",
           f"<p>各ケース {sc.get('sample')} 回の抽出（乱数の種 {sc.get('seed')}）。劣化の抽出値は項目（塩害・杭・洗掘）ごとの乱数系列で、"
           "ケースの間で共通。供用年数の間でも抽出値は共通で、年数による劣化の量だけが違う。地震後は地震（地盤変位・慣性力・"
           "流動層の低減）と土留部の比を同時に抽出し、年数の間で共通にした（同じ地震で劣化の有無を比べられる）。</p>",
           "<p>解析は rcframe の事象追跡型の荷重増分解析（地盤ばねモデル、杭の塑性ヒンジ、除荷）。1段目は粗い分割と cos 式の"
           "ヒンジ、2段目はファイバーの M–φ（εcu 式B）のヒンジ。常時は全てを 2段目で計算し、地震後は杭が降伏したか残留変位が "
           "0.05 m 以上のものを 2段目で計算した"
           + ("（1段目の残留変位 0.5 m 以上は利用不可に決まるので除く）。</p>" if any(
               (e["data"]["scenarios"].get("refine") or {}).get("skip_above") for e in ctx["eq"]) else "。</p>"),
           "<h3>判定基準と必要な荷重</h3>"]
    if any(_methods(cc) for cc in ctx["cases"]):
        for cc in (ctx["eq"][:1] or ctx["normal"][:1]):
            out.append(criteria_html(cc["opts"]))
        out.append('<p class="note">常時のケースは地震がないので、本震時・余震・土留部の条件はなく、利用方法ごとの利用時の荷重だけで'
                   "判定する。数値は仮の値（想定船舶の牽引力・接岸力、余震の震度、土留部・渡版・段差の閾値）。</p>")
        return "".join(out)
    trs = []
    for cc in (ctx["normal"][:1] + ctx["eq"][:1]):
        lim = cc["opts"]["limits"]
        names = {"residual_displacement": "上部工の残留変位", "bridge_opening": "渡版の開き（上部工 − 土留部）",
                 "bridge_closing": "土留部が上部工に寄る量", "apron_step": "土留部の背後の段差"}
        for k, name in names.items():
            if k in lim:
                trs.append([cc["level"], name, f"{lim[k][0] / 1e3:g} m 以上", f"{lim[k][1] / 1e3:g} m 以上"])
        if "pile_rotation" in lim:
            trs.append([cc["level"], "杭の残留の塑性回転角", f"{lim['pile_rotation'][0]:g} rad 以上", f"{lim['pile_rotation'][1]:g} rad 以上"])
        trs.append([cc["level"], "杭の損傷", "塑性化", "局部座屈"])
        trs.append([cc["level"], "許容荷重（必要な値: " + "、".join(
            f"{LOAD_NAME[k]} {load_text(k, v)}" for k, v in required_values(cc["opts"]).items()) + "）",
            "必要な値を下回る", "0（載せられない）"])
    out.append(table(["", "項目", "注意事項あり", "利用不可"], trs))
    out.append('<p class="note">いずれも仮の値。許容荷重は、被災状態（常時は長期荷重の状態）から単位荷重を増やし、杭の塑性回転の'
               '増加・梁の |M| = Mu に最初に達する荷重係数。</p>')
    return "".join(out)


def sec_normal(study, ctx):
    if not ctx["normal"]:
        return None
    out = []
    trs = []
    for c in ctx["normal"]:
        for a in c["ages"]:
            g = at(c["rows"], a)
            p = probs(g)
            rs = Counter()
            for lv in STATUS[1:]:
                rs.update(reason_counts(g, lv))
            trs.append([c["label"], f"{a:g}"] + [f"{p[s][0]:.1%}" for s in STATUS]
                       + ["、".join(f"{k} {v}" for k, v in rs.most_common(4)) or "-"])
    out.append(table(["ケース", "年数"] + list(STATUS) + ["注意事項あり以上の理由（件数、重複あり）"], trs, num=(1, 2, 3, 4)))
    out.append("<h3>許容荷重</h3>")
    trs = []
    for c in ctx["normal"]:
        req = required_values(c["opts"])
        for a in c["ages"]:
            g = at(c["rows"], a)
            row = [c["label"], f"{a:g}"]
            for k in ("surcharge", "berthing", "tractive"):
                v = [r[f"allow_{k}"] for r in g if _num(r.get(f"allow_{k}"))]
                row.append(f"{np.median(v):,.0f}（{quant(v, 0.05):,.0f}）・{np.mean([x < req[k] for x in v]):.1%}")
            trs.append(row)
    out.append(table(["ケース", "年数", "上載荷重 [kN/m²]", "接岸力 [kN]", "牽引力 [kN]"], trs, num=(1,)))
    out.append('<p class="note">各欄: 中央値（5% 値）・必要な値を下回る確率。</p>')
    full = max(ctx["normal"], key=lambda c: len(c["scenarios"]))
    a = full["ages"][-1]
    g = at(full["rows"], a)
    pts = [(r["corrosion_head"], r["allow_berthing"], f"pt{STATUS.index(r['status'])}",
            f"{r['status']}: 杭頭 {r['corrosion_head']:.1f} mm、許容接岸力 {r['allow_berthing']:,.0f} kN")
           for r in g if _num(r.get("corrosion_head")) and _num(r.get("allow_berthing"))]
    if pts:
        out.append(figure(Raw(svg.scatter_chart(pts, "杭頭側の腐食量 [mm]", "許容接岸力 [kN]",
                                                LEGEND_ST, yfmt="{:,.0f}")),
                          f"{full['label']} {a:g} 年: 杭頭側の腐食量と許容接岸力（色は判定）"))
    pts = [(r["rebar_loss_bottom"] * 100, r["allow_surcharge"], f"pt{STATUS.index(r['status'])}",
            f"{r['status']}: 欠損率 {r['rebar_loss_bottom']:.1%}、許容上載荷重 {r['allow_surcharge']:.0f}")
           for r in g if _num(r.get("rebar_loss_bottom")) and _num(r.get("allow_surcharge"))]
    if pts:
        out.append(figure(Raw(svg.scatter_chart(pts, "梁の下端筋の断面欠損率 [%]", "許容上載荷重 [kN/m²]",
                                                LEGEND_ST)),
                          f"{full['label']} {a:g} 年: 下端筋の断面欠損率と許容上載荷重（色は判定）"))
    return "".join(out)


def _bins(rows, key, edges, flag):
    out = []
    for lo, hi in zip(edges, edges[1:]):
        g = [r for r in rows if _num(r.get(key)) and lo <= r[key] < hi]
        out.append((lo, hi, len(g), sum(flag(r) for r in g)))
    return out


def sec_eq(study, ctx):
    if not ctx["eq"]:
        return None
    out = []
    trs = []
    for c in ctx["eq"]:
        for a in c["ages"]:
            g = at(c["rows"], a)
            p = probs(g)
            noret = Counter(rejudge(g, c["opts"], RETAIN_LIMITS))
            rs = reason_counts(g, STATUS[2])
            trs.append([c["label"], f"{a:g}"] + [f"{p[s][0]:.1%}±{p[s][1]:.1%}" for s in STATUS]
                       + [f"{noret[STATUS[2]] / len(g):.1%}", "、".join(f"{k} {v}" for k, v in rs.most_common(5)) or "-"])
    out.append(wrapt(table(["ケース", "年数"] + list(STATUS) + ["利用不可（土留部の基準を除く）", "利用不可の理由（件数、重複あり）"],
                           trs, num=(1, 2, 3, 4, 5))))
    out.append('<p class="note">± は標準誤差。年数の間で地震の抽出値は共通。</p>')
    out.append("<h3>同じ地震での年数の比較</h3>")
    trs = []
    for c in ctx["eq"]:
        a0 = c["ages"][0]
        for a in c["ages"][1:]:
            pr = [(v[a0], v[a]) for v in paired(c["rows"]) if a0 in v and a in v]
            tr = Counter((x["status"], y["status"]) for x, y in pr if x["status"] != y["status"])
            worse = sum(STATUS.index(y["status"]) > STATUS.index(x["status"]) for x, y in pr)
            better = sum(STATUS.index(y["status"]) < STATUS.index(x["status"]) for x, y in pr)
            trs.append([c["level"], f"{a0:g} → {a:g}", f"{worse}", f"{better}", f"{len(pr)}",
                        "、".join(f"{x}→{y} {n}" for (x, y), n in tr.most_common())])
    out.append(wrapt(table(["地震", "年数", "悪化", "改善", "抽出", "判定の変化（件数）"], trs, num=(2, 3, 4))))
    for c in ctx["eq"]:
        worse, other, stats = _driver(c)
        if len(worse) >= MIN_WORSE and stats:
            out.append(f"<p>{escape(c['level'])} で {c['ages'][-1]:g} 年に悪化した {len(worse)} 件と、それ以外の劣化の量の平均:</p>")
            out.append(table(["変数", "悪化した抽出", "それ以外", "差 / 標準偏差"],
                             [[name, fmt.format(w), fmt.format(o), f"{d:+.2f}"] for k, name, fmt, w, o, d in stats],
                             num=(1, 2, 3)))
    # 地盤変位の区間ごとの利用不可の割合
    for c in ctx["eq"]:
        series, cls = [], ["c0", "c1", "c4", "c2", "c3"]
        trs = []
        for i, a in enumerate(c["ages"]):
            b = _bins(at(c["rows"], a), "surface", SURF_BINS, lambda r: r["status"] == STATUS[2])
            series.append((f"{a:g} 年", [((lo + (hi if math.isfinite(hi) else lo * 1.5)) / 2, k / n) for lo, hi, n, k in b if n >= 5],
                           cls[i % len(cls)]))
            for lo, hi, n, k in b:
                if n:
                    trs.append((lo, hi, a, n, k))
        if any(len(s[1]) > 1 for s in series):
            out.append(figure(Raw(svg.xy_chart(series, "陸側の列の海底面の地盤変位 [m]（区間の中央）", "利用不可の割合",
                                               xfmt="{:.2f}", yfmt="{:.0%}")),
                              f"{c['level']} 地震後: 地盤変位の区間ごとの利用不可の割合（5 件以上の区間）"))
        edges = sorted({(lo, hi) for lo, hi, *_ in trs})
        rows = []
        for lo, hi in edges:
            cells = []
            for a in c["ages"]:
                x = next((t for t in trs if t[0] == lo and t[1] == hi and t[2] == a), None)
                cells.append("-" if x is None else f"{x[4]}/{x[3]}")
            rows.append([f"{lo:g}〜{hi:g}" if math.isfinite(hi) else f"{lo:g}〜"] + cells)
        out.append(f"<details><summary>{escape(c['level'])}: 地盤変位の区間ごとの利用不可の件数</summary>"
                   + table(["地盤変位 [m]"] + [f"{a:g} 年" for a in c["ages"]], rows, num=tuple(range(1, len(c["ages"]) + 1)))
                   + "</details>")
        fr = []
        for a in c["ages"]:
            g = at(c["rows"], a)
            x = [float(r["surface"]) for r in g]
            f1 = safe_fit(x, [r["status"] != STATUS[0] for r in g])
            f2 = safe_fit(x, [r["status"] == STATUS[2] for r in g])
            fmt = lambda f: "推定できない" if f is None or f[1] < 0.02 or f[1] > 5 else f"θ = {f[0]:.3f} m、β = {f[1]:.2f}"
            fr.append([f"{a:g}", fmt(f1), fmt(f2)])
        out.append(f"<details><summary>{escape(c['level'])}: フラジリティ曲線 P = Φ(ln(x/θ)/β) の最尤推定</summary>"
                   + table(["年数", "注意事項あり以上", "利用不可"], fr)
                   + '<p class="note">劣化した状態では、地盤変位が小さくても杭頭の局部座屈などで利用不可になる抽出があり、'
                     '対数正規のフラジリティ曲線にはよく当てはまらない（β が大きくなる）。判定がほぼ全て同じ場合は推定できない。</p></details>')
    # 杭頭の腐食量と局部座屈（最初の地震のケースの最後の年数）
    c = ctx["eq"][0]
    a = c["ages"][-1]
    g = at(c["rows"], a)
    pts = [(r["corrosion_head"], float(r["surface"]), f"pt{STATUS.index(r['status'])}",
            f"{r['status']}: 杭頭 {r['corrosion_head']:.1f} mm、地盤変位 {r['surface']:.3f} m")
           for r in g if _num(r.get("corrosion_head"))]
    if pts:
        out.append(f"<h3>杭頭の腐食と局部座屈（{escape(c['level'])}、{a:g} 年）</h3>")
        out.append(figure(Raw(svg.scatter_chart(pts, "杭頭側の腐食量 [mm]", "陸側の列の海底面の地盤変位 [m]",
                                                LEGEND_ST, yfmt="{:.2f}")),
                          f"{c['level']} 地震後 {a:g} 年: 杭頭側の腐食量と地盤変位（色は判定）"))
        b = _bins(g, "corrosion_head", HEAD_BINS, lambda r: bool(r.get("n_buckled")))
        bu = _bins(g, "corrosion_head", HEAD_BINS, lambda r: r["status"] == STATUS[2])
        out.append(table(["杭頭側の腐食量 [mm]", "抽出", "局部座屈あり", "利用不可"],
                         [[f"{lo:g}〜{hi:g}" if math.isfinite(hi) else f"{lo:g}〜", n, f"{k}（{k / n:.0%}）" if n else "-",
                           f"{k2}（{k2 / n:.0%}）" if n else "-"] for (lo, hi, n, k), (_, _, _, k2) in zip(b, bu)],
                         num=(1, 2, 3)))
    return "".join(out)


def _pct(p):
    return f"{p:.1%}"


def sec_discussion(study, ctx):
    by = {c["name"]: c for c in ctx["cases"]}
    nm = {s: f"{s} {SC_NAME.get(s, '')}" for s in ("S1", "S2", "S3", "S4", "S5")}
    out = []

    def case_with(ids, eq=False):
        cs = [c for c in ctx["cases"] if c["eq"] == eq and set(c["scenarios"]) == set(ids)]
        return cs[0] if cs else None

    # S1
    c = case_with(["S1"])
    if c:
        a = c["ages"][-1]
        g = at(c["rows"], a)
        v = [r["allow_surcharge"] for r in g if _num(r.get("allow_surcharge"))]
        g0 = at(c["rows"], c["ages"][0])
        v0 = [r["allow_surcharge"] for r in g0 if _num(r.get("allow_surcharge"))]
        gov = Counter(r.get("allow_surcharge_by", "").split("（")[0] for r in g).most_common(1)[0]
        req_q = required_values(c["opts"])["surcharge"]                 # 利用方法があれば最も大きい値
        out.append(f"<h3>{escape(nm['S1'])}</h3><p>常時の許容上載荷重の中央値は {c['ages'][0]:g} 年 {np.median(v0):.0f}、"
                   f"{a:g} 年 {np.median(v):.0f} kN/m²（5% 値 {quant(v, 0.05):.0f}）で、{gov[1]}/{len(g)} 件で{escape(gov[0])}"
                   f"が限界になる。必要な値 {req_q:g} kN/m² を下回るのは {a:g} 年で "
                   f"{_pct(float(np.mean([x < req_q for x in v])))}。下端筋の欠損は梁の Mu だけに効くため、地震後の判定への影響は小さい"
                   "（6 章の悪化した抽出とそれ以外で欠損率の差は小さい）。予防保全では、腐食が外観に現れる前の塩化物イオン量・"
                   "かぶり（点検 (8) の詳細調査）で腐食の開始時期を絞ることが効く。</p>")
    # S2
    c = case_with(["S2"])
    if c:
        a = c["ages"][-1]
        g = at(c["rows"], a)
        p = probs(g)
        perf = sum(bool(r.get("perforation")) for r in g)
        req = required_values(c["opts"])
        pb = np.mean([r["allow_berthing"] < req["berthing"] for r in g if _num(r.get("allow_berthing"))])
        s = (f"<h3>{escape(nm['S2'])}</h3><p>常時の {a:g} 年で、利用可 {_pct(p[STATUS[0]][0])}・注意事項あり "
             f"{_pct(p[STATUS[1]][0])}・利用不可 {_pct(p[STATUS[2]][0])}。許容接岸力が必要な値を下回る確率は {_pct(pb)}、"
             f"腐食量が板厚に達した杭（開孔）のある抽出は {perf} 件。")
        if ctx["eq"]:
            e = ctx["eq"][0]
            ea = e["ages"][-1]
            b = _bins(at(e["rows"], ea), "corrosion_head", HEAD_BINS, lambda r: bool(r.get("n_buckled")))
            hi = [(lo, n, k) for lo, hi_, n, k in b if lo >= 7 and n]
            lo_ = [(n, k) for lo, hi_, n, k in b if lo < 7 and n]
            if hi and lo_:
                s += (f"{e['level']} 地震後の {ea:g} 年では、杭頭側の腐食量 7 mm 以上の抽出の "
                      f"{sum(k for _, _, k in hi) / sum(n for _, n, _ in hi):.0%} で杭頭が局部座屈した（7 mm 未満は "
                      f"{sum(k for _, k in lo_) / sum(n for n, _ in lo_):.0%}）。")
        s += ("地震後の利用可否を最も左右するのはこの経路で、変状連鎖図の上流の被覆防食・陽極（点検 (6)(7)、Ⅱ類）を"
              "見て、鋼管杭の腐食（(5)、Ⅰ類）になる前に手当てする予防保全の意味が数値で表れている。</p>")
        out.append(s)
    # S3
    for e in ctx["eq"]:
        a0 = e["ages"][0]
        g = at(e["rows"], a0)
        p = probs(g)
        f2 = safe_fit([float(r["surface"]) for r in g], [r["status"] == STATUS[2] for r in g])
        s = (f"<h3>{escape(nm['S3'])}（{escape(e['level'])}、劣化なし）</h3><p>{a0:g} 年（劣化なし）の利用不可 {_pct(p[STATUS[2]][0])}、"
             f"注意事項あり {_pct(p[STATUS[1]][0])}。")
        if f2 and 0.02 < f2[1] < 5:
            s += f"利用不可のフラジリティ曲線は中央値 {f2[0]:.3f} m（β {f2[1]:.2f}）。"
        rs = reason_counts(g, STATUS[2])
        if rs:
            s += "利用不可の主な理由は " + "、".join(f"{k}（{v} 件）" for k, v in rs.most_common(3)) + "。"
        out.append(s + "</p>")
    # S4
    c = case_with(["S4"])
    if c:
        a = c["ages"][-1]
        g = sorted((r for r in at(c["rows"], a) if _num(r.get("scour"))), key=lambda r: r["scour"])
        q = max(len(g) // 4, 1)
        lo_, hi_ = g[:q], g[-q:]
        p = probs(at(c["rows"], a))
        s = (f"<h3>{escape(nm['S4'])}</h3><p>常時の {a:g} 年で利用可 {_pct(p[STATUS[0]][0])}。洗掘深さの下位 1/4（平均 "
             f"{np.mean([r['scour'] for r in lo_]):.2f} m）と上位 1/4（平均 {np.mean([r['scour'] for r in hi_]):.2f} m）で、"
             f"許容接岸力の中央値は {np.median([r['allow_berthing'] for r in lo_]):,.0f} → "
             f"{np.median([r['allow_berthing'] for r in hi_]):,.0f} kN。")
        for e in ctx["eq"]:
            ea = e["ages"][-1]
            ge = sorted((r for r in at(e["rows"], ea) if _num(r.get("scour"))), key=lambda r: r["scour"])
            qe = max(len(ge) // 4, 1)
            pl = float(np.mean([r["status"] == STATUS[2] for r in ge[:qe]]))
            ph = float(np.mean([r["status"] == STATUS[2] for r in ge[-qe:]]))
            sd = math.sqrt(pl * (1 - pl) / qe + ph * (1 - ph) / qe)
            s += (f"{e['level']} 地震後の {ea:g} 年の利用不可は、洗掘の下位 1/4 で {_pct(pl)}、上位 1/4 で {_pct(ph)}"
                  + ("（差は標準誤差 " + _pct(sd) + " の 2 倍未満で、はっきりした違いはない）。" if abs(ph - pl) < 2 * sd
                     else f"（差 {ph - pl:+.1%}、標準誤差 {_pct(sd)}）。"))
        s += ("平衡の洗掘深さの中央値 0.5 m の仮定では、判定への影響は小さい。洗掘深さの分布は仮定なので、"
              "深浅測量（点検 (1)）で確かめることが前提になる。</p>")
        out.append(s)
    # S5
    for e in ctx["eq"]:
        if "S5" not in e["scenarios"]:
            continue
        trs = []
        for a in e["ages"]:
            g = at(e["rows"], a)
            p = probs(g)
            nr = Counter(rejudge(g, e["opts"], RETAIN_LIMITS))
            n_step = sum(any("段差" in x for x in (r.get("reasons") or "").split(" / ")) for r in g)
            n_open = sum(any("渡版の開き" in x for x in (r.get("reasons") or "").split(" / ")) for r in g)
            trs.append([f"{a:g}", _pct(1 - p[STATUS[0]][0]), _pct(1 - nr[STATUS[0]] / len(g)), n_step, n_open])
        out.append(f"<h3>{escape(nm['S5'])}（{escape(e['level'])}）</h3>")
        out.append(table(["年数", "注意事項あり以上", "同（土留部の基準を除く）", "段差が理由", "渡版の開きが理由"], trs,
                         num=(1, 2, 3, 4)))
    out.append('<p class="note">土留部の移動・沈下は地盤変位に比を掛けた仮定で、土留部の構造形式（点検 (4)）に応じた検討が必要。</p>')
    return "".join(out)


def _known(ctx):
    """留意事項（ケースの設定に合わせる）"""
    guide = any(_methods(c) for c in ctx["cases"])
    skip = any((e["data"]["scenarios"].get("refine") or {}).get("skip_above") for e in ctx["eq"])
    out = []
    for x in KNOWN:
        if x.startswith("地震後は、1段目の残留変位が skip_above") and not skip:
            continue
        if x.startswith("判定基準・必要な荷重は仮の値") and guide:
            x = ("判定はガイドライン第2版の表 D-5 に合わせたが、利用方法の荷重（想定船舶の牽引力・接岸力）、余震の震度（L1 相当）、"
                 "土留部の残留水平変位・渡版の開き・段差の閾値は仮の値")
        out.append(x)
    if guide:
        out += ["余震は本震後の状態に慣性力 kh·W を静的に漸増させた（余震による地盤変位の増加・液状化なし）。梁の限界は Mu"
                "（材料係数 1.0）で、ガイドラインの設計断面耐力とは係数が異なる",
                "杭の支持力は設計支持力の入力があるときだけ確かめる（この計算例は入力なし）。フェーズ 1 の特性値は設計値で代える"]
    return out


def sec_notes(study, ctx):
    return ("<h3>仮定と簡略化</h3><ul class='list'>" + "".join(f"<li>{escape(x)}</li>" for x in _known(ctx)) + "</ul>"
            + "<h3>結果を確かにするために必要なデータ</h3>"
            + table(["データ（点検診断の項目）", "用途"], NEEDS))


def sec_appendix(study, ctx):
    out = [f"<h3>再現の方法</h3><pre class='src'>python examples/{escape(ctx['root'].name)}.py -j 6        # 計算（再開できる）\n"
           f"python -m rcframe.deterioration_report {escape(str(ctx['root'].as_posix()))} -o 報告書.html</pre>",
           "<h3>照合（テスト）</h3>",
           table(["項目", "照合した理論値・手計算", "テスト"], [
               ["劣化の変数（D1）", "下端筋の欠損で Mu = As·fy·(d − a/2)、部位ごとのヒンジ = 一様な腐食の断面・手計算 Mp(N)、"
                "洗掘で自由長が増える", "tests/test_deterioration.py"],
               ["劣化のモデル（D2）", "erfc(1) で t_i = x²/(4D)、t_i で濃度 = Clim、欠損率・腐食量の手計算、抽出の統計量",
                "tests/test_deterioration_model.py"],
               ["土留部と渡版（D3）", "判定の境界、移動・開き・段差の手計算", "tests/test_retaining.py"],
               ["報告書（D5）", "表の確率 = 結果の行の集計、外部参照なし", "tests/test_deterioration_report.py"],
               ["本震時の記録（G1）", "梁の M/Mu = section_forces_at の最大、長期の軸力 = 線形解析", "tests/test_main_shock.py"],
               ["余震（G2）", "弾性範囲の許容震度 = 両向きの線形解析の重ね合わせ", "tests/test_aftershock.py"],
               ["利用方法ごとの判定（G3）", "表 D-5 の項目ごとの不可の範囲、注意事項、荷重", "tests/test_methods.py"]])]
    for c in ctx["cases"]:
        out.append(f"<details><summary>入力 {escape(c['name'])}.yaml（{escape(c['label'])}）</summary>"
                   f"<pre class='src'>{escape(c['text'])}</pre></details>")
    return "".join(out)


# ================================================================ ガイドライン（G4）
D5_ROWS = [
    ["杭の損傷", "フェーズ 1: 限界曲率以下（局部座屈 = θu に達したヒンジがない）／フェーズ 2・3: 降伏以下（降伏したヒンジがない）",
     "降伏以下: 本震後に kh·W を両向きに漸増し、杭の塑性回転が増える震度 ≧ 想定の余震の震度", "降伏以下: 許容荷重 ≧ 利用方法の荷重"],
    ["杭の支持力", "杭の軸力 / 設計支持力 < 1（入力があるとき）", "同（余震の漸増の限界）", "同（許容荷重の限界）"],
    ["上部工の断面破壊", "本震の履歴中の梁の M/Mu < 1", "梁の M ≦ Mu（余震の漸増の限界）", "梁の M ≦ Mu（許容荷重の限界）"],
    ["土留部の変状", "土留部の残留水平変位（比 × 陸側の地盤変位）≦ 閾値（フェーズ 2・3 / 全て）", "-", "-"],
]


def _methods(c):
    return c["opts"].get("methods") or {}


def _usable(r, m):
    return m in (r.get("methods_ok") or "").split(METHOD_SEP)


def _res(r):
    return abs(r["deck_residual"])


def _pse(k, n):
    p = k / n if n else math.nan
    return f"{p:.1%}±{math.sqrt(p * (1 - p) / n):.1%}" if n else "-"


def _counts_text(cnt, top=5):
    return "、".join(f"{k} {v}" for k, v in cnt.most_common(top)) or "-"


def _phase_method(ms, phase):
    """フェーズ phase の利用方法（なければ最初）の名前"""
    return next((m for m, v in ms.items() if v["phase"] == phase), next(iter(ms)))


def _head_allowed(opts):
    """杭頭の限界曲率の超過を許す（フェーズ 1 は可）判定の設定"""
    lim = {**opts["limits"], "pile_damage": {**opts["limits"]["pile_damage"], "杭頭の局部座屈": "limited"}}
    return {**opts, "limits": lim}


def sec_guideline(study, ctx):
    cs = [c for c in ctx["cases"] if _methods(c)]
    if not cs:
        return None
    ms = _methods(cs[0])
    out = ["<p>港湾施設の利用可否判断に係るガイドライン 第2版（国土交通省港湾局、令和8年5月）の付録 D（表 D-5）に合わせ、"
           "利用方法（利用フェーズ × 利用形態、表 D-1）ごとに、本震時・余震時・地震後の利用時の部材の条件で判定した。"
           "区分は能登半島地震の際の公表と同じ「利用可／利用可（注意事項あり）／利用不可」で、利用可は全ての利用方法が使えて"
           "注意事項（緊急復旧）がないもの、利用不可はどの利用方法も使えないもの。</p>",
           "<h3>利用方法</h3>",
           table(["利用方法", "フェーズ", "内容"] + [LOAD_NAME[k] for k in ("surcharge", "berthing", "tractive")],
                 [[m, str(v["phase"]), v["label"] or "-"] + [load_text(k, v["loads"][k]) if k in v["loads"] else "-"
                                                             for k in ("surcharge", "berthing", "tractive")]
                  for m, v in ms.items()]),
           "<h3>表 D-5 の条件と解析での扱い</h3>",
           wrapt(table(["確認項目", "本震時", "余震時", "地震後の利用時"], D5_ROWS)),
           '<p class="note">本震時の条件は地震の履歴の全体で満たす条件（静的な増分解析の載荷から除荷まで）。余震は '
           + escape(", ".join(f"{c['level']} kh = {c['opts']['required']['aftershock']:g}" for c in cs
                              if "aftershock" in c["opts"]["required"]) or "なし")
           + "。施設の利用の観点（渡版の開き・背後の段差）は緊急復旧を前提とした注意事項とした。</p>"]
    # 利用方法ごとの使える割合
    trs = []
    for c in cs:
        for a in c["ages"]:
            g = at(c["rows"], a)
            p = probs(g)
            trs.append([c["label"], f"{a:g}"] + [_pse(sum(_usable(r, m) for r in g), len(g)) for m in ms]
                       + [f"{p[s][0]:.1%}" for s in STATUS])
    out.append("<h3>利用方法ごとの使える割合</h3>")
    out.append(wrapt(table(["ケース", "年数"] + [f"{m} が使える" for m in ms] + list(STATUS), trs,
                           num=tuple(range(1, len(ms) + 5)))))
    out.append('<p class="note">± は標準誤差。利用可 = 全ての利用方法が使えて注意事項がない割合。利用不可 = 想定した利用方法'
               "（利用時の荷重・船舶）のどれも使えないこと。許容荷重が想定の値に満たなくても、より小さい船舶・載荷重なら使える場合が"
               "ある（8 章の許容荷重の図を参照）。</p>")
    for c in [c for c in cs if c["eq"]]:
        series = [(m, [(a, sum(_usable(r, m) for r in at(c["rows"], a)) / len(at(c["rows"], a))) for a in c["ages"]],
                   f"c{i % 3}") for i, m in enumerate(ms)]
        same = len({tuple(_usable(r, m) for r in c["rows"]) for m in ms}) == 1
        out.append(figure(Raw(svg.xy_chart(series, "供用年数 [年]", "使える割合", xfmt="{:g}", yfmt="{:.0%}")),
                          f"{c['level']} 地震後: 利用方法ごとの使える割合と供用年数（同じ地震の抽出）"
                          + ("。全ての抽出で利用方法の可否が同じなので線が重なる" if same else "")))
        if same:
            out.append(f'<p class="note">{escape(c["level"])}: 全ての抽出で、どの利用方法も同時に使える か 同時に使えない。'
                       "利用方法を分ける条件（フェーズ 2・3 の本震時の降伏以下、上載荷重）だけで不可になる抽出は、"
                       "全ての利用方法に効く条件（余震・限界曲率・上部工）でも不可になっている。</p>")
    # 不可・注意事項の理由
    trs = []
    for c in cs:
        for a in c["ages"]:
            g = at(c["rows"], a)
            trs.append([c["label"], f"{a:g}", _counts_text(reason_counts(g, STATUS[2])),
                        _counts_text(reason_counts(g, STATUS[1]))])
    out.append("<h3>判定を決めた条件</h3>")
    out.append(wrapt(table(["ケース", "年数", "利用不可の理由（件数、重複あり）", "注意事項ありの理由（件数、重複あり）"], trs,
                           num=(1,))))
    # 本震時の上部工・杭頭・余震
    trs = []
    for c in [c for c in cs if c["eq"]]:
        req = c["opts"]["required"].get("aftershock")
        for a in c["ages"]:
            g = at(c["rows"], a)
            n = len(g)
            af = [r["allow_aftershock"] for r in g if _num(r.get("allow_aftershock"))]
            bk = [r for r in g if (r.get("n_buckled") or 0) > 0]
            head = sum(r.get("n_buckled_head") == r["n_buckled"] for r in bk)
            trs.append([c["label"], f"{a:g}", _pse(sum((r.get("deck_mu_max") or 0) >= 1 for r in g), n),
                        _pse(len(bk), n), f"{head}/{len(bk)}",
                        f"{np.median(af):.3f}（{quant(af, 0.1):.3f}）" if af else "-",
                        _pse(sum(x < req for x in af), n) if af and req is not None else "-"])
    if trs:
        out.append("<h3>本震時の上部工・杭と余震</h3>")
        out.append(wrapt(table(["ケース", "年数", "本震で梁が Mu", "局部座屈あり", "うち杭頭だけ",
                                "余震の許容震度（10% 値）", "許容震度 < 想定"], trs, num=(1, 2, 3, 4, 5, 6))))
        out.append('<p class="note">余震の許容震度は、杭の降伏・梁の Mu・杭の支持力に最初に達する震度（両向きの小さい方）。</p>')
    # 杭頭の限界曲率の超過を許した場合（港空研資料 No.1429）
    trs = []
    for c in [c for c in cs if c["eq"]]:
        o2 = _head_allowed(c["opts"])
        m1 = _phase_method(_methods(c), 1)
        for a in c["ages"]:
            g = at(c["rows"], a)
            j2 = [judge_detail(r, o2) for r in g]
            n = len(g)
            trs.append([c["label"], f"{a:g}", f"{probs(g)[STATUS[2]][0]:.1%}", f"{sum(j[0] == STATUS[2] for j in j2) / n:.1%}",
                        f"{sum(_usable(r, m1) for r in g) / n:.1%}", f"{sum(m1 in j[2] for j in j2) / n:.1%}"])
    expl = []
    for c in [c for c in cs if c["eq"]]:
        req = c["opts"]["required"].get("aftershock")
        ho = [r for r in at(c["rows"], c["ages"][-1]) if (r.get("n_buckled") or 0) > 0 and r.get("n_buckled_head") == r["n_buckled"]]
        if ho and req is not None:
            k = sum(_num(r.get("allow_aftershock")) and r["allow_aftershock"] < req for r in ho)
            expl.append(f"{c['level']} の {c['ages'][-1]:g} 年で杭頭だけが局部座屈した {len(ho)} 件のうち {k} 件は、"
                        f"余震の許容震度が想定（kh = {req:g}）に満たない（局部座屈した杭頭は余震で塑性回転が増える）。")
    if trs:
        out.append("<h3>杭頭の限界曲率の超過を許した場合</h3>")
        if expl:
            out.append("<p>" + escape("".join(expl)) + "そのため、杭頭の超過を許しても余震の条件で不可になる。</p>")
        out.append(wrapt(table(["ケース", "年数", "利用不可（既定）", "利用不可（杭頭を許す）", f"{m1} が使える（既定）",
                                f"{m1} が使える（杭頭を許す）"], trs, num=(1, 2, 3, 4, 5))))
        out.append('<p class="note">ガイドラインの表 D-5 の注は、鋼管杭の杭頭は一定程度まで限界曲率の超過を許容できる可能性がある'
                   "（港湾空港技術研究所資料 No.1429）として、専門技術者への相談を求めている。ここでは杭頭だけの局部座屈をフェーズ 1 では"
                   "可とした場合を、結果の行から判定し直した（腐食した杭頭の局部座屈後の耐力は確かめていない）。</p>")
    return "".join(out)


def _yield_disp(pier, corr):
    """表 A-1 の簡易な目安: 杭を両端固定の梁とみなし、片側を変位させて降伏する水平変位 δy = My·L²/(6EI)。
    L = 上部工の下面 − 仮想固定点（海底面 − 1/β）、My = fy·Z（外面の腐食 corr [mm] の後、軸力 0、残りの板厚は 1 mm 以上）、
    EI は腐食前。戻り値: {杭の列: δy [m]}（列の中の最小）"""
    out = {}
    for (i, j), p in pier.piles.items():
        sec, mat = pier.pile_sec[(i, j)]
        Do, Di = sec.dims(corroded=False)
        Dc = max(Do - 2 * corr, Di + 2.0)
        Z = math.pi / 32 * (Dc ** 4 - Di ** 4) / Dc
        EI = mat.E * math.pi / 64 * (Do ** 4 - Di ** 4)
        L = pier.z_soffit - p["z_fix"]
        name = pier.gy[j][0]
        out[name] = min(out.get(name, math.inf), mat.fy * Z * L ** 2 / (6 * EI) / 1e3)
    return out


def sec_indicator(study, ctx):
    cs = [c for c in ctx["eq"] if _methods(c)]
    if not cs:
        return None
    out = ["<p>ガイドライン（3.5・3.6、付録 D.6）は、事前の数値解析で法線の残留水平変位と構造物の状態の関係を整理しておき、"
           "発災後に計測した残留水平変位で利用可否を判断することを求めている（図 D-8・D-9）。ここでは抽出したシナリオの"
           "上部工の残留水平変位（防災基準点からの相対変位に当たる）を指標として、その関係を年数ごとに示す。"
           "ガイドラインの横軸は地震動の大きさ（速度 PSI 値など）だが、ここでは陸側の列の海底面の地盤変位で代える。</p>"]
    for c in cs:
        ms = _methods(c)
        ages = c["ages"]
        out.append(f"<h3>{escape(c['level'])} 地震後: 地震の大きさと残留水平変位（図 D-8 に当たる）</h3>")
        for a in (ages[0], ages[-1]):
            pts = [(float(r["surface"]), _res(r), f"pt{STATUS.index(r['status'])}",
                    f"{r['status']}: 地盤変位 {float(r['surface']):.3f} m、残留 {_res(r):.3f} m、使える {r.get('methods_ok') or 'なし'}")
                   for r in at(c["rows"], a) if _num(r.get("surface"))]
            out.append(figure(Raw(svg.scatter_chart(pts, "陸側の列の海底面の地盤変位 [m]", "上部工の残留水平変位 [m]",
                                                    LEGEND_ST, xfmt="{:.2f}",
                                                    yfmt="{:.2f}")),
                              f"{c['level']} 地震後 {a:g} 年: 地盤変位と上部工の残留水平変位（色は判定）"))
        out.append(f"<h3>{escape(c['level'])} 地震後: 残留水平変位と許容荷重（図 D-9 に当たる）</h3>")
        req = c["opts"]["required"].get("aftershock")
        tv = sorted({v["loads"]["tractive"] for v in ms.values() if "tractive" in v["loads"]})
        bv = sorted({v["loads"]["berthing"] for v in ms.values() if "berthing" in v["loads"]})
        for key, name, marks, fmt in (("allow_tractive", "許容牽引力 [kN]", [(v, f"利用方法の牽引力 {v:g} kN") for v in tv], "{:,.0f}"),
                                      ("allow_berthing", "許容接岸力 [kN]", [(v, f"利用方法の接岸力 {v:g} kN") for v in bv], "{:,.0f}"),
                                      ("allow_aftershock", "余震の許容震度 kh", [(req, f"想定の余震 kh = {req:g}")] if req else [],
                                       "{:.2f}")):
            top = 3 * max([m_[0] for m_ in marks] + [1e-9])
            pts = [(_res(r), min(r[key], top), f"pa{i % 3}", f"{a:g} 年: 残留 {_res(r):.3f} m、{name} {r[key]:.3g}")
                   for i, a in enumerate(ages) for r in at(c["rows"], a) if _num(r.get(key))]
            if pts:
                out.append(figure(Raw(svg.scatter_chart(pts, "上部工の残留水平変位 [m]", name,
                                                        [(f"{a:g} 年", f"pa{i % 3}") for i, a in enumerate(ages)],
                                                        xfmt="{:.2f}", yfmt=fmt, hmarks=marks)),
                                  f"{c['level']} 地震後: 残留水平変位と{name.split(' [')[0]}（色は供用年数、利用方法の値の 3 倍で頭打ち）"))
        # 閾値 A
        a0 = ages[0]
        trs = []
        for m in ms:
            row = [m]
            A = {}
            for a in ages:
                bad = [_res(r) for r in at(c["rows"], a) if not _usable(r, m)]
                A[a] = min(bad) if bad else None
                row.append("-" if not bad else f"{min(bad):.3f}（{len(bad)} 件）")
            for a in ages[1:]:
                if A[a0] is None:
                    row.append("-")
                    continue
                g = [r for r in at(c["rows"], a) if _res(r) < A[a0]]
                k = sum(not _usable(r, m) for r in g)
                row.append(f"{k}/{len(g)}（{k / len(g):.1%}）" if g else "-")
            trs.append(row)
        out.append(f"<h3>{escape(c['level'])} 地震後: 残留水平変位の閾値</h3>")
        out.append(wrapt(table(["利用方法"] + [f"{a:g} 年の A [m]（件数）" for a in ages]
                               + [f"{a:g} 年に {a0:g} 年の A で見逃す" for a in ages[1:]], trs)))
        out.append(f'<p class="note">閾値 A = その利用方法が使えない抽出の残留水平変位の最小値（A 未満の抽出は全て使える。'
                   f"ガイドラインの図 D-8 の A に当たる）。{a0:g} 年（劣化なし）の解析で決めた A を劣化した年数に使うと、"
                   "右の列の割合で使えない施設を使えると判断することになる。</p>")
        m_last = list(ms)[-1]
        A0 = min((_res(r) for r in at(c["rows"], a0) if not _usable(r, m_last)), default=None)
        if A0 is not None:
            trs = []
            for a in ages[1:]:
                miss = [r for r in at(c["rows"], a) if _res(r) < A0 and not _usable(r, m_last)]
                cnt = Counter()
                for r in miss:
                    cnt.update({n for x in reason_lines(r, r["status"]) for key, n in CATS if key in x})
                trs.append([f"{a:g}", str(len(miss)), _counts_text(cnt, 6)])
            out.append(table(["年数", f"{a0:g} 年の A（{m_last}、{A0:.3f} m）未満で使えない抽出", "理由（件数、重複あり）"], trs,
                             num=(1,)))
        # フラジリティ（横軸 = 残留水平変位）
        m1 = _phase_method(ms, 1)
        trs = []
        for a in ages:
            g = [r for r in at(c["rows"], a) if _res(r) > 0]
            x = [_res(r) for r in g]
            fmt_ = lambda f: "推定できない" if f is None or f[1] < 0.02 or f[1] > 5 else f"θ = {f[0]:.3f} m、β = {f[1]:.2f}"
            trs.append([f"{a:g}", fmt_(safe_fit(x, [r["status"] != STATUS[0] for r in g])),
                        fmt_(safe_fit(x, [not _usable(r, m1) for r in g]))])
        out.append(f"<details><summary>{escape(c['level'])}: 残留水平変位に対するフラジリティ曲線 P = Φ(ln(x/θ)/β)</summary>"
                   + table(["年数", "利用可でない（いずれかの利用方法が使えない）", f"{m1} が使えない"], trs)
                   + '<p class="note">横軸を現地で測れる残留水平変位にした（6 章の地盤変位は現地で測れない）。</p></details>')
    out.append(_simple_criterion(cs[0]))
    return "".join(out)


def _simple_criterion(c):
    """表 A-1 の簡易な目安 δy と、解析による判定の比較（最初の地震後のケース）"""
    from .pier import Pier
    d = {k: v for k, v in c["data"].items() if k != "scenarios"}
    pier = Pier(d)
    ms = _methods(c)
    m2 = _phase_method(ms, 2)
    trs, cmp = [], []
    for a in c["ages"]:
        g = at(c["rows"], a)
        ch = [r["corrosion_head"] for r in g if _num(r.get("corrosion_head"))]
        corr = float(np.median(ch)) if ch else 0.0
        dy = _yield_disp(pier, corr)
        trs.append([f"{a:g}", f"{corr:.1f}"] + [f"{v:.3f}" for v in dy.values()])
        lim = min(dy.values())
        below = [r for r in g if _res(r) <= lim]
        above = [r for r in g if _res(r) > lim]
        miss = sum(not _usable(r, m2) for r in below)
        safe = sum(_usable(r, m2) for r in above)
        cmp.append([f"{a:g}", f"{lim:.3f}", f"{miss}/{len(below)}" + (f"（{miss / len(below):.1%}）" if below else ""),
                    f"{safe}/{len(above)}" + (f"（{safe / len(above):.1%}）" if above else "")])
    out = [f"<h3>表 A-1 の簡易な目安との比較（{escape(c['level'])} 地震後）</h3>",
           "<p>ガイドラインの表 A-1（利用フェーズ 2・利用形態 A の例）は、事前の数値解析がない場合の桟橋の残留水平変位の目安を、"
           "杭を両端固定（仮想固定点・上部工）の梁とみなし片側を変位させて降伏する水平変位 δy = My·L²/(6EI) としている。"
           "列ごとの δy（杭頭側の腐食量はその年数の中央値）と、δy（最小の列）で判断した場合と解析による判定"
           f"（利用方法 {escape(m2)}）の食い違いを示す。</p>",
           table(["年数", "杭頭側の腐食量の中央値 [mm]"] + [f"δy 列 {k} [m]" for k in _yield_disp(pier, 0.0)], trs,
                 num=tuple(range(1, 2 + len(pier.pile_rows)))),
           table(["年数", "目安 δy [m]", f"δy 以下なのに {m2} が使えない", f"δy を超えるが {m2} が使える"], cmp, num=(1,)),
           '<p class="note">My は軸力 0・腐食後の外径の断面、EI は腐食前、L = 上部工の下面 − 仮想固定点（海底面 − 1/β）。'
           "目安は地盤ごと上部工が動く側方流動の変位も杭の変形とみなすので安全側に外れ、腐食による局部座屈の早まり"
           "（径厚比の増加）は表せない。</p>"]
    return "".join(out)


SECTIONS = (
    ("sec-outline", "概要", sec_outline),
    ("sec-chain", "変状連鎖図と重点シナリオ", sec_chain),
    ("sec-dists", "確率分布と劣化の量", sec_dists),
    ("sec-method", "解析と判定の方法", sec_method),
    ("sec-normal", "常時の結果", sec_normal),
    ("sec-eq", "地震後の結果", sec_eq),
    ("sec-guideline", "ガイドラインによる判定", sec_guideline),
    ("sec-indicator", "残留水平変位による判断基準", sec_indicator),
    ("sec-discussion", "シナリオごとの考察", sec_discussion),
    ("sec-notes", "留意事項", sec_notes),
    ("sec-appendix", "付録", sec_appendix),
)


def build_deterioration_report(root, title=None):
    """報告書の HTML（文字列）"""
    ctx = _compute(root)
    name = ctx["cases"][0]["data"].get("name", "").split("（")[0]
    guide = any(_methods(c) for c in ctx["cases"])
    study = SimpleNamespace(name=name, report={}, title=title or (f"変状連鎖の重点シナリオと利用可否判断（ガイドライン第2版）{name}"
                                                                  if guide else f"変状連鎖の重点シナリオの確率評価 {name}"))
    return assemble_html(study, SECTIONS, ctx, _cover(study, ctx), extra_css=CARD_CSS + EXTRA_CSS, title=study.title)


def write_deterioration_report(root, path, title=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_deterioration_report(root, title), encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="変状連鎖の重点シナリオの確率評価の報告書（HTML）")
    ap.add_argument("root", help="結果のフォルダ（inputs/ と results/ がある）")
    ap.add_argument("-o", "--out", default=None, help="出力先（既定は 結果のフォルダ/報告書.html）")
    a = ap.parse_args(argv)
    p = write_deterioration_report(a.root, a.out or Path(a.root) / "報告書.html")
    print(f"報告書 → {p}")


if __name__ == "__main__":
    main()
