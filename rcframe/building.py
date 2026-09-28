"""通り芯・階・断面リストからの解析モデル自動生成

入力は dict（YAML から読む場合は io_yaml.load_building を使う）。
書式の詳細は README.md「入力ファイル（YAML）」を参照。

階と部材の対応（日本の慣用に合わせる）
    柱 floor=1F : 1FL〜2FL 間の柱（1F柱）
    梁 floor=2F : 2FL 上の梁（2F梁）
    最下階の節点は全自由度固定、最下階以外は床区画から剛床を自動設定する。

位置の指定
    柱     "X2/Y1"（交点）, "X2/*", "*/Y1", "*"（全交点）
    梁     "Y1/X1-X3"（Y1通り上 X1〜X3 間の X方向梁）, "X2/*", "*"（全スパン）
    吹抜け "X1-X2/Y1-Y2"（X範囲/Y範囲で囲まれた床区画）, "*" も可

荷重の自動生成（floor_loads 等があるとき）: G, P, Ex, Ey（, Ex±e, Ey±e）
    計算そのものは loads.py、ここでは入力の解釈と載荷を行う

単位: mm, N（例外: 荷重欄 floor_loads/self_weight/extra_weights は kN/m², kN/m³, kN）
"""
import math

from .loads import TC, seismic_story_forces, tributary_profiles
from .materials import (STEELS, Bars, BeamBars, BeamRebar, ColumnRebar, Concrete, Hoop,
                        RectSection, SteelGrade)
from .model import Model

# YAML の荷重欄の単位（実務の慣用単位）→ 内部単位 mm, N
KN_M2 = 1e-3     # kN/m² → N/mm²
KN_M3 = 1e-6     # kN/m³ → N/mm³
KN = 1e3         # kN → N

ALL = "*"
_FORBIDDEN = ("/", "-", "*")
_NONE = (None, "none", "None", "NONE")


class InputError(ValueError):
    """入力データの誤り"""


def _num(v, what):
    if isinstance(v, bool):
        raise InputError(f"{what}: 数値ではありません: {v!r}")
    try:
        return float(v)   # PyYAML は 300e3 を文字列として読むため float() で変換
    except (TypeError, ValueError):
        raise InputError(f"{what}: 数値ではありません: {v!r}") from None


def _vec(v, n, what):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise InputError(f"{what}: {n} 成分のリストで指定してください: {v!r}")
    return [_num(c, what) for c in v]


def _check_keys(d, allowed, what):
    if not isinstance(d, dict):
        raise InputError(f"{what}: 辞書形式で指定してください: {d!r}")
    unknown = set(d) - set(allowed)
    if unknown:
        raise InputError(f"{what}: 未知の項目 {sorted(map(str, unknown))}")


def _as_list(v):
    return v if isinstance(v, list) else [v]


class Building:
    """通り芯・階・配置ルールから Model を生成する

    生成後は self.model に Model が入る。通り芯名から番号を引くには
    node(), column(), beam(), master(), drift_levels() を使う。
    """
    TOP_KEYS = ("name", "materials", "sections", "defaults", "grids", "floors",
                "columns", "beams", "openings", "loads",
                "floor_loads", "floor_use", "self_weight", "extra_weights", "seismic",
                "steels", "design", "nonlinear", "ultimate")

    def __init__(self, data: dict):
        _check_keys(data, self.TOP_KEYS, "トップレベル")
        self.name = str(data.get("name", "building"))
        self._read_materials(data.get("materials") or {})
        self._read_steels(data.get("steels") or {})
        self._read_sections(data.get("sections") or {})
        self._read_design(data.get("design") or {})
        self._read_nonlinear(data.get("nonlinear") or {})
        d = data.get("defaults") or {}
        _check_keys(d, ("material",), "defaults")
        self.default_material = d.get("material")
        self._read_grids(data.get("grids"))
        self._read_floors(data.get("floors"))
        self._read_ultimate(data.get("ultimate") or {})      # 階名を参照するので floors の後

        self.cols = {}       # (f, ix, iy) -> (RectSection, Concrete)
        self.beams = {}      # ("x", f, iy, ix) / ("y", f, ix, iy) -> (RectSection, Concrete)
        for r in data.get("columns") or []:
            self._apply_rule(r, "column")
        for r in data.get("beams") or []:
            self._apply_rule(r, "beam")
        self.openings = set()   # (f, ix, iy) 床のない区画
        for r in data.get("openings") or []:
            self._apply_opening(r)

        self._build()
        self.weights = {}          # f -> (W [N], xg, yg) 地震用重量と重心
        self.seismic = None        # 地震力の計算表
        if any(k in data for k in ("floor_loads", "floor_use", "self_weight",
                                   "extra_weights", "seismic")):
            self._generate_loads(data)
        for case, spec in (data.get("loads") or {}).items():   # 手入力の荷重は加算
            self._apply_loads(str(case), spec)

    # ============================================================ 読み込み
    def _read_materials(self, mats):
        self.materials = {}
        for name, d in mats.items():
            _check_keys(d, ("Fc", "gamma", "nu", "E"), f"材料 {name}")
            if "Fc" not in d:
                raise InputError(f"材料 {name}: Fc がありません")
            kw = {k: _num(d[k], f"材料 {name}.{k}") for k in d}
            self.materials[str(name)] = Concrete(str(name), **kw)

    def _read_steels(self, steels):
        self.steels = dict(STEELS)
        for name, d in steels.items():
            what = f"鋼種 {name}"
            d = d or {}
            _check_keys(d, ("F", "ftL", "ftL_D29"), what)
            base = STEELS.get(str(name))
            if base is None and "F" not in d:
                raise InputError(f"{what}: 既定表にない鋼種は F を指定してください")
            F = _num(d.get("F", base.F if base else None), f"{what}.F")
            ftL = _num(d.get("ftL", base.ftL if base else min(F, 195.0)), f"{what}.ftL")
            ftL29 = _num(d.get("ftL_D29", base.ftL_D29 if base else ftL), f"{what}.ftL_D29")
            self.steels[str(name)] = SteelGrade(str(name), F, ftL, ftL29)

    def _read_sections(self, secs):
        self.sections, self.section_material = {}, {}
        for name, d in secs.items():
            _check_keys(d, ("b", "D", "phi_y", "phi_z", "kappa", "material", "rebar"), f"断面 {name}")
            for k in ("b", "D"):
                if k not in d:
                    raise InputError(f"断面 {name}: {k} がありません")
            kw = {k: _num(v, f"断面 {name}.{k}") for k, v in d.items()
                  if k not in ("material", "rebar")}
            sec = RectSection(str(name), **kw)
            if d.get("rebar") is not None:
                sec.rebar = self._read_rebar(d["rebar"], f"断面 {name} の配筋")
            self.sections[str(name)] = sec
            self.section_material[str(name)] = d.get("material")

    def _read_rebar(self, r, what):
        """梁（end/center）または柱（x/y/hoop_x/hoop_y）の配筋"""
        is_beam = "end" in r or "center" in r
        keys = ("main", "shear", "dt") + (("end", "center") if is_beam
                                          else ("x", "y", "hoop_x", "hoop_y"))
        _check_keys(r, keys, what)
        missing = [k for k in keys if k not in r]
        if missing:
            raise InputError(f"{what}: {missing} がありません")
        steel = {}
        for k in ("main", "shear"):
            if str(r[k]) not in self.steels:
                raise InputError(f"{what}: 未定義の鋼種 {r[k]!r}")
            steel[k] = self.steels[str(r[k])]
        dt = _num(r["dt"], f"{what}.dt")
        try:
            if is_beam:
                pos = {}
                for p in ("end", "center"):
                    _check_keys(r[p], ("top", "bottom", "stirrup"), f"{what}.{p}")
                    if set(r[p]) != {"top", "bottom", "stirrup"}:
                        raise InputError(f"{what}.{p}: top, bottom, stirrup をすべて指定してください")
                    pos[p] = BeamBars(Bars.parse(r[p]["top"]), Bars.parse(r[p]["bottom"]),
                                      Hoop.parse(r[p]["stirrup"]))
                return BeamRebar(steel["main"], steel["shear"], dt, pos["end"], pos["center"])
            x, y = Bars.parse(r["x"]), Bars.parse(r["y"])
            if x.n < 2 or y.n < 2:
                raise InputError(f"{what}: 1面の本数は隅筋を含めて 2 本以上です")
            return ColumnRebar(steel["main"], steel["shear"], dt, x, y,
                               Hoop.parse(r["hoop_x"]), Hoop.parse(r["hoop_y"]))
        except ValueError as ex:
            if isinstance(ex, InputError):
                raise
            raise InputError(f"{what}: {ex}") from None

    def _read_design(self, d):
        _check_keys(d, ("shear_factor", "young_ratio", "long", "seismic"), "design")
        self.design = {
            "shear_factor": _num(d.get("shear_factor", 1.5), "design.shear_factor"),
            "young_ratio": _num(d.get("young_ratio", 15), "design.young_ratio"),
            "long": [str(c) for c in _as_list(d["long"])] if d.get("long") else None,
            "seismic": [str(c) for c in _as_list(d["seismic"])] if d.get("seismic") else None,
        }

    def _read_nonlinear(self, d):
        _check_keys(d, ("sigma_y_factor", "post_yield_ratio", "pushover"), "nonlinear")
        p = d.get("pushover") or {}
        _check_keys(p, ("directions", "max_drift", "step_drift"), "nonlinear.pushover")
        r = _num(d.get("post_yield_ratio", 0.001), "nonlinear.post_yield_ratio")
        if not 0 < r < 1:
            raise InputError("nonlinear.post_yield_ratio: 0 < r < 1 としてください")
        dirs = [str(x) for x in _as_list(p.get("directions", ["+x", "-x", "+y", "-y"]))]
        for x in dirs:
            if x.lstrip("+-").lower() not in ("x", "y"):
                raise InputError(f"nonlinear.pushover.directions: +x, -x, +y, -y から指定してください: {x!r}")
        self.nonlinear = {
            "given": bool(d),        # YAML に nonlinear の指定があるか（計算例の実行判定用）
            "sigma_y_factor": _num(d.get("sigma_y_factor", 1.1), "nonlinear.sigma_y_factor"),
            "post_yield_ratio": r,
            "pushover": {
                "directions": dirs,
                "max_drift": _num(p.get("max_drift", 0.02), "nonlinear.pushover.max_drift"),
                "step_drift": _num(p.get("step_drift", 0.0002), "nonlinear.pushover.step_drift"),
            },
        }

    def _read_ultimate(self, d):
        _check_keys(d, ("qsu", "qsu_factor", "drift_limit", "stop_at_column_shear", "ds_override"),
                    "ultimate")
        qsu = str(d.get("qsu", "min"))
        if qsu not in ("min", "mean"):
            raise InputError(f"ultimate.qsu: min または mean を指定してください: {qsu!r}")
        ds = {}
        for name, v in (d.get("ds_override") or {}).items():
            if str(name) not in self.fi or self.fi[str(name)] == len(self.floors) - 1:
                raise InputError(f"ultimate.ds_override: 層（最上階以外の階名）ではありません: {name!r}")
            ds[str(name)] = _num(v, f"ultimate.ds_override.{name}")
        self.ultimate = {
            "given": bool(d),
            "qsu": qsu,
            "qsu_factor": _num(d.get("qsu_factor", 1.0), "ultimate.qsu_factor"),
            "drift_limit": _num(d.get("drift_limit", 0.01), "ultimate.drift_limit"),
            "stop_at_column_shear": bool(d.get("stop_at_column_shear", True)),
            "ds_override": ds,
        }

    def _read_grids(self, grids):
        if not isinstance(grids, dict) or set(grids) != {"x", "y"}:
            raise InputError("grids: x と y の両方を指定してください")
        self.gx, self.gy = [], []          # [(name, pos)]
        seen = set()
        for axis, out in (("x", self.gx), ("y", self.gy)):
            g = grids[axis]
            if not isinstance(g, dict) or not g:
                raise InputError(f"grids.{axis}: {{通り芯名: 位置}} の形式で指定してください")
            for name, pos in g.items():
                name = str(name)
                if any(c in name for c in _FORBIDDEN):
                    raise InputError(f"通り芯名 {name!r} に / - * は使えません")
                if name in seen:
                    raise InputError(f"通り芯名 {name!r} が重複しています")
                seen.add(name)
                out.append((name, _num(pos, f"通り芯 {name}")))
            ps = [p for _, p in out]
            if any(b <= a for a, b in zip(ps, ps[1:])):
                raise InputError(f"grids.{axis}: 位置は昇順で指定してください")
        self.ix = {n: k for k, (n, _) in enumerate(self.gx)}
        self.iy = {n: k for k, (n, _) in enumerate(self.gy)}

    def _read_floors(self, floors):
        if not isinstance(floors, dict) or len(floors) < 2:
            raise InputError("floors: {階名: 床レベル} で 2 階分以上指定してください")
        self.floors = []                   # [(name, z)] 下から
        for name, z in floors.items():
            name = str(name)
            if name in (f for f, _ in self.floors):
                raise InputError(f"階名 {name!r} が重複しています")
            self.floors.append((name, _num(z, f"階 {name}")))
        zs = [z for _, z in self.floors]
        if any(b <= a for a, b in zip(zs, zs[1:])):
            raise InputError("floors: 床レベルは下の階から昇順で指定してください")
        self.fi = {n: k for k, (n, _) in enumerate(self.floors)}

    # ============================================================ 位置の解釈
    def _floor_indices(self, spec, allowed, what):
        if spec is None:
            return list(allowed)
        out = []
        for name in _as_list(spec):
            name = str(name)
            if name not in self.fi:
                raise InputError(f"{what}: 未定義の階 {name!r}")
            if self.fi[name] not in allowed:
                raise InputError(f"{what}: 階 {name} には指定できません"
                                 "（柱は最上階、梁・吹抜けは最下階に置けません）")
            out.append(self.fi[name])
        return out

    def _grid_index(self, name, table, what):
        if name not in table:
            raise InputError(f"{what}: 未定義の通り芯 {name!r}")
        return table[name]

    def _range(self, s, table, what):
        """"A-B" または "*" -> (i0, i1)（i0 < i1）"""
        n = len(table)
        if s == ALL:
            return 0, n - 1
        parts = s.split("-")
        if len(parts) != 2:
            raise InputError(f"{what}: 範囲は \"始点-終点\" で指定してください: {s!r}")
        a, b = (self._grid_index(p, table, what) for p in parts)
        if a == b:
            raise InputError(f"{what}: 範囲の始点と終点が同じです: {s!r}")
        return min(a, b), max(a, b)

    def _column_points(self, at, what):
        pts = []
        for s in _as_list(at):
            s = str(s)
            if s == ALL:
                s = "*/*"
            parts = s.split("/")
            if len(parts) != 2:
                raise InputError(f"{what}: 柱の位置は \"X通り/Y通り\" で指定してください: {s!r}")
            xs = range(len(self.gx)) if parts[0] == ALL else [self._grid_index(parts[0], self.ix, what)]
            ys = range(len(self.gy)) if parts[1] == ALL else [self._grid_index(parts[1], self.iy, what)]
            pts += [(i, j) for i in xs for j in ys]
        return pts

    def _beam_spans(self, at, what):
        """-> [("x", iy, ix) / ("y", ix, iy)]（ix/iy はスパン始点側の通り芯番号）"""
        spans = []
        for s in _as_list(at):
            s = str(s)
            if s == ALL:
                spans += [("x", j, i) for j in range(len(self.gy)) for i in range(len(self.gx) - 1)]
                spans += [("y", i, j) for i in range(len(self.gx)) for j in range(len(self.gy) - 1)]
                continue
            parts = s.split("/")
            if len(parts) != 2 or parts[0] == ALL:
                raise InputError(f"{what}: 梁の位置は \"通り/始点-終点\" で指定してください: {s!r}")
            line, rng = parts
            if line in self.iy:            # Y通り上の X方向梁
                j = self.iy[line]
                i0, i1 = self._range(rng, self.ix, what)
                spans += [("x", j, i) for i in range(i0, i1)]
            elif line in self.ix:          # X通り上の Y方向梁
                i = self.ix[line]
                j0, j1 = self._range(rng, self.iy, what)
                spans += [("y", i, j) for j in range(j0, j1)]
            else:
                raise InputError(f"{what}: 未定義の通り芯 {line!r}")
        return spans

    # ============================================================ 配置ルール
    def _apply_rule(self, r, kind):
        what = f"{'柱' if kind == 'column' else '梁'}ルール {r}"
        _check_keys(r, ("floor", "at", "section", "material"), what)
        if "section" not in r:
            raise InputError(f"{what}: section がありません（部材なしは none）")
        nf = len(self.floors)
        allowed = range(nf - 1) if kind == "column" else range(1, nf)
        floors = self._floor_indices(r.get("floor"), allowed, what)

        sec = r["section"]
        if sec in _NONE:
            val = None
        else:
            sec = str(sec)
            if sec not in self.sections:
                raise InputError(f"{what}: 未定義の断面 {sec!r}")
            mname = r.get("material") or self.section_material[sec] or self.default_material
            if mname is None:
                raise InputError(f"{what}: 材料が指定されていません（defaults.material など）")
            if str(mname) not in self.materials:
                raise InputError(f"{what}: 未定義の材料 {mname!r}")
            val = (self.sections[sec], self.materials[str(mname)])

        if kind == "column":
            keys = [(f, i, j) for f in floors for i, j in self._column_points(r.get("at", ALL), what)]
            target = self.cols
        else:
            keys = [(d, f, a, b) for f in floors for d, a, b in self._beam_spans(r.get("at", ALL), what)]
            target = self.beams
        for k in keys:
            if val is None:
                target.pop(k, None)
            else:
                target[k] = val

    def _area_cells(self, area, what):
        """"X範囲/Y範囲" または "*" -> 床区画 [(ix, iy)]"""
        area = str(area)
        parts = ["*", "*"] if area == ALL else area.split("/")
        if len(parts) != 2:
            raise InputError(f"{what}: area は \"X範囲/Y範囲\" で指定してください: {area!r}")
        i0, i1 = self._range(parts[0], self.ix, what)
        j0, j1 = self._range(parts[1], self.iy, what)
        return [(i, j) for i in range(i0, i1) for j in range(j0, j1)]

    def _apply_opening(self, r):
        what = f"吹抜け {r}"
        _check_keys(r, ("floor", "area"), what)
        floors = self._floor_indices(r.get("floor"), range(1, len(self.floors)), what)
        cells = self._area_cells(r.get("area", ""), what)
        for f in floors:
            self.openings |= {(f, i, j) for i, j in cells}

    # ============================================================ モデル生成
    def _build(self):
        m = Model(self.name)
        nf, nx, ny = len(self.floors), len(self.gx), len(self.gy)

        # 部材端となる点だけに節点を作る
        need = set()
        for f, i, j in self.cols:
            need |= {(f, i, j), (f + 1, i, j)}
        for d, f, a, b in self.beams:
            need |= {(f, b, a), (f, b + 1, a)} if d == "x" else {(f, a, b), (f, a, b + 1)}
        self.node_ids = {}
        self.labels = {}
        for nid, (f, i, j) in enumerate(sorted(need), 1):
            m.add_node(nid, self.gx[i][1], self.gy[j][1], self.floors[f][1],
                       fix=[1] * 6 if f == 0 else None)
            self.node_ids[(f, i, j)] = nid
            self.labels[("node", nid)] = f"{self.gx[i][0]}/{self.gy[j][0]}/{self.floors[f][0]}"

        # 要素: 階ごとに 柱（下の階）→ X方向梁 → Y方向梁
        self.elem_ids = {}
        eid = 1
        for f in range(1, nf):
            for i in range(nx):
                for j in range(ny):
                    if (f - 1, i, j) in self.cols:
                        sec, mat = self.cols[(f - 1, i, j)]
                        m.add_element(eid, self.node_ids[(f - 1, i, j)], self.node_ids[(f, i, j)],
                                      mat, sec, "column")
                        self.elem_ids[("c", f - 1, i, j)] = eid
                        self.labels[("elem", eid)] = (f"{self.floors[f - 1][0]}柱 "
                                                      f"{self.gx[i][0]}/{self.gy[j][0]}")
                        eid += 1
            for j in range(ny):
                for i in range(nx - 1):
                    if ("x", f, j, i) in self.beams:
                        sec, mat = self.beams[("x", f, j, i)]
                        m.add_element(eid, self.node_ids[(f, i, j)], self.node_ids[(f, i + 1, j)],
                                      mat, sec)
                        self.elem_ids[("x", f, j, i)] = eid
                        self.labels[("elem", eid)] = (f"{self.floors[f][0]}梁 {self.gy[j][0]}/"
                                                      f"{self.gx[i][0]}-{self.gx[i + 1][0]}")
                        eid += 1
            for i in range(nx):
                for j in range(ny - 1):
                    if ("y", f, i, j) in self.beams:
                        sec, mat = self.beams[("y", f, i, j)]
                        m.add_element(eid, self.node_ids[(f, i, j)], self.node_ids[(f, i, j + 1)],
                                      mat, sec)
                        self.elem_ids[("y", f, i, j)] = eid
                        self.labels[("elem", eid)] = (f"{self.floors[f][0]}梁 {self.gx[i][0]}/"
                                                      f"{self.gy[j][0]}-{self.gy[j + 1][0]}")
                        eid += 1

        m.auto_rigid_zones()

        # 剛床: 床区画（吹抜けを除く）の面積図心にマスター節点
        self.masters = {}
        next_nid = max(m.nodes, default=0) + 1
        for f in range(1, nf):
            cells = [(i, j) for i in range(nx - 1) for j in range(ny - 1)
                     if (f, i, j) not in self.openings]
            slaves = sorted({self.node_ids[(f, i + di, j + dj)]
                             for i, j in cells for di in (0, 1) for dj in (0, 1)
                             if (f, i + di, j + dj) in self.node_ids})
            if not slaves:
                continue
            area = sx = sy = 0.0
            for i, j in cells:
                (x0, x1), (y0, y1) = (self.gx[i][1], self.gx[i + 1][1]), (self.gy[j][1], self.gy[j + 1][1])
                a = (x1 - x0) * (y1 - y0)
                area += a
                sx += a * (x0 + x1) / 2
                sy += a * (y0 + y1) / 2
            m.add_node(next_nid, sx / area, sy / area, self.floors[f][1])
            m.add_diaphragm(next_nid, slaves)
            self.masters[f] = next_nid
            self.labels[("node", next_nid)] = f"{self.floors[f][0]} 床マスター"
            next_nid += 1

        self.model = m

    # ============================================================ 荷重
    def _apply_loads(self, case, spec):
        what = f"荷重ケース {case}"
        _check_keys(spec, ("beams", "floors"), what)
        m = self.model
        for r in spec.get("beams") or []:
            w_what = f"{what} の梁荷重 {r}"
            _check_keys(r, ("floor", "at", "w"), w_what)
            if "w" not in r:
                raise InputError(f"{w_what}: w がありません")
            w = _vec(r["w"], 3, w_what)
            floors = self._floor_indices(r.get("floor"), range(1, len(self.floors)), w_what)
            eids = [self.elem_ids[(d, f, a, b)]
                    for f in floors for d, a, b in self._beam_spans(r.get("at", ALL), w_what)
                    if (d, f, a, b) in self.elem_ids]
            if not eids:
                raise InputError(f"{w_what}: 該当する梁がありません")
            for e in eids:
                m.add_member_load(case, e, w)
        for fname, F in (spec.get("floors") or {}).items():
            f_what = f"{what} の床荷重 {fname}"
            fname = str(fname)
            if fname not in self.fi:
                raise InputError(f"{f_what}: 未定義の階 {fname!r}")
            if self.fi[fname] not in self.masters:
                raise InputError(f"{f_what}: 階 {fname} には剛床（マスター節点）がありません")
            fx, fy, mz = _vec(F, 3, f_what + "（[Fx, Fy, Mz]）")
            m.add_nodal_load(case, self.masters[self.fi[fname]], [fx, fy, 0, 0, 0, mz])

    # ============================================================ 荷重の自動生成
    def _cell_name(self, f, i, j):
        return (f"{self.floors[f][0]} 区画 {self.gx[i][0]}-{self.gx[i + 1][0]}/"
                f"{self.gy[j][0]}-{self.gy[j + 1][0]}")

    def _generate_loads(self, data):
        """床荷重の梁への分配（G, P）、部材自重（G）、地震用重量、Ai 分布の地震力"""
        m = self.model
        nf = len(self.floors)
        upper = range(1, nf)

        # ---- 入力（YAML の単位 kN/m², kN/m³, kN を N, mm に換算）
        self.floor_loads = {}
        for name, d in (data.get("floor_loads") or {}).items():
            what = f"床荷重 {name}"
            _check_keys(d, ("DL", "LL_frame", "LL_seismic"), what)
            if "DL" not in d:
                raise InputError(f"{what}: DL がありません")
            self.floor_loads[str(name)] = {k: _num(d.get(k, 0), f"{what}.{k}") * KN_M2
                                           for k in ("DL", "LL_frame", "LL_seismic")}
        self.cell_use = {}                  # (f, ix, iy) -> 用途名
        for r in data.get("floor_use") or []:
            what = f"床用途 {r}"
            _check_keys(r, ("floor", "area", "use"), what)
            if "use" not in r:
                raise InputError(f"{what}: use がありません（床荷重なしは none）")
            use = None if r["use"] in _NONE else str(r["use"])
            if use is not None and use not in self.floor_loads:
                raise InputError(f"{what}: floor_loads に未定義の用途 {use!r}")
            for f in self._floor_indices(r.get("floor"), upper, what):
                for i, j in self._area_cells(r.get("area", ALL), what):
                    if use is None:
                        self.cell_use.pop((f, i, j), None)
                    else:
                        self.cell_use[(f, i, j)] = use
        sw = data.get("self_weight") or {}
        _check_keys(sw, ("gamma",), "self_weight")
        gamma = _num(sw.get("gamma", 24.0), "self_weight.gamma") * KN_M3
        extra = {}
        for name, w in (data.get("extra_weights") or {}).items():
            what = f"extra_weights.{name}"
            f = self._floor_indices(str(name), upper, what)[0]
            if f not in self.masters:
                raise InputError(f"{what}: 階 {name} には剛床（マスター節点）がありません")
            extra[f] = _num(w, what) * KN

        # ---- 地震用重量の集計: f -> [ΣW, ΣW·x, ΣW·y]
        acc = {f: [0.0, 0.0, 0.0] for f in upper}

        def add_weight(f, w, x, y):
            if f in acc:
                acc[f][0] += w
                acc[f][1] += w * x
                acc[f][2] += w * y

        # ---- 床荷重: 亀甲分割で梁へ（G: DL, P: LL_frame）
        for (f, i, j), use in sorted(self.cell_use.items()):
            if (f, i, j) in self.openings:
                continue
            (x0, x1), (y0, y1) = (self.gx[i][1], self.gx[i + 1][1]), (self.gy[j][1], self.gy[j + 1][1])
            keys = {"x0": ("x", f, j, i), "x1": ("x", f, j + 1, i),
                    "y0": ("y", f, i, j), "y1": ("y", f, i + 1, j)}
            edges = {e for e, k in keys.items() if k in self.elem_ids}
            try:
                prof = tributary_profiles(x1 - x0, y1 - y0, edges)
            except ValueError as ex:
                raise InputError(f"{self._cell_name(f, i, j)}: {ex}") from None
            fl = self.floor_loads[use]
            for case, w in (("G", fl["DL"]), ("P", fl["LL_frame"])):
                if w:
                    for e, pr in prof.items():
                        m.add_member_load(case, self.elem_ids[keys[e]], [0, 0, -w], pr)
            add_weight(f, (fl["DL"] + fl["LL_seismic"]) * (x1 - x0) * (y1 - y0),
                       (x0 + x1) / 2, (y0 + y1) / 2)

        # ---- 部材自重（G）: 梁はその階、柱は上下の階に半分ずつ（基礎側は除く）
        for key, eid in self.elem_ids.items():
            e = m.elements[eid]
            ni, nj = m.nodes[e.ni], m.nodes[e.nj]
            q = gamma * e.sec.A
            if not q:
                continue
            m.add_member_load("G", eid, [0, 0, -q])
            L = math.dist(ni.xyz, nj.xyz)
            if key[0] == "c":
                add_weight(key[1], q * L / 2, ni.x, ni.y)
                add_weight(key[1] + 1, q * L / 2, ni.x, ni.y)
            else:
                add_weight(key[1], q * L, (ni.x + nj.x) / 2, (ni.y + nj.y) / 2)

        # ---- 追加重量（床の図心＝マスター節点位置）
        for f, w in extra.items():
            nd = m.nodes[self.masters[f]]
            add_weight(f, w, nd.x, nd.y)

        self.weights = {f: (W, Wx / W, Wy / W) for f, (W, Wx, Wy) in acc.items() if W > 0}

        if data.get("seismic") is not None:
            self._seismic_loads(data["seismic"])

    def _seismic_loads(self, sd):
        """Ai 分布の地震力を各階の重心に作用させる（マスター節点へ Fx, Fy, Mz として載荷）"""
        _check_keys(sd, ("Z", "soil", "C0", "T", "directions", "eccentricity"), "seismic")
        Z = _num(sd.get("Z", 1.0), "seismic.Z")
        C0 = _num(sd.get("C0", 0.2), "seismic.C0")
        soil = sd.get("soil", 2)
        if soil not in TC:
            raise InputError(f"seismic.soil: 地盤種別は 1, 2, 3 のいずれかです: {soil!r}")
        T = None if sd.get("T") is None else _num(sd["T"], "seismic.T")
        dirs = [str(d) for d in _as_list(sd.get("directions", ["x", "y"]))]
        if not dirs or any(d not in ("x", "y") for d in dirs):
            raise InputError(f"seismic.directions: x, y から指定してください: {dirs!r}")
        ecc = _num(sd.get("eccentricity", 0.0), "seismic.eccentricity")

        nf = len(self.floors)
        for f in range(1, nf):
            name = self.floors[f][0]
            if f not in self.masters:
                raise InputError(f"seismic: 階 {name} に剛床がないため Ai 分布の地震力を作用させられません")
            if f not in self.weights:
                raise InputError(f"seismic: 階 {name} の地震用重量が 0 です")
        W = [self.weights[f][0] for f in range(1, nf)]
        h_m = (self.floors[-1][1] - self.floors[0][1]) / 1000
        T, Rt, rows = seismic_story_forces(W, h_m, Z, soil, C0, T)

        m = self.model
        for k, row in enumerate(rows):
            f = k + 1
            row.update(story=self.floors[k][0], floor=self.floors[f][0], W=W[k],
                       xg=self.weights[f][1], yg=self.weights[f][2])
        self.seismic = {"Z": Z, "soil": soil, "Tc": TC[soil], "C0": C0, "T": T, "Rt": Rt,
                        "h": h_m, "eccentricity": ecc, "rows": rows}

        variants = [("", 0)] + ([("+e", 1), ("-e", -1)] if ecc else [])
        for d in dirs:
            for tag, sgn in variants:
                case = f"E{d}{tag}"
                for k, row in enumerate(rows):
                    f = k + 1
                    mst = m.nodes[self.masters[f]]
                    bx, by = self._floor_extent(f)
                    if d == "x":
                        yg = row["yg"] + sgn * ecc * by
                        F = [row["P"], 0, 0, 0, 0, -(yg - mst.y) * row["P"]]
                    else:
                        xg = row["xg"] + sgn * ecc * bx
                        F = [0, row["P"], 0, 0, 0, (xg - mst.x) * row["P"]]
                    m.add_nodal_load(case, self.masters[f], F)

    def _floor_extent(self, f):
        """床区画（吹抜けを除く）の X, Y 方向の外形寸法（偶発偏心の基準）"""
        cells = [(i, j) for i in range(len(self.gx) - 1) for j in range(len(self.gy) - 1)
                 if (f, i, j) not in self.openings]
        xs = [self.gx[i][1] for i, _ in cells] + [self.gx[i + 1][1] for i, _ in cells]
        ys = [self.gy[j][1] for _, j in cells] + [self.gy[j + 1][1] for _, j in cells]
        return max(xs) - min(xs), max(ys) - min(ys)

    def print_seismic_table(self):
        """地震用重量と Ai 分布の計算表（kN）"""
        s = self.seismic
        if s is None:
            print("地震力は生成されていません（seismic 未指定）")
            return
        print(f"\n■ 地震力（Ai 分布）  Z={s['Z']}  地盤種別 {s['soil']} (Tc={s['Tc']} s)  "
              f"h={s['h']:.2f} m  T={s['T']:.3f} s  Rt={s['Rt']:.3f}  C0={s['C0']}")
        print(f"{'層':>4} {'Wi':>9} {'ΣWi':>9} {'αi':>7} {'Ai':>7} {'Ci':>7} {'Qi':>9} "
              f"{'Pi':>9} {'重心 xg':>9} {'yg':>9}")
        for r in reversed(s["rows"]):
            print(f"{r['story']:>4} {r['W'] / KN:9.1f} {r['sumW'] / KN:9.1f} {r['alpha']:7.3f} "
                  f"{r['Ai']:7.3f} {r['Ci']:7.3f} {r['Q'] / KN:9.1f} {r['P'] / KN:9.1f} "
                  f"{r['xg']:9.0f} {r['yg']:9.0f}")
        print("  Wi・重心: 層の上の床の値、Pi: 上の床に作用する外力 [kN, mm]")

    # ============================================================ 検索
    def _f(self, floor):
        if str(floor) not in self.fi:
            raise KeyError(f"未定義の階 {floor!r}")
        return self.fi[str(floor)]

    def node(self, x, y, floor):
        """通り芯交点の節点番号"""
        key = (self._f(floor), self.ix[x], self.iy[y])
        if key not in self.node_ids:
            raise KeyError(f"節点 {x}/{y}/{floor} はありません")
        return self.node_ids[key]

    def column(self, x, y, floor):
        """floor 階の柱（floor〜1つ上の階）の要素番号"""
        key = ("c", self._f(floor), self.ix[x], self.iy[y])
        if key not in self.elem_ids:
            raise KeyError(f"{floor}柱 {x}/{y} はありません")
        return self.elem_ids[key]

    def beam(self, line, a, b, floor):
        """floor 階の line 通り上、隣り合う通り芯 a-b 間の梁の要素番号"""
        f = self._f(floor)
        if line in self.iy:
            i0, i1 = sorted((self.ix[a], self.ix[b]))
            key = ("x", f, self.iy[line], i0)
        else:
            i0, i1 = sorted((self.iy[a], self.iy[b]))
            key = ("y", f, self.ix[line], i0)
        if i1 - i0 != 1 or key not in self.elem_ids:
            raise KeyError(f"{floor}梁 {line}/{a}-{b} はありません")
        return self.elem_ids[key]

    def master(self, floor):
        """floor 階の床マスター節点（剛床がなければ None）"""
        return self.masters.get(self._f(floor))

    def drift_levels(self):
        """Result.story_drifts に渡す [(階名, 上の床のマスター節点)]"""
        out = []
        for f in range(1, len(self.floors)):
            if f not in self.masters:
                raise ValueError(f"階 {self.floors[f][0]} に剛床がないため層間変形角を求められません")
            out.append((self.floors[f - 1][0], self.masters[f]))
        return out

    def label(self, kind, id_):
        """kind = "node" / "elem" の通り芯表記（例: "1F柱 X1/Y1"）"""
        return self.labels.get((kind, id_), str(id_))
