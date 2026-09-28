"""直杭式横桟橋（鋼管杭）の解析モデル自動生成

    上部工: 縦梁・横梁の格子を deck.model_level の高さに配置し、上部工全体を 1つの剛床とする
            （マスター節点は上部工の平面の図心）。床版は荷重の分配のみ（モデル化しない）
    鋼管杭: 杭頭（model_level）から仮想固定点までを 1本の部材とし、仮想固定点で固定
            仮想固定点 = 海底面 − 1/β、β = (kh·D / 4EI)^(1/4)（Chang）
            杭頭は上部工に剛結し、model_level から soffit（上部工下面）までを剛域とする

YAML の単位（港湾の実務に合わせる）: 高さ・通り芯 m（D.L. 基準）、断面寸法 mm、
    荷重 kN・kN/m・kN·m、kh kN/m³。内部はすべて mm, N に換算する。
"""
from .grid import ALL, _NONE, GridMixin, InputError, _as_list, _check_keys, _num, _vec
from .materials import RectSection
from .model import Model
from .steel import JsceConcrete, PipeSection, Steel, chang_beta

M = 1e3          # m → mm
KN = 1e3         # kN → N
KNM = 1e6        # kN·m → N·mm
KH = 1e-6        # kN/m³ → N/mm³


class Pier(GridMixin):
    """桟橋（直杭式横桟橋）の入力から Model を生成する

    生成後は self.model に Model が入る。node(), pile(), beam(), master() で番号を引く。
    杭ごとの仮想固定点などは self.piles[(ix, iy)]（dict）に入る。
    """
    TOP_KEYS = ("type", "name", "materials", "sections", "grids", "deck", "beams", "piles",
                "loads", "report")

    def __init__(self, data: dict):
        _check_keys(data, self.TOP_KEYS, "トップレベル")
        if str(data.get("type", "pier")) != "pier":
            raise InputError("type: 桟橋の入力は type: pier です")
        self.name = str(data.get("name", "pier"))
        self._read_materials(data.get("materials") or {})
        self._read_sections(data.get("sections") or {})
        g = data.get("grids")
        if isinstance(g, dict):          # 通り芯の位置 m → mm
            g = {a: ({k: _num(v, f"通り芯 {k}") * M for k, v in (g[a] or {}).items()}
                     if isinstance(g.get(a), dict) else g.get(a)) for a in g}
        self._read_grids(g)
        self._read_deck(data.get("deck"))
        rp = data.get("report") or {}
        _check_keys(rp, ("title", "author", "date", "notes"), "report")
        self.report = {k: str(rp[k]) for k in rp if rp[k] is not None}
        self.source_text = None

        self.beams = {}                  # ("x", iy, ix) / ("y", ix, iy) -> (断面, 材料)
        for r in data.get("beams") or []:
            self._apply_beam_rule(r)
        self._read_piles(data.get("piles"))
        self._build()
        for case, spec in (data.get("loads") or {}).items():
            self._apply_loads(str(case), spec)

    # ============================================================ 読み込み
    def _read_materials(self, mats):
        self.materials = {}
        for name, d in mats.items():
            what = f"材料 {name}"
            if d.get("steel"):
                _check_keys(d, ("steel", "fy", "E", "nu"), what)
                if "fy" not in d:
                    raise InputError(f"{what}: fy がありません")
                kw = {k: _num(d[k], f"{what}.{k}") for k in ("fy", "E", "nu") if k in d}
                self.materials[str(name)] = Steel(str(name), **kw)
            else:
                _check_keys(d, ("fck", "E", "nu", "gamma"), what)
                if "fck" not in d:
                    raise InputError(f"{what}: fck がありません（鋼材は steel: true）")
                kw = {k: _num(d[k], f"{what}.{k}") for k in d}
                try:
                    self.materials[str(name)] = JsceConcrete(str(name), **kw)
                except ValueError as ex:
                    raise InputError(f"{what}: {ex}") from None

    def _read_sections(self, secs):
        self.sections, self.section_material = {}, {}
        for name, d in secs.items():
            what = f"断面 {name}"
            if "pipe" in d:
                _check_keys(d, ("pipe", "material", "corrosion", "stiffness"), what)
                p = d["pipe"]
                _check_keys(p, ("D", "t"), f"{what}.pipe")
                c = d.get("corrosion") or {}
                _check_keys(c, ("outer", "inner"), f"{what}.corrosion")
                try:
                    sec = PipeSection(str(name), _num(p.get("D"), f"{what}.D"), _num(p.get("t"), f"{what}.t"),
                                      _num(c.get("outer", 0), f"{what}.corrosion.outer"),
                                      _num(c.get("inner", 0), f"{what}.corrosion.inner"),
                                      str(d.get("stiffness", "intact")))
                except ValueError as ex:
                    if isinstance(ex, InputError):
                        raise
                    raise InputError(f"{what}: {ex}") from None
            else:
                _check_keys(d, ("b", "D", "phi_y", "phi_z", "kappa", "material"), what)
                for k in ("b", "D"):
                    if k not in d:
                        raise InputError(f"{what}: {k} がありません（鋼管は pipe: {{D, t}}）")
                kw = {k: _num(v, f"{what}.{k}") for k, v in d.items() if k != "material"}
                sec = RectSection(str(name), **kw)
            m = d.get("material")
            if m is None or str(m) not in self.materials:
                raise InputError(f"{what}: 材料 {m!r} が未定義です")
            self.sections[str(name)] = sec
            self.section_material[str(name)] = self.materials[str(m)]

    def _read_deck(self, d):
        _check_keys(d, ("top", "model_level", "soffit"), "deck")
        for k in ("top", "model_level", "soffit"):
            if d is None or k not in d:
                raise InputError(f"deck: {k} がありません")
        self.deck_top = _num(d["top"], "deck.top") * M
        self.z_model = _num(d["model_level"], "deck.model_level") * M
        self.z_soffit = _num(d["soffit"], "deck.soffit") * M
        if not self.deck_top >= self.z_model >= self.z_soffit:
            raise InputError("deck: top ≧ model_level ≧ soffit としてください")

    def _section(self, name, what, kind):
        name = str(name)
        if name not in self.sections:
            raise InputError(f"{what}: 未定義の断面 {name!r}")
        sec = self.sections[name]
        if kind == "pile" and not isinstance(sec, PipeSection):
            raise InputError(f"{what}: 杭の断面は鋼管（pipe）です: {name!r}")
        if kind == "beam" and isinstance(sec, PipeSection):
            raise InputError(f"{what}: 梁に鋼管断面は使えません: {name!r}")
        return sec, self.section_material[name]

    def _apply_beam_rule(self, r):
        what = f"梁ルール {r}"
        _check_keys(r, ("at", "section"), what)
        if "section" not in r:
            raise InputError(f"{what}: section がありません（梁なしは none）")
        val = None if r["section"] in _NONE else self._section(r["section"], what, "beam")
        for d, a, b in self._beam_spans(r.get("at", ALL), what):
            if val is None:
                self.beams.pop((d, a, b), None)
            else:
                self.beams[(d, a, b)] = val

    def _read_piles(self, p):
        _check_keys(p, ("section", "rows", "overrides"), "piles")
        if p is None or "rows" not in p:
            raise InputError("piles: rows（杭の列ごとの海底面と kh）がありません")
        rows = {}
        for name, d in p["rows"].items():
            what = f"piles.rows.{name}"
            name = str(name)
            if name not in self.iy:
                raise InputError(f"{what}: 未定義の Y通り（杭の列）です")
            _check_keys(d, ("seabed", "kh"), what)
            if "seabed" not in d or "kh" not in d:
                raise InputError(f"{what}: seabed と kh を指定してください")
            kh = _num(d["kh"], f"{what}.kh")
            if kh <= 0:
                raise InputError(f"{what}: kh は正の値です")
            sb = _num(d["seabed"], f"{what}.seabed") * M
            if sb >= self.z_soffit:
                raise InputError(f"{what}: 海底面が上部工の下面以上です")
            rows[self.iy[name]] = (sb, kh * KH)
        self.pile_sec = {}
        if p.get("section") not in _NONE:
            val = self._section(p["section"], "piles.section", "pile")
            self.pile_sec = {(i, j): val for i in range(len(self.gx)) for j in range(len(self.gy))}
        for r in p.get("overrides") or []:
            what = f"piles.overrides {r}"
            _check_keys(r, ("at", "section"), what)
            val = None if r.get("section") in _NONE else self._section(r["section"], what, "pile")
            for ij in self._column_points(r.get("at", ALL), what):
                if val is None:
                    self.pile_sec.pop(ij, None)
                else:
                    self.pile_sec[ij] = val
        missing = sorted({self.gy[j][0] for (_, j) in self.pile_sec if j not in rows})
        if missing:
            raise InputError(f"piles.rows: 杭のある列 {missing} の seabed・kh がありません")
        self.pile_rows = rows

    # ============================================================ モデル生成
    def _build(self):
        m = Model(self.name)
        nx, ny = len(self.gx), len(self.gy)
        need = set(self.pile_sec)
        for d, a, b in self.beams:
            need |= {(b, a), (b + 1, a)} if d == "x" else {(a, b), (a, b + 1)}
        self.node_ids, self.elem_ids, self.labels, self.piles = {}, {}, {}, {}
        nid = 1
        for i, j in sorted(need):
            m.add_node(nid, self.gx[i][1], self.gy[j][1], self.z_model)
            self.node_ids[(i, j)] = nid
            self.labels[("node", nid)] = f"{self.gx[i][0]}/{self.gy[j][0]}"
            nid += 1
        eid = 1
        # 杭: i端 = 仮想固定点（固定）、j端 = 杭頭
        for (i, j), (sec, mat) in sorted(self.pile_sec.items()):
            seabed, kh = self.pile_rows[j]
            EI = mat.E * sec.Iy
            beta = chang_beta(kh, sec.D, EI)
            z_fix = seabed - 1.0 / beta
            m.add_node(nid, self.gx[i][1], self.gy[j][1], z_fix, fix=[1] * 6)
            self.labels[("node", nid)] = f"{self.gx[i][0]}/{self.gy[j][0]} 仮想固定点"
            m.add_element(eid, nid, self.node_ids[(i, j)], mat, sec, "column",
                          rigid_j=self.z_model - self.z_soffit)
            self.elem_ids[("p", i, j)] = eid
            self.labels[("elem", eid)] = f"杭 {self.gx[i][0]}/{self.gy[j][0]}"
            self.piles[(i, j)] = dict(eid=eid, fixed_node=nid, seabed=seabed, kh=kh, beta=beta,
                                      z_fix=z_fix, length=self.z_model - z_fix,
                                      free_length=self.z_soffit - z_fix)
            nid += 1
            eid += 1
        for key in sorted(self.beams, key=lambda k: (k[0], k[1], k[2])):
            d, a, b = key
            sec, mat = self.beams[key]
            if d == "x":
                n0, n1 = self.node_ids[(b, a)], self.node_ids[(b + 1, a)]
                lab = f"梁 {self.gy[a][0]}/{self.gx[b][0]}-{self.gx[b + 1][0]}"
            else:
                n0, n1 = self.node_ids[(a, b)], self.node_ids[(a, b + 1)]
                lab = f"梁 {self.gx[a][0]}/{self.gy[b][0]}-{self.gy[b + 1][0]}"
            m.add_element(eid, n0, n1, mat, sec)
            self.elem_ids[key] = eid
            self.labels[("elem", eid)] = lab
            eid += 1
        # 上部工の剛床（マスター節点は平面の図心）
        xs = [p for _, p in self.gx]
        ys = [p for _, p in self.gy]
        self.master_id = None
        slaves = sorted(self.node_ids.values())
        if len(slaves) > 1 and (nx > 1 or ny > 1):
            self.master_id = nid
            m.add_node(nid, (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2, self.z_model)
            m.add_diaphragm(nid, slaves)
            self.labels[("node", nid)] = "上部工 マスター"
        self.model = m

    # ============================================================ 荷重（P1: 手入力）
    def _apply_loads(self, case, spec):
        what = f"荷重ケース {case}"
        _check_keys(spec, ("deck", "beams", "nodes"), what)
        m = self.model
        if spec.get("deck") is not None:
            if self.master_id is None:
                raise InputError(f"{what}: 上部工の剛床がないため deck 荷重は使えません")
            fx, fy, mz = _vec(spec["deck"], 3, f"{what}.deck（[Fx, Fy, Mz] kN, kN·m）")
            m.add_nodal_load(case, self.master_id, [fx * KN, fy * KN, 0, 0, 0, mz * KNM])
        for r in spec.get("beams") or []:
            w_what = f"{what} の梁荷重 {r}"
            _check_keys(r, ("at", "w"), w_what)
            w = _vec(r.get("w"), 3, w_what + "（[wx, wy, wz] kN/m）")     # kN/m = N/mm
            eids = [self.elem_ids[k] for k in self._beam_spans(r.get("at", ALL), w_what)
                    if k in self.elem_ids]
            if not eids:
                raise InputError(f"{w_what}: 該当する梁がありません")
            for e in eids:
                m.add_member_load(case, e, w)
        for r in spec.get("nodes") or []:
            n_what = f"{what} の節点荷重 {r}"
            _check_keys(r, ("at", "F"), n_what)
            F = _vec(r.get("F"), 6, n_what + "（[Fx, Fy, Fz, Mx, My, Mz] kN, kN·m）")
            for ij in self._column_points(r.get("at"), n_what):
                if ij not in self.node_ids:
                    raise InputError(f"{n_what}: 節点がありません")
                m.add_nodal_load(case, self.node_ids[ij], [F[0] * KN, F[1] * KN, F[2] * KN,
                                                           F[3] * KNM, F[4] * KNM, F[5] * KNM])

    # ============================================================ 検索
    def node(self, x, y):
        """上部工の節点番号"""
        key = (self.ix[x], self.iy[y])
        if key not in self.node_ids:
            raise KeyError(f"節点 {x}/{y} はありません")
        return self.node_ids[key]

    def pile(self, x, y):
        """杭の要素番号"""
        key = ("p", self.ix[x], self.iy[y])
        if key not in self.elem_ids:
            raise KeyError(f"杭 {x}/{y} はありません")
        return self.elem_ids[key]

    def pile_info(self, x, y):
        return self.piles[(self.ix[x], self.iy[y])]

    def beam(self, line, a, b):
        if line in self.iy:
            i0, i1 = sorted((self.ix[a], self.ix[b]))
            key = ("x", self.iy[line], i0)
        else:
            i0, i1 = sorted((self.iy[a], self.iy[b]))
            key = ("y", self.ix[line], i0)
        if i1 - i0 != 1 or key not in self.elem_ids:
            raise KeyError(f"梁 {line}/{a}-{b} はありません")
        return self.elem_ids[key]

    def master(self):
        return self.master_id

    def label(self, kind, id_):
        return self.labels.get((kind, id_), str(id_))
