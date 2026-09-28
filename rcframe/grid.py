"""通り芯と位置指定の解釈（建物 building.py・桟橋 pier.py で共通）

位置の指定
    交点   "X2/Y1", "X2/*", "*/Y1", "*"
    梁     "Y1/X1-X3"（Y1通り上 X1〜X3 間の X方向梁）, "X2/*", "*"（全スパン）
    区画   "X1-X2/Y1-Y2"（X範囲/Y範囲で囲まれた区画）, "*"
"""

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


class GridMixin:
    """通り芯（self.gx, self.gy, self.ix, self.iy）と位置指定の解釈"""

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
        """交点の指定 -> [(ix, iy)]"""
        pts = []
        for s in _as_list(at):
            s = str(s)
            if s == ALL:
                s = "*/*"
            parts = s.split("/")
            if len(parts) != 2:
                raise InputError(f"{what}: 位置は \"X通り/Y通り\" で指定してください: {s!r}")
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

    def _area_cells(self, area, what):
        """"X範囲/Y範囲" または "*" -> 区画 [(ix, iy)]"""
        area = str(area)
        parts = ["*", "*"] if area == ALL else area.split("/")
        if len(parts) != 2:
            raise InputError(f"{what}: area は \"X範囲/Y範囲\" で指定してください: {area!r}")
        i0, i1 = self._range(parts[0], self.ix, what)
        j0, j1 = self._range(parts[1], self.iy, what)
        return [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
