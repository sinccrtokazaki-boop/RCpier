"""3次元梁柱要素の剛性マトリクスと各種変換

局所自由度の並び（1節点6自由度 × 2節点）:
    [ux, uy, uz, rx, ry, rz]_i  [ux, uy, uz, rx, ry, rz]_j
"""
import numpy as np


def local_stiffness(E, G, A, Iy, Iz, J, L, Asy=None, Asz=None):
    """せん断変形を考慮した（Timoshenko）12x12 局所剛性マトリクス

    Asy, Asz を None にするとせん断変形を無視（Euler-Bernoulli）
    """
    k = np.zeros((12, 12))

    ea = E * A / L
    k[0, 0] = k[6, 6] = ea
    k[0, 6] = k[6, 0] = -ea

    gj = G * J / L
    k[3, 3] = k[9, 9] = gj
    k[3, 9] = k[9, 3] = -gj

    # xy 面内の曲げ（uy, rz）: Iz
    p = 12.0 * E * Iz / (G * Asy * L ** 2) if Asy else 0.0
    c = E * Iz / ((1.0 + p) * L ** 3)
    idx = [1, 5, 7, 11]
    k[np.ix_(idx, idx)] += c * np.array([
        [12, 6 * L, -12, 6 * L],
        [6 * L, (4 + p) * L ** 2, -6 * L, (2 - p) * L ** 2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, (2 - p) * L ** 2, -6 * L, (4 + p) * L ** 2],
    ])

    # xz 面内の曲げ（uz, ry）: Iy
    p = 12.0 * E * Iy / (G * Asz * L ** 2) if Asz else 0.0
    c = E * Iy / ((1.0 + p) * L ** 3)
    idx = [2, 4, 8, 10]
    k[np.ix_(idx, idx)] += c * np.array([
        [12, -6 * L, -12, -6 * L],
        [-6 * L, (4 + p) * L ** 2, 6 * L, (2 - p) * L ** 2],
        [-12, 6 * L, 12, 6 * L],
        [-6 * L, (2 - p) * L ** 2, 6 * L, (4 + p) * L ** 2],
    ])
    return k


def local_axes(xi, xj, beta_deg=0.0):
    """部材局所座標軸（行ベクトル ex, ey, ez を並べた 3x3）

    規約:
      ・一般部材: 参照ベクトル = 全体 Z  →  水平梁の局所 z は鉛直上向き
      ・鉛直部材: 参照ベクトル = 全体 Y  →  柱の局所 y = 全体 X, 局所 z = 全体 Y
      ・beta: 局所 x 軸まわりのコードアングル [deg]
    """
    xi, xj = np.asarray(xi, float), np.asarray(xj, float)
    d = xj - xi
    L = np.linalg.norm(d)
    ex = d / L
    v = np.array([0.0, 1.0, 0.0]) if abs(ex[2]) > 0.9999 else np.array([0.0, 0.0, 1.0])
    ey = np.cross(v, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    if beta_deg:
        b = np.radians(beta_deg)
        ey, ez = np.cos(b) * ey + np.sin(b) * ez, -np.sin(b) * ey + np.cos(b) * ez
    return np.vstack([ex, ey, ez]), L


def transformation_matrix(R):
    """全体→局所の 12x12 変換マトリクス"""
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return T


def rigid_zone_matrix(ai, aj):
    """剛域変換: 可撓部端の変位 = H @ 節点の変位（局所座標）

    u' = u + θ × r,  r_i = (ai,0,0),  r_j = (-aj,0,0)
    """
    H = np.eye(12)
    H[1, 5] = ai
    H[2, 4] = -ai
    H[7, 11] = -aj
    H[8, 10] = aj
    return H


def udl_equivalent_loads(w_local, Lf, ai, aj):
    """局所座標の等分布荷重 w=(wx,wy,wz) [N/mm] に対する等価節点荷重

    戻り値:
      p_flex  : 可撓部（長さ Lf）両端の等価節点力（局所, 12）
      p_rigid : 剛域上の荷重を節点へ直接伝達した分（局所, 12）
    """
    wx, wy, wz = w_local
    p = np.zeros(12)
    p[0] = p[6] = wx * Lf / 2
    p[1] = p[7] = wy * Lf / 2
    p[5] = wy * Lf ** 2 / 12
    p[11] = -wy * Lf ** 2 / 12
    p[2] = p[8] = wz * Lf / 2
    p[4] = -wz * Lf ** 2 / 12
    p[10] = wz * Lf ** 2 / 12

    r = np.zeros(12)
    r[0:3] = np.array([wx, wy, wz]) * ai
    r[5] = wy * ai ** 2 / 2
    r[4] = -wz * ai ** 2 / 2
    r[6:9] = np.array([wx, wy, wz]) * aj
    r[11] = -wy * aj ** 2 / 2
    r[10] = wz * aj ** 2 / 2
    return p, r


_GX, _GW = np.polynomial.legendre.leggauss(3)   # 3点 Gauss: 5次まで厳密


def _profile_samples(profile, a, b):
    """区分線形の分布 profile=[(x, f)] の区間 [a, b] 上の積分点 (x, f*重み)
    被積分関数は f（1次）× 高々3次の多項式なので各区間 3点 Gauss で厳密"""
    xs = [x for x, _ in profile]
    fs = [f for _, f in profile]
    cuts = sorted({a, b, *[x for x in xs if a < x < b]})
    pts, wts = [], []
    for s0, s1 in zip(cuts, cuts[1:]):
        lo, hi = max(s0, xs[0]), min(s1, xs[-1])
        if hi - lo <= 0:
            continue
        x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _GX
        pts.append(x)
        wts.append(0.5 * (hi - lo) * _GW * np.interp(x, xs, fs))
    if not pts:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(pts), np.concatenate(wts)


def _fixed_end_transverse(s, q, Lf, EI, GA):
    """両端固定 Timoshenko 梁の横荷重 q(s) に対する等価節点荷重（θ = dw/ds 基準）

    柔性法: i端固定の片持ち梁に置き換え、j端の反力 (Rj, Mj) を適合条件から求める
        片持ち梁の j端たわみ（点荷重 P が s にあるとき）: P s²(3Lf−s)/(6EI) + P s/GA
        j端回転角: P s²/(2EI)
    戻り値: (Pi, Mi, Pj, Mj) = −(固定端反力)
    """
    cs = 1.0 / GA if GA else 0.0
    dq = np.sum(q * (s ** 2 * (3 * Lf - s) / (6 * EI) + s * cs))
    tq = np.sum(q * s ** 2 / (2 * EI))
    F = np.array([[Lf ** 3 / (3 * EI) + Lf * cs, Lf ** 2 / (2 * EI)],
                  [Lf ** 2 / (2 * EI), Lf / EI]])
    Rj, Mj = np.linalg.solve(F, [-dq, -tq])
    Ri = -np.sum(q) - Rj
    Mi = -Mj - Rj * Lf - np.sum(q * s)
    return -Ri, -Mi, -Rj, -Mj


def distributed_equivalent_loads(w_local, profile, L, ai, aj, EIy, EIz, GAy=None, GAz=None):
    """区分線形分布荷重（台形・三角形など）に対する等価節点荷重

    荷重 = w_local × f(x)  [N/mm]
      w_local : 局所座標の荷重方向と大きさ (wx, wy, wz)
      profile : [(x, f)] x は i節点からの距離 [mm]（昇順）。範囲外は 0
      L       : 節点間距離, ai/aj : 剛域長
      EIy, EIz: 可撓部の曲げ剛性（局所 y/z 軸まわり）
      GAy, GAz: せん断剛性 G*As（None でせん断変形無視）。GAy は uy 方向の変形用
    せん断変形を考慮した固定端力を柔性法で厳密に求める（非対称荷重では
    せん断変形が固定端モーメントに影響するため）。
    戻り値は udl_equivalent_loads と同じ (p_flex, p_rigid)。
    """
    wx, wy, wz = w_local
    Lf = L - ai - aj
    p = np.zeros(12)
    r = np.zeros(12)

    # 可撓部（s = x − ai）
    x, q = _profile_samples(profile, ai, L - aj)
    if len(x):
        s = x - ai
        p[0], p[6] = wx * np.sum(q * (1 - s / Lf)), wx * np.sum(q * s / Lf)
        Pi, Mi, Pj, Mj = _fixed_end_transverse(s, q, Lf, EIz, GAy)   # xy 面: θ = rz
        p[1], p[5], p[7], p[11] = wy * Pi, wy * Mi, wy * Pj, wy * Mj
        Pi, Mi, Pj, Mj = _fixed_end_transverse(s, q, Lf, EIy, GAz)   # xz 面: ry = −dw/dx
        p[2], p[4], p[8], p[10] = wz * Pi, -wz * Mi, wz * Pj, -wz * Mj

    # 剛域上の荷重は節点へ直接伝達（モーメントは節点まわり）
    for (a, b), node, off in (((0.0, ai), 0, 0.0), ((L - aj, L), 6, L)):
        if b - a <= 0:
            continue
        x, q = _profile_samples(profile, a, b)
        if not len(x):
            continue
        Q, Qd = np.sum(q), np.sum(q * (x - off))
        r[node:node + 3] += np.array([wx, wy, wz]) * Q
        r[node + 5] += wy * Qd
        r[node + 4] += -wz * Qd
    return p, r
