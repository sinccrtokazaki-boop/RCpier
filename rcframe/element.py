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
