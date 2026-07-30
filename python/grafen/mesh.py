"""Volume mesh generators (hexahedra) for cubes, spheres, and ellipsoids."""

from __future__ import annotations

import numpy as np


def _ellipsoid_point(req: float, rpl: float, b: float, lam: float) -> np.ndarray:
    """Point on ellipsoid of revolution (C++ Ellipsoid::getPoint), H=0."""
    l = 1.0 / np.sqrt(req * req * np.cos(b) ** 2 + rpl * rpl * np.sin(b) ** 2)
    x = (req * req * l) * np.cos(b) * np.cos(lam)
    y = (req * req * l) * np.cos(b) * np.sin(lam)
    z = (rpl * rpl * l) * np.sin(b)
    return np.array([x, y, z], dtype=float)


def _mirror_hex(corners: np.ndarray, axis: int) -> np.ndarray:
    """Mirror hexahedron corners across a coordinate axis (C++ mirrorX/Y/Z)."""
    out = corners.copy()
    out[:, axis] *= -1.0
    # Swap pairs so outward orientation stays consistent with C++
    if axis == 0:  # X
        swaps = ((0, 1), (2, 3), (4, 5), (6, 7))
    elif axis == 1:  # Y
        swaps = ((0, 2), (1, 3), (4, 6), (5, 7))
    else:  # Z
        swaps = ((0, 4), (1, 5), (2, 6), (3, 7))
    for i, j in swaps:
        out[[i, j]] = out[[j, i]]
    return out


def ellipsoid_mesh(
    req: float,
    rpl: float,
    nl: int,
    nb: int,
    nr: int,
    magnetization: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Body-fitted hexahedral mesh of an ellipsoid of revolution (or sphere).

    Builds one octant then mirrors to fill the full body (C++ ellipsoidGen).

    Returns
    -------
    corners : (N, 8, 3)
    dens : (N, 3)
    """
    if magnetization is None:
        magnetization = np.zeros(3)

    lam = np.linspace(0.0, 0.5 * np.pi, nl + 1)
    b = np.linspace(0.0, 0.5 * np.pi, nb + 1)
    reqs = np.linspace(0.0, req, nr + 1)
    rpls = np.linspace(0.0, rpl, nr + 1)

    cells: list[np.ndarray] = []
    for ri in range(nr):
        for li in range(nl):
            for bi in range(nb):
                if ri > 0:
                    ei_req, ei_rpl = reqs[ri], rpls[ri]
                    internal = np.array(
                        [
                            _ellipsoid_point(ei_req, ei_rpl, b[bi + 1], lam[li + 1]),
                            _ellipsoid_point(ei_req, ei_rpl, b[bi], lam[li + 1]),
                            _ellipsoid_point(ei_req, ei_rpl, b[bi + 1], lam[li]),
                            _ellipsoid_point(ei_req, ei_rpl, b[bi], lam[li]),
                        ]
                    )
                else:
                    # Degenerate inner face at the origin (pyramid-like first shell)
                    internal = np.zeros((4, 3))

                ee_req, ee_rpl = reqs[ri + 1], rpls[ri + 1]
                external = np.array(
                    [
                        _ellipsoid_point(ee_req, ee_rpl, b[bi + 1], lam[li + 1]),
                        _ellipsoid_point(ee_req, ee_rpl, b[bi], lam[li + 1]),
                        _ellipsoid_point(ee_req, ee_rpl, b[bi + 1], lam[li]),
                        _ellipsoid_point(ee_req, ee_rpl, b[bi], lam[li]),
                    ]
                )
                # Hexahedron(qUpper=external, qLower=internal) →
                # p = [ext.p1..p4, int.p1..p4]
                corners = np.vstack([external, internal])
                cells.append(corners)

    octant = np.stack(cells, axis=0)

    def dbl(axis: int, arr: np.ndarray) -> np.ndarray:
        mirrored = np.stack([_mirror_hex(c, axis) for c in arr], axis=0)
        return np.concatenate([arr, mirrored], axis=0)

    full = dbl(0, octant)
    full = dbl(1, full)
    full = dbl(2, full)

    dens = np.tile(np.asarray(magnetization, dtype=float), (full.shape[0], 1))
    return full, dens


def sphere_mesh(
    radius: float,
    nl: int,
    nb: int,
    nr: int,
    magnetization: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical body mesh (ellipsoid with equal axes)."""
    return ellipsoid_mesh(radius, radius, nl, nb, nr, magnetization)


def cube_mesh(
    x: tuple[float, float, int],
    y: tuple[float, float, int],
    z: tuple[float, float, int],
    magnetization: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Regular Cartesian brick mesh (C++ cubeGen).

    Each of x,y,z is (lower, upper, n_cells).
    Corner order matches C++: upper quad then lower quad in z.
    """
    if magnetization is None:
        magnetization = np.zeros(3)

    x0, x1, nx = x
    y0, y1, ny = y
    z0, z1, nz = z
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    zs = np.linspace(z0, z1, nz + 1)

    n = nx * ny * nz
    corners = np.empty((n, 8, 3), dtype=float)
    idx = 0
    for zi in range(nz):
        for yi in range(ny):
            for xi in range(nx):
                # C++ Quadrangle cur = { (x+1,y+1), (x+1,y), (x,y+1), (x,y) }
                # hex = Hexahedron(cur + z_upper, cur + z_lower)
                # so p[0..3] at z.at(zi+1), p[4..7] at z.at(zi)
                xu0, xu1 = xs[xi], xs[xi + 1]
                yu0, yu1 = ys[yi], ys[yi + 1]
                zu0, zu1 = zs[zi], zs[zi + 1]
                upper = np.array(
                    [
                        [xu1, yu1, zu1],
                        [xu1, yu0, zu1],
                        [xu0, yu1, zu1],
                        [xu0, yu0, zu1],
                    ]
                )
                lower = np.array(
                    [
                        [xu1, yu1, zu0],
                        [xu1, yu0, zu0],
                        [xu0, yu1, zu0],
                        [xu0, yu0, zu0],
                    ]
                )
                corners[idx] = np.vstack([upper, lower])
                idx += 1

    dens = np.tile(np.asarray(magnetization, dtype=float), (n, 1))
    return corners, dens


def translate_mesh(corners: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate all cell corners by offset."""
    return corners + np.asarray(offset, dtype=float).reshape(1, 1, 3)


def merge_meshes(
    *meshes: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate several (corners, dens) meshes."""
    corners = np.concatenate([m[0] for m in meshes], axis=0)
    dens = np.concatenate([m[1] for m in meshes], axis=0)
    return corners, dens
