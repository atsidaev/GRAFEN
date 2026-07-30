"""Magnetic field of uniformly magnetized polyhedra via triangular faces.

Matches GRAFEN C++ (calcField.cu): for each triangle S of a cell with
magnetization I,

    contrib = g_S(a) * (n_S · I)

and

    H_snd(a) = -1/(4π) * sum(contrib).
"""

from __future__ import annotations

import numpy as np

_FIELD_CONST = -1.0 / (4.0 * np.pi)


def _are_points_on_one_side(line_p1, line_p2, t1, t2) -> bool:
    m1 = np.cross(line_p2 - line_p1, t1 - line_p2)
    m2 = np.cross(line_p2 - line_p1, t2 - line_p2)
    return (np.min(m1) > 0) == (np.min(m2) > 0)


def hexahedron_triangles(corners: np.ndarray) -> np.ndarray:
    """Return 12 triangles (12, 3, 3) for one hexahedron with corners (8, 3).

    Corner indexing matches C++ Hexahedron / HexahedronWid::getTri.
    """
    p = corners
    tris = np.empty((12, 3, 3), dtype=float)

    if not _are_points_on_one_side(p[2], p[1], p[0], p[3]):
        tris[0] = (p[2], p[3], p[1])
        tris[1] = (p[2], p[1], p[0])
    else:
        tris[0] = (p[3], p[1], p[0])
        tris[1] = (p[3], p[0], p[2])

    tris[2] = (p[0], p[4], p[6])
    tris[3] = (p[0], p[6], p[2])
    tris[4] = (p[3], p[7], p[5])
    tris[5] = (p[3], p[5], p[1])
    tris[6] = (p[0], p[1], p[5])
    tris[7] = (p[0], p[5], p[4])
    tris[8] = (p[6], p[7], p[3])
    tris[9] = (p[6], p[3], p[2])

    if not _are_points_on_one_side(p[6], p[5], p[4], p[7]):
        tris[10] = (p[6], p[4], p[5])
        tris[11] = (p[6], p[5], p[7])
    else:
        tris[10] = (p[4], p[5], p[7])
        tris[11] = (p[4], p[7], p[6])

    return tris


def hexahedra_to_triangles(
    corners: np.ndarray,
    magnetization: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten hexahedra to a triangle list.

    Parameters
    ----------
    corners : (N, 8, 3)
    magnetization : (N, 3)

    Returns
    -------
    triangles : (N*12, 3, 3)
    dens : (N*12, 3)  magnetization copied to each face of its parent cell
    """
    n = corners.shape[0]
    tris = np.empty((n * 12, 3, 3), dtype=float)
    dens = np.empty((n * 12, 3), dtype=float)
    for i in range(n):
        tris[i * 12 : (i + 1) * 12] = hexahedron_triangles(corners[i])
        dens[i * 12 : (i + 1) * 12] = magnetization[i]
    return tris, dens


def triangle_integral(p0: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Closed-form ∇∫_S 1/|q-a| dS contribution g_S(a) (C++ intTrAn)."""
    a1 = tri[0] - p0
    a2 = tri[1] - p0
    a3 = tri[2] - p0
    a1m = np.linalg.norm(a1)
    a2m = np.linalg.norm(a2)
    a3m = np.linalg.norm(a3)
    a12 = tri[1] - tri[0]
    a23 = tri[2] - tri[1]
    a31 = tri[0] - tri[2]
    a12m = np.linalg.norm(a12)
    a23m = np.linalg.norm(a23)
    a31m = np.linalg.norm(a31)

    res = a31 * (np.log((a3m + a1m + a31m) / (a3m + a1m - a31m)) / a31m)
    res = res + a12 * (np.log((a1m + a2m + a12m) / (a1m + a2m - a12m)) / a12m)
    res = res + a23 * (np.log((a2m + a3m + a23m) / (a2m + a3m - a23m)) / a23m)

    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    n = n / np.linalg.norm(n)
    # C++: res = N * res with Point3D::* = cross product (not projection)
    res = np.cross(n, res)

    triple = np.dot(a1, np.cross(a2, a3))
    denom = (
        a1m * a2m * a3m
        + a3m * np.dot(a1, a2)
        + a2m * np.dot(a1, a3)
        + a1m * np.dot(a2, a3)
    )
    res = res + n * (2.0 * np.arctan2(triple, denom))
    return res


def field_from_triangles(p0: np.ndarray, triangles: np.ndarray, dens: np.ndarray) -> np.ndarray:
    """H_snd at p0 from magnetized triangles."""
    from .field_fast import field_from_triangles_fast

    return field_from_triangles_fast(p0, triangles, dens)


def field_from_hexahedra(p0: np.ndarray, corners: np.ndarray, dens: np.ndarray) -> np.ndarray:
    """H_snd at p0 from magnetized hexahedra."""
    tris, tdens = hexahedra_to_triangles(corners, dens)
    return field_from_triangles(p0, tris, tdens)


def field_at_points(
    points: np.ndarray,
    corners: np.ndarray,
    dens: np.ndarray,
) -> np.ndarray:
    """Evaluate H_snd at many points. points (M,3) -> (M,3)."""
    from .field_fast import field_at_points_fast

    tris, tdens = hexahedra_to_triangles(corners, dens)
    return field_at_points_fast(points, tris, tdens)


def mass_centers(corners: np.ndarray) -> np.ndarray:
    """Centroid of each hexahedron via 6-tetrahedron split (C++ massCenter)."""
    # Tet corners relative to hexahedron corner indices (C++ splitTh)
    tets = (
        (0, 4, 5, 6),
        (0, 7, 5, 6),
        (0, 2, 6, 3),
        (0, 7, 6, 3),
        (0, 3, 1, 7),
        (0, 5, 1, 7),
    )
    n = corners.shape[0]
    centers = np.zeros((n, 3), dtype=float)
    for i in range(n):
        p = corners[i]
        r = np.zeros(3)
        mass = 0.0
        for a, b, c, d in tets:
            # Tetrahedron(p[a] as apex? C++: Tetrahedron(p, b1, b2, b3) with base triangle
            # Tetrahedron(p[0], p[4], p[5], p[6]) means apex p[0], base p4,p5,p6
            apex, b1, b2, b3 = p[a], p[b], p[c], p[d]
            vol = abs(np.dot(np.cross(b2 - b1, b3 - b2), apex - b1) / 6.0)
            # C++ uses abs((base.p2-base.p1)*(base.p3-base.p2)^(p-base.p1)/6)
            # base is Triangle(b1,b2,b3), so base.p1=b1, p2=b2, p3=b3
            cm = 0.25 * (apex + b1 + b2 + b3)
            r += cm * vol
            mass += vol
        centers[i] = r / mass if mass > 0 else p.mean(axis=0)
    return centers
