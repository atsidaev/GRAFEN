"""Closed-form reference solutions for validation."""

from __future__ import annotations

import numpy as np

from .field import field_from_hexahedra


def sphere_magnetization(i0: np.ndarray, kappa: float) -> np.ndarray:
    """Uniform magnetization of a sphere: I = I0 / (1 + K/3)."""
    return np.asarray(i0, dtype=float) / (1.0 + kappa / 3.0)


def ellipsoid_magnetization(
    i0: np.ndarray,
    kappa: float,
    req: float,
    rpl: float,
) -> np.ndarray:
    """Uniform magnetization of an ellipsoid of revolution (C++ formula).

    Requires rpl >= req (prolate along z). For a sphere (req==rpl) delegates
    to sphere_magnetization.
    """
    i0 = np.asarray(i0, dtype=float)
    if abs(rpl - req) < 1e-15 * max(req, rpl, 1.0):
        return sphere_magnetization(i0, kappa)
    if rpl < req:
        raise ValueError("ellipsoid_magnetization expects rpl >= req (prolate along z)")

    m = rpl / req
    s = np.sqrt(m * m - 1.0)
    ln_pt = np.log((m + s) / (m - s))
    # Demagnetizing factors L (along z), M (along x,y)
    l_fac = (1.0 / (m * m - 1.0)) * ((m / (2.0 * s)) * ln_pt - 1.0)
    m_fac = (m / (2.0 * (m * m - 1.0))) * (m - (1.0 / (2.0 * s)) * ln_pt)
    return i0 / (1.0 + kappa * np.array([m_fac, m_fac, l_fac]))


def sphere_field_exterior(
    center: np.ndarray,
    radius: float,
    magnetization: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Exterior H of a uniformly magnetized sphere (SI dipole).

    H(r) = (R³/3) [3 (I·r̂) r̂ - I] / r³
    which equals (1/(4π)) [3(m·r)r/r⁵ - m/r³] with m = (4π/3) R³ I.

    Note: C++ field_sphere_H multiplies by an extra 4π and does not match
    the polyhedron kernel (or field_sphere_H_in_Hz); this function matches
    the SI polyhedron convention used by field_from_triangles.
    """
    center = np.asarray(center, dtype=float)
    j = np.asarray(magnetization, dtype=float)
    pts = np.asarray(points, dtype=float)

    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        rvec = p - center
        r = np.linalg.norm(rvec)
        if r <= radius * (1.0 + 1e-9):
            out[i] = -j / 3.0
            continue
        out[i] = (radius**3 / 3.0) * (3.0 * np.dot(j, rvec) * rvec / r**5 - j / r**3)
    return out


def cuboid_field_uniform(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    magnetization: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Exact H_snd of one uniformly magnetized axis-aligned cuboid.

    Implemented as the analytic triangle-face sum on the single hexahedron
    (same closed-form as GRAFEN; exact for planar faces).
    """
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners = np.array(
        [
            [
                [x1, y1, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x0, y0, z1],
                [x1, y1, z0],
                [x1, y0, z0],
                [x0, y1, z0],
                [x0, y0, z0],
            ]
        ],
        dtype=float,
    )
    dens = np.asarray(magnetization, dtype=float).reshape(1, 3)
    pts = np.asarray(points, dtype=float)
    out = np.empty_like(pts)
    for i, p0 in enumerate(pts):
        out[i] = field_from_hexahedra(p0, corners, dens)
    return out
