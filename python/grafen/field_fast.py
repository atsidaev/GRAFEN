"""Numba-accelerated triangle field kernel."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap

    def prange(*args):
        return range(*args)


@njit(cache=True)
def _triangle_integral(p0, tri):
    a1 = tri[0] - p0
    a2 = tri[1] - p0
    a3 = tri[2] - p0
    a1m = np.sqrt(a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2])
    a2m = np.sqrt(a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2])
    a3m = np.sqrt(a3[0] * a3[0] + a3[1] * a3[1] + a3[2] * a3[2])
    a12 = tri[1] - tri[0]
    a23 = tri[2] - tri[1]
    a31 = tri[0] - tri[2]
    a12m = np.sqrt(a12[0] * a12[0] + a12[1] * a12[1] + a12[2] * a12[2])
    a23m = np.sqrt(a23[0] * a23[0] + a23[1] * a23[1] + a23[2] * a23[2])
    a31m = np.sqrt(a31[0] * a31[0] + a31[1] * a31[1] + a31[2] * a31[2])

    res = np.zeros(3)
    if a31m > 1e-30:
        res = res + a31 * (np.log((a3m + a1m + a31m) / (a3m + a1m - a31m)) / a31m)
    if a12m > 1e-30:
        res = res + a12 * (np.log((a1m + a2m + a12m) / (a1m + a2m - a12m)) / a12m)
    if a23m > 1e-30:
        res = res + a23 * (np.log((a2m + a3m + a23m) / (a2m + a3m - a23m)) / a23m)

    # n = cross(tri[1]-tri[0], tri[2]-tri[0])
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    n = np.empty(3)
    n[0] = e1[1] * e2[2] - e1[2] * e2[1]
    n[1] = e1[2] * e2[0] - e1[0] * e2[2]
    n[2] = e1[0] * e2[1] - e1[1] * e2[0]
    nn = np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    if nn < 1e-30:
        return np.array([np.nan, np.nan, np.nan])
    n /= nn
    # Cross product n × res (matches C++ Point3D operator*)
    cx = n[1] * res[2] - n[2] * res[1]
    cy = n[2] * res[0] - n[0] * res[2]
    cz = n[0] * res[1] - n[1] * res[0]
    res[0], res[1], res[2] = cx, cy, cz

    triple = a1[0] * (a2[1] * a3[2] - a2[2] * a3[1]) + a1[1] * (a2[2] * a3[0] - a2[0] * a3[2]) + a1[2] * (
        a2[0] * a3[1] - a2[1] * a3[0]
    )
    denom = (
        a1m * a2m * a3m
        + a3m * (a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2])
        + a2m * (a1[0] * a3[0] + a1[1] * a3[1] + a1[2] * a3[2])
        + a1m * (a2[0] * a3[0] + a2[1] * a3[1] + a2[2] * a3[2])
    )
    res = res + n * (2.0 * np.arctan2(triple, denom))
    return res


@njit(cache=True)
def _field_from_triangles(p0, triangles, dens):
    total = np.zeros(3)
    for i in range(triangles.shape[0]):
        tri = triangles[i]
        e1 = tri[1] - tri[0]
        e2 = tri[2] - tri[0]
        n = np.empty(3)
        n[0] = e1[1] * e2[2] - e1[2] * e2[1]
        n[1] = e1[2] * e2[0] - e1[0] * e2[2]
        n[2] = e1[0] * e2[1] - e1[1] * e2[0]
        nn = np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if nn < 1e-30:
            continue
        n /= nn
        g = _triangle_integral(p0, tri)
        if not (np.isfinite(g[0]) and np.isfinite(g[1]) and np.isfinite(g[2])):
            continue
        dot = n[0] * dens[i, 0] + n[1] * dens[i, 1] + n[2] * dens[i, 2]
        total = total + g * dot
    return total * (-1.0 / (4.0 * np.pi))


@njit(parallel=True, cache=True)
def _field_at_points(points, triangles, dens):
    out = np.empty_like(points)
    for i in prange(points.shape[0]):
        out[i] = _field_from_triangles(points[i], triangles, dens)
    return out


def field_from_triangles_fast(p0, triangles, dens):
    return _field_from_triangles(
        np.asarray(p0, dtype=np.float64),
        np.asarray(triangles, dtype=np.float64),
        np.asarray(dens, dtype=np.float64),
    )


def field_at_points_fast(points, triangles, dens):
    return _field_at_points(
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.float64),
        np.asarray(dens, dtype=np.float64),
    )
