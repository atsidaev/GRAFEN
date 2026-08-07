"""Convert a closed STL surface into a GRAFEN volumetric hex VTU model.

Default path: axis-aligned hex voxelization of the STL AABB, keeping cells
whose centers lie inside the closed surface (ray casting).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mesh import cube_mesh
from .model_io import save_model
from .stl_io import read_stl, stl_bbox


def _ray_triangle_hits(
    origins: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Count Möller–Trumbore intersections of rays with triangles.

    Parameters
    ----------
    origins : (P, 3)
    direction : (3,) unit-ish direction (same for all rays)
    triangles : (T, 3, 3)

    Returns
    -------
    hits : (P,) int — number of intersections with t > eps
    """
    origins = np.asarray(origins, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    tris = np.asarray(triangles, dtype=np.float64)
    p = origins.shape[0]
    tcount = tris.shape[0]
    if p == 0 or tcount == 0:
        return np.zeros(p, dtype=np.int64)

    v0 = tris[:, 0]  # (T, 3)
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    # Process in origin chunks to bound memory (P * T)
    hits = np.zeros(p, dtype=np.int64)
    chunk = max(1, min(p, 512))
    for start in range(0, p, chunk):
        o = origins[start : start + chunk]  # (C, 3)
        c = o.shape[0]
        # Broadcast: (C, 1, 3) vs (1, T, 3)
        tvec = o[:, None, :] - v0[None, :, :]  # (C, T, 3)
        pvec = np.cross(direction, e2)  # (T, 3)
        det = np.einsum("tj,tj->t", e1, pvec)  # (T,)
        valid = np.abs(det) > eps
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]

        u = np.einsum("ctj,tj->ct", tvec, pvec) * inv_det[None, :]  # (C, T)
        qvec = np.cross(tvec, e1[None, :, :])  # (C, T, 3)
        v = np.einsum("ctj,j->ct", qvec, direction) * inv_det[None, :]
        t_param = np.einsum("ctj,tj->ct", qvec, e2) * inv_det[None, :]

        inside = (
            valid[None, :]
            & (u >= 0.0)
            & (v >= 0.0)
            & ((u + v) <= 1.0)
            & (t_param > eps)
        )
        hits[start : start + c] = inside.sum(axis=1)
    return hits


def points_inside_stl(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Odd-parity ray test: True if point is inside a closed triangle mesh."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # Slightly skewed ray reduces hits on shared edges / coplanar faces
    direction = np.array([1.0, 1e-5, 1e-6], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    hits = _ray_triangle_hits(points, direction, triangles, eps=eps)
    return (hits % 2) == 1


def _ray_first_hit_t(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Smallest t > eps for origin + t*direction hitting the mesh; nan if none."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = np.linalg.norm(direction)
    if dn < eps:
        return float("nan")
    direction = direction / dn
    tris = np.asarray(triangles, dtype=np.float64)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("tj,tj->t", e1, pvec)
    valid = np.abs(det) > eps
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin - v0
    u = np.einsum("tj,tj->t", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.einsum("tj,j->t", qvec, direction) * inv_det
    t_param = np.einsum("tj,tj->t", qvec, e2) * inv_det
    hit = valid & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t_param > eps)
    if not np.any(hit):
        return float("nan")
    return float(np.min(t_param[hit]))


def _spherical_unit(b: float, lam: float) -> np.ndarray:
    """Unit direction: latitude b (β), longitude lam (λ); z = sin(b)."""
    cb = np.cos(b)
    return np.array([cb * np.cos(lam), cb * np.sin(lam), np.sin(b)], dtype=np.float64)


def stl_geometric_center(triangles: np.ndarray) -> np.ndarray:
    """BBox center; must lie inside a star-convex body for polar meshing."""
    lo, hi = stl_bbox(triangles)
    return 0.5 * (lo + hi)


def polar_mesh_stl(
    triangles: np.ndarray,
    nl: int,
    nb: int,
    nr: int,
    magnetization: np.ndarray,
    kappa: float,
    *,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar (tesseroid-like) hex mesh from a center out to the STL surface.

    Angular grid covers the full sphere; each radial column is subdivided into
    ``nr`` shells. Cell faces are planar quads (flat “top/bottom” along radius).
    Body must be star-convex w.r.t. ``center`` (default: STL bbox center).

    Parameters
    ----------
    nl, nb, nr : longitude, latitude, and radial cell counts
    """
    if nl < 1 or nb < 1 or nr < 1:
        raise ValueError("nl, nb, nr must be >= 1")
    tris = np.asarray(triangles, dtype=np.float64)
    magnetization = np.asarray(magnetization, dtype=float).reshape(3)
    if center is None:
        center = stl_geometric_center(tris)
    else:
        center = np.asarray(center, dtype=np.float64).reshape(3)

    if not bool(points_inside_stl(center.reshape(1, 3), tris)[0]):
        raise ValueError(
            "polar_mesh_stl: center is not inside the STL; "
            "pass --center inside a star-convex body"
        )

    verts = tris.reshape(-1, 3)
    rel_v = verts - center.reshape(1, 3)
    vnorm = np.linalg.norm(rel_v, axis=1)
    vnorm_safe = np.maximum(vnorm, 1e-30)
    vdir = rel_v / vnorm_safe[:, None]

    def surface_radius(direction: np.ndarray) -> float:
        """Exit distance along unit ``direction``; robust to MT edge misses."""
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        dn = np.linalg.norm(direction)
        if dn < 1e-30:
            return float("nan")
        direction = direction / dn

        t = _ray_first_hit_t(center, direction, tris)
        if np.isfinite(t) and t > 0.0:
            return t

        # Grazing hit on STL edges (common when ray lies in a coordinate plane):
        # retry with tiny angular jitters.
        for scale in (1e-4, 1e-3, 1e-2):
            for axis in (0, 1, 2):
                jitter = np.zeros(3)
                jitter[axis] = scale
                for sign in (1.0, -1.0):
                    d2 = direction + sign * jitter
                    d2 = d2 / np.linalg.norm(d2)
                    t2 = _ray_first_hit_t(center, d2, tris)
                    if np.isfinite(t2) and t2 > 0.0:
                        return t2

        # Last resort: support among vertices nearly along this ray (not all verts)
        cosang = vdir @ direction
        for thr in (0.995, 0.98, 0.95, 0.9):
            sel = cosang >= thr
            if np.any(sel):
                return float(np.max(rel_v[sel] @ direction))
        return float("nan")

    # Full sphere: β ∈ [-π/2, π/2]; λ offset by half step so rays avoid
    # coordinate-plane triangle edges (λ=0,π) that make Möller–Trumbore miss.
    lam = (np.arange(nl) + 0.5) * (2.0 * np.pi / float(nl))
    b = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nb + 1)

    # Surface distance along each angular node ray (λ periodic: store nl columns)
    r_surf = np.empty((nb + 1, nl), dtype=np.float64)
    for bi in range(nb + 1):
        for li in range(nl):
            d = _spherical_unit(float(b[bi]), float(lam[li]))
            t = surface_radius(d)
            if not np.isfinite(t) or t <= 0.0:
                raise ValueError(
                    f"polar_mesh_stl: no surface hit for direction "
                    f"b={b[bi]:g}, lam={lam[li]:g}; body may not be star-convex"
                )
            r_surf[bi, li] = t

    # Angular node indices for hex corners (external/internal order = ellipsoid_mesh)
    # k: (b_idx, lam_idx) relative to cell (bi, li)
    corner_idx = (
        (1, 1),  # b[bi+1], lam[li+1]
        (0, 1),  # b[bi],   lam[li+1]
        (1, 0),  # b[bi+1], lam[li]
        (0, 0),  # b[bi],   lam[li]
    )

    cells: list[np.ndarray] = []
    for ri in range(nr):
        t0 = float(ri) / float(nr)
        t1 = float(ri + 1) / float(nr)
        for li in range(nl):
            for bi in range(nb):
                external = np.empty((4, 3), dtype=np.float64)
                internal = np.empty((4, 3), dtype=np.float64)
                for k, (db, dl) in enumerate(corner_idx):
                    bi_k = bi + db
                    li_k = (li + dl) % nl
                    r = r_surf[bi_k, li_k]
                    d = _spherical_unit(float(b[bi_k]), float(lam[li_k]))
                    if ri > 0:
                        internal[k] = center + (t0 * r) * d
                    else:
                        internal[k] = center.copy()
                    external[k] = center + (t1 * r) * d
                cells.append(np.vstack([external, internal]))

    corners = np.stack(cells, axis=0)
    dens = np.tile(magnetization, (corners.shape[0], 1))
    kappa_arr = np.full(corners.shape[0], float(kappa), dtype=float)
    return corners, dens, kappa_arr


def voxelize_stl(
    triangles: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    magnetization: np.ndarray,
    kappa: float,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    pad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill STL AABB with hex voxels; keep cells whose centers are inside.

    Parameters
    ----------
    triangles : (T, 3, 3) closed surface
    nx, ny, nz : subdivision of the bounding box
    magnetization : (3,) initial I for every kept cell
    kappa : scalar susceptibility for every kept cell
    bounds : optional AABB override ((x0,x1),(y0,y1),(z0,z1)); default STL bbox
    pad : expand AABB by this amount on each side (before subdividing)

    Returns
    -------
    corners : (N, 8, 3), dens : (N, 3), kappa_arr : (N,)
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    tris = np.asarray(triangles, dtype=np.float64)
    if bounds is None:
        lo, hi = stl_bbox(tris)
        bounds = (
            (float(lo[0]) - pad, float(hi[0]) + pad),
            (float(lo[1]) - pad, float(hi[1]) + pad),
            (float(lo[2]) - pad, float(hi[2]) + pad),
        )
    (x0, x1), (y0, y1), (z0, z1) = bounds
    magnetization = np.asarray(magnetization, dtype=float).reshape(3)

    corners_full, dens_full = cube_mesh(
        (x0, x1, nx),
        (y0, y1, ny),
        (z0, z1, nz),
        magnetization=magnetization,
    )
    centers = corners_full.mean(axis=1)
    inside = points_inside_stl(centers, tris)
    if not np.any(inside):
        raise ValueError(
            "voxelize_stl: no cell centers inside the STL; "
            "check that the surface is closed and watertight, or refine/pad the grid"
        )
    corners = corners_full[inside]
    dens = dens_full[inside]
    kappa_arr = np.full(corners.shape[0], float(kappa), dtype=float)
    return corners, dens, kappa_arr


def stl_to_hex_model(
    stl_path: str | Path,
    *,
    magnetization: np.ndarray,
    kappa: float,
    method: str = "voxel",
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    nl: int = 16,
    nb: int = 8,
    nr: int = 4,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    pad: float = 0.0,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read STL and build a volumetric hex model (``voxel`` or ``polar``)."""
    tris = read_stl(stl_path)
    method = method.lower().strip()
    if method == "voxel":
        return voxelize_stl(
            tris,
            nx,
            ny,
            nz,
            magnetization,
            kappa,
            bounds=bounds,
            pad=pad,
        )
    if method == "polar":
        return polar_mesh_stl(
            tris,
            nl,
            nb,
            nr,
            magnetization,
            kappa,
            center=center,
        )
    raise ValueError(f"Unknown meshing method {method!r}; use voxel|polar")


def convert_stl_to_vtu(
    stl_path: str | Path,
    vtu_path: str | Path,
    *,
    magnetization: np.ndarray,
    kappa: float,
    method: str = "voxel",
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    nl: int = 16,
    nb: int = 8,
    nr: int = 4,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    pad: float = 0.0,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners, dens, kappa_arr = stl_to_hex_model(
        stl_path,
        magnetization=magnetization,
        kappa=kappa,
        method=method,
        nx=nx,
        ny=ny,
        nz=nz,
        nl=nl,
        nb=nb,
        nr=nr,
        bounds=bounds,
        pad=pad,
        center=center,
    )
    save_model(vtu_path, corners, dens, kappa_arr)
    return corners, dens, kappa_arr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Convert a closed STL surface to a GRAFEN hex VTU. "
            "Default: AABB voxel fill. With --polar: spherical sectors from the "
            "body center out to the surface (planar-faced radial hexes)."
        ),
    )
    p.add_argument("stl", type=Path, help="Input .stl (closed / watertight)")
    p.add_argument("vtu", type=Path, help="Output .vtu")
    p.add_argument("-k", "--kappa", type=float, default=0.0, help="Susceptibility κ")
    p.add_argument(
        "-H",
        "--Hprime",
        dest="h_prime",
        type=float,
        nargs=3,
        default=None,
        metavar=("Hx", "Hy", "Hz"),
        help="Inducing field H'; store I0 = κ H' in the VTU (preferred over -I)",
    )
    p.add_argument(
        "-I",
        dest="magnetization",
        type=float,
        nargs=3,
        default=None,
        metavar=("Ix", "Iy", "Iz"),
        help="Magnetization I (alternative to -H; default 0 if neither given)",
    )
    p.add_argument(
        "--polar",
        action="store_true",
        help="Polar meshing from geometric center (star-convex bodies)",
    )
    p.add_argument("--nx", type=int, default=16, help="voxel: cells along X (default 16)")
    p.add_argument("--ny", type=int, default=16, help="voxel: cells along Y (default 16)")
    p.add_argument("--nz", type=int, default=16, help="voxel: cells along Z (default 16)")
    p.add_argument("--nl", type=int, default=16, help="polar: longitude cells (default 16)")
    p.add_argument("--nb", type=int, default=8, help="polar: latitude cells (default 8)")
    p.add_argument("--nr", type=int, default=4, help="polar: radial shells (default 4)")
    p.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("x", "y", "z"),
        help="polar: mesh center (default: STL bbox center; must be inside)",
    )
    p.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("x0", "x1", "y0", "y1", "z0", "z1"),
        help="voxel: AABB override (default: STL bbox)",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="voxel: expand STL bbox by this amount on each side",
    )
    args = p.parse_args(argv)

    bounds = None
    if args.bounds is not None:
        x0, x1, y0, y1, z0, z1 = args.bounds
        bounds = ((x0, x1), (y0, y1), (z0, z1))

    if args.h_prime is not None and args.magnetization is not None:
        p.error("use either -H/--Hprime or -I, not both")
    if args.h_prime is not None:
        magnetization = np.array(args.h_prime, dtype=float) * float(args.kappa)
    elif args.magnetization is not None:
        magnetization = np.array(args.magnetization, dtype=float)
    else:
        magnetization = np.zeros(3)

    center = None if args.center is None else np.array(args.center, dtype=float)
    method = "polar" if args.polar else "voxel"
    corners, dens, kappa = convert_stl_to_vtu(
        args.stl,
        args.vtu,
        magnetization=magnetization,
        kappa=args.kappa,
        method=method,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nl=args.nl,
        nb=args.nb,
        nr=args.nr,
        bounds=bounds,
        pad=args.pad,
        center=center,
    )
    print(f"Wrote {args.vtu} ({corners.shape[0]} hex cells, method={method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
