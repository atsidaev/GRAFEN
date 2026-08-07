"""Observation-grid helpers and Surfer GRD field output (no demagnetization)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .field import field_at_points
from .model_io import load_model
from .pygrid import Grid, create


def parse_axis_spec(
    spec: str,
    *,
    n: int | None = None,
    name: str = "axis",
) -> tuple[float, float, int, float]:
    """Parse ``lo,hi[,step]`` into ``(lo, hi, n_nodes, step)``.

    Forms
    -----
    ``lo,hi,step`` — node count from step: ``n = round((hi-lo)/step) + 1``
    ``lo,hi`` — requires ``n`` (``--nCol`` / ``--nRow``); step = ``(hi-lo)/(n-1)``
    """
    parts = [p.strip() for p in str(spec).replace(";", ",").split(",") if p.strip()]
    if len(parts) not in (2, 3):
        raise ValueError(
            f"--{name} expects lo,hi or lo,hi,step (got {spec!r})"
        )
    lo = float(parts[0])
    hi = float(parts[1])
    if hi <= lo:
        raise ValueError(f"--{name}: upper bound must be > lower ({lo} .. {hi})")

    if len(parts) == 3:
        if n is not None:
            raise ValueError(
                f"--{name}: give either step in the range or --nCol/--nRow, not both"
            )
        step = float(parts[2])
        if step <= 0:
            raise ValueError(f"--{name}: step must be > 0")
        n_nodes = int(round((hi - lo) / step)) + 1
        if n_nodes < 2:
            raise ValueError(f"--{name}: need at least 2 nodes (got {n_nodes})")
        # Recompute hi so Surfer header is consistent with integer n and step
        hi_exact = lo + step * (n_nodes - 1)
        return lo, hi_exact, n_nodes, step

    if n is None:
        raise ValueError(
            f"--{name}={spec} needs --nCol/--nRow (or pass step as third value)"
        )
    if n < 2:
        raise ValueError(f"node count for --{name} must be >= 2 (got {n})")
    step = (hi - lo) / float(n - 1)
    return lo, hi, int(n), step


def make_xy_points(
    xmin: float,
    xmax: float,
    xnum: int,
    ymin: float,
    ymax: float,
    ynum: int,
    z: float,
) -> tuple[np.ndarray, Grid]:
    """Build (N,3) observation points in Surfer row-major order and empty Grid."""
    grid = create(xmin, xmax, xnum, ymin, ymax, ynum)
    xs = xmin + grid.dx * np.arange(xnum)
    ys = ymin + grid.dy * np.arange(ynum)
    # iy slow, ix fast — matches pygrid / Surfer GS binary
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [
            xx.ravel(order="C"),
            yy.ravel(order="C"),
            np.full(xnum * ynum, float(z), dtype=np.float64),
        ]
    )
    return pts, grid


def component_values(h: np.ndarray, component: str) -> np.ndarray:
    """Select a scalar field from vector H (N, 3)."""
    c = component.lower().strip()
    if c in ("hx", "x", "bx"):
        return h[:, 0]
    if c in ("hy", "y", "by"):
        return h[:, 1]
    if c in ("hz", "z", "bz"):
        return h[:, 2]
    if c in ("bt", "b", "abs", "mag", "|h|"):
        return np.linalg.norm(h, axis=1)
    raise ValueError(
        f"Unknown component {component!r}; use hx|hy|hz|bt"
    )


def field_grid_from_model(
    corners: np.ndarray,
    dens: np.ndarray,
    *,
    xmin: float,
    xmax: float,
    xnum: int,
    ymin: float,
    ymax: float,
    ynum: int,
    z: float,
    component: str = "hz",
) -> Grid:
    """Compute H from VTU magnetizations (no demag) on a planar XY grid → Surfer Grid."""
    pts, grid = make_xy_points(xmin, xmax, xnum, ymin, ymax, ynum, z)
    h = field_at_points(pts, corners, dens)
    grid.data = np.ascontiguousarray(component_values(h, component), dtype=np.float64)
    grid.fixzminmax()
    return grid


def magnetization_for_field(
    dens: np.ndarray,
    kappa: np.ndarray,
    *,
    h_prime: np.ndarray | None = None,
    kappa_override: float | np.ndarray | None = None,
) -> np.ndarray:
    """Magnetization used for no-demag field: stored ``I``, or ``I0 = κ H′``.

    If ``h_prime`` is given, ignore stored ``dens`` and set
    ``I = kappa * H'`` (per cell). ``kappa_override`` replaces VTU κ when set.
    """
    n = dens.shape[0]
    if kappa_override is not None:
        k = np.broadcast_to(np.asarray(kappa_override, dtype=float), (n,)).astype(float)
    else:
        k = np.asarray(kappa, dtype=float).reshape(n)
    if h_prime is not None:
        hp = np.asarray(h_prime, dtype=float).reshape(3)
        return k[:, None] * hp
    return np.asarray(dens, dtype=float).reshape(n, 3)


def vtu_field_to_grd(
    vtu_path: str | Path,
    grd_path: str | Path,
    *,
    x_spec: str,
    y_spec: str,
    z: float = 0.0,
    n_col: int | None = None,
    n_row: int | None = None,
    component: str = "hz",
    h_prime: np.ndarray | None = None,
    kappa: float | None = None,
) -> Grid:
    """Load model, compute no-demag field on grid, write Surfer GS binary (.grd).

    Magnetization: VTU ``I`` by default. With ``h_prime``, use ``I0 = κ H'``
    (κ from VTU, or ``kappa`` if given).
    """
    corners, dens, kappa_arr = load_model(vtu_path)
    dens_use = magnetization_for_field(
        dens, kappa_arr, h_prime=h_prime, kappa_override=kappa
    )
    z_lo = float(corners[:, :, 2].min())
    z_hi = float(corners[:, :, 2].max())
    if z_lo - 1e-9 <= z <= z_hi + 1e-9:
        import warnings

        warnings.warn(
            f"Observation plane z={z:g} intersects the model Z-range "
            f"[{z_lo:g}, {z_hi:g}]; field singularities / NaNs are expected. "
            f"Use --z outside the body (C++ examples: bodies at z<0, grid at z=0).",
            UserWarning,
            stacklevel=2,
        )
    if not np.any(np.abs(dens_use) > 0):
        import warnings

        warnings.warn(
            "Magnetization I is all zeros — GRD will be ~0. "
            "Pass -H Hx Hy Hz (and -k / VTU kappa) or store nonzero I in the VTU.",
            UserWarning,
            stacklevel=2,
        )
    xmin, xmax, xnum, _dx = parse_axis_spec(x_spec, n=n_col, name="x")
    ymin, ymax, ynum, _dy = parse_axis_spec(y_spec, n=n_row, name="y")
    grid = field_grid_from_model(
        corners,
        dens_use,
        xmin=xmin,
        xmax=xmax,
        xnum=xnum,
        ymin=ymin,
        ymax=ymax,
        ynum=ynum,
        z=z,
        component=component,
    )
    grid.write_grd7(str(grd_path))
    return grid
