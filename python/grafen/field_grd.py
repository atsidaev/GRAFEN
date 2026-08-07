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
) -> Grid:
    """Load model, compute no-demag field on grid, write Surfer GS binary (.grd)."""
    corners, dens, _kappa = load_model(vtu_path)
    xmin, xmax, xnum, _dx = parse_axis_spec(x_spec, n=n_col, name="x")
    ymin, ymax, ynum, _dy = parse_axis_spec(y_spec, n=n_row, name="y")
    grid = field_grid_from_model(
        corners,
        dens,
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
