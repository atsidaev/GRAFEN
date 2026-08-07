#!/usr/bin/env python3
"""Compute magnetic field (no demagnetization) from a VTU model onto a Surfer GRD.

Uses magnetization ``I`` stored in the VTU as-is (no demag CG). Output is
Surfer 7 binary grid via ``grafen.pygrid``.

Examples
--------
Spacing form (xLL, xRR, xSize)::

  python tools/vtu_to_grd.py model.vtu hz.grd --x=-15,15,1 --y=-10,10,1 --z=5

Node-count form::

  python tools/vtu_to_grd.py model.vtu hz.grd \\
      --x=-15,15 --y=-10,10 --nCol=256 --nRow=256 --z=5 --component hz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grafen.field_grd import parse_axis_spec, vtu_field_to_grd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "GRAFEN field (no demag) from VTU → Surfer GRD. "
            "Grid: --x=lo,hi,step or --x=lo,hi with --nCol; same for --y / --nRow."
        ),
    )
    p.add_argument("vtu", type=Path, help="Input model (.vtu or .npz)")
    p.add_argument("grd", type=Path, help="Output Surfer 7 binary .grd")
    p.add_argument(
        "--x",
        required=True,
        help="X range: lo,hi,step  or  lo,hi (then --nCol)",
    )
    p.add_argument(
        "--y",
        required=True,
        help="Y range: lo,hi,step  or  lo,hi (then --nRow)",
    )
    p.add_argument(
        "--nCol",
        type=int,
        default=None,
        dest="n_col",
        help="Number of X nodes (with --x=lo,hi)",
    )
    p.add_argument(
        "--nRow",
        type=int,
        default=None,
        dest="n_row",
        help="Number of Y nodes (with --y=lo,hi)",
    )
    p.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="Observation height Z (constant plane, default 0)",
    )
    p.add_argument(
        "--component",
        choices=("hx", "hy", "hz", "bt"),
        default="hz",
        help="Scalar written to GRD (default: hz). bt = |H|",
    )
    args = p.parse_args(argv)

    # Validate axis specs early for clearer errors
    xmin, xmax, xnum, dx = parse_axis_spec(args.x, n=args.n_col, name="x")
    ymin, ymax, ynum, dy = parse_axis_spec(args.y, n=args.n_row, name="y")

    grid = vtu_field_to_grd(
        args.vtu,
        args.grd,
        x_spec=args.x,
        y_spec=args.y,
        z=args.z,
        n_col=args.n_col,
        n_row=args.n_row,
        component=args.component,
    )
    print(
        f"Wrote {args.grd}  {grid.xnum}x{grid.ynum}  "
        f"X=[{xmin:g},{xmax:g}] dx={dx:g}  Y=[{ymin:g},{ymax:g}] dy={dy:g}  "
        f"z={args.z:g}  component={args.component}  "
        f"zmin={grid.zmin:g} zmax={grid.zmax:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
