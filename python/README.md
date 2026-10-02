# GRAFEN (Python)

Forward magnetic modelling with self-demagnetization — Python port of the C++/GPU GRAFEN solver.

## Install

```bash
cd python
pip install -e ".[test]"
```

## Run tests

```bash
cd python
pytest -q
```

## Model files

Save/load volumetric hex models (corners + magnetization ``I`` + ``kappa``):

- ``.npz`` — exact NumPy round-trip
- ``.vtu`` — VTK UnstructuredGrid (ParaView); hex cells with cell data ``I`` and ``kappa``

```python
from grafen import save_model, load_model, cube_mesh

corners, dens = cube_mesh((-5, 5, 4), (-2, 2, 2), (-2, 2, 2), magnetization=[2.8, 2.8, 7.0])
save_model("body.vtu", corners, dens, kappa=0.2)
corners, dens, kappa = load_model("body.vtu")
```

STL is not supported as a model format (surface-only; no hex cells / ``I`` / ``kappa``).

## STL → VTU

Convert a closed (watertight) STL into a GRAFEN hex VTU by voxelizing the
bounding box and keeping cells whose centers lie inside the surface:

```bash
# AABB voxel fill (default)
python tools/stl_to_vtu.py model.stl model.vtu --nx 32 --ny 32 --nz 32 \
  -k 2 -H 14 14 35

# Polar / tesseroid-like hexes from the body center (star-convex shapes)
python tools/stl_to_vtu.py ellipsoid.stl ell.vtu --polar --nl 24 --nb 12 --nr 6 \
  -k 2 -H 14 14 35

# Cubed-sphere: inner equal cube + 6-face shells (more uniform cell size)
python tools/stl_to_vtu.py ellipsoid.stl ell.vtu --cubed-sphere -n 12 --nr 6 \
  -k 2 -H 14 14 35

# Talwani stacked prisms: same AABB grid as voxel, XY fill of horizontal slices
# (fast on large STLs; 67P-scale meshes)
python tools/stl_to_vtu.py Comet_67P.stl comet.vtu --talwani \
    --nx 16 --ny 16 --nz 16 -k 2 -H 14 14 35
```

(``-H`` stores ``I0 = κ H'``; ``-I`` is an optional alternative.
``--polar`` / ``--cubed-sphere`` need a center inside the body; default is the STL bbox center.)


Optional `--bounds x0 x1 y0 y1 z0 z1` overrides the STL AABB; `--pad` expands it.

Test surfaces: `tests/data/{cube,sphere,ellipsoid}.stl` — fully below z=0 (tops at z=-5);
regenerate with `python tools/generate_test_stls.py`.

## Field → Surfer GRD (no demag)

Compute the anomalous field from a VTU model using stored ``I`` (no demagnetization
solve) on a constant-Z plane and write Surfer 7 binary GRD via ``grafen.pygrid``:

```bash
# spacing form: --x=xLL,xRR,xSize ; I0 = κ H' (no demag)
python tools/vtu_to_grd.py body.vtu hz.grd --x=-15,15,1 --y=-10,10,1 --z=5 \
  -H 14 14 35 -k 0.2

# node-count form
python tools/vtu_to_grd.py body.vtu hz.grd \
  --x=-15,15 --y=-10,10 --nCol=256 --nRow=256 --z=5 --component hz \
  -H 14 14 35 -k 0.2
```

``--component`` is ``hx`` | ``hy`` | ``hz`` (default) | ``bt`` (|H|).
Without ``-H``, uses magnetization ``I`` already stored in the VTU.
``stl_to_vtu`` likewise accepts ``-H`` / ``-k`` (stores ``I0 = κ H'``).

CUDA twin (same CLI flags): ``make -C src/tools`` then
``src/tools/vtu_to_grd model.vtu out.grd --x=-20,20,10 --y=-20,20,10 -z 0 -H 2 2 2 -k 1``.

## Field accuracy vs element count

For each target hex count, mesh **voxel, polar, and cubed-sphere**, compute **H**
on a survey plane (uniform ``I0 = κ H'``, no demag), and print relative RMS vs
the analytic field (sphere dipole or cuboid). ``-n`` is the only mesh control:
the 1-D size is ``m = round(N^(1/3))`` (voxel ``m³``, polar ``m×m×m``; cubed-sphere
picks ``n, nr`` so ``n³ + 6 n² nr`` is closest to ``N``).

```bash
python tools/field_accuracy.py tests/data/sphere.stl /tmp/acc -n 64,216,512
```

Writes ``outdir/accuracy.csv`` and one ``.vtu`` per method×N. ``--grd`` also
writes a Surfer GRD per run. ``--shape auto`` infers sphere or cube from the
STL name (ellipsoid has no closed-form exterior field for uniform I).

