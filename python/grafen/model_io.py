"""Save/load GRAFEN volumetric hex models (.npz and .vtu).

NPZ arrays (exact round-trip):
  corners: float64 (N, 8, 3)  — GRAFEN Hexahedron corner order
  dens:    float64 (N, 3)     — magnetization I
  kappa:   float64 (N,)       — susceptibility

VTU UnstructuredGrid uses VTK_HEXAHEDRON connectivity (mapped from GRAFEN
order) with cell data ``I`` (3-vector) and ``kappa`` (scalar).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# GRAFEN corner i -> VTK_HEXAHEDRON local node (axis-aligned convention)
_GRAFEN_TO_VTK = np.array([7, 5, 4, 6, 3, 1, 0, 2], dtype=np.int64)
_VTK_TO_GRAFEN = np.argsort(_GRAFEN_TO_VTK)

_VTK_HEXAHEDRON = 12


def save_model(
    path: str | Path,
    corners: np.ndarray,
    dens: np.ndarray,
    kappa: np.ndarray | float | None = None,
) -> None:
    """Write a model to ``.npz`` or ``.vtu`` (by suffix)."""
    path = Path(path)
    corners, dens, kappa = _normalize(corners, dens, kappa)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        _save_npz(path, corners, dens, kappa)
    elif suffix == ".vtu":
        _save_vtu(path, corners, dens, kappa)
    else:
        raise ValueError(f"Unsupported model format: {path.suffix} (use .npz or .vtu)")


def load_model(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(corners, dens, kappa)`` from ``.npz`` or ``.vtu``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".vtu":
        return _load_vtu(path)
    raise ValueError(f"Unsupported model format: {path.suffix} (use .npz or .vtu)")


def _normalize(
    corners: np.ndarray,
    dens: np.ndarray,
    kappa: np.ndarray | float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(corners, dtype=np.float64)
    dens = np.asarray(dens, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(f"corners must be (N, 8, 3), got {corners.shape}")
    n = corners.shape[0]
    if dens.ndim == 1:
        dens = np.tile(dens, (n, 1))
    dens = dens.reshape(n, 3)
    if kappa is None:
        kappa_arr = np.zeros(n, dtype=np.float64)
    else:
        kappa_arr = np.broadcast_to(np.asarray(kappa, dtype=np.float64), (n,)).copy()
    return corners, dens, kappa_arr


def _save_npz(path: Path, corners: np.ndarray, dens: np.ndarray, kappa: np.ndarray) -> None:
    np.savez_compressed(path, corners=corners, dens=dens, kappa=kappa)


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        corners = np.asarray(data["corners"], dtype=np.float64)
        dens = np.asarray(data["dens"], dtype=np.float64)
        if "kappa" in data:
            kappa = np.asarray(data["kappa"], dtype=np.float64)
        else:
            kappa = np.zeros(corners.shape[0], dtype=np.float64)
    return _normalize(corners, dens, kappa)


def _save_vtu(path: Path, corners: np.ndarray, dens: np.ndarray, kappa: np.ndarray) -> None:
    n = corners.shape[0]
    # Duplicate vertices (8 per cell) — simple, exact round-trip
    points = corners.reshape(n * 8, 3)
    # Connectivity in VTK order
    conn = (np.arange(n * 8, dtype=np.int64).reshape(n, 8))[:, _GRAFEN_TO_VTK]
    offsets = np.arange(8, n * 8 + 1, 8, dtype=np.int64)
    types = np.full(n, _VTK_HEXAHEDRON, dtype=np.uint8)

    def fmt(arr: np.ndarray) -> str:
        flat = np.asarray(arr).ravel()
        return " ".join(f"{x:.16g}" for x in flat)

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{n * 8}" NumberOfCells="{n}">',
        "      <Points>",
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
        f"          {fmt(points)}",
        "        </DataArray>",
        "      </Points>",
        "      <Cells>",
        '        <DataArray type="Int64" Name="connectivity" format="ascii">',
        f"          {fmt(conn)}",
        "        </DataArray>",
        '        <DataArray type="Int64" Name="offsets" format="ascii">',
        f"          {fmt(offsets)}",
        "        </DataArray>",
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        f"          {fmt(types)}",
        "        </DataArray>",
        "      </Cells>",
        '      <CellData Scalars="kappa" Vectors="I">',
        '        <DataArray type="Float64" Name="I" NumberOfComponents="3" format="ascii">',
        f"          {fmt(dens)}",
        "        </DataArray>",
        '        <DataArray type="Float64" Name="kappa" format="ascii">',
        f"          {fmt(kappa)}",
        "        </DataArray>",
        "      </CellData>",
        "    </Piece>",
        "  </UnstructuredGrid>",
        "</VTKFile>",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _text_array(elem: ET.Element) -> np.ndarray:
    return np.fromstring(elem.text or "", sep=" ", dtype=np.float64)


def _find_data_array(parent: ET.Element, name: str | None = None) -> ET.Element:
    for da in parent.findall("DataArray"):
        if name is None or da.get("Name") == name:
            return da
    raise ValueError(f"VTU: DataArray {name!r} not found under {parent.tag}")


def _load_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError(f"VTU: no UnstructuredGrid/Piece in {path}")

    n_cells = int(piece.get("NumberOfCells", "0"))
    points_el = piece.find("Points")
    cells_el = piece.find("Cells")
    cell_data = piece.find("CellData")
    if points_el is None or cells_el is None:
        raise ValueError("VTU: missing Points or Cells")

    points = _text_array(_find_data_array(points_el)).reshape(-1, 3)
    conn = _text_array(_find_data_array(cells_el, "connectivity")).astype(np.int64)
    types = _text_array(_find_data_array(cells_el, "types")).astype(np.int64)
    if n_cells == 0:
        n_cells = len(types)
    if not np.all(types == _VTK_HEXAHEDRON):
        raise ValueError("VTU: only VTK_HEXAHEDRON (type 12) cells are supported")
    if conn.size != n_cells * 8:
        raise ValueError(f"VTU: expected {n_cells * 8} connectivity ids, got {conn.size}")

    conn = conn.reshape(n_cells, 8)
    # VTK local order -> GRAFEN corner slots
    corners = points[conn][:, _VTK_TO_GRAFEN, :]

    if cell_data is None:
        dens = np.zeros((n_cells, 3), dtype=np.float64)
        kappa = np.zeros(n_cells, dtype=np.float64)
    else:
        try:
            dens = _text_array(_find_data_array(cell_data, "I")).reshape(n_cells, 3)
        except ValueError:
            dens = np.zeros((n_cells, 3), dtype=np.float64)
        try:
            kappa = _text_array(_find_data_array(cell_data, "kappa")).reshape(n_cells)
        except ValueError:
            kappa = np.zeros(n_cells, dtype=np.float64)

    return _normalize(corners, dens, kappa)
