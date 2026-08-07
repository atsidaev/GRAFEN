"""ASCII/binary STL surface mesh I/O (triangles only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_stl_ascii(path: str | Path, triangles: np.ndarray, name: str = "grafen") -> None:
    """Write triangles (N, 3, 3) as ASCII STL."""
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"triangles must be (N, 3, 3), got {tris.shape}")
    lines = [f"solid {name}"]
    for tri in tris:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn > 0:
            n = n / nn
        else:
            n = np.zeros(3)
        lines.append(f"  facet normal {n[0]:.8e} {n[1]:.8e} {n[2]:.8e}")
        lines.append("    outer loop")
        for p in tri:
            lines.append(f"      vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def read_stl(path: str | Path) -> np.ndarray:
    """Read STL (ASCII or binary) → triangles (N, 3, 3)."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.lstrip().startswith(b"solid") and b"facet" in raw[:2000]:
        return _read_stl_ascii(raw.decode("ascii", errors="replace"))
    return _read_stl_binary(raw)


def _read_stl_ascii(text: str) -> np.ndarray:
    verts: list[list[float]] = []
    tris: list[np.ndarray] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(verts) == 3:
                tris.append(np.array(verts, dtype=np.float64))
                verts = []
    if not tris:
        raise ValueError("STL ASCII: no triangles found")
    return np.stack(tris, axis=0)


def _read_stl_binary(raw: bytes) -> np.ndarray:
    if len(raw) < 84:
        raise ValueError("STL binary: file too short")
    n = int.from_bytes(raw[80:84], "little", signed=False)
    need = 84 + n * 50
    if len(raw) < need:
        raise ValueError(f"STL binary: expected {need} bytes for {n} triangles")
    tris = np.empty((n, 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        # skip normal (12 bytes), read 9 floats, skip attribute (2 bytes)
        vals = np.frombuffer(raw, dtype="<f4", count=12, offset=off)
        tris[i] = vals[3:12].reshape(3, 3).astype(np.float64)
        off += 50
    return tris


def cube_surface_stl(bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """12 triangles for an axis-aligned box."""
    (x0, x1), (y0, y1), (z0, z1) = bounds
    # 8 corners
    p = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = [
        (0, 1, 2, 3),  # z0
        (4, 7, 6, 5),  # z1
        (0, 4, 5, 1),  # y0
        (2, 6, 7, 3),  # y1
        (0, 3, 7, 4),  # x0
        (1, 5, 6, 2),  # x1
    ]
    tris = []
    for a, b, c, d in faces:
        tris.append([p[a], p[b], p[c]])
        tris.append([p[a], p[c], p[d]])
    return np.array(tris, dtype=np.float64)


def uv_sphere_surface_stl(radius: float, n_lat: int = 32, n_lon: int = 64) -> np.ndarray:
    """UV sphere triangle surface, radius R, centered at origin."""
    return _uv_ellipsoid_surface_stl(radius, radius, n_lat, n_lon)


def _uv_ellipsoid_surface_stl(req: float, rpl: float, n_lat: int, n_lon: int) -> np.ndarray:
    """Ellipsoid of revolution (x,y equatorial req; z polar rpl)."""
    lats = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, n_lon + 1)
    tris: list[np.ndarray] = []
    for i in range(n_lat):
        for j in range(n_lon):
            def pt(la: float, lo: float) -> np.ndarray:
                return np.array(
                    [
                        req * np.cos(la) * np.cos(lo),
                        req * np.cos(la) * np.sin(lo),
                        rpl * np.sin(la),
                    ],
                    dtype=np.float64,
                )

            p00 = pt(lats[i], lons[j])
            p01 = pt(lats[i], lons[j + 1])
            p10 = pt(lats[i + 1], lons[j])
            p11 = pt(lats[i + 1], lons[j + 1])
            # skip degenerate caps
            if np.linalg.norm(p00 - p01) > 1e-14 and np.linalg.norm(p00 - p10) > 1e-14:
                tris.append(np.stack([p00, p10, p11]))
            if np.linalg.norm(p00 - p01) > 1e-14 and np.linalg.norm(p01 - p11) > 1e-14:
                tris.append(np.stack([p00, p11, p01]))
    return np.stack(tris, axis=0)


def ellipsoid_surface_stl(req: float, rpl: float, n_lat: int = 32, n_lon: int = 64) -> np.ndarray:
    return _uv_ellipsoid_surface_stl(req, rpl, n_lat, n_lon)


def stl_bbox(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz_min, xyz_max) of triangle vertices."""
    v = triangles.reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)
