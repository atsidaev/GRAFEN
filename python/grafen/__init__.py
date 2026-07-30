"""GRAFEN: forward magnetic modelling with self-demagnetization (Python port)."""

from .analytic import (
    cuboid_field_uniform,
    ellipsoid_magnetization,
    sphere_field_exterior,
    sphere_magnetization,
)
from .demag import solve_demagnetization
from .field import field_at_points, field_from_hexahedra, hexahedra_to_triangles
from .mesh import cube_mesh, ellipsoid_mesh, merge_meshes, sphere_mesh, translate_mesh

__all__ = [
    "cuboid_field_uniform",
    "ellipsoid_magnetization",
    "sphere_field_exterior",
    "sphere_magnetization",
    "solve_demagnetization",
    "field_at_points",
    "field_from_hexahedra",
    "hexahedra_to_triangles",
    "cube_mesh",
    "ellipsoid_mesh",
    "merge_meshes",
    "sphere_mesh",
    "translate_mesh",
]

__version__ = "0.1.0"
