"""GRAFEN: forward magnetic modelling with self-demagnetization (Python port)."""

from .analytic import (
    cuboid_field_uniform,
    ellipsoid_magnetization,
    sphere_field_exterior,
    sphere_magnetization,
)
from .demag import solve_magnetic
from .field import field_at_points, field_from_hexahedra, hexahedra_to_triangles
from .mesh import cube_mesh, ellipsoid_mesh, merge_meshes, sphere_mesh, translate_mesh
from .model_io import load_model, save_model
from .stl_convert import convert_stl_to_vtu, polar_mesh_stl, stl_to_hex_model, voxelize_stl

__all__ = [
    "cuboid_field_uniform",
    "ellipsoid_magnetization",
    "sphere_field_exterior",
    "sphere_magnetization",
    "solve_magnetic",
    "field_at_points",
    "field_from_hexahedra",
    "hexahedra_to_triangles",
    "cube_mesh",
    "ellipsoid_mesh",
    "merge_meshes",
    "sphere_mesh",
    "translate_mesh",
    "load_model",
    "save_model",
    "convert_stl_to_vtu",
    "polar_mesh_stl",
    "stl_to_hex_model",
    "voxelize_stl",
]

__version__ = "0.1.0"
