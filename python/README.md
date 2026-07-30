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

Tests compare numerical results to closed-form references for a sphere, two spheres, an ellipsoid, a cube, and two cubes.
