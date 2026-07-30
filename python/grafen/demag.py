"""Self-demagnetization via conjugate gradients (C++ demagCG / CG.h)."""

from __future__ import annotations

import numpy as np

from .field import field_at_points, mass_centers


def solve_demagnetization(
    corners: np.ndarray,
    kappa: np.ndarray | float,
    i0: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 20,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Solve I = I0 + K H_snd(I)  (Fredholm II / discrete demag system).

    Parameters
    ----------
    corners : (N, 8, 3)
    kappa : float or (N,) susceptibility
    i0 : (N, 3) primary magnetization K*H' (+ remanence)
    tol, max_iter : CG stopping criteria (relative residual)
    x0 : initial guess (defaults to i0)

    Returns
    -------
    I : (N, 3) magnetization including self-demagnetization
    """
    n = corners.shape[0]
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (n,)).astype(float)
    i0 = np.asarray(i0, dtype=float)
    if i0.ndim == 1:
        i0 = np.tile(i0, (n, 1))
    x = np.array(i0 if x0 is None else x0, dtype=float, copy=True)

    centers = mass_centers(corners)

    def apply_a(z: np.ndarray) -> np.ndarray:
        # A z = z - K H_snd(z)
        h = field_at_points(centers, corners, z)
        return z - kappa[:, None] * h

    # CG for A x = b with b = i0 (same as C++ CG.h, vector "dot" = sum of ^)
    r = i0 - apply_a(x)
    zdir = r.copy()
    b_norm2 = _frobenius2(i0)
    if b_norm2 == 0.0:
        return x

    err = np.sqrt(_frobenius2(r) / b_norm2)
    for _ in range(max_iter):
        if err <= tol:
            break
        az = apply_a(zdir)
        r_dot = _frobenius2(r)
        denom = np.sum(az * zdir)
        if abs(denom) < 1e-30:
            break
        alpha = r_dot / denom
        x = x + alpha * zdir
        r = r - alpha * az
        beta = _frobenius2(r) / r_dot
        zdir = r + beta * zdir
        err = np.sqrt(_frobenius2(r) / b_norm2)

    return x


def _frobenius2(a: np.ndarray) -> float:
    return float(np.sum(a * a))
