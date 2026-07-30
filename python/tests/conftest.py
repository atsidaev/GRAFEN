"""Shared helpers for numerical vs analytic comparisons."""

from __future__ import annotations

import numpy as np


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(a) ** 2, axis=-1))))


def relative_rms(num: np.ndarray, ref: np.ndarray) -> float:
    """RMS(|num-ref|) / RMS(|ref|)."""
    ref = np.asarray(ref, dtype=float)
    num = np.asarray(num, dtype=float)
    err = rms(num - ref)
    denom = rms(ref)
    if denom == 0.0:
        return err
    return err / denom


def check_relative_rms(
    num: np.ndarray,
    ref: np.ndarray,
    tol: float,
    label: str = "",
) -> float:
    """Print absolute and relative RMS, then assert relative RMS < tol."""
    ref = np.asarray(ref, dtype=float)
    num = np.asarray(num, dtype=float)
    abs_err = rms(num - ref)
    denom = rms(ref)
    rel = abs_err if denom == 0.0 else abs_err / denom
    prefix = f"{label}: " if label else ""
    print(f"{prefix}RMS={abs_err:.6e}  relative_RMS={rel:.6e}  (tol={tol:g})")
    assert rel < tol, f"{prefix}relative RMS {rel:.6e} >= {tol:g}"
    return rel


def mean_magnetization(i: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(i), axis=0)
