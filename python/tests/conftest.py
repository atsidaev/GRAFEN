"""Shared helpers for numerical vs analytic comparisons."""

from __future__ import annotations

import numpy as np


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(a) ** 2, axis=-1))))


def relative_rms(num: np.ndarray, ref: np.ndarray) -> float:
    """RMS(|num-ref|) / RMS(|ref|)."""
    ref = np.asarray(ref, dtype=float)
    num = np.asarray(num, dtype=float)
    denom = rms(ref)
    if denom == 0.0:
        return rms(num - ref)
    return rms(num - ref) / denom


def mean_magnetization(i: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(i), axis=0)
