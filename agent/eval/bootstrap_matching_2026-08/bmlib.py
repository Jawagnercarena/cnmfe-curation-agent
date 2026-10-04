"""
bmlib.py — shared helpers for the bootstrap-matching orientation investigation.

Everything here is READ-ONLY with respect to D:\\Julian_CNMFe. Analysis outputs
belong in this eval directory, never in session dirs.

Pixel-order conventions
-----------------------
* MATLAB `neuron.A` is (d1*d2, N) with column-major linear pixel index
  p = row + col*d1  ("F-order rows").
* `spatial_footprints.mat` is written per-plane via
  `reshape(neuron.A(:,n), d1, d2)`, so scipy.io.loadmat returns a logically
  correct (N, d1, d2) = [n, row, col] stack ("image stack").
* numpy C-order flattening of an image stack gives p = row*d2 + col, which is
  NOT the same as MATLAB's linearization. Mixing the two compares one side
  against the transposed image of the other — the production bug.

Use `stack_to_F(fp3d)` / `Fcols_to_stack(A, d1, d2)` to convert explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

DATA_ROOT = Path(r"D:\Julian_CNMFe")
AREAS = ("BLA", "vCA1")

# The four surviving candidate-footprint caches (agent sessions, BLA).
# NEVER modify or delete these.
SANDBOX_SESSIONS = [
    DATA_ROOT / "BLA" / "2tones" / "AVG5x-TSeries-093025-bla21-313um-38z-000",
    DATA_ROOT / "BLA" / "2tones" / "AVG5x-TSeries-100125-bla12-639um-23z-000",
    DATA_ROOT / "BLA" / "4odorDO" / "AVG5x-TSeries-02092026-bla12-681um-22z-000",
    DATA_ROOT / "BLA" / "Valence" / "AVG5x-TSeries-121225-bla12-652um-23z-000",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stack(mat_path: Path) -> np.ndarray:
    """Load a spatial_footprints.mat as a float32 (N, d1, d2) image stack.

    Handles both classic .mat and v7.3/HDF5. h5py exposes a MATLAB (N, d1, d2)
    array with reversed dims (d2, d1, N), so it is transposed back explicitly.
    """
    try:
        data = sio.loadmat(str(mat_path))
        return data["spatial_footprints"].astype(np.float32)
    except NotImplementedError:
        import h5py
        with h5py.File(str(mat_path), "r") as f:
            arr = f["spatial_footprints"][()]          # (d2, d1, N)
        return np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.float32)


def load_curated_stack(session_dir: Path) -> np.ndarray:
    return load_stack(session_dir / "spatial_footprints.mat")


def load_sandbox_candidates(session_dir: Path) -> np.ndarray:
    return load_stack(session_dir / "_bootstrap_validate" / "spatial_footprints.mat")


# ---------------------------------------------------------------------------
# Orientation conversion
# ---------------------------------------------------------------------------

def stack_to_C(fp3d: np.ndarray) -> np.ndarray:
    """Image stack -> (N, pixels) rows in numpy C-order (row*d2 + col)."""
    return fp3d.reshape(fp3d.shape[0], -1)


def stack_to_F(fp3d: np.ndarray) -> np.ndarray:
    """Image stack -> (N, pixels) rows in MATLAB F-order (row + col*d1)."""
    return fp3d.transpose(0, 2, 1).reshape(fp3d.shape[0], -1)


def Fcols_to_stack(A: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """MATLAB-style (pixels, N) F-order columns -> (N, d1, d2) image stack."""
    return A.T.reshape(-1, d2, d1).transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def cosine_matrix(rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two (N, pixels) row matrices -> (Na, Nb)."""
    na = np.linalg.norm(rows_a, axis=1, keepdims=True) + 1e-12
    nb = np.linalg.norm(rows_b, axis=1, keepdims=True) + 1e-12
    return (rows_a / na) @ (rows_b / nb).T


def hungarian_pairs(corr: np.ndarray):
    """linear_sum_assignment on -corr; returns (rows, cols, sims) sorted desc."""
    ri, ci = linear_sum_assignment(-corr)
    sims = corr[ri, ci]
    order = np.argsort(-sims)
    return ri[order], ci[order], sims[order]


def centroids(stack: np.ndarray) -> np.ndarray:
    """(N, 2) [row, col] intensity-weighted centroids of an image stack."""
    n = stack.shape[0]
    out = np.zeros((n, 2))
    rr = np.arange(stack.shape[1])
    cc = np.arange(stack.shape[2])
    for i in range(n):
        m = stack[i]
        tot = m.sum() + 1e-12
        out[i, 0] = (m.sum(axis=1) * rr).sum() / tot
        out[i, 1] = (m.sum(axis=0) * cc).sum() / tot
    return out
