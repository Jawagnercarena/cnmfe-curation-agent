"""
band_scores.py -- for one or more cohort-5 candidate sets, the candidates by Cn-peak
band (3x3-mean Cn over the footprint core A > 0.5 max, the Analysis A definition)
with the deployed vCA1 model's score distribution and the curator's auto-reject flag.

Use: band_scores.py NAME[:SUBDIR] ...   (SUBDIR = an archived set such as
.rerun_mincorr070_2026-10-06; default = the live session folder).  Cn.mat and
pnr.mat are taken from the set folder when present, else from the session folder;
a 256-res Cn is kron-upsampled to 512.  Read-only.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[2]
sys.path.insert(0, str(AGENT))
import local_config  # noqa: E402

TASK = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
MODEL = AGENT / "model" / "vCA1" / "classifier.joblib"
BANDS = [(-1.0, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 9.0)]


def label(lo, hi):
    return "< 0.60" if lo < 0 else (">= 0.80" if hi > 1 else f"{lo:.2f}-{hi:.2f}")


def load_set(d, sd):
    A = np.asarray(loadmat(str(d / "spatial_footprints.mat"))["spatial_footprints"], dtype=float)
    z = np.load(str(d / "candidate_features.npz"), allow_pickle=True)
    X = np.asarray(z["feature_matrix"], dtype=float)
    n = A.shape[0]
    raw = np.asarray(z["auto_rejected"]).ravel()
    if raw.size == n and raw.dtype == bool:
        ar = raw
    else:
        ar = np.zeros(n, dtype=bool)
        if raw.size:
            ar[raw.astype(int)] = True
    cn_path = d / "Cn.mat" if (d / "Cn.mat").exists() else sd / "Cn.mat"
    Cn = np.nan_to_num(np.asarray(loadmat(str(cn_path))["Cn"], dtype=float))
    if Cn.shape[0] * 2 == A.shape[1]:
        Cn = np.kron(Cn, np.ones((2, 2)))
    assert Cn.shape == A.shape[1:], (Cn.shape, A.shape)
    return A, X, ar, Cn


def cn_peak(A, Cn):
    sm = uniform_filter(Cn, 3)
    out = np.full(A.shape[0], np.nan)
    for i, a in enumerate(A):
        m = a.max()
        if m > 0:
            out[i] = sm[a > 0.5 * m].max()
    return out


def main(args):
    o = joblib.load(str(MODEL))
    thr = float(o["reject_threshold"])
    print(f"model {MODEL.name}: n_sessions {o.get('n_sessions')}, reject_threshold {thr}")
    print("set | N | auto-rej | >=0.5 | 0.2-0.5 | 0.01-0.2 | <0.01 || per band: n (auto-rej / >=0.5)")
    for arg in args:
        name, _, sub = arg.partition(":")
        sd = TASK / name
        d = sd / sub if sub else sd
        if not (d / "spatial_footprints.mat").exists():
            print(f"{arg} | missing spatial_footprints.mat in {d}")
            continue
        A, X, ar, Cn = load_set(d, sd)
        s = o["clf"].predict_proba(o["scaler"].transform(X))[:, 1]
        cn = cn_peak(A, Cn)
        cells = []
        for lo, hi in BANDS:
            m = (cn >= lo) & (cn < hi)
            cells.append(f"{label(lo, hi)}: {m.sum()} ({ar[m].sum()} / {(s[m] >= 0.5).sum()})")
        short = name.replace("AVG5x-TSeries-", "") + (f" [{sub}]" if sub else " [live]")
        print(f"{short} | {len(s)} | {ar.sum()} | {(s >= 0.5).sum()} | {((s >= 0.2) & (s < 0.5)).sum()} | "
              f"{((s >= thr) & (s < 0.2)).sum()} | {(s < thr).sum()} || " + "; ".join(cells)
              + f" | Cn peak min {np.nanmin(cn):.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
