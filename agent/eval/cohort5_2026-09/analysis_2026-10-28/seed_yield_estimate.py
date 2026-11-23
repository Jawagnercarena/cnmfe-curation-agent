"""
seed_yield_estimate.py -- what a lower min_corr seed would yield on a cohort-5
session, estimated from its full-movie Cn/PNR images.  Read-only.

Per session: Cn percentiles inside a 20-px border, PNR p90, the number of Cn
local maxima (3x3 mean, 37-px window, border excluded) above each threshold that
also pass the session's min_pnr gate, the live candidate count, and the Cn-peak
median of the live candidates (footprint core, A > 0.5 max).  The maxima count is
a proxy for the seeding (CNMF-E seeds on the 256-res Cn.*PNR image after a
median filter), calibrated on the 12 pilots whose true counts at 0.70 / 0.60 are
known.

Usage: python seed_yield_estimate.py [--sessions NAME ...]   (default: every
cohort-5 session folder that has Cn.mat)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from local_config import DATA_PARENT   # noqa: E402

TASK = DATA_PARENT / "vCA1" / "6odorDualDiffRew"
ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
THS = (0.55, 0.60, 0.65, 0.70, 0.80)
BD = 20


def load_img(sd, name, key):
    p = sd / name
    if not p.exists():
        return None
    return np.nan_to_num(np.asarray(loadmat(str(p))[key], dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()
    if a.sessions:
        dirs = [TASK / s for s in a.sessions]
    else:
        dirs = sorted(p for p in TASK.iterdir() if p.is_dir() and any(f"-{an}-" in p.name for an in ANIMALS))
    print("session | N live | seed mc/mp | Cn p50/p85/p90/p95 | PNR p90 | Cn maxima passing PNR gate at " +
          "/".join("%.2f" % t for t in THS) + " | live cand Cn-peak median (min)")
    for sd in dirs:
        Cn = load_img(sd, "Cn.mat", "Cn")
        pnr = load_img(sd, "pnr.mat", "pnr")
        if Cn is None or pnr is None:
            print("%s | no Cn/pnr yet" % sd.name[14:])
            continue
        if pnr.shape != Cn.shape:
            pnr = np.kron(pnr, np.ones((Cn.shape[0] // pnr.shape[0], Cn.shape[1] // pnr.shape[1])))
        log = (sd / "agent_run.log").read_text(encoding="utf-8", errors="replace") if (sd / "agent_run.log").exists() else ""
        m = re.findall(r"Final parameters: \{.*?'min_corr': ([0-9.]+), 'min_pnr': ([0-9.]+)", log)
        mc, mp = (float(m[-1][0]), float(m[-1][1])) if m else (float("nan"), float("nan"))
        crop = Cn[BD:-BD, BD:-BD]
        pct = np.percentile(crop, [50, 85, 90, 95])
        pnr90 = float(np.percentile(pnr[BD:-BD, BD:-BD], 90))
        sm = ndi.uniform_filter(Cn, 3)
        mx = sm == ndi.maximum_filter(sm, size=37)
        mx[:BD, :] = mx[-BD:, :] = mx[:, :BD] = mx[:, -BD:] = False
        gate = pnr > mp if np.isfinite(mp) else np.ones_like(mx)
        counts = [int((mx & gate & (sm > t)).sum()) for t in THS]
        n_live = "?"
        cn_med = ""
        npz = sd / "candidate_features.npz"
        fp = sd / "spatial_footprints.mat"
        if npz.exists():
            z = np.load(str(npz), allow_pickle=True)
            n_live = int(np.asarray(z["n_candidates"]).ravel()[0])
        if fp.exists():
            A = np.asarray(loadmat(str(fp))["spatial_footprints"], dtype=float)
            A = A.reshape(A.shape[0], Cn.shape[0], Cn.shape[1]) if A.ndim == 2 else A
            pk = np.array([sm[x > 0.5 * x.max()].max() for x in A])
            cn_med = "%.2f (%.2f)" % (np.median(pk), pk.min())
        print("%s | %s | %.2f/%.1f | %.2f/%.2f/%.2f/%.2f | %4.1f | %s | %s" % (
            sd.name[14:], str(n_live).rjust(4), mc, mp, *pct, pnr90, " / ".join("%3d" % c for c in counts), cn_med))


if __name__ == "__main__":
    main()
