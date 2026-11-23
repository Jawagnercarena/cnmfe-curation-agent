"""Labelled candidates of the 12 threshold-0 pilot sessions (read-only).

Candidate footprints: <session>/.pre_ingest/spatial_footprints.mat (the headless output the
reviewer saw, written before ingest replaced the live file).  Labels: <session>/labels.mat
('labels', 1 = kept).  On these pilots the review threshold was 0, so labels cover every
candidate; candidates() refuses the session otherwise, and checks the join by matching the
reviewer's final kept footprints (live spatial_footprints.mat) back onto the candidates.
"""
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

PILOT_DATES = ("082626", "082726", "082826")
FINAL_RE = re.compile(r"Final parameters: \{.*'min_corr': ([0-9.]+), 'min_pnr': ([0-9.]+)")


def load_sf(p):
    try:
        a = loadmat(str(p))["spatial_footprints"]
    except NotImplementedError:
        with h5py.File(str(p), "r") as f:
            a = np.array(f["spatial_footprints"]).transpose(2, 1, 0)
    return np.asarray(a, dtype=np.float32)


def run_params(sd):
    mc = mp = None
    for ln in open(sd / "agent_run.log", encoding="utf-8", errors="replace"):
        m = FINAL_RE.search(ln)
        if m:
            mc, mp = float(m.group(1)), float(m.group(2))
    return mc, mp


def pilots():
    return [s for s in rc.sessions() if s[0] == "6odor" and s[1][:6] in PILOT_DATES]


def candidates(sd):
    """dict: A (N, 256, 256) 2 x 2 mean of the footprints; core (N, 256, 256) bool, a >= half
    its max; labels (N,) 0/1; peak (N, 2) row / col of the footprint maximum at 256;
    join (min cosine, exact set match) of the label join check."""
    pre = load_sf(sd / ".pre_ingest" / "spatial_footprints.mat")
    z = loadmat(str(sd / "labels.mat"))
    lab = np.asarray(z["labels"]).ravel().astype(int)
    if len(lab) != pre.shape[0]:
        raise RuntimeError("%s: %d labels for %d candidates" % (sd.name, len(lab), pre.shape[0]))
    fin = load_sf(sd / "spatial_footprints.mat")
    P = pre.reshape(pre.shape[0], -1)
    F = fin.reshape(fin.shape[0], -1)
    S = (F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)) @ (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)).T
    best = S.argmax(axis=1)
    join_ok = sorted(set(best.tolist())) == np.nonzero(lab == 1)[0].tolist() and len(set(best.tolist())) == fin.shape[0]
    A = pre.reshape(pre.shape[0], 256, 2, 256, 2).mean(axis=(2, 4))
    mx = A.reshape(A.shape[0], -1).max(axis=1)
    core = A >= 0.5 * mx[:, None, None]
    core &= mx[:, None, None] > 0
    flat = A.reshape(A.shape[0], -1).argmax(axis=1)
    peak = np.stack([flat // 256, flat % 256], axis=1)
    return dict(A=A, core=core, labels=lab, peak=peak, join_min_cos=float(S.max(axis=1).min()), join_ok=bool(join_ok))


def ceiling(cn, pnr, p):
    """Per pixel: the highest min_corr at which the pixel still seeds given min_pnr p, i.e.
    min(own Cn, 5 x 5 median of Cn) with Cn counted as 0 wherever PNR < p (the 13th largest of
    the 25 window values is their median; zero padded like medfilt2)."""
    c = np.nan_to_num(np.asarray(cn, float)).copy()
    c[np.nan_to_num(np.asarray(pnr, float)) < p] = 0
    return np.minimum(c, ndi.median_filter(c, size=rc.MED, mode="constant", cval=0.0))


def cell_ceiling(core, cn, pnr, p, shift=(0, 0)):
    """Per cell: max over its core of ceiling(); the highest min_corr at which the cell can
    seed from inside its own core.  shift moves the cores by (rows, cols) first."""
    ce = ceiling(cn, pnr, p)
    if shift != (0, 0):
        core = np.roll(core, shift, axis=(1, 2))
    out = np.zeros(core.shape[0])
    pk = np.zeros(core.shape[0])
    c0 = np.nan_to_num(np.asarray(cn, float))
    for i in range(core.shape[0]):
        m = core[i]
        if m.any():
            out[i] = ce[m].max()
            pk[i] = c0[m].max()
    return out, pk
