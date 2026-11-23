"""Read-only.  Replay the FIRST seed-selection pass of greedyROI_endoscope on the
saved pre-extraction Cn/PNR images (native 256x256) for several min_corr values,
with the session's own min_pnr, gSiz 36 and bd 10:
    v = Cn.*PNR, zeroed where Cn < min_corr or PNR < min_pnr
    v = medfilt2(v, 9x9, zero padded); v(ind_search) = 0; v(border) = 0
    seeds = v == ordfilt2(v, 19x19 max) & v > 0
Later passes add seeds from the residual and extraction can reject seeds, so
this is a proxy; the column at the estimated min_corr is compared with the
actual candidate count to calibrate it."""
import os
import re
import numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi

ROOT = r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew"
PILOT = {"pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109"}
REF = {"pnb97", "pnb98"}
GSIZ = 36
MED = 2 * round(GSIZ / 4) + 1 - 10   # medfilt2 window = round(gSiz/4) = 9
WIN = 2 * round(GSIZ / 4) + 1        # ordfilt2 window = 19
MCS = [None, 0.80, 0.75, 0.70, 0.65]


def first_pass_seeds(Cn, PNR, mc, mp, bd):
    v = Cn * PNR
    v[(Cn < mc) | (PNR < mp)] = 0
    ind_search = v == 0
    v = ndi.median_filter(v, size=MED, mode="constant", cval=0.0)
    v[ind_search] = 0
    vmax = ndi.maximum_filter(v, size=WIN, mode="constant", cval=0.0)
    v[:bd, :] = 0
    v[-bd:, :] = 0
    v[:, :bd] = 0
    v[:, -(bd + 1):] = 0
    return int(((v == vmax) & (vmax > 0) & (v > 0)).sum())


print("session | N cand | est min_corr / min_pnr / bd | first-pass seeds at: est, 0.80, 0.75, 0.70, 0.65")
groups = {}
for d in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, d)
    m = re.search(r"-(pnb\d+)-", d)
    if not os.path.isdir(sd) or d.startswith(".") or not m:
        continue
    animal = m.group(1)
    if animal not in PILOT | REF:
        continue
    t = open(os.path.join(sd, "agent_run.log"), encoding="utf-8", errors="replace").read()
    m1 = re.search(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
    m2 = re.search(r"'bd': (\d+)", t)
    mc_est, mp, bd = float(m1.group(1)), float(m1.group(2)), int(m2.group(1))
    z = np.load(os.path.join(sd, "candidate_features.npz"), allow_pickle=True)
    N = int(np.asarray(z["n_candidates"]).ravel()[0])
    Cn = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "Cn.mat"))["Cn"], dtype=float))
    PNR = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "pnr.mat"))["pnr"], dtype=float))
    if Cn.shape != PNR.shape:
        Cn = Cn.reshape(PNR.shape[0], 2, PNR.shape[1], 2).mean(axis=(1, 3))
    counts = [first_pass_seeds(Cn, PNR, mc_est if mc is None else mc, mp, bd) for mc in MCS]
    grp = "ref" if animal in REF else ("pilot_d1" if d.split("-")[2] == "082626" else "pilot_d2")
    groups.setdefault(grp, []).append((N, counts))
    print(" | ".join([d, str(N), "%.2f / %.1f / %d" % (mc_est, mp, bd), " ".join(str(c) for c in counts)]))

print()
for grp, rows in groups.items():
    Ns = np.array([r[0] for r in rows])
    C = np.array([r[1] for r in rows])
    print("== %s (%d sessions): median N cand %d; median first-pass seeds at est/0.80/0.75/0.70/0.65 = %s; "
          "median ratio seeds(0.75)/seeds(est) = %.2f, seeds(0.70)/seeds(est) = %.2f" % (
              grp, len(rows), int(np.median(Ns)), " / ".join("%d" % x for x in np.median(C, axis=0)),
              np.median(C[:, 2] / np.maximum(C[:, 0], 1)), np.median(C[:, 3] / np.maximum(C[:, 0], 1))))
