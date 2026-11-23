"""Read-only: count local maxima of the full-run Cn image above absolute levels
(cell-scale window, so dense clusters are not merged), split into seeded (within
R px of a candidate peak) and unseeded.  Reference sessions take candidate peaks
from review_peaks.csv (MATLAB export of the review set); pilot sessions from
spatial_footprints.mat (all candidates)."""
import os
import re
import csv
import numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi

ROOT = r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = {"pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109"}
LEVELS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
R = 9            # local-max window half-width and seeded-match radius, px at 512x512
BD = 20          # ignore a border band (bd=10 at 256x256 = 20 px here)

peaks = {}
with open(os.path.join(HERE, "review_peaks.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        peaks.setdefault(r["session"], []).append((int(r["peak_row"]) - 1, int(r["peak_col"]) - 1))

summary = {}
print("session | N cand | min_corr | " + " | ".join("Cn>%.2f maxima (seeded/unseeded)" % l for l in LEVELS)
      + " | unseeded maxima above own min_corr | median Cn of candidate peaks")
for d in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, d)
    if not os.path.isdir(sd) or d.startswith("."):
        continue
    if not all(os.path.exists(os.path.join(sd, f)) for f in ["agent_run.log", "Cn.mat", "candidate_features.npz"]):
        continue
    t = open(os.path.join(sd, "agent_run.log"), encoding="utf-8", errors="replace").read()
    m1 = re.search(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
    mc = float(m1.group(1))
    animal = re.search(r"-(pnb\d+)-", d).group(1)
    N = int(np.asarray(np.load(os.path.join(sd, "candidate_features.npz"), allow_pickle=True)["n_candidates"]).ravel()[0])
    Cn = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "Cn.mat"))["Cn"], dtype=float))
    if d in peaks:
        pk = peaks[d]
    else:
        A = np.asarray(loadmat(os.path.join(sd, "spatial_footprints.mat"))["spatial_footprints"], dtype=float)
        if A.shape[0] != N:
            print("SKIP", d)
            continue
        pk = [tuple(np.unravel_index(np.argmax(A[k]), A[k].shape)) for k in range(N)]
    pkmask = np.zeros(Cn.shape, bool)
    for r, c in pk:
        pkmask[r, c] = True
    near = ndi.binary_dilation(pkmask, structure=np.ones((2 * R + 1, 2 * R + 1)))
    sm = ndi.uniform_filter(Cn, 3)
    ismax = (sm == ndi.maximum_filter(sm, size=2 * R + 1))
    ismax[:BD, :] = ismax[-BD:, :] = ismax[:, :BD] = ismax[:, -BD:] = False
    cells = []
    for lv in LEVELS:
        m = ismax & (sm > lv)
        n_all, n_seed = int(m.sum()), int((m & near).sum())
        cells.append("%d (%d/%d)" % (n_all, n_seed, n_all - n_seed))
    m = ismax & (sm > mc)
    un_mc = int((m & ~near).sum())
    cn_pk = np.median([Cn[r, c] for r, c in pk])
    grp = ("pilot_d1" if d.split("-")[2] == "082626" else "pilot_d2") if animal in PILOT else "ref"
    summary.setdefault(grp, []).append((cells, un_mc, N))
    print(" | ".join([d, str(N), "%.2f" % mc] + cells + [str(un_mc), "%.3f" % cn_pk]))

print()
for grp, rows in summary.items():
    print("== %s (%d sessions): median per session, total (seeded/unseeded); median N cand %d" % (
        grp, len(rows), int(np.median([r[2] for r in rows]))))
    for j, lv in enumerate(LEVELS):
        tot = np.median([int(r[0][j].split(" ")[0]) for r in rows])
        se = np.median([int(r[0][j].split("(")[1].split("/")[0]) for r in rows])
        un = np.median([int(r[0][j].split("/")[1].rstrip(")")) for r in rows])
        print("  Cn>%.2f: %.0f (%.0f/%.0f)" % (lv, tot, se, un))
    print("  unseeded maxima above own min_corr (median): %.0f" % np.median([r[1] for r in rows]))
