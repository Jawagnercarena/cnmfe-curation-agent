"""Read-only: count cell-sized blobs in each session's full-run Cn image at fixed
absolute Cn levels, split into seeded (contains a candidate's peak pixel) and
unseeded, and for unseeded blobs above the session's min_corr check whether PNR
was the binding criterion.  Also writes a 3-panel figure (mean raw frame, Cn with
candidate peaks and unseeded blobs, PNR) for selected sessions."""
import os
import re
import csv
import numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
PEAKS = os.path.join(HERE, "review_peaks.csv")
PILOT = {"pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109"}
LEVELS = [0.6, 0.65, 0.7, 0.75, 0.8]
AREA_MIN, AREA_MAX = 60, 2500          # px at 512x512; a 9-px-radius cell is ~250 px
FIG_SESSIONS = ["AVG5x-TSeries-082726-pnb109-405um-25z-000",
                "AVG5x-TSeries-062626-pnb97-630um-24z-000",
                "AVG5x-TSeries-082726-pnb105-391um-35z-000",
                "AVG5x-TSeries-082626-pnb105-397um-35z-000"]
FRAME_BYTES = 512 * 512

peaks = {}
if os.path.exists(PEAKS):
    with open(PEAKS, newline="") as fh:
        for r in csv.DictReader(fh):
            peaks.setdefault(r["session"], []).append((int(r["peak_row"]) - 1, int(r["peak_col"]) - 1))


def mean_raw(sd, name, n=20):
    tif = os.path.join(sd, name + ".tif")
    if not os.path.exists(tif):
        return None
    import tifffile
    with tifffile.TiffFile(tif) as t:
        nf = int(t.imagej_metadata.get("images", 0))
        off = t.pages[0].dataoffsets[0]
    acc = np.zeros((512, 512), np.float64)
    with open(tif, "rb") as fh:
        for i in np.linspace(0, nf - 1, n).astype(int):
            fh.seek(off + int(i) * FRAME_BYTES)
            acc += np.frombuffer(fh.read(FRAME_BYTES), dtype=np.uint8).reshape(512, 512)
    return acc / n


summary = {}
print("session | N cand | min_corr/min_pnr | " + " | ".join("Cn>%.2f blobs (seeded/unseeded)" % l for l in LEVELS)
      + " | unseeded blobs above min_corr: PNR>=min_pnr / PNR<min_pnr")
for d in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, d)
    if not os.path.isdir(sd) or d.startswith("."):
        continue
    need = ["agent_run.log", "Cn.mat", "pnr.mat", "candidate_features.npz"]
    if not all(os.path.exists(os.path.join(sd, f)) for f in need):
        continue
    t = open(os.path.join(sd, "agent_run.log"), encoding="utf-8", errors="replace").read()
    m1 = re.search(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
    mc, mp = float(m1.group(1)), float(m1.group(2))
    animal = re.search(r"-(pnb\d+)-", d).group(1)
    z = np.load(os.path.join(sd, "candidate_features.npz"), allow_pickle=True)
    N = int(np.asarray(z["n_candidates"]).ravel()[0])
    Cn = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "Cn.mat"))["Cn"], dtype=float))
    pnr = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "pnr.mat"))["pnr"], dtype=float))
    if pnr.shape != Cn.shape:
        pnr = np.kron(pnr, np.ones((2, 2)))
    # candidate peak pixels
    if d in peaks:
        pk = peaks[d]
    else:
        A = np.asarray(loadmat(os.path.join(sd, "spatial_footprints.mat"))["spatial_footprints"], dtype=float)
        if A.shape[0] != N:
            print("SKIP", d, "footprints", A.shape[0], "N", N)
            continue
        pk = [tuple(np.unravel_index(np.argmax(A[k]), A[k].shape)) for k in range(N)]
    if len(pk) != N:
        print("SKIP", d, "peaks", len(pk), "N", N)
        continue
    pkmask = np.zeros(Cn.shape, bool)
    for r, c in pk:
        pkmask[r, c] = True
    cells = []
    unseeded_lvl = {}
    for lv in LEVELS:
        lab, nlab = ndi.label(Cn > lv, structure=np.ones((3, 3)))
        if nlab == 0:
            cells.append("0 (0/0)")
            unseeded_lvl[lv] = []
            continue
        areas = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        hit = ndi.maximum(pkmask.astype(int), lab, index=np.arange(1, nlab + 1))
        ok = (areas >= AREA_MIN) & (areas <= AREA_MAX)
        n_all = int(ok.sum())
        n_seed = int((ok & (hit > 0)).sum())
        cells.append("%d (%d/%d)" % (n_all, n_seed, n_all - n_seed))
        ids = np.arange(1, nlab + 1)[ok & (hit == 0)]
        unseeded_lvl[lv] = (lab, ids)
    # unseeded blobs above the session's own min_corr: was PNR the binding criterion?
    lab, nlab = ndi.label(Cn > mc, structure=np.ones((3, 3)))
    pass_pnr = fail_pnr = 0
    if nlab:
        ids = np.arange(1, nlab + 1)
        areas = ndi.sum(np.ones_like(lab), lab, index=ids)
        hit = ndi.maximum(pkmask.astype(int), lab, index=ids)
        pmax = ndi.maximum(pnr, lab, index=ids)
        sel = (areas >= AREA_MIN) & (areas <= AREA_MAX) & (hit == 0)
        pass_pnr = int((sel & (pmax >= mp)).sum())
        fail_pnr = int((sel & (pmax < mp)).sum())
    grp = ("pilot_d1" if d.split("-")[2] == "082626" else "pilot_d2") if animal in PILOT else "ref"
    summary.setdefault(grp, []).append((d, N, cells, pass_pnr, fail_pnr))
    print(" | ".join([d, str(N), "%.2f / %.1f" % (mc, mp)] + cells + ["%d / %d" % (pass_pnr, fail_pnr)]))

    if d in FIG_SESSIONS:
        raw = mean_raw(sd, d)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        if raw is not None:
            ax[0].imshow(raw, cmap="gray", vmin=np.percentile(raw, 1), vmax=np.percentile(raw, 99.5))
        ax[0].set_title("mean of 20 raw frames")
        ax[1].imshow(Cn, cmap="viridis", vmin=0.3, vmax=1.0)
        rr = [p[0] for p in pk]
        cc = [p[1] for p in pk]
        ax[1].plot(cc, rr, "r+", ms=7, mew=1.2, label="candidate peaks (%d)" % N)
        lv = 0.7
        lab2, ids2 = unseeded_lvl[lv]
        if len(ids2):
            cen = ndi.center_of_mass(np.ones_like(lab2), lab2, index=ids2)
            ax[1].plot([c for _, c in cen], [r for r, _ in cen], "wo", mfc="none", ms=9, mew=1.2,
                       label="unseeded Cn>%.2f blobs (%d)" % (lv, len(ids2)))
        ax[1].legend(loc="lower left", fontsize=8)
        ax[1].set_title("Cn (full run), min_corr=%.2f" % mc)
        ax[2].imshow(pnr, cmap="magma", vmin=0, vmax=30)
        ax[2].set_title("PNR, min_pnr=%.1f" % mp)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(d)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "fig_%s.png" % d), dpi=80)
        plt.close(fig)

print()
for grp, rows in summary.items():
    print("== %s (%d sessions): median blobs per session at each level, as total (seeded/unseeded)" % (grp, len(rows)))
    for j, lv in enumerate(LEVELS):
        tot = np.median([int(r[2][j].split(" ")[0]) for r in rows])
        se = np.median([int(r[2][j].split("(")[1].split("/")[0]) for r in rows])
        un = np.median([int(r[2][j].split("/")[1].rstrip(")")) for r in rows])
        print("  Cn>%.2f: %.0f (%.0f/%.0f)" % (lv, tot, se, un))
    print("  unseeded blobs above own min_corr with PNR>=min_pnr (median): %.0f ; with PNR<min_pnr: %.0f" % (
        np.median([r[3] for r in rows]), np.median([r[4] for r in rows])))
