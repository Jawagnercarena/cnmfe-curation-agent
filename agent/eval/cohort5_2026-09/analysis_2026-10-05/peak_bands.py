"""Read-only.  For each pilot session: local maxima of the saved pre-extraction Cn
image (cell-scale window, border excluded), split into seeded (within R px of a
candidate peak) and unseeded.  For every unseeded maximum, by Cn band:
  - does the PNR image pass the session's min_pnr near the peak (the gate the
    seeding applies besides min_corr)?
  - does the mean raw image show a soma there?  contrast = (disk mean - annulus
    mean) / annulus mean on the mean of 200 raw frames; compared with the same
    score at seeded peaks and at random interior points (null).
Writes a figure for one session with unseeded maxima coloured by band."""
import os
import re
import numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = {"pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109"}
EDGES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]
R = 9
BD = 20
NFR = 200
FRAME_BYTES = 512 * 512
FIG = "AVG5x-TSeries-082726-pnb109-405um-25z-000"
rng = np.random.default_rng(0)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(float)


K_IN = disk(5)
K_IN /= K_IN.sum()
K_OUT = disk(14) - np.pad(disk(9), 5)
K_OUT /= K_OUT.sum()


def mean_image(sd, name):
    tif = os.path.join(sd, name + ".tif")
    with tifffile.TiffFile(tif) as t:
        nf = int(t.imagej_metadata.get("images", 0))
        off = t.pages[0].dataoffsets[0]
    acc = np.zeros((512, 512), np.float64)
    with open(tif, "rb") as fh:
        for i in np.linspace(0, nf - 1, NFR).astype(int):
            fh.seek(off + int(i) * FRAME_BYTES)
            acc += np.frombuffer(fh.read(FRAME_BYTES), dtype=np.uint8).reshape(512, 512)
    return acc / NFR


pool = []      # rows: (session, band_idx, seeded, pnr_pass, pnr_val, contrast)
nulls = {}
print("session | N cand | min_corr/min_pnr | seeded maxima | unseeded maxima per band " + str(EDGES))
for d in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, d)
    m = re.search(r"-(pnb\d+)-", d)
    if not os.path.isdir(sd) or d.startswith(".") or not m or m.group(1) not in PILOT:
        continue
    t = open(os.path.join(sd, "agent_run.log"), encoding="utf-8", errors="replace").read()
    m1 = re.search(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
    mc, mp = float(m1.group(1)), float(m1.group(2))
    Cn = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "Cn.mat"))["Cn"], dtype=float))
    pnr = np.nan_to_num(np.asarray(loadmat(os.path.join(sd, "pnr.mat"))["pnr"], dtype=float))
    if pnr.shape != Cn.shape:
        pnr = np.kron(pnr, np.ones((2, 2)))
    A = np.asarray(loadmat(os.path.join(sd, "spatial_footprints.mat"))["spatial_footprints"], dtype=float)
    N = A.shape[0]
    pk = [tuple(np.unravel_index(np.argmax(A[k]), A[k].shape)) for k in range(N)]
    pkmask = np.zeros(Cn.shape, bool)
    for r, c in pk:
        pkmask[r, c] = True
    near = ndi.binary_dilation(pkmask, structure=np.ones((2 * R + 1, 2 * R + 1)))
    sm = ndi.uniform_filter(Cn, 3)
    ismax = (sm == ndi.maximum_filter(sm, size=2 * R + 1)) & (sm > EDGES[0])
    ismax[:BD, :] = ismax[-BD:, :] = ismax[:, :BD] = ismax[:, -BD:] = False
    pnr_loc = ndi.maximum_filter(pnr, size=11)
    img = mean_image(sd, d)
    inner = ndi.maximum_filter(ndi.convolve(img, K_IN, mode="nearest"), size=9)
    outer = ndi.convolve(img, K_OUT, mode="nearest")
    contrast = (inner - outer) / (outer + 1e-6)
    rr, cc = np.nonzero(ismax)
    counts = [0] * (len(EDGES) - 1)
    nseed = 0
    for r, c in zip(rr, cc):
        b = int(np.searchsorted(EDGES, sm[r, c], side="right") - 1)
        s = bool(near[r, c])
        if s:
            nseed += 1
        else:
            counts[b] += 1
        pool.append((d, b, s, pnr_loc[r, c] >= mp, pnr_loc[r, c], contrast[r, c], sm[r, c]))
    rnd = contrast[rng.integers(BD, 512 - BD, 400), rng.integers(BD, 512 - BD, 400)]
    nulls[d] = rnd
    print(" | ".join([d, str(N), "%.2f / %.1f" % (mc, mp), str(nseed), " ".join(str(x) for x in counts)]))
    if d == FIG:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        for a in ax:
            a.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
            a.set_xticks([])
            a.set_yticks([])
        ax[0].set_title("%s: mean of %d raw frames" % (d, NFR), fontsize=9)
        cols = {3: ("c", "0.75-0.80"), 4: ("y", "0.80-0.85"), 5: ("w", ">=0.85"), 2: ("m", "0.70-0.75")}
        ax[1].plot([c for _, c in pk], [r for r, _ in pk], "r+", ms=8, mew=1.2, label="candidate peaks (%d)" % N)
        for b, (col, lab) in cols.items():
            pts = [(r, c) for r, c in zip(rr, cc) if not near[r, c]
                   and int(np.searchsorted(EDGES, sm[r, c], side="right") - 1) in ([5, 6] if b == 5 else [b])]
            if pts:
                ax[1].plot([c for _, c in pts], [r for r, _ in pts], "o", mfc="none", mec=col, ms=10, mew=1.3,
                           label="unseeded Cn %s (%d)" % (lab, len(pts)))
        ax[1].legend(loc="lower left", fontsize=8)
        ax[1].set_title("candidate peaks and unseeded Cn maxima by band (min_corr used: %.2f)" % mc, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "fig_bands_%s.png" % d), dpi=85)
        plt.close(fig)

arr = pool
print()
print("POOLED over the 12 pilot sessions. contrast = (disk - annulus)/annulus on the mean image; "
      "soma-like = contrast above the 90th percentile of random interior points of the same session")
null90 = {d: np.percentile(v, 90) for d, v in nulls.items()}
null_med = np.median(np.concatenate(list(nulls.values())))
print("random points: median contrast %.3f" % null_med)
seeded = [x for x in arr if x[2]]
print("seeded peaks (candidates): n %d, median Cn %.3f, median PNR %.1f, median contrast %.3f, soma-like %.0f%%" % (
    len(seeded), np.median([x[6] for x in seeded]), np.median([x[4] for x in seeded]),
    np.median([x[5] for x in seeded]), 100.0 * np.mean([x[5] > null90[x[0]] for x in seeded])))
print("band | unseeded n | PNR>=min_pnr | PNR>=5 | median PNR | median contrast | soma-like")
for b in range(len(EDGES) - 1):
    rows = [x for x in arr if (not x[2]) and x[1] == b]
    if not rows:
        continue
    print("%.2f-%.2f | %d | %.0f%% | %.0f%% | %.1f | %.3f | %.0f%%" % (
        EDGES[b], min(EDGES[b + 1], 1.0), len(rows), 100.0 * np.mean([x[3] for x in rows]),
        100.0 * np.mean([x[4] >= 5 for x in rows]), np.median([x[4] for x in rows]),
        np.median([x[5] for x in rows]), 100.0 * np.mean([x[5] > null90[x[0]] for x in rows])))
print()
print("Per-session soma-like counts among UNSEEDED maxima with Cn >= 0.75 (and 0.70) and PNR >= min_pnr:")
for d in sorted(nulls):
    for lo in (0.75, 0.70):
        rows = [x for x in arr if x[0] == d and not x[2] and x[6] >= lo and x[3]]
        print("  %s  Cn>=%.2f: %d unseeded pass PNR, %d soma-like" % (
            d, lo, len(rows), sum(1 for x in rows if x[5] > null90[d])))
