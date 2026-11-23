"""Read-only, v2: candidate support normalised PER candidate (v1 used the global max,
which dropped dim candidates and mislabelled their peaks as unseeded).  Also draws
a zoom of the dense central patch of 082726-pnb105 (mean raw image + Cn, kept/deleted)."""
import sys, re, os, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
import tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); EX = Path(local_config.EXCHANGE_ROOT)
LOCAL = DP / "vCA1" / "6odorDualDiffRew"; NEW = ".rerun_mincorr070_2026-10-06"
INB = EX / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
def mean_image(sd, name, nfr=200):
    tif = sd / (name + ".tif")
    with tifffile.TiffFile(str(tif)) as t:
        nf = int(t.imagej_metadata.get("images", 0)); off = t.pages[0].dataoffsets[0]
    acc = np.zeros((512, 512), np.float64)
    with open(tif, "rb") as fh:
        for i in np.linspace(0, nf - 1, nfr).astype(int):
            fh.seek(off + int(i) * 512 * 512); acc += np.frombuffer(fh.read(512 * 512), dtype=np.uint8).reshape(512, 512)
    return acc / nfr
print("session | min_pnr | N cand | Cn maxima>0.70 (21px win) | inside a candidate | outside: total / PNR ok / blob>=150px / Cn>=0.75 among blob-ok")
store = {}
for sd in sorted(p for p in LOCAL.iterdir() if (p / NEW).exists()):
    s = sd.name
    mp = float(re.findall(r"min_pnr=(\d+(?:\.\d+)?)", (sd / "agent_run.log").read_text(encoding="utf-8", errors="replace"))[-1])
    Cn = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float)); sm = ndi.uniform_filter(Cn, 3)
    pnr = np.nan_to_num(np.asarray(loadmat(str(sd / "pnr.mat"))["pnr"], dtype=float))
    if pnr.shape != Cn.shape: pnr = np.kron(pnr, np.ones((2, 2)))
    pnr_loc = ndi.maximum_filter(pnr, size=7)
    A = foot(sd / NEW / "spatial_footprints.mat")
    An = A / (A.reshape(len(A), -1).max(1)[:, None, None] + 1e-12)
    support = ndi.binary_dilation(An.max(0) > 0.3, structure=np.ones((5, 5)))
    mx = (sm == ndi.maximum_filter(sm, size=21)) & (sm > 0.70); mx[:20, :] = mx[-20:, :] = mx[:, :20] = mx[:, -20:] = False
    lab, _ = ndi.label(Cn > 0.70); sizes = ndi.sum(np.ones_like(Cn), lab, index=np.arange(lab.max() + 1))
    rr, cc = np.nonzero(mx); inside = support[rr, cc]
    out = [(sm[r, c], pnr_loc[r, c], sizes[lab[r, c]], r, c) for r, c in zip(rr[~inside], cc[~inside])]
    out = np.array(out) if out else np.zeros((0, 5))
    ok = out[out[:, 1] >= mp] if len(out) else out
    big = ok[ok[:, 2] >= 150] if len(ok) else ok
    hi = big[big[:, 0] >= 0.75] if len(big) else big
    print("%s | %.1f | %d | %d | %d | %d / %d / %d / %d" % (s[14:], mp, len(A), len(rr), int(inside.sum()), len(out), len(ok), len(big), len(hi)))
    store[s] = (A, sm, out, big)
s = [x for x in store if "082726-pnb105" in x][0]; A, sm, out, big = store[s]
print(); print("082726-pnb105: unseeded maxima outside every candidate (Cn | PNR | blob px | row,col):")
for x in out[np.argsort(-out[:, 0])]: print("  %.2f | %5.1f | %5d | %d,%d" % (x[0], x[1], x[2], x[3], x[4]))
L = np.asarray(loadmat(str(INB / s / "labels.mat"))["labels"]).ravel()
img = mean_image(LOCAL / s, s)
# dense patch: find the 160x160 window with the most candidate centroids
cents = np.array([ndi.center_of_mass(a > 0.5 * a.max()) for a in A])
best, bw = None, -1
for r0 in range(0, 352, 16):
    for c0 in range(0, 352, 16):
        n = int(((cents[:, 0] >= r0) & (cents[:, 0] < r0 + 160) & (cents[:, 1] >= c0) & (cents[:, 1] < c0 + 160)).sum())
        if n > bw: bw, best = n, (r0, c0)
r0, c0 = best; sl = (slice(r0, r0 + 160), slice(c0, c0 + 160))
print("densest 160x160 window at row %d col %d holds %d candidates" % (r0, c0, bw))
fig, ax = plt.subplots(1, 3, figsize=(21, 7.5))
ax[0].imshow(img[sl], cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8)); ax[0].set_title("mean of 200 raw frames (dense patch)")
ax[1].imshow(img[sl], cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8)); ax[1].set_title("same + candidates: green kept, red deleted")
ax[2].imshow(sm[sl], cmap="gray", vmin=0.3, vmax=1.0); ax[2].set_title("Cn (0.70 run) + candidates; yellow = Cn max>0.70 outside all")
for k in range(len(A)):
    a = A[k][sl]
    if a.max() < 0.3 * A[k].max(): continue
    for j in (1, 2): ax[j].contour(a > 0.5 * A[k].max(), levels=[0.5], colors=["lime" if L[k] == 1 else "red"], linewidths=0.9)
for x in out:
    if r0 <= x[3] < r0 + 160 and c0 <= x[4] < c0 + 160: ax[2].plot(x[4] - c0, x[3] - r0, "o", mfc="none", mec="yellow", ms=10, mew=1.3)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("%s  (window rows %d-%d, cols %d-%d of 512)" % (s[14:], r0, r0 + 160, c0, c0 + 160)); fig.tight_layout()
outp = os.path.join(HERE, "fig_pnb105_082726_dense_patch.png"); fig.savefig(outp, dpi=80); print("figure:", outp)
