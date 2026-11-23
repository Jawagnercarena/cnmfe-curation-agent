"""Read-only.  (A) packing of Taylor's kept cells per reviewed session (centroid
nearest-neighbour distance, footprint radius); (B) pnb105 sweep arms (gSig 7 / 11)
vs the 0.70 gSig-9 set: recall of kept cells, unmatched extras; (C) Cn local maxima
vs window size; (D) figure for 082726-pnb105."""
import os, sys, re, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
EX = Path(local_config.EXCHANGE_ROOT); DP = Path(local_config.DATA_PARENT)
TASK = "vCA1/6odorDualDiffRew"; LOCAL = DP / TASK; INB = EX / "inbox" / "Taylor" / TASK
SWEEP = DP / "vCA1" / ".sweep"; NEW = ".rerun_mincorr070_2026-10-06"
HERE = os.path.dirname(os.path.abspath(__file__))
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 and A.shape[1] == 512 * 512 else A
def geom(A):
    """centroid (row, col), equivalent radius of the core (> 0.5 max), area of core"""
    cents, rads = [], []
    for k in range(A.shape[0]):
        a = A[k]; core = a > 0.5 * a.max()
        r, c = np.nonzero(core); w = a[core]
        cents.append((np.average(r, weights=w), np.average(c, weights=w)))
        rads.append(np.sqrt(core.sum() / np.pi))
    return np.array(cents), np.array(rads)
def nn(cents):
    if len(cents) < 2: return np.array([])
    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    return d.min(1)
def cos_match(A, B):
    An = A.reshape(len(A), -1); Bn = B.reshape(len(B), -1)
    An = An / (np.linalg.norm(An, axis=1, keepdims=True) + 1e-12); Bn = Bn / (np.linalg.norm(Bn, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T
sessions = sorted(d.name for d in LOCAL.iterdir() if d.is_dir() and (d / NEW).exists())
print("(A) kept-cell packing per reviewed session (0.70 set, Taylor's labels). NN = centroid distance to nearest other kept cell, px at 512.")
print("session | N | kept | kept radius med (px) | kept NN med / 10th pct (px) | kept pairs NN<18 px (2*gSig) | all-candidate NN med")
packing = {}
for s in sessions:
    lab = INB / s / "labels.mat"
    A = foot(LOCAL / s / NEW / "spatial_footprints.mat")
    cents, rads = geom(A); nn_all = nn(cents)
    if not lab.exists():
        print("%s | %d | - | - | - | - | %.1f" % (s[14:], len(A), np.median(nn_all))); continue
    L = np.asarray(loadmat(str(lab))["labels"]).ravel(); keep = np.flatnonzero(L == 1)
    nk = nn(cents[keep])
    packing[s] = (cents, rads, keep)
    print("%s | %d | %d | %.1f | %.1f / %.1f | %d of %d | %.1f" % (s[14:], len(A), len(keep), np.median(rads[keep]),
          np.median(nk), np.percentile(nk, 10), int((nk < 18).sum()), len(keep), np.median(nn_all)))
print()
print("(B) pnb105 sweep arms vs the 0.70 gSig-9 set (cosine >= 0.45 = same object)")
for s in [x for x in sessions if "pnb105" in x]:
    A9 = foot(LOCAL / s / NEW / "spatial_footprints.mat")
    lab = INB / s / "labels.mat"
    keep = np.flatnonzero(np.asarray(loadmat(str(lab))["labels"]).ravel() == 1) if lab.exists() else None
    for arm in ("gSig07", "gSig11"):
        p = SWEEP / s / arm / "spatial_footprints.mat"
        if not p.exists(): print("  %s %s: no footprints" % (s[14:], arm)); continue
        Aa = foot(p); S = cos_match(Aa, A9)
        extras = np.flatnonzero(S.max(1) < 0.45)
        line = "  %s %s: %d cand; %d match a 0.70 candidate, %d extras (no 0.70 counterpart)" % (s[14:], arm, len(Aa), len(Aa) - len(extras), len(extras))
        if keep is not None:
            rec = int((S[:, keep].max(0) >= 0.45).sum())
            line += "; recovers %d of Taylor's %d kept cells" % (rec, len(keep))
        print(line)
        if "082726" in s and arm == "gSig07": extras07 = (Aa, extras)
print()
print("(C) Cn local maxima > 0.70 vs window size (px at 512; the seeding window is ~37), border 20")
print("session | w15 | w21 | w27 | w37 | N cand 0.70")
cn_store = {}
for s in sessions:
    Cn = np.nan_to_num(np.asarray(loadmat(str(LOCAL / s / "Cn.mat"))["Cn"], dtype=float)); sm = ndi.uniform_filter(Cn, 3)
    N = foot(LOCAL / s / NEW / "spatial_footprints.mat").shape[0]
    row = []
    for w in (15, 21, 27, 37):
        mx = (sm == ndi.maximum_filter(sm, size=w)) & (sm > 0.70); mx[:20, :] = mx[-20:, :] = mx[:, :20] = mx[:, -20:] = False
        row.append(int(mx.sum()))
    cn_store[s] = sm
    print("%s | %s | %d" % (s[14:], " | ".join(str(x) for x in row), N))
# (D) figure
s = [x for x in sessions if "082726-pnb105" in x][0]
cents, rads, keep = packing[s]; A9 = foot(LOCAL / s / NEW / "spatial_footprints.mat"); sm = cn_store[s]
L = np.asarray(loadmat(str(INB / s / "labels.mat"))["labels"]).ravel()
fig, ax = plt.subplots(1, 2, figsize=(18, 9))
for a in ax:
    a.imshow(sm, cmap="gray", vmin=0.3, vmax=1.0); a.set_xticks([]); a.set_yticks([])
ax[0].set_title("%s: Cn (0.70 run). green = kept (%d), red = deleted (%d)" % (s[14:], len(keep), int((L == 0).sum())), fontsize=10)
for k in range(len(A9)):
    a = A9[k]; ax[0].contour(a > 0.5 * a.max(), levels=[0.5], colors=["lime" if L[k] == 1 else "red"], linewidths=0.8)
mx = (sm == ndi.maximum_filter(sm, size=21)) & (sm > 0.70); mx[:20, :] = mx[-20:, :] = mx[:, :20] = mx[:, -20:] = False
rr, cc = np.nonzero(mx)
near = ndi.binary_dilation(A9.max(0) > 0.3 * A9.max(), structure=np.ones((5, 5)))
un = [(r, c) for r, c in zip(rr, cc) if not near[r, c]]
ax[1].set_title("yellow = Cn maxima > 0.70 (21 px window) outside every candidate (%d); cyan = gSig-7 arm extras (%d)" % (len(un), len(extras07[1])), fontsize=10)
for k in keep:
    a = A9[k]; ax[1].contour(a > 0.5 * a.max(), levels=[0.5], colors=["lime"], linewidths=0.8)
ax[1].plot([c for _, c in un], [r for r, _ in un], "o", mfc="none", mec="yellow", ms=9, mew=1.2)
Aa, ex = extras07
for k in ex:
    a = Aa[k]; ax[1].contour(a > 0.5 * a.max(), levels=[0.5], colors=["cyan"], linewidths=0.9)
fig.tight_layout(); out = os.path.join(HERE, "fig_pnb105_082726_density.png"); fig.savefig(out, dpi=80); print("figure:", out)
