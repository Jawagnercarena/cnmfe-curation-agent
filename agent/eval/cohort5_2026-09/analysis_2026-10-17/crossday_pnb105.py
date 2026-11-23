"""Read-only.  pnb105 day 0 (266 cand) vs day 1 (79 cand, 24 kept), same field (zero
shift): which day-0 candidates have a day-1 counterpart, and what the day-1 Cn image
shows at the positions of the day-0 candidates that have none.  4-panel figure."""
import sys, os, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
import tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); EX = Path(local_config.EXCHANGE_ROOT)
LOCAL = DP / "vCA1" / "6odorDualDiffRew"; NEW = ".rerun_mincorr070_2026-10-06"; INB = EX / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
D0 = "AVG5x-TSeries-082626-pnb105-397um-35z-000"; D1 = "AVG5x-TSeries-082726-pnb105-391um-35z-000"
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
def cents(A): return np.array([ndi.center_of_mass(a > 0.5 * a.max()) for a in A])
means = np.load(os.path.join(HERE, "pnb105_means.npy"))          # day0, day1 mean images (300 frames) from day1_vs_day2.py
A0 = foot(LOCAL / D0 / NEW / "spatial_footprints.mat"); A1 = foot(LOCAL / D1 / NEW / "spatial_footprints.mat")
L1 = np.asarray(loadmat(str(INB / D1 / "labels.mat"))["labels"]).ravel(); keep1 = np.flatnonzero(L1 == 1)
c0, c1 = cents(A0), cents(A1)
Cn0 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D0 / "Cn.mat"))["Cn"], dtype=float)); Cn1 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D1 / "Cn.mat"))["Cn"], dtype=float))
sm0, sm1 = ndi.uniform_filter(Cn0, 3), ndi.uniform_filter(Cn1, 3)
dist = np.sqrt(((c0[:, None, :] - c1[None, :, :]) ** 2).sum(-1))       # day0 x day1
R = 8
d0_has = dist.min(1) <= R; d1_has = dist.min(0) <= R
print("day-0 candidates %d, day-1 candidates %d (kept %d). Match radius %d px (cell core radius ~9)." % (len(A0), len(A1), len(keep1), R))
print("day-1 candidates with a day-0 candidate within %d px: %d of %d; among Taylor's kept: %d of %d" % (R, int(d1_has.sum()), len(A1), int(d1_has[keep1].sum()), len(keep1)))
print("day-0 candidates with a day-1 candidate within %d px: %d of %d" % (R, int(d0_has.sum()), len(A0)))
# what does day 1 show at the day-0-only positions?
cn1_at = np.array([ndi.maximum_filter(sm1, size=9)[int(r), int(c)] for r, c in c0])
cn0_at = np.array([ndi.maximum_filter(sm0, size=9)[int(r), int(c)] for r, c in c0])
only0 = ~d0_has
print("day-0-only candidates (%d): day-1 Cn at their position: median %.2f; n >= 0.70: %d; 0.60-0.70: %d; < 0.60: %d  (their own day-0 Cn median %.2f)" % (
    only0.sum(), np.median(cn1_at[only0]), int((cn1_at[only0] >= 0.70).sum()), int(((cn1_at[only0] >= 0.6) & (cn1_at[only0] < 0.7)).sum()), int((cn1_at[only0] < 0.6).sum()), np.median(cn0_at[only0])))
print("matched day-0 candidates (%d): day-1 Cn at their position median %.2f (day-0 Cn median %.2f)" % (d0_has.sum(), np.median(cn1_at[d0_has]), np.median(cn0_at[d0_has])))
# figure
fig, ax = plt.subplots(2, 2, figsize=(18, 18))
for j, (img, A, ttl) in enumerate(((means[0], A0, "day 0 (082626) mean image + %d candidates" % len(A0)), (means[1], A1, "day 1 (082726) mean image + %d candidates (green = kept 24)" % len(A1)))):
    ax[0, j].imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8)); ax[0, j].set_title(ttl, fontsize=11)
    for k in range(len(A)):
        col = "red" if j == 0 else ("lime" if L1[k] == 1 else "red")
        ax[0, j].contour(A[k] > 0.5 * A[k].max(), levels=[0.5], colors=[col], linewidths=0.6)
for j, (cn, ttl) in enumerate(((sm0, "day 0 Cn image (0.70 run)"), (sm1, "day 1 Cn image (0.70 run); red x = day-0 candidates with no day-1 counterpart"))):
    ax[1, j].imshow(cn, cmap="gray", vmin=0.3, vmax=1.0); ax[1, j].set_title(ttl, fontsize=11)
ax[1, 1].plot(c0[only0, 1], c0[only0, 0], "rx", ms=5, mew=1)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); out = os.path.join(HERE, "fig_pnb105_day0_vs_day1.png"); fig.savefig(out, dpi=70); print("figure:", out)
