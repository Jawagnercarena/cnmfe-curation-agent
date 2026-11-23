"""Read-only.  With the day-1 field shifted (73, 2) px relative to day 0: which day-0
cells lie inside the day-1 field, which of those day 1 seeded, and what the day-1 Cn
is at the ones it did not.  Per-cell Cn day 0 -> day 1 for the matched pairs."""
import sys, os, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); EX = Path(local_config.EXCHANGE_ROOT)
LOCAL = DP / "vCA1" / "6odorDualDiffRew"; NEW = ".rerun_mincorr070_2026-10-06"; INB = EX / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
D0 = "AVG5x-TSeries-082626-pnb105-397um-35z-000"; D1 = "AVG5x-TSeries-082726-pnb105-391um-35z-000"
DR, DC = 73, 2            # day-1 content sits 73 rows lower than the same content on day 0: day0 (r, c) -> day1 (r - 73, c - 2)
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float); return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
def cents(A): return np.array([ndi.center_of_mass(a > 0.5 * a.max()) for a in A])
A0 = foot(LOCAL / D0 / NEW / "spatial_footprints.mat"); A1 = foot(LOCAL / D1 / NEW / "spatial_footprints.mat")
L1 = np.asarray(loadmat(str(INB / D1 / "labels.mat"))["labels"]).ravel(); keep1 = np.flatnonzero(L1 == 1)
Cn0 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D0 / "Cn.mat"))["Cn"], dtype=float)); Cn1 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D1 / "Cn.mat"))["Cn"], dtype=float))
m0 = ndi.maximum_filter(ndi.uniform_filter(Cn0, 3), 9); m1 = ndi.maximum_filter(ndi.uniform_filter(Cn1, 3), 9)
c0 = cents(A0); c0in1 = c0 - np.array([DR, DC]); c1 = cents(A1)
BD = 24
inside = (c0in1[:, 0] >= BD) & (c0in1[:, 0] < 512 - BD) & (c0in1[:, 1] >= BD) & (c0in1[:, 1] < 512 - BD)
dist = np.sqrt(((c0in1[:, None, :] - c1[None, :, :]) ** 2).sum(-1)); R = 12
matched0 = dist.min(1) <= R
cn1_at0 = np.array([m1[int(np.clip(r, 0, 511)), int(np.clip(c, 0, 511))] for r, c in c0in1])
cn0_at0 = np.array([m0[int(r), int(c)] for r, c in c0])
print("day-0 candidates %d: inside the day-1 field (border %d) %d, outside %d" % (len(A0), BD, inside.sum(), (~inside).sum()))
ins = inside; um = ins & ~matched0
print("inside: seeded on day 1 (counterpart within %d px) %d; NOT seeded %d" % (R, (ins & matched0).sum(), um.sum()))
print("NOT seeded on day 1 (%d): day-1 Cn at their position >= 0.70: %d | 0.60-0.70: %d | 0.50-0.60: %d | < 0.50: %d  (median %.2f); their day-0 Cn median %.2f" % (
    um.sum(), (cn1_at0[um] >= .7).sum(), ((cn1_at0[um] >= .6) & (cn1_at0[um] < .7)).sum(), ((cn1_at0[um] >= .5) & (cn1_at0[um] < .6)).sum(), (cn1_at0[um] < .5).sum(), np.median(cn1_at0[um]), np.median(cn0_at0[um])))
# strongest day-0 cells (Cn >= 0.90 on day 0) inside the day-1 field: how many did day 1 seed?
strong = ins & (cn0_at0 >= 0.90)
print("day-0 cells with day-0 Cn >= 0.90 inside the day-1 field: %d; seeded on day 1: %d; day-1 Cn at the unseeded ones: median %.2f" % (
    strong.sum(), (strong & matched0).sum(), np.median(cn1_at0[strong & ~matched0]) if (strong & ~matched0).any() else -1))
# matched pairs: per-cell Cn drop
pairs = np.flatnonzero(ins & matched0); j = dist[pairs].argmin(1)
cn1_pair = np.array([m1[int(c1[k, 0]), int(c1[k, 1])] for k in j]); cn0_pair = cn0_at0[pairs]
print("matched pairs %d: day-0 Cn median %.2f -> day-1 Cn median %.2f; median per-cell drop %.2f" % (len(pairs), np.median(cn0_pair), np.median(cn1_pair), np.median(cn0_pair - cn1_pair)))
kept_pair = np.isin(j, keep1)
print("  of which Taylor kept the day-1 counterpart: %d (their day-1 Cn median %.2f); deleted: %d (day-1 Cn median %.2f)" % (
    kept_pair.sum(), np.median(cn1_pair[kept_pair]), (~kept_pair).sum(), np.median(cn1_pair[~kept_pair]) if (~kept_pair).any() else -1))
# figure: day-1 Cn with day-0-only cells coloured by day-1 Cn band, day-1 candidates kept/deleted
sm1 = ndi.uniform_filter(Cn1, 3)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(sm1, cmap="gray", vmin=0.3, vmax=1.0); ax[0].set_title("day 1 Cn; green = kept (24), red = deleted (55); day-0 cells in this field without a day-1 candidate: o coloured by day-1 Cn", fontsize=9)
for k in range(len(A1)): ax[0].contour(A1[k] > 0.5 * A1[k].max(), levels=[0.5], colors=["lime" if L1[k] == 1 else "red"], linewidths=0.7)
for col, lo, hi, lab in (("yellow", .7, 2, ">=0.70"), ("orange", .6, .7, "0.60-0.70"), ("cyan", .5, .6, "0.50-0.60"), ("magenta", 0, .5, "<0.50")):
    sel = um & (cn1_at0 >= lo) & (cn1_at0 < hi)
    ax[0].plot(c0in1[sel, 1], c0in1[sel, 0], "o", mfc="none", mec=col, ms=9, mew=1.2, label="day-0 cell, no day-1 candidate, day-1 Cn %s (%d)" % (lab, sel.sum()))
ax[0].legend(loc="lower left", fontsize=8)
sm0 = ndi.uniform_filter(Cn0, 3); sh0 = np.roll(np.roll(sm0, -DR, 0), -DC, 1); sh0[512 - DR:, :] = 0
ax[1].imshow(sh0, cmap="gray", vmin=0.3, vmax=1.0); ax[1].set_title("day 0 Cn shifted into day-1 coordinates (bottom %d rows not imaged on day 0); same markers" % DR, fontsize=9)
for col, lo, hi in (("yellow", .7, 2), ("orange", .6, .7), ("cyan", .5, .6), ("magenta", 0, .5)):
    sel = um & (cn1_at0 >= lo) & (cn1_at0 < hi); ax[1].plot(c0in1[sel, 1], c0in1[sel, 0], "o", mfc="none", mec=col, ms=9, mew=1.2)
for k in keep1: ax[1].contour(A1[k] > 0.5 * A1[k].max(), levels=[0.5], colors=["lime"], linewidths=0.7)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); out = os.path.join(HERE, "fig_pnb105_overlap_aligned.png"); fig.savefig(out, dpi=72); print("figure:", out)
