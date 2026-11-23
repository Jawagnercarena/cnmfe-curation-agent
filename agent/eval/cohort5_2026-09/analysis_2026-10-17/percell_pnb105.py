"""Read-only.  For the pnb105 day-0 cells inside the aligned day-1 field: soma contrast
on each day's mean image (disk minus annulus over annulus), per-cell Cn drop, split by
seeded / not seeded on day 1.  Optical (haze, focus) vs activity: if contrast holds but
Cn drops, the cells are visible but less active; if contrast drops too, focus/haze."""
import sys, os, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); LOCAL = DP / "vCA1" / "6odorDualDiffRew"; NEW = ".rerun_mincorr070_2026-10-06"
HERE = os.path.dirname(os.path.abspath(__file__))
D0 = "AVG5x-TSeries-082626-pnb105-397um-35z-000"; D1 = "AVG5x-TSeries-082726-pnb105-391um-35z-000"; DR, DC = 73, 2
means = np.load(os.path.join(HERE, "pnb105_means.npy")); img0, img1 = means
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float); return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
def cents(A): return np.array([ndi.center_of_mass(a > 0.5 * a.max()) for a in A])
def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return (x * x + y * y <= r * r).astype(float)
K_IN = disk(5); K_IN /= K_IN.sum(); K_OUT = disk(14) - np.pad(disk(9), 5); K_OUT /= K_OUT.sum()
def contrast(img):
    inner = ndi.maximum_filter(ndi.convolve(img, K_IN, mode="nearest"), 5); outer = ndi.convolve(img, K_OUT, mode="nearest"); return (inner - outer) / (outer + 1e-6)
con0, con1 = contrast(img0), contrast(img1)
A0 = foot(LOCAL / D0 / NEW / "spatial_footprints.mat"); A1 = foot(LOCAL / D1 / NEW / "spatial_footprints.mat")
Cn0 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D0 / "Cn.mat"))["Cn"], dtype=float)); Cn1 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D1 / "Cn.mat"))["Cn"], dtype=float))
m0 = ndi.maximum_filter(ndi.uniform_filter(Cn0, 3), 9); m1 = ndi.maximum_filter(ndi.uniform_filter(Cn1, 3), 9)
c0 = cents(A0); c0in1 = c0 - np.array([DR, DC]); c1 = cents(A1); BD = 24
inside = (c0in1[:, 0] >= BD) & (c0in1[:, 0] < 512 - BD) & (c0in1[:, 1] >= BD) & (c0in1[:, 1] < 512 - BD)
dist = np.sqrt(((c0in1[:, None, :] - c1[None, :, :]) ** 2).sum(-1)); matched = dist.min(1) <= 12
rng = np.random.default_rng(0); null1 = con1[rng.integers(BD, 512 - BD, 3000), rng.integers(BD, 512 - BD, 3000)]; null0 = con0[rng.integers(BD, 512 - BD, 3000), rng.integers(BD, 512 - BD, 3000)]
t0, t1 = np.percentile(null0, 90), np.percentile(null1, 90)
def at(img, pts): return np.array([img[int(np.clip(r, 0, 511)), int(np.clip(c, 0, 511))] for r, c in pts])
for lab, sel in (("seeded on day 1", inside & matched), ("NOT seeded on day 1", inside & ~matched)):
    k0 = at(con0, c0[sel]); k1 = at(con1, c0in1[sel]); q0 = at(m0, c0[sel]); q1 = at(m1, c0in1[sel])
    print("%s (%d): soma contrast day0 median %.3f -> day1 %.3f (soma-like above the 90th pct of random points: day0 %.0f%%, day1 %.0f%%); Cn day0 %.2f -> day1 %.2f, per-cell drop median %.2f, IQR %.2f-%.2f" % (
        lab, sel.sum(), np.median(k0), np.median(k1), 100 * (k0 > t0).mean(), 100 * (k1 > t1).mean(), np.median(q0), np.median(q1), np.median(q0 - q1), *np.percentile(q0 - q1, [25, 75])))
print("random interior points: contrast 90th pct day0 %.3f, day1 %.3f; median day0 %.3f, day1 %.3f" % (t0, t1, np.median(null0), np.median(null1)))
# global image statistics: haze = ratio of high-pass energy to low-pass energy
for lab, img in (("day 0", img0), ("day 1", img1)):
    hp = img - ndi.gaussian_filter(img, 12); print("%s: mean-image high-pass std / image std = %.3f" % (lab, hp[BD:-BD, BD:-BD].std() / img[BD:-BD, BD:-BD].std()))
