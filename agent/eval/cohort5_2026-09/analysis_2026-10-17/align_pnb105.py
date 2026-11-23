"""Read-only.  Cell-level alignment of pnb105 day 0 vs day 1: best integer shift
(+-80 px) between HIGH-PASS filtered mean images and between the Cn images, then the
candidate matching redone at that shift."""
import sys, os, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); EX = Path(local_config.EXCHANGE_ROOT)
LOCAL = DP / "vCA1" / "6odorDualDiffRew"; NEW = ".rerun_mincorr070_2026-10-06"; INB = EX / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__))
D0 = "AVG5x-TSeries-082626-pnb105-397um-35z-000"; D1 = "AVG5x-TSeries-082726-pnb105-391um-35z-000"
means = np.load(os.path.join(HERE, "pnb105_means.npy"))
def hp(img, s=12):
    x = img - ndi.gaussian_filter(img, s); x = ndi.gaussian_filter(x, 1.5); return x
def best_shift(a, b, rng=80):
    """maximise Pearson corr of a and b shifted by (dr, dc), central 320x320 window; brute force on a 4-px grid then refine"""
    m = slice(96, 416)
    def corr(dr, dc):
        b2 = np.roll(np.roll(b, dr, 0), dc, 1); return np.corrcoef(a[m, m].ravel(), b2[m, m].ravel())[0, 1]
    best = (-2, 0, 0)
    for dr in range(-rng, rng + 1, 4):
        for dc in range(-rng, rng + 1, 4):
            c = corr(dr, dc)
            if c > best[0]: best = (c, dr, dc)
    c0, r0, c0_ = best
    for dr in range(r0 - 4, r0 + 5):
        for dc in range(c0_ - 4, c0_ + 5):
            c = corr(dr, dc)
            if c > best[0]: best = (c, dr, dc)
    return best, corr(0, 0)
Cn0 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D0 / "Cn.mat"))["Cn"], dtype=float)); Cn1 = np.nan_to_num(np.asarray(loadmat(str(LOCAL / D1 / "Cn.mat"))["Cn"], dtype=float))
(bm, dr, dc), z = best_shift(hp(means[0]), hp(means[1])); print("high-pass mean images: best shift of day1 onto day0 (dr, dc) = (%d, %d), corr %.3f; at zero shift %.3f" % (dr, dc, bm, z))
(bc, dr2, dc2), z2 = best_shift(hp(Cn0, 8), hp(Cn1, 8)); print("Cn images: best shift (%d, %d), corr %.3f; at zero shift %.3f" % (dr2, dc2, bc, z2))
# self-consistency control: day 0 first half vs second half mean image (same session) should give shift 0 and high corr
import tifffile
def mean_frames(s, idx):
    tif = LOCAL / s / (s + ".tif")
    with tifffile.TiffFile(str(tif)) as t: off = t.pages[0].dataoffsets[0]
    acc = np.zeros((512, 512)); 
    with open(tif, "rb") as fh:
        for i in idx: fh.seek(off + int(i) * 512 * 512); acc += np.frombuffer(fh.read(512 * 512), dtype=np.uint8).reshape(512, 512)
    return acc / len(idx)
h1 = mean_frames(D0, np.linspace(0, 5000, 150).astype(int)); h2 = mean_frames(D0, np.linspace(5100, 10100, 150).astype(int))
(bs, drs, dcs), zs = best_shift(hp(h1), hp(h2), rng=20); print("control, day-0 first half vs second half: shift (%d, %d), corr %.3f" % (drs, dcs, bs))
# redo candidate matching at the mean-image shift
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float); return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
def cents(A): return np.array([ndi.center_of_mass(a > 0.5 * a.max()) for a in A])
A0 = foot(LOCAL / D0 / NEW / "spatial_footprints.mat"); A1 = foot(LOCAL / D1 / NEW / "spatial_footprints.mat")
L1 = np.asarray(loadmat(str(INB / D1 / "labels.mat"))["labels"]).ravel(); keep1 = np.flatnonzero(L1 == 1)
c0, c1 = cents(A0), cents(A1) + np.array([dr, dc])
dist = np.sqrt(((c0[:, None, :] - c1[None, :, :]) ** 2).sum(-1))
for R in (8, 12):
    print("at shift (%d, %d), radius %d px: day-1 candidates with a day-0 counterpart %d of %d (kept: %d of %d); day-0 with a day-1 counterpart %d of %d" % (
        dr, dc, R, int((dist.min(0) <= R).sum()), len(A1), int((dist.min(0)[keep1] <= R).sum()), len(keep1), int((dist.min(1) <= R).sum()), len(A0)))
np.save(os.path.join(HERE, "pnb105_shift.npy"), np.array([dr, dc]))
