"""Read-only.  Soma-like blobs in the mean raw image (cell-scale DoG, contrast above
the 95th pct of random interior points) that have NO 0.70 candidate on them, and the
Cn they sit at.  pnb105 both days + two references.  Figure for 082726-pnb105."""
import sys, os, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
import tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); EX = Path(local_config.EXCHANGE_ROOT)
LOCAL = DP / "vCA1" / "6odorDualDiffRew"; NEW = ".rerun_mincorr070_2026-10-06"; INB = EX / "inbox" / "Taylor" / "vCA1" / "6odorDualDiffRew"
HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
S = ["AVG5x-TSeries-082626-pnb105-397um-35z-000", "AVG5x-TSeries-082726-pnb105-391um-35z-000",
     "AVG5x-TSeries-082726-pnb109-405um-25z-000", "AVG5x-TSeries-082826-pnb99-436um-3z-000-000", "AVG5x-TSeries-082726-pnb104-416um-3z-000"]
def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A
def mean_image(sd, name, nfr=300):
    tif = sd / (name + ".tif")
    with tifffile.TiffFile(str(tif)) as t: nf = int(t.imagej_metadata.get("images", 0)); off = t.pages[0].dataoffsets[0]
    acc = np.zeros((512, 512), np.float64)
    with open(tif, "rb") as fh:
        for i in np.linspace(0, nf - 1, nfr).astype(int):
            fh.seek(off + int(i) * 512 * 512); acc += np.frombuffer(fh.read(512 * 512), dtype=np.uint8).reshape(512, 512)
    return acc / nfr
def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]; return (x * x + y * y <= r * r).astype(float)
K_IN = disk(5); K_IN /= K_IN.sum(); K_OUT = disk(14) - np.pad(disk(9), 5); K_OUT /= K_OUT.sum()
BD = 24
print("session | N cand | soma-like blobs (mean image) | with a candidate | WITHOUT any candidate | Cn at the uncovered ones: median / n>=0.70 / n 0.60-0.70 / n<0.60")
for s in S:
    sd = LOCAL / s; img = mean_image(sd, s)
    inner = ndi.convolve(img, K_IN, mode="nearest"); outer = ndi.convolve(img, K_OUT, mode="nearest"); con = (inner - outer) / (outer + 1e-6)
    null = con[rng.integers(BD, 512 - BD, 2000), rng.integers(BD, 512 - BD, 2000)]; thr = np.percentile(null, 95)
    pk = (con == ndi.maximum_filter(con, size=15)) & (con > thr); pk[:BD, :] = pk[-BD:, :] = pk[:, :BD] = pk[:, -BD:] = False
    A = foot(sd / NEW / "spatial_footprints.mat"); An = A / (A.reshape(len(A), -1).max(1)[:, None, None] + 1e-12)
    support = ndi.binary_dilation(An.max(0) > 0.3, structure=np.ones((7, 7)))
    Cn = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float)); cnl = ndi.maximum_filter(ndi.uniform_filter(Cn, 3), size=9)
    rr, cc = np.nonzero(pk); cov = support[rr, cc]; cn_un = cnl[rr[~cov], cc[~cov]]
    print("%s | %d | %d | %d | %d | %.2f / %d / %d / %d" % (s[14:], len(A), len(rr), int(cov.sum()), int((~cov).sum()),
          np.median(cn_un) if len(cn_un) else 0, int((cn_un >= 0.70).sum()), int(((cn_un >= 0.60) & (cn_un < 0.70)).sum()), int((cn_un < 0.60).sum())))
    if "082726-pnb105" in s:
        L = np.asarray(loadmat(str(INB / s / "labels.mat"))["labels"]).ravel()
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        for a in ax: a.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8)); a.set_xticks([]); a.set_yticks([])
        ax[0].set_title("%s mean of 300 raw frames; green kept (%d), red deleted (%d)" % (s[14:], int((L == 1).sum()), int((L == 0).sum())), fontsize=10)
        for k in range(len(A)): ax[0].contour(A[k] > 0.5 * A[k].max(), levels=[0.5], colors=["lime" if L[k] == 1 else "red"], linewidths=0.7)
        for k in np.flatnonzero(L == 1): ax[1].contour(A[k] > 0.5 * A[k].max(), levels=[0.5], colors=["lime"], linewidths=0.7)
        un = list(zip(rr[~cov], cc[~cov], cn_un))
        for col, lo, hi, lab in (("yellow", 0.70, 2, "Cn>=0.70"), ("orange", 0.60, 0.70, "0.60-0.70"), ("magenta", 0, 0.60, "Cn<0.60")):
            pts = [(r, c) for r, c, v in un if lo <= v < hi]
            ax[1].plot([c for _, c in pts], [r for r, _ in pts], "o", mfc="none", mec=col, ms=11, mew=1.4, label="soma-like, no candidate, %s (%d)" % (lab, len(pts)))
        ax[1].legend(loc="lower left", fontsize=9); ax[1].set_title("soma-like blobs in the mean image with no candidate, coloured by local Cn", fontsize=10)
        fig.tight_layout(); out = os.path.join(HERE, "fig_pnb105_082726_visible_unseeded.png"); fig.savefig(out, dpi=80); print("figure:", out)
