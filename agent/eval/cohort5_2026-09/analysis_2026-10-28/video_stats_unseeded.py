"""
video_stats_unseeded.py -- are there cells visible in the video that CNMFe headless
did not seed?  Read-only.

For each cohort-5 session with a local tif: raw-level statistics of the uint8 AVG5x
movie (300 frames spread over the session for levels and the mean image; 200
consecutive mid-session frames for temporal noise), the Cn/PNR percentiles of the
full-movie images, the candidate count, and two unseeded-structure counts:
  (a) soma-like blobs in the mean raw image (cell-scale difference-of-means
      contrast above the 95th percentile of random interior points, 15-px local
      maxima, 24-px border) with no candidate support on them, binned by the local
      Cn; the "strong" subset is contrast above the 99th percentile.
  (b) Cn local maxima (3x3 mean, 21-px window, 20-px border) above 0.60 / 0.70 that
      pass the session's min_pnr gate, sit on a Cn blob of >= 150 px, and lie
      outside every candidate support.
Candidate supports are normalised per candidate (A > 0.3 max, dilated 5 px), as in
analysis_2026-10-17/unseeded_why2.py.  Figures (--fig): mean raw image with the
candidate contours and the uncovered blobs, and the Cn image with the (b) maxima.

The candidate set is the live one unless --prev NAME:SUBDIR names an archived set
for a session.  The 12 reviewed pilots NEED it (.rerun_mincorr070_2026-10-06, or
.rerun_mincorr060_2026-10-18 for 082726-pnb105): after ingest the live
spatial_footprints.mat holds only the reviewer's kept cells.  A session whose
re-run is in progress needs it too (its live Cn.mat is the 256-res pre-pass image);
Cn/pnr are read from the archived set when it holds them.

Usage: python video_stats_unseeded.py [--sessions NAME ...] [--fig NAME ...]
                                      [--prev NAME:SUBDIR ...] [--out DIR]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from local_config import DATA_PARENT   # noqa: E402

TASK = DATA_PARENT / "vCA1" / "6odorDualDiffRew"
ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
BD = 24
RNG = np.random.default_rng(0)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(float)


K_IN = disk(5)
K_IN /= K_IN.sum()
K_OUT = disk(14) - np.pad(disk(9), 5)
K_OUT /= K_OUT.sum()


def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A


def tif_frames(tif, idx):
    """uint8 512x512 frames at the given indices from a contiguous ImageJ stack."""
    with tifffile.TiffFile(str(tif)) as t:
        p0 = t.pages[0]
        if p0.dtype != np.uint8 or p0.shape != (512, 512):
            raise RuntimeError("unexpected tif layout %s %s" % (p0.dtype, p0.shape))
        nf = int(t.imagej_metadata.get("images", 0))
        off = p0.dataoffsets[0]
    out = np.empty((len(idx), 512, 512), np.uint8)
    with open(tif, "rb") as fh:
        for j, i in enumerate(idx):
            fh.seek(off + int(i) * 512 * 512)
            out[j] = np.frombuffer(fh.read(512 * 512), dtype=np.uint8).reshape(512, 512)
    return out, nf


def raw_stats(sd, name):
    tif = sd / (name + ".tif")
    if not tif.exists():
        return None
    with tifffile.TiffFile(str(tif)) as t:
        nf = int(t.imagej_metadata.get("images", 0))
    spread, _ = tif_frames(tif, np.linspace(0, nf - 1, 300).astype(int))
    img = spread.mean(0)
    block, _ = tif_frames(tif, np.arange(nf // 2, nf // 2 + 200))
    B = block.astype(np.float32)
    m = B.mean(0)
    d = np.median(np.abs(np.diff(B, axis=0)), axis=0) / 0.6745 / np.sqrt(2)
    sel = (m >= 60) & (m < 100)
    return {"nf": nf, "img": img, "p1": float(np.percentile(spread, 1)), "p50": float(np.percentile(spread, 50)),
            "p99": float(np.percentile(spread, 99)), "sat": 100.0 * float((spread == 255).mean()),
            "noise60_100": float(np.median(d[sel])) if sel.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--fig", nargs="*", default=[])
    ap.add_argument("--prev", nargs="*", default=[], help="NAME:SUBDIR archived candidate set to use for NAME")
    ap.add_argument("--out", type=Path, default=HERE / "figs")
    a = ap.parse_args()
    prev = dict(x.split(":", 1) for x in a.prev)
    if a.sessions:
        dirs = [TASK / s for s in a.sessions]
    else:
        dirs = sorted(p for p in TASK.iterdir() if p.is_dir() and any(f"-{an}-" in p.name for an in ANIMALS))
    a.out.mkdir(parents=True, exist_ok=True)
    print("session | set | N cand | frames | raw p1/p50/p99 | %sat | noise(60-100) | Cn p50/p90/p95 | %Cn>0.8 | PNR p90 | "
          "blobs: total / covered / uncovered Cn>=0.70, 0.60-0.70, <0.60 | strong uncovered Cn>=0.70 | "
          "Cn maxima outside all candidates >0.60 / >0.70 (PNR ok, blob>=150px)")
    for sd in dirs:
        s = sd.name
        setd = sd / prev[s] if s in prev else sd
        if not (setd / "spatial_footprints.mat").exists() or not (sd / "Cn.mat").exists():
            print("%s | not processed yet" % s[14:])
            continue
        rs = raw_stats(sd, s)
        if rs is None:
            print("%s | no local tif" % s[14:])
            continue
        A = foot(setd / "spatial_footprints.mat")
        An = A / (A.reshape(len(A), -1).max(1)[:, None, None] + 1e-12)
        support = ndi.binary_dilation(An.max(0) > 0.3, structure=np.ones((5, 5)))
        imgd = setd if (setd / "Cn.mat").exists() and (setd / "pnr.mat").exists() else sd
        Cn = np.nan_to_num(np.asarray(loadmat(str(imgd / "Cn.mat"))["Cn"], dtype=float))
        pnr = np.nan_to_num(np.asarray(loadmat(str(imgd / "pnr.mat"))["pnr"], dtype=float))
        if Cn.shape != (512, 512):   # a 256-res pre-pass image (a run in progress)
            Cn = np.kron(Cn, np.ones((512 // Cn.shape[0], 512 // Cn.shape[1])))
        if pnr.shape != Cn.shape:
            pnr = np.kron(pnr, np.ones((Cn.shape[0] // pnr.shape[0], Cn.shape[1] // pnr.shape[1])))
        logp = setd / "agent_run.log" if (setd / "agent_run.log").exists() else sd / "agent_run.log"
        mm = re.findall(r"min_pnr=(\d+(?:\.\d+)?)", logp.read_text(encoding="utf-8", errors="replace")) if logp.exists() else []
        mp = float(mm[-1]) if mm else float("nan")
        sm = ndi.uniform_filter(Cn, 3)
        cnl = ndi.maximum_filter(sm, size=9)
        crop = Cn[20:-20, 20:-20]
        # (a) soma-like blobs in the mean raw image
        img = rs["img"]
        inner = ndi.convolve(img, K_IN, mode="nearest")
        outer = ndi.convolve(img, K_OUT, mode="nearest")
        con = (inner - outer) / (outer + 1e-6)
        null = con[RNG.integers(BD, 512 - BD, 2000), RNG.integers(BD, 512 - BD, 2000)]
        thr95, thr99 = np.percentile(null, 95), np.percentile(null, 99)
        pk = (con == ndi.maximum_filter(con, size=15)) & (con > thr95)
        pk[:BD, :] = pk[-BD:, :] = pk[:, :BD] = pk[:, -BD:] = False
        rr, cc = np.nonzero(pk)
        cov = support[rr, cc]
        cn_un = cnl[rr[~cov], cc[~cov]]
        strong_un = con[rr[~cov], cc[~cov]] > thr99
        n_hi, n_mid, n_lo = int((cn_un >= 0.70).sum()), int(((cn_un >= 0.60) & (cn_un < 0.70)).sum()), int((cn_un < 0.60).sum())
        n_strong_hi = int((strong_un & (cn_un >= 0.70)).sum())
        # (b) Cn maxima outside every candidate
        pnr_loc = ndi.maximum_filter(pnr, size=7)
        res_b = {}
        pts_b = []
        for th in (0.60, 0.70):
            mx = (sm == ndi.maximum_filter(sm, size=21)) & (sm > th)
            mx[:20, :] = mx[-20:, :] = mx[:, :20] = mx[:, -20:] = False
            lab, _ = ndi.label(Cn > th)
            sizes = ndi.sum(np.ones_like(Cn), lab, index=np.arange(lab.max() + 1))
            r2, c2 = np.nonzero(mx)
            outside = ~support[r2, c2]
            ok = outside & (pnr_loc[r2, c2] >= mp) & (sizes[lab[r2, c2]] >= 150)
            res_b[th] = int(ok.sum())
            if th == 0.60:
                pts_b = list(zip(r2[ok], c2[ok], sm[r2[ok], c2[ok]]))
        print("%s | %s | %d | %d | %d/%d/%d | %.2f | %.2f | %.2f/%.2f/%.2f | %.1f | %.1f | %d / %d / %d, %d, %d | %d | %d / %d" % (
            s[14:], "prev" if s in prev else "live", len(A), rs["nf"], rs["p1"], rs["p50"], rs["p99"], rs["sat"], rs["noise60_100"],
            np.percentile(crop, 50), np.percentile(crop, 90), np.percentile(crop, 95), 100.0 * (crop > 0.8).mean(),
            np.percentile(pnr[20:-20, 20:-20], 90), len(rr), int(cov.sum()), n_hi, n_mid, n_lo, n_strong_hi, res_b[0.60], res_b[0.70]))
        if s in a.fig:
            fig, ax = plt.subplots(1, 2, figsize=(18, 9))
            v0, v1 = np.percentile(img, 1), np.percentile(img, 99.8)
            ax[0].imshow(img, cmap="gray", vmin=v0, vmax=v1)
            ax[1].imshow(sm, cmap="gray", vmin=0.3, vmax=1.0)
            for k in range(len(A)):
                for j in (0, 1):
                    ax[j].contour(A[k] > 0.5 * A[k].max(), levels=[0.5], colors=["cyan"], linewidths=0.7)
            un = list(zip(rr[~cov], cc[~cov], cn_un, strong_un))
            for col, lo, hi, lab_ in (("yellow", 0.70, 2, "Cn>=0.70"), ("orange", 0.60, 0.70, "Cn 0.60-0.70"), ("magenta", 0, 0.60, "Cn<0.60")):
                pts = [(r, c, st) for r, c, v, st in un if lo <= v < hi]
                ax[0].plot([c for _, c, _ in pts], [r for r, _, _ in pts], "o", mfc="none", mec=col, ms=11, mew=1.2,
                           label="soma-like blob, no candidate, %s (%d, strong %d)" % (lab_, len(pts), sum(1 for p in pts if p[2])))
            ax[1].plot([c for _, c, _ in pts_b], [r for r, _, _ in pts_b], "o", mfc="none", mec="yellow", ms=11, mew=1.2,
                       label="Cn max > 0.60 outside all candidates, PNR ok, blob >= 150 px (%d; > 0.70: %d)" % (res_b[0.60], res_b[0.70]))
            ax[0].set_title("%s: mean of 300 raw frames, %d candidates (cyan)" % (s[14:], len(A)), fontsize=10)
            ax[1].set_title("Cn (full movie, 3x3 mean) with the same candidates", fontsize=10)
            for x in ax:
                x.legend(loc="lower left", fontsize=8)
                x.set_xticks([])
                x.set_yticks([])
            fig.tight_layout()
            outp = a.out / ("fig_%s_unseeded.png" % s[14:])
            fig.savefig(str(outp), dpi=80)
            plt.close(fig)
            print("   figure:", outp)


if __name__ == "__main__":
    main()
