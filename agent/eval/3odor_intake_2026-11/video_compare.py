"""Read-only.  Image statistics of the new 3odor AVG5x tifs vs the six animals'
processed 6odorDualDiffRew tifs, computed IDENTICALLY on both sets from the tif.

Per tif (uint8 512x512 contiguous ImageJ stack):
  levels   300 frames spread over the session: p1/p50/p99, % pixels at 0 and 255.
           The 8-bit mapping is set per session by Fiji from the first averaged
           frame (scripts/README_avg_project.md), so absolute levels are NOT
           comparable between sessions; the clipping fractions are.
  noise    1000 consecutive mid-session frames: per-pixel MAD of frame differences
           (uint8 units), and that noise divided by the p1-p99 range of the block's
           mean image (scale- and offset-free).
  Cn/PNR   the same 1000 frames, downsampled 2x2 in space and 2x in time like the
           headless run (ssub=2, tsub=2), then a port of
           greedyROI_endoscope.m (gSig 5, gSiz 18 at 256 = 9/36 halved, center_psf, noise by
           get_noise_fft logmexp on 0.25-0.5, threshold 3 sigma, 8-neighbour
           correlation).  Stats on the 256 image with a 10-px border removed.
  motion   residual rigid shift (phase correlation, DoG-filtered, subpixel parabola)
           of 5-frame means of the block vs the block mean, and of the 300 spread
           frames vs their mean.
For 6odor sessions it also reads the pipeline's full-movie Cn.mat / pnr.mat and the
candidate count in review_summary.txt, to check the subsample metric against them.
Writes a TSV and one npz of images per session under --out.  Never writes to D:.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.io import loadmat
from scipy.signal import fftconvolve

ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
NEW_DIR = Path(r"D:/2p data/3odor")
sys.path.append(str(Path(__file__).resolve().parents[2]))     # agent/, for local_config
from local_config import DATA_PARENT                           # noqa: E402
OLD_DIR = Path(DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
ANIMAL_RE = re.compile(r"-(pnb\d+)-")
NBLOCK = 1000
NSPREAD = 300
GSIG, GSIZ = 5, 18          # gSig 9 / gSiz 36 halved by downSample.m (ceil(x/ssub))


def tif_list(only=None):
    out = []
    for t in sorted(NEW_DIR.glob("AVG5x-TSeries-*.tif")):
        m = ANIMAL_RE.search(t.name)
        if m and m.group(1) in ANIMALS:
            out.append(("3odor", t.stem, t, None))
    for d in sorted(p for p in OLD_DIR.iterdir() if p.is_dir()):
        m = ANIMAL_RE.search(d.name)
        if m and m.group(1) in ANIMALS:
            t = d / (d.name + ".tif")
            if t.exists():
                out.append(("6odor", d.name, t, d))
    if only:
        out = [x for x in out if any(o in x[1] for o in only)]
    return out


def layout(tif):
    with tifffile.TiffFile(str(tif)) as t:
        p0 = t.pages[0]
        if p0.dtype != np.uint8 or p0.shape != (512, 512):
            raise RuntimeError("unexpected tif layout %s %s" % (p0.dtype, p0.shape))
        nf = int((t.imagej_metadata or {}).get("images", 0))
        off = int(p0.dataoffsets[0])
    return nf, off


def read_idx(tif, off, idx):
    out = np.empty((len(idx), 512, 512), np.uint8)
    with open(tif, "rb") as fh:
        for j, i in enumerate(idx):
            fh.seek(off + int(i) * 262144)
            out[j] = np.frombuffer(fh.read(262144), dtype=np.uint8).reshape(512, 512)
    return out


def read_block(tif, off, start, n):
    with open(tif, "rb") as fh:
        fh.seek(off + int(start) * 262144)
        buf = fh.read(n * 262144)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, 512, 512)


def psf_centered(siz, sig):
    # fspecial('gaussian', siz, sig) then the center_psf step of correlation_image_endoscope.m
    h = (siz - 1) / 2.0
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sig * sig))
    g /= g.sum()
    nz = g >= g[:, 0].max()
    g = g - g[nz].mean()
    g[~nz] = 0
    return g.astype(np.float32)


PSF = psf_centered(GSIZ, GSIG)


def spatial_filter(Y):
    """imfilter(Y, psf, 'replicate') per frame, via FFT on an edge-padded stack."""
    p = GSIZ // 2 + 1
    out = np.empty_like(Y)
    k = PSF[::-1, ::-1][None]
    for i in range(0, len(Y), 100):
        Yp = np.pad(Y[i:i + 100], ((0, 0), (p, p), (p, p)), mode="edge")
        f = fftconvolve(Yp, k, mode="same", axes=(1, 2))
        out[i:i + 100] = f[:, p:-p, p:-p]
    return out


def noise_fft(X):
    """get_noise_fft, logmexp, range (0.25, 0.5]; X is T x P."""
    T = X.shape[0]
    ff = np.arange(T // 2 + 1) / T
    ind = (ff > 0.25) & (ff <= 0.5)
    sn = np.empty(X.shape[1], np.float64)
    for j in range(0, X.shape[1], 16384):
        xd = np.fft.rfft(X[:, j:j + 16384], axis=0)
        psd = (np.abs(xd) ** 2) / T
        psd[1:-1] *= 2
        sn[j:j + 16384] = np.sqrt(np.exp(np.mean(np.log(psd[ind] / 2 + 1e-30), axis=0)))
    return sn


def downsample(block_u8):
    Y = block_u8.astype(np.float32)
    T = (len(Y) // 2) * 2
    Y = Y[:T].reshape(T // 2, 2, 512, 512).mean(axis=1)            # tsub = 2
    return Y.reshape(T // 2, 256, 2, 256, 2).mean(axis=(2, 4))      # ssub = 2


def full_downsampled(tif, off, nf, chunk=2000):
    """The whole movie at 256 x 256 and half rate, read in chunks (low memory)."""
    T = (nf // 2) * 2
    out = np.empty((T // 2, 256, 256), np.float32)
    for s in range(0, T, chunk):
        n = min(chunk, T - s)
        out[s // 2:(s + n) // 2] = downsample(read_block(tif, off, s, n))
    return out


def cn_pnr(block_u8):
    return cn_pnr_ds(downsample(block_u8))


def img_stats(cn, pnr, prefix):
    """Stats on 256 x 256 Cn / PNR images, 10-px border removed."""
    c = cn[10:-10, 10:-10]
    pc = pnr[10:-10, 10:-10]
    sm = ndi.uniform_filter(cn, 3)
    r = {}
    for th in (0.70, 0.80):
        pk = (sm == ndi.maximum_filter(sm, size=9)) & (sm > th)
        pk[:10, :] = pk[-10:, :] = pk[:, :10] = pk[:, -10:] = False
        r["%s_npk%03d" % (prefix, int(th * 100))] = int(pk.sum())
        r["%s_pct%03d" % (prefix, int(th * 100))] = 100.0 * (c > th).mean()
    r.update({prefix + "_cn_p50": np.percentile(c, 50), prefix + "_cn_p90": np.percentile(c, 90),
              prefix + "_pnr_p50": np.percentile(pc, 50), prefix + "_pnr_p85": np.percentile(pc, 85),
              prefix + "_pnr_p90": np.percentile(pc, 90)})
    return r


def cn_pnr_ds(Y):
    HY = spatial_filter(Y.astype(np.float32))
    n = HY.shape[0]
    X = HY.reshape(n, -1)
    X = X - np.median(X, axis=0)[None]
    hmax = X.max(axis=0)
    sn = noise_fft(X)
    pnr = (hmax / sn).reshape(256, 256)
    X[X < (sn * 3)[None]] = 0            # greedyROI_endoscope.m sig = 3
    X = X - X.mean(axis=0)[None]
    s = np.sqrt((X * X).mean(axis=0))
    s[s == 0] = 1
    X = (X / s[None]).reshape(n, 256, 256)
    acc = np.zeros((256, 256), np.float64)
    cnt = np.zeros((256, 256), np.float64)
    ones = np.ones((256, 256))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.zeros_like(X)
            ys = slice(max(dy, 0), 256 + min(dy, 0))
            yd = slice(max(-dy, 0), 256 + min(-dy, 0))
            xs = slice(max(dx, 0), 256 + min(dx, 0))
            xd = slice(max(-dx, 0), 256 + min(-dx, 0))
            sh[:, yd, xd] = X[:, ys, xs]
            acc += (sh * X).mean(axis=0)
            c = np.zeros((256, 256))
            c[yd, xd] = ones[ys, xs]
            cnt += c
    return (acc / cnt).astype(np.float32), pnr.astype(np.float32)


def dog(img):
    return ndi.gaussian_filter(img, 1.5) - ndi.gaussian_filter(img, 8)


WIN = np.outer(np.hanning(512), np.hanning(512)).astype(np.float32)


def shifts(frames, ref):
    R = np.fft.fft2(dog(ref) * WIN)
    out = []
    for f in frames:
        F = np.fft.fft2(dog(f) * WIN)
        X = F * np.conj(R)
        X /= np.abs(X) + 1e-9
        c = np.real(np.fft.ifft2(X))
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        sub = []
        for i, n, ax in ((iy, 512, 0), (ix, 512, 1)):
            if ax == 0:
                a, b, d = c[(i - 1) % n, ix], c[i, ix], c[(i + 1) % n, ix]
            else:
                a, b, d = c[iy, (i - 1) % n], c[iy, i], c[iy, (i + 1) % n]
            den = a - 2 * b + d
            o = 0.5 * (a - d) / den if den != 0 else 0.0
            v = i + o
            if v > n / 2:
                v -= n
            sub.append(v)
        out.append(np.hypot(*sub))
    return np.asarray(out)


def read_cand(sd):
    p = sd / "review_summary.txt"
    if not p.exists():
        return ""
    m = re.search(r"Total candidates:\s+(\d+)", p.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else ""


def mat_img(p, key):
    a = np.nan_to_num(np.asarray(loadmat(str(p))[key], dtype=float))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--port-full", nargs="*", default=[],
                    help="6odor sessions on which to ALSO run the full-movie port (validation)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "img").mkdir(exist_ok=True)
    stat_keys = ["cn_p50", "cn_p90", "pct070", "pct080", "npk070", "npk080", "pnr_p50", "pnr_p85", "pnr_p90"]
    cols = (["group", "session", "animal", "frames", "complete", "p1", "p50", "p99", "pct0", "pct255",
             "noise_u8", "noise_rel"] + ["blk_" + k for k in stat_keys] + ["full_src"] +
            ["full_" + k for k in stat_keys] + ["port_" + k for k in stat_keys] +
            ["mot_blk_med", "mot_blk_p95", "mot_blk_max", "mot_spr_med", "mot_spr_p95", "mot_spr_max",
             "n_cand", "secs"])
    tsv = a.out / "video_compare.tsv"
    done = set()
    if tsv.exists():
        for ln in tsv.read_text().splitlines()[1:]:
            done.add(ln.split("\t")[1])
    else:
        tsv.write_text("\t".join(cols) + "\n")
    for group, stem, tif, sd in tif_list(a.only):
        name = stem.replace("AVG5x-TSeries-", "")
        if name in done:
            continue
        t0 = time.time()
        nf, off = layout(tif)
        complete = tif.stat().st_size >= off + nf * 262144
        r = {"group": group, "session": name, "animal": ANIMAL_RE.search(stem).group(1), "frames": nf,
             "complete": int(complete)}
        spread = read_idx(tif, off, np.linspace(0, nf - 1, NSPREAD).astype(int))
        r.update(p1=np.percentile(spread, 1), p50=np.percentile(spread, 50), p99=np.percentile(spread, 99),
                 pct0=100.0 * (spread == 0).mean(), pct255=100.0 * (spread == 255).mean())
        st = max(0, nf // 2 - NBLOCK // 2)
        blk = read_block(tif, off, st, min(NBLOCK, nf - st))
        B = blk.astype(np.float32)
        bm = B.mean(0)
        d = np.median(np.abs(np.diff(B, axis=0)), axis=0) / 0.6745 / np.sqrt(2)
        r["noise_u8"] = float(np.median(d[24:-24, 24:-24]))
        rng = np.percentile(bm[24:-24, 24:-24], 99) - np.percentile(bm[24:-24, 24:-24], 1)
        r["noise_rel"] = r["noise_u8"] / rng if rng > 0 else float("nan")
        cn, pnr = cn_pnr(blk)
        r.update(img_stats(cn, pnr, "blk"))
        del blk
        fcn = fpnr = None
        if group == "3odor" or any(p in name for p in a.port_full):
            fcn, fpnr = cn_pnr_ds(full_downsampled(tif, off, nf))
            r.update(img_stats(fcn, fpnr, "port"))
        if group == "3odor":
            r["full_src"] = "port"
            r.update({"full_" + k: r["port_" + k] for k in stat_keys})
        elif (sd / "Cn.mat").exists() and (sd / "pnr.mat").exists():
            C = mat_img(sd / "Cn.mat", "Cn")
            P = mat_img(sd / "pnr.mat", "pnr")
            if C.shape == (512, 512):
                C = C.reshape(256, 2, 256, 2).mean(axis=(1, 3))
            if C.shape == (256, 256) and P.shape == (256, 256):
                r["full_src"] = "pipeline"
                r.update(img_stats(C, P, "full"))
            else:
                r["full_src"] = "shape%sx%s" % (C.shape, P.shape)
        nb = len(B) // 5
        b5 = B[:nb * 5].reshape(nb, 5, 512, 512).mean(1)
        sb = shifts(b5, bm)
        sp = spread.astype(np.float32)
        ss = shifts(sp, sp.mean(0))
        r.update(mot_blk_med=np.median(sb), mot_blk_p95=np.percentile(sb, 95), mot_blk_max=sb.max(),
                 mot_spr_med=np.median(ss), mot_spr_p95=np.percentile(ss, 95), mot_spr_max=ss.max())
        if sd is not None:
            r["n_cand"] = read_cand(sd)
        r["secs"] = time.time() - t0
        extra = {} if fcn is None else {"full_cn": fcn, "full_pnr": fpnr}
        np.savez_compressed(a.out / "img" / (name + ".npz"), spread_mean=spread.mean(0).astype(np.float32),
                            block_mean=bm.astype(np.float32), cn=cn, pnr=pnr, mot_blk=sb.astype(np.float32),
                            mot_spr=ss.astype(np.float32), **extra)
        vals = []
        for k in cols:
            v = r.get(k, "")
            vals.append("%.4f" % v if isinstance(v, (float, np.floating)) else str(v))
        with open(tsv, "a") as fh:
            fh.write("\t".join(vals) + "\n")
        print("%s %s %.0fs full(%s) cn_p90 %s pnr_p90 %s | port cn_p90 %s | noise %.2f mot %.2f/%.2f" % (
            group, name, r["secs"], r.get("full_src"), r.get("full_cn_p90"), r.get("full_pnr_p90"),
            r.get("port_cn_p90"), r["noise_u8"], r["mot_blk_p95"], r["mot_spr_p95"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
