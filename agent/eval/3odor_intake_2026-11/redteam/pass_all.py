"""One read of every tif (71 3odor + 40 6odor): images and noise diagnostics, cached.

Per session -> cache/<short name>.npz:
  cn, pnr          greedyROI seeding images on the FULL half-rate movie (sig 3), 256 x 256
  pcn, ppnr        pre-pass images (1000 linspace frames, 3 x 3 patches, sig 5)
  mean256          mean of the downsampled movie
  sn, hy_std, hy_mad   per pixel: get_noise_fft sigma, std and 1.4826 * MAD of the filtered trace
  rx1 ry1 rx2 ry2  per-pixel maps: spatial correlation of the temporal residual
                   r(t) = y(t) - (y(t-1) + y(t+1)) / 2 with the neighbour 1 / 2 px away
  rt1 rt2          per-pixel maps: lag-1 / lag-2 temporal autocorrelation of r
                   (white noise: -2/3 and +1/6)
  psd_hy_low, psd_y_low, psd_hy_all   32-band mean periodogram (each pixel normalised by
                   its own mean power in (0.25, 0.5]) of the filtered / raw traces, over
                   low-PNR pixels (PNR <= the interior 25th percentile) / all interior pixels
  scalars          frames, T, pct0, pct255, lv_p1/p50/p99, noise_u8, noise_rel (the
                   definition of video_compare.py, re-implemented), est_mc, est_mp,
                   raw_mc, raw_mp, pre_zero (share of pre-pass Cn pixels that are 0)
Usage: pass_all.py [--only substr ...] [--workers 4]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

NB = 32


def band_psd(X, cols):
    """Mean over the pixels in cols of the periodogram, each normalised by its own mean power
    in (0.25, 0.5], averaged into NB equal bands of (0, 0.5]."""
    T = X.shape[0]
    ff = np.arange(0, T // 2 + 1) / T
    hi = (ff > 0.25) & (ff <= 0.5)
    edges = np.linspace(0, 0.5, NB + 1)
    which = np.clip(np.searchsorted(edges, ff, side="left") - 1, 0, NB - 1)
    acc = np.zeros(NB)
    n = 0
    for j in range(0, len(cols), 4096):
        c = cols[j:j + 4096]
        b = X[:, c].astype(np.float32)
        b = b - b.mean(axis=0)[None]
        xd = sfft.rfft(b, axis=0, workers=rc.NW)
        p = xd.real.astype(np.float64) ** 2 + xd.imag.astype(np.float64) ** 2
        p /= p[hi].mean(axis=0)[None]
        for k in range(NB):
            sel = (which == k) & (ff > 0)
            acc[k] += p[sel].mean(axis=0).sum()
        n += len(c)
    return acc / max(n, 1)


def residual_maps(Y):
    T, h, w = Y.shape
    s1 = np.zeros((h, w))
    s2 = np.zeros((h, w))
    prod = {k: np.zeros((h, w)) for k in ("x1", "x2", "y1", "y2", "t1", "t2")}
    n = 0
    step = 512
    for i in range(1, T - 1, step):
        e = min(i + step, T - 1)
        r = Y[i:e] - 0.5 * (Y[i - 1:e - 1] + Y[i + 1:e + 1])
        e2 = min(e, T - 3)                                   # frames that have r(t+1), r(t+2)
        if e2 > i:
            ra = r[:e2 - i]
            rb = Y[i + 1:e2 + 1] - 0.5 * (Y[i:e2] + Y[i + 2:e2 + 2])
            rcc = Y[i + 2:e2 + 2] - 0.5 * (Y[i + 1:e2 + 1] + Y[i + 3:e2 + 3])
            prod["t1"] += np.einsum("tij,tij->ij", ra, rb, dtype=np.float64)
            prod["t2"] += np.einsum("tij,tij->ij", ra, rcc, dtype=np.float64)
        s1 += r.sum(axis=0, dtype=np.float64)
        s2 += np.einsum("tij,tij->ij", r, r, dtype=np.float64)
        prod["x1"][:, :-1] += np.einsum("tij,tij->ij", r[:, :, :-1], r[:, :, 1:], dtype=np.float64)
        prod["x2"][:, :-2] += np.einsum("tij,tij->ij", r[:, :, :-2], r[:, :, 2:], dtype=np.float64)
        prod["y1"][:-1, :] += np.einsum("tij,tij->ij", r[:, :-1, :], r[:, 1:, :], dtype=np.float64)
        prod["y2"][:-2, :] += np.einsum("tij,tij->ij", r[:, :-2, :], r[:, 2:, :], dtype=np.float64)
        n += r.shape[0]
    mu = s1 / n
    var = s2 / n - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-12))
    out = {}

    def cc(p, dy, dx):
        m = np.full((h, w), np.nan)
        a = (slice(0, h - dy), slice(0, w - dx))
        b = (slice(dy, h), slice(dx, w))
        m[a] = (p[a] / n - mu[a] * mu[b]) / (sd[a] * sd[b])
        return m.astype(np.float32)

    out["rx1"], out["rx2"] = cc(prod["x1"], 0, 1), cc(prod["x2"], 0, 2)
    out["ry1"], out["ry2"] = cc(prod["y1"], 1, 0), cc(prod["y2"], 2, 0)
    out["rt1"] = ((prod["t1"] / max(n - 2, 1) - mu * mu) / np.maximum(var, 1e-12)).astype(np.float32)
    out["rt2"] = ((prod["t2"] / max(n - 2, 1) - mu * mu) / np.maximum(var, 1e-12)).astype(np.float32)
    return out


def one(job):
    group, name, tif, sd, an = job
    dst = rc.CACHE / (name + ".npz")
    if dst.exists():
        return name + " (cached)"
    t0 = time.time()
    try:
        mm = rc.open_movie(tif)
        nf = mm.shape[0]
        sc = {"frames": nf}
        spread = np.asarray(mm[np.linspace(0, nf - 1, 300).astype(int)])
        sc.update(pct0=100.0 * (spread == 0).mean(), pct255=100.0 * (spread == 255).mean(),
                  lv_p1=float(np.percentile(spread, 1)), lv_p50=float(np.percentile(spread, 50)),
                  lv_p99=float(np.percentile(spread, 99)))
        st = max(0, nf // 2 - 500)
        B = np.asarray(mm[st:st + min(1000, nf - st)]).astype(np.float32)
        bm = B.mean(0)
        d = np.median(np.abs(np.diff(B, axis=0)), axis=0) / 0.6745 / np.sqrt(2)
        sc["noise_u8"] = float(np.median(d[24:-24, 24:-24]))
        rng = float(np.percentile(bm[24:-24, 24:-24], 99) - np.percentile(bm[24:-24, 24:-24], 1))
        sc["noise_rel"] = sc["noise_u8"] / rng if rng > 0 else float("nan")
        del B, d
        Y = rc.load_ds(tif, mm=mm)
        T = Y.shape[0]
        sc["T"] = T
        mean256 = Y.mean(axis=0)
        cn, pnr, X, sn = rc.seeding_images(Y, want_traces=True)
        hy_std = X.std(axis=0)
        hy_mad = np.empty(X.shape[1], np.float32)
        for j in range(0, X.shape[1], 8192):
            hy_mad[j:j + 8192] = 1.4826 * np.median(np.abs(X[:, j:j + 8192]), axis=0)   # median is 0 already
        inner = np.zeros((256, 256), bool)
        inner[10:-10, 10:-10] = True
        thr = np.percentile(pnr[inner], 25)
        low = np.nonzero((inner & (pnr <= thr)).ravel())[0]
        allp = np.nonzero(inner.ravel())[0][::4]
        psd_hy_low = band_psd(X, low)
        psd_hy_all = band_psd(X, allp)
        del X
        psd_y_low = band_psd(Y.reshape(T, -1), low)
        res = residual_maps(Y)
        pcn, ppnr = rc.prepass_images(Y)
        emc, emp = rc.estimate_params(pcn, ppnr)
        rmc, rmp = rc.estimate_params(pcn, ppnr, raw=True)
        sc.update(est_mc=emc, est_mp=emp, raw_mc=rmc, raw_mp=rmp, pre_zero=float((pcn == 0).mean()))
        rc.CACHE.mkdir(exist_ok=True)
        tmp = rc.CACHE / (name + ".tmp.npz")
        np.savez_compressed(tmp, cn=cn.astype(np.float32), pnr=pnr.astype(np.float32),
                            pcn=pcn.astype(np.float32), ppnr=ppnr.astype(np.float32),
                            mean256=mean256.astype(np.float32), sn=sn.astype(np.float32).reshape(256, 256),
                            hy_std=hy_std.reshape(256, 256), hy_mad=hy_mad.reshape(256, 256),
                            psd_hy_low=psd_hy_low, psd_hy_all=psd_hy_all, psd_y_low=psd_y_low,
                            group=group, animal=an, **res, **sc)
        tmp.replace(dst)
        return "%s %s T %d est %.2f/%.1f noise_rel %.3f %.0fs" % (group, name, T, emc, emp, sc["noise_rel"],
                                                                 time.time() - t0)
    except Exception as e:                                   # report, keep going
        return "FAILED %s: %s: %s" % (name, type(e).__name__, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rc.CACHE.mkdir(exist_ok=True)
    jobs = rc.sessions(a.only)
    print("%d sessions" % len(jobs), flush=True)
    if a.workers <= 1:
        for j in jobs:
            print(one(j), flush=True)
    else:
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(one, jobs):
                print(msg, flush=True)


if __name__ == "__main__":
    main()
