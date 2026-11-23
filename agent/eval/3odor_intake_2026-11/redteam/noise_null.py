"""Noise-only fields through the seeding port: where do first-pass seeds stop? (C2, C3, C4)

Variants (one noise movie = one "field", 256 x 256 x T at half rate):
  gauss      Gaussian, white in time, lag-1 spatial correlation rhox / rhoy built at 256
             (the model under test; own implementation)
  gauss2     same plus lag-2 spatial correlation (5-tap kernels fitted numerically)
  u8         built at 512 x 512, full rate (2T frames): level + sigma * correlated Gaussian,
             rounded and clipped to uint8, then downsampled exactly like the pipeline
             (--level, --sigma in uint8 units)
  poisson    as u8 but gain * Poisson(lam) per full-rate pixel (skewed, signal-dependent)
  colour     gauss with a temporal power spectrum shaped by a 32-band profile taken from a
             cached real session (--shape <cache npz>, key psd_hy_low)
  white      Gaussian, no spatial correlation (control)
Each field: Cn / PNR quantiles (border 10 removed) and first-pass seeds on a grid
min_corr 0.40 .. 0.80 step 0.01 x min_pnr 3, 4, 5, 6.
Output: one .npz per field in cache/null/ and a line on stdout; summarise with null_report.py.
Usage: noise_null.py --variant gauss --T 6400 --seeds 1 2 3 ... [--workers 4]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

C_GRID = np.round(np.arange(0.40, 0.801, 0.01), 2)
P_GRID = np.array([3.0, 4.0, 5.0, 6.0])
OUT = rc.CACHE / "null"


def taps3(rho):
    """[a 1 a] with lag-1 autocorrelation 2a / (1 + 2a^2) = rho."""
    if rho <= 0:
        return np.array([1.0])
    a = (1.0 - np.sqrt(1.0 - 2.0 * rho * rho)) / (2.0 * rho)
    return np.array([a, 1.0, a])


def taps5(rho1, rho2):
    """[b a 1 a b] with lag-1 / lag-2 autocorrelation rho1 / rho2 (numerical fit)."""
    def f(p):
        a, b = p
        k = np.array([b, a, 1.0, a, b])
        ac = np.correlate(k, k, "full")
        m = len(k) - 1
        return [ac[m + 1] / ac[m] - rho1, ac[m + 2] / ac[m] - rho2]
    s = least_squares(f, [rho1 / 2, rho2 / 2])
    a, b = s.x
    return np.array([b, a, 1.0, a, b])


def taps512(rho):
    """[a 1 a] at 512 so that 2-pixel bins have lag-1 correlation rho:
    rho = (a + a^2) / (1 + 2a + 2a^2)."""
    if rho <= 0:
        return np.array([1.0])
    A, B, C = 1 - 2 * rho, 1 - 2 * rho, -rho
    a = (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    return np.array([a, 1.0, a])


def smooth(z, kx, ky):
    if len(kx) > 1:
        z = ndi.correlate1d(z, kx.astype(np.float32), axis=2, mode="wrap")
    if len(ky) > 1:
        z = ndi.correlate1d(z, ky.astype(np.float32), axis=1, mode="wrap")
    return z


def make_field(a, seed):
    rng = np.random.default_rng([seed, a.T])
    T = a.T
    Y = np.empty((T, 256, 256), np.float32)
    if a.variant in ("gauss", "gauss2", "colour", "white"):
        if a.variant == "white":
            kx = ky = np.array([1.0])
        elif a.variant == "gauss2":
            kx, ky = taps5(a.rhox, a.rhox2), taps5(a.rhoy, a.rhoy2)
        else:
            kx, ky = taps3(a.rhox), taps3(a.rhoy)
        for i in range(0, T, 512):
            n = min(512, T - i)
            z = rng.standard_normal((n, 256, 256), dtype=np.float32)
            Y[i:i + n] = smooth(z, kx, ky)
        if a.variant == "colour":
            prof = np.load(a.shape)[a.shape_key].astype(float)          # 32 bands over (0, 0.5]
            ff = np.arange(0, T // 2 + 1) / T
            centres = (np.arange(len(prof)) + 0.5) * 0.5 / len(prof)
            gain = np.sqrt(np.interp(ff, centres, prof)).astype(np.float32)
            gain[0] = 0
            flat = Y.reshape(T, -1)
            for j in range(0, flat.shape[1], 8192):
                xd = sfft.rfft(flat[:, j:j + 8192], axis=0, workers=rc.NW)
                xd *= gain[:, None]
                flat[:, j:j + 8192] = sfft.irfft(xd, n=T, axis=0, workers=rc.NW)
    elif a.variant in ("u8", "poisson"):
        kx, ky = taps512(a.rhox), taps512(a.rhoy)
        norm = np.sqrt((kx ** 2).sum() * (ky ** 2).sum())
        for i in range(0, T, 128):
            n = min(128, T - i)
            if a.variant == "u8":
                z = rng.standard_normal((2 * n, 512, 512), dtype=np.float32)
                z = smooth(z, kx, ky) / norm
                raw = a.level + a.sigma * z
            else:
                gain = a.level / a.lam
                raw = gain * rng.poisson(a.lam, (2 * n, 512, 512)).astype(np.float32)
                raw = smooth(raw, kx, ky) / (kx.sum() * ky.sum())
            raw = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
            Y[i:i + n] = rc.ds_block(raw)
    else:
        raise SystemExit("unknown variant")
    return Y


def tag(a):
    t = a.variant
    if a.variant == "u8":
        t += "_L%g_S%g" % (a.level, a.sigma)
    if a.variant == "poisson":
        t += "_L%g_lam%g" % (a.level, a.lam)
    if a.variant == "colour":
        t += "_" + Path(a.shape).stem[:13] + "_" + a.shape_key
    if a.variant == "gauss2":
        t += "_%g_%g" % (a.rhox2, a.rhoy2)
    if a.variant in ("gauss", "colour") and (a.rhox, a.rhoy) != (0.18, 0.13):
        t += "_rx%g_ry%g" % (a.rhox, a.rhoy)
    return t


def one(job):
    a, seed = job
    t0 = time.time()
    dst = OUT / ("%s_T%d_s%d.npz" % (tag(a), a.T, seed))
    if dst.exists():
        return "%s (cached)" % dst.name
    Y = make_field(a, seed)
    # measured spatial / temporal structure of the field itself
    m = min(1000, Y.shape[0] - 2)
    r = Y[1:m + 1] - 0.5 * (Y[0:m] + Y[2:m + 2])
    r = (r - r.mean(0)) / (r.std(0) + 1e-9)
    rx1 = float((r[:, :, 1:] * r[:, :, :-1]).mean())
    ry1 = float((r[:, 1:] * r[:, :-1]).mean())
    rx2 = float((r[:, :, 2:] * r[:, :, :-2]).mean())
    cn, pnr, X, sn = rc.seeding_images(Y, want_traces=True)
    ratio = float(np.median(X.std(axis=0) / sn))
    del X, Y
    grid = np.zeros((len(C_GRID), len(P_GRID)), np.int32)
    for i, c in enumerate(C_GRID):
        for j, p in enumerate(P_GRID):
            grid[i, j] = rc.first_pass(cn, pnr, c, p)
    ci, pi = cn[10:-10, 10:-10], pnr[10:-10, 10:-10]
    np.savez_compressed(dst, cn=cn.astype(np.float32), pnr=pnr.astype(np.float32), grid=grid, c_grid=C_GRID,
                        p_grid=P_GRID, T=a.T, seed=seed, tag=tag(a), rx1=rx1, ry1=ry1, rx2=rx2, std_over_sn=ratio)
    return ("%s T %d seed %d | corr x %.3f y %.3f x2 %.3f | std/sn %.3f | Cn p50 %.3f p99.9 %.3f max %.3f | "
            "PNR p50 %.2f p99.9 %.2f max %.2f | seeds@p3 c .50/.55/.60/.65: %d/%d/%d/%d | %.0fs" % (
                tag(a), a.T, seed, rx1, ry1, rx2, ratio, np.median(ci), np.percentile(ci, 99.9), ci.max(),
                np.median(pi), np.percentile(pi, 99.9), pi.max(), grid[10, 0], grid[15, 0], grid[20, 0],
                grid[25, 0], time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--T", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rhox", type=float, default=0.18)
    ap.add_argument("--rhoy", type=float, default=0.13)
    ap.add_argument("--rhox2", type=float, default=0.015)
    ap.add_argument("--rhoy2", type=float, default=0.015)
    ap.add_argument("--level", type=float, default=80.0)
    ap.add_argument("--sigma", type=float, default=12.0)
    ap.add_argument("--lam", type=float, default=17.0)
    ap.add_argument("--shape", default=None)
    ap.add_argument("--shape-key", dest="shape_key", default="psd_hy_low")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fft-workers", type=int, default=4)
    a = ap.parse_args()
    rc.NW = a.fft_workers
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for T in a.T:
        b = argparse.Namespace(**vars(a))
        b.T = T
        jobs += [(b, s) for s in a.seeds]
    if a.workers <= 1:
        for j in jobs:
            print(one(j), flush=True)
    else:
        with Pool(a.workers, initializer=_init, initargs=(a.fft_workers,)) as pool:
            for msg in pool.imap_unordered(one, jobs):
                print(msg, flush=True)


def _init(nw):
    rc.NW = nw


if __name__ == "__main__":
    main()
