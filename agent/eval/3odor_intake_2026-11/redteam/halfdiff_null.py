"""Real-data noise null (C2 / C3 / C4): the half-DIFFERENCE movie.

The pipeline's half-rate movie is S(t) = (y(2t) + y(2t+1)) / 2.  D(t) = (y(2t) - y(2t+1)) / 2
has the same noise as S if the noise of consecutive frames is independent and symmetric,
with the real spatial structure, the real 8-bit quantisation and any residual frame-to-frame
motion, while everything slower than one frame (cells, neuropil, drift) cancels.  What does
not cancel is the part of a transient that changes within a frame pair (onsets), so D is
noise PLUS a little signal: seeds found on D are an upper bound on noise-only seeds.

D goes through the same seeding port and first-pass grid as the simulated fields and is
saved next to them (cache/null/halfdiff_<session>_T<T>_s0.npz).
Usage: halfdiff_null.py --only <substr> ... [--workers 3]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import noise_null as nn  # noqa: E402


def one(job):
    group, name, tif, sd, an = job
    t0 = time.time()
    mm = rc.open_movie(tif)
    n = (mm.shape[0] // 2) * 2
    T = n // 2
    tag = "halfdiff_" + name[:13].rstrip("-")
    dst = nn.OUT / ("%s_T%d_s0.npz" % (tag, T))
    if dst.exists():
        return dst.name + " (cached)"
    D = np.empty((T, 256, 256), np.float32)
    for s in range(0, n, 1024):
        e = min(s + 1024, n)
        y = np.asarray(mm[s:e]).astype(np.float32).reshape((e - s) // 2, 2, 512, 512)
        d = 0.5 * (y[:, 0] - y[:, 1])
        D[s // 2:e // 2] = d.reshape(-1, 256, 2, 256, 2).mean(axis=(2, 4))
    m = min(1000, T - 2)
    r = D[1:m + 1] - 0.5 * (D[0:m] + D[2:m + 2])
    r = (r - r.mean(0)) / (r.std(0) + 1e-9)
    rx1 = float((r[:, :, 1:] * r[:, :, :-1]).mean())
    ry1 = float((r[:, 1:] * r[:, :-1]).mean())
    rx2 = float((r[:, :, 2:] * r[:, :, :-2]).mean())
    cn, pnr, X, sn = rc.seeding_images(D, want_traces=True)
    ratio = float(np.median(X.std(axis=0) / sn))
    del X, D
    grid = np.zeros((len(nn.C_GRID), len(nn.P_GRID)), np.int32)
    for i, c in enumerate(nn.C_GRID):
        for j, p in enumerate(nn.P_GRID):
            grid[i, j] = rc.first_pass(cn, pnr, c, p)
    nn.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, cn=cn.astype(np.float32), pnr=pnr.astype(np.float32), grid=grid, c_grid=nn.C_GRID,
                        p_grid=nn.P_GRID, T=T, seed=0, tag=tag, rx1=rx1, ry1=ry1, rx2=rx2, std_over_sn=ratio)
    ci = cn[10:-10, 10:-10]
    return ("%s %s T %d | corr x %.3f y %.3f x2 %.3f | std/sn %.3f | Cn p50 %.3f p99.9 %.3f max %.3f | "
            "seeds@p3 c .50/.55/.60/.65/.70: %d/%d/%d/%d/%d | %.0fs" % (
                group, name, T, rx1, ry1, rx2, ratio, np.median(ci), np.percentile(ci, 99.9), ci.max(),
                grid[10, 0], grid[15, 0], grid[20, 0], grid[25, 0], grid[30, 0], time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    rc.NW = 8
    jobs = rc.sessions(a.only)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
