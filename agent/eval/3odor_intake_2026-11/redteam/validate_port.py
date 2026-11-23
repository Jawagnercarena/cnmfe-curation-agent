"""Validation of rt_core against what ACORN itself wrote, on one session (console only).

  1. imfilter_replicate vs scipy.ndimage.correlate (direct, no FFT) on 3 frames
  2. PNR0 port vs pnr.mat, pixel by pixel (256 x 256)
  3. Cn0 port -> imresize x2 vs Cn.mat, pixel by pixel (512 x 512)
  4. pre-pass port -> 85th-percentile estimate vs "Estimated from Cn/pnr images" in agent_run.log
Usage: validate_port.py <session substring>
"""
import re
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402


def main():
    g, name, tif, sd, an = [s for s in rc.sessions([sys.argv[1]]) if s[0] == "6odor"][0]
    t0 = time.time()
    Y = rc.load_ds(tif)
    print("%s  T %d  load %.0fs" % (name, Y.shape[0], time.time() - t0), flush=True)

    fr = Y[[0, Y.shape[0] // 2, Y.shape[0] - 1]]
    a = rc.imfilter_replicate(fr)
    k = rc.PSF
    b = np.stack([ndi.correlate(f.astype(np.float64), k, mode="nearest", origin=(-1, -1)) for f in fr])
    b0 = np.stack([ndi.correlate(f.astype(np.float64), k, mode="nearest") for f in fr])
    print("1. FFT filter vs ndimage.correlate origin -1: max abs diff %.2e (origin 0: %.2e; data range %.1f)" % (
        np.abs(a - b).max(), np.abs(a - b0).max(), np.ptp(b)), flush=True)

    t0 = time.time()
    cn, pnr = rc.seeding_images(Y)
    print("   seeding images %.0fs" % (time.time() - t0), flush=True)
    P = rc.loadmat_img(sd / "pnr.mat", "pnr")
    C = rc.loadmat_img(sd / "Cn.mat", "Cn")
    print("   pnr.mat %s, Cn.mat %s, NaN in Cn.mat %d" % (P.shape, C.shape, int(np.isnan(C).sum())))
    d = pnr - P
    print("2. PNR port - pnr.mat: max abs %.4f, median abs %.5f, p99.9 abs %.4f (PNR median %.2f)" % (
        np.abs(d).max(), np.median(np.abs(d)), np.percentile(np.abs(d), 99.9), np.median(P)))
    for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        print("   (control) port shifted by %s: median abs %.4f" % (sh, np.median(np.abs(np.roll(pnr, sh, (0, 1)) - P))))
    up = rc.matlab_resize2x(cn)
    d = up - np.nan_to_num(C)
    print("3. imresize(Cn port) - Cn.mat: max abs %.4f, median abs %.5f, p99.9 abs %.4f" % (
        np.abs(d).max(), np.median(np.abs(d)), np.percentile(np.abs(d), 99.9)))
    d2 = cn - rc.block2(np.nan_to_num(C))
    print("   Cn port (256) - block-mean(Cn.mat): max abs %.4f, median abs %.5f" % (np.abs(d2).max(), np.median(np.abs(d2))))
    for kk in (25, 50, 100):
        print("   pk%d: port %.4f | block-mean(Cn.mat) %.4f | Cn.mat at 512 (11x11 at 512 res) %.4f" % (
            kk, rc.pk(cn, kk), rc.pk(rc.block2(np.nan_to_num(C)), kk), rc.pk(np.nan_to_num(C), kk)))

    t0 = time.time()
    pc, pp = rc.prepass_images(Y)
    est = rc.estimate_params(pc, pp)
    raw = rc.estimate_params(pc, pp, raw=True)
    logged = []
    for ln in open(sd / "agent_run.log", encoding="utf-8", errors="replace"):
        m = re.search(r"Estimated from Cn/pnr images: min_corr=([0-9]+(?:\.[0-9]+)?), min_pnr=([0-9]+(?:\.[0-9]+)?)", ln)
        if m:
            logged.append((float(m.group(1)), float(m.group(2))))
    print("4. pre-pass port (%.0fs): min_corr %.2f min_pnr %.1f (raw %.4f / %.3f) | logged %s" % (
        time.time() - t0, est[0], est[1], raw[0], raw[1], logged))
    print("   pre-pass Cn: zero pixels %.1f%%, <= 0.1 %.1f%%; p50 of the rest %.3f" % (
        100 * (pc == 0).mean(), 100 * (pc <= 0.1).mean(), np.median(pc[pc > 0.1])))
    print("   same frames at sig 3 (what the earlier simulation assumed): 85th pct of Cn > 0.1 = %.3f" % (
        rc.estimate_params(rc.prepass_images(Y, sig=3.0)[0], pp, raw=True)[0]))
    print("   full-movie seeding Cn: 85th pct of Cn > 0.1 = %.3f; PNR 85th pct %.2f" % rc.estimate_params(cn, pnr, raw=True))


if __name__ == "__main__":
    main()
