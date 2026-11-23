"""Residual-motion metrics of a registered, frame-averaged movie.

Read-only on the tif. Writes one npz per session into --out (scratch).

Per frame:
  dy, dx        rigid displacement of the frame relative to the session template
                (phase correlation, sub-pixel by a parabola fit), pixels
  pc            phase-correlation peak value
  r_raw         Pearson r, band-passed frame vs template, centre crop
  r_al          same after undoing (dy, dx) with a Fourier shift
  bdy, bdx      displacement of each block of a 4x4 grid (128 px) vs the template,
                measured AFTER undoing (dy, dx): the non-rigid remainder
  bpc           phase-correlation peak per block
  br            Pearson r per block after undoing (dy, dx)
  mean_f        mean raw intensity
  e_hi, e_mid   variance of the 1-2 px band and of the 2-15 px band (centre crop)
Per session:
  template      mean of the better-correlated half of ~600 sampled frames
  seg_means     mean image of each of NSEG equal time segments
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import fft as sfft

NSEG = 24
MAXSH = 40          # search range, px
MARGIN = 48         # crop for correlations, px
NB = 4              # blocks per side
BMAX = 16           # block search range, px
WORKERS = 8


def freq_grids(ly: int, lx: int):
    fy = np.fft.fftfreq(ly)[:, None].astype(np.float32)
    fx = np.fft.rfftfreq(lx)[None, :].astype(np.float32)
    return fy, fx


def gauss_tf(fy, fx, sigma: float):
    """Transfer function of a real-space Gaussian of the given sigma (px)."""
    return np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2)).astype(np.float32)


def taper(n: int, edge: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(edge) + 0.5) / edge)
    w[:edge] = ramp
    w[-edge:] = ramp[::-1]
    return w


def peak_subpixel(cc: np.ndarray, maxsh: int):
    """cc: (B, Ly, Lx) correlation surfaces. Returns dy, dx, peak (sub-pixel)."""
    idx = np.r_[0:maxsh + 1, -maxsh:0]
    order = np.argsort(np.where(idx > maxsh, idx - cc.shape[1], idx))
    iy = np.r_[0:maxsh + 1, cc.shape[1] - maxsh:cc.shape[1]]
    ix = np.r_[0:maxsh + 1, cc.shape[2] - maxsh:cc.shape[2]]
    sy = np.r_[0:maxsh + 1, -maxsh:0]
    oy = np.argsort(sy)
    win = cc[:, iy][:, :, ix][:, oy][:, :, oy]          # (B, 2m+1, 2m+1), index = shift + m
    del order
    b = win.shape[0]
    n = win.shape[1]
    flat = win.reshape(b, -1).argmax(axis=1)
    py, px = np.divmod(flat, n)
    peak = win[np.arange(b), py, px]
    dy = (py - maxsh).astype(np.float32)
    dx = (px - maxsh).astype(np.float32)
    ar = np.arange(b)
    oky = (py > 0) & (py < n - 1)
    okx = (px > 0) & (px < n - 1)
    pym = np.clip(py - 1, 0, n - 1); pyp = np.clip(py + 1, 0, n - 1)
    pxm = np.clip(px - 1, 0, n - 1); pxp = np.clip(px + 1, 0, n - 1)
    c0 = peak
    cym, cyp = win[ar, pym, px], win[ar, pyp, px]
    cxm, cxp = win[ar, py, pxm], win[ar, py, pxp]
    deny = (cym - 2 * c0 + cyp)
    denx = (cxm - 2 * c0 + cxp)
    with np.errstate(divide="ignore", invalid="ignore"):
        offy = np.where(oky & (deny < 0), 0.5 * (cym - cyp) / deny, 0.0)
        offx = np.where(okx & (denx < 0), 0.5 * (cxm - cxp) / denx, 0.0)
    dy += np.clip(offy, -0.5, 0.5).astype(np.float32)
    dx += np.clip(offx, -0.5, 0.5).astype(np.float32)
    return dy, dx, peak.astype(np.float32)


def pearson_crop(a: np.ndarray, t: np.ndarray, m: int) -> np.ndarray:
    """a: (B, Ly, Lx), t: (Ly, Lx). Pearson r in the centre crop."""
    ac = a[:, m:-m, m:-m]
    tc = t[m:-m, m:-m]
    ac = ac - ac.mean(axis=(1, 2), keepdims=True)
    tc = tc - tc.mean()
    num = (ac * tc[None]).sum(axis=(1, 2))
    den = np.sqrt((ac ** 2).sum(axis=(1, 2)) * (tc ** 2).sum())
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tif", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (test)")
    args = ap.parse_args(argv)

    t0 = time.time()
    mov = tifffile.memmap(args.tif, mode="r")
    nt, ly, lx = mov.shape
    if args.limit:
        nt = min(nt, args.limit)
    name = args.tif.stem
    print(f"[{name}] {nt} frames {ly}x{lx} {mov.dtype}", flush=True)

    fy, fx = freq_grids(ly, lx)
    bp = gauss_tf(fy, fx, 2.0) - gauss_tf(fy, fx, 15.0)      # 2-15 px band
    hi = gauss_tf(fy, fx, 1.0) - gauss_tf(fy, fx, 2.0)       # 1-2 px band
    sm = bp                                                  # band-limited phase correlation (weight = the 2-15 px band)
    win = taper(ly, MARGIN)[:, None] * taper(lx, MARGIN)[None, :]

    bs = ly // NB
    bfy, bfx = freq_grids(bs, bs)
    bsm = gauss_tf(bfy, bfx, 2.0) - gauss_tf(bfy, bfx, 15.0)
    bwin = taper(bs, 16)[:, None] * taper(bs, 16)[None, :]

    # ---- template: mean of the better half of ~600 sampled frames
    samp = np.unique(np.linspace(0, nt - 1, min(nt, 600)).astype(int))
    stack = np.empty((len(samp), ly, lx), dtype=np.float32)
    for i, k in enumerate(samp):
        stack[i] = mov[k]
    f = sfft.rfft2(stack, workers=WORKERS)
    stack_bp = sfft.irfft2(f * bp, s=(ly, lx), workers=WORKERS).astype(np.float32)
    t_bp0 = stack_bp.mean(axis=0)
    r0 = pearson_crop(stack_bp, t_bp0, MARGIN)
    keep = r0 >= np.median(r0)
    template_raw = stack[keep].mean(axis=0)
    t_bp = stack_bp[keep].mean(axis=0)
    del stack, stack_bp, f
    tf_c = np.conj(sfft.rfft2(t_bp * win, workers=WORKERS))
    t_blocks = t_bp.reshape(NB, bs, NB, bs).transpose(0, 2, 1, 3) * bwin
    tbf_c = np.conj(sfft.rfft2(t_blocks, workers=WORKERS))
    print(f"[{name}] template from {int(keep.sum())} of {len(samp)} sampled frames "
          f"(sample r median {np.median(r0):.3f})", flush=True)

    dy = np.zeros(nt, np.float32); dx = np.zeros(nt, np.float32); pc = np.zeros(nt, np.float32)
    r_raw = np.zeros(nt, np.float32); r_al = np.zeros(nt, np.float32)
    bdy = np.zeros((nt, NB, NB), np.float32); bdx = np.zeros((nt, NB, NB), np.float32)
    bpc = np.zeros((nt, NB, NB), np.float32); br = np.zeros((nt, NB, NB), np.float32)
    mean_f = np.zeros(nt, np.float32); e_hi = np.zeros(nt, np.float32); e_mid = np.zeros(nt, np.float32)
    seg_sum = np.zeros((NSEG, ly, lx), np.float64); seg_n = np.zeros(NSEG, np.int64)
    seg_of = np.minimum((np.arange(nt) * NSEG) // nt, NSEG - 1)

    for a in range(0, nt, args.chunk):
        b = min(nt, a + args.chunk)
        x = np.asarray(mov[a:b], dtype=np.float32)
        mean_f[a:b] = x.mean(axis=(1, 2))
        for s in np.unique(seg_of[a:b]):
            sel = seg_of[a:b] == s
            seg_sum[s] += x[sel].sum(axis=0, dtype=np.float64)
            seg_n[s] += int(sel.sum())
        xf = sfft.rfft2(x, workers=WORKERS)
        x_bp = sfft.irfft2(xf * bp, s=(ly, lx), workers=WORKERS).astype(np.float32)
        x_hi = sfft.irfft2(xf * hi, s=(ly, lx), workers=WORKERS).astype(np.float32)
        e_hi[a:b] = x_hi[:, MARGIN:-MARGIN, MARGIN:-MARGIN].var(axis=(1, 2))
        e_mid[a:b] = x_bp[:, MARGIN:-MARGIN, MARGIN:-MARGIN].var(axis=(1, 2))
        del x_hi
        r_raw[a:b] = pearson_crop(x_bp, t_bp, MARGIN)

        # rigid phase correlation on the tapered band-passed frame
        wf = sfft.rfft2(x_bp * win, workers=WORKERS)
        cross = wf * tf_c
        cross /= (np.abs(cross) + 1e-6)
        cc = sfft.irfft2(cross * sm, s=(ly, lx), workers=WORKERS)
        dy[a:b], dx[a:b], pc[a:b] = peak_subpixel(cc, MAXSH)
        del cc, cross, wf

        # undo the rigid displacement, correlate again
        ph = np.exp(2j * np.pi * (dy[a:b, None, None] * fy[None] + dx[a:b, None, None] * fx[None]))
        x_al = sfft.irfft2(xf * bp * ph, s=(ly, lx), workers=WORKERS).astype(np.float32)
        r_al[a:b] = pearson_crop(x_al, t_bp, MARGIN)
        del ph, xf

        # block-wise displacement (4x4 grid) of the RIGIDLY ALIGNED frame, so the
        # values are the non-rigid remainder (the fixed window biases a block
        # estimate toward zero in proportion to its size; small after alignment)
        xb = x_al.reshape(b - a, NB, bs, NB, bs).transpose(0, 1, 3, 2, 4)
        tb = t_bp.reshape(NB, bs, NB, bs).transpose(0, 2, 1, 3)
        xc = xb - xb.mean(axis=(3, 4), keepdims=True)
        tc = tb - tb.mean(axis=(2, 3), keepdims=True)
        br[a:b] = (xc * tc[None]).sum(axis=(3, 4)) / np.maximum(
            np.sqrt((xc ** 2).sum(axis=(3, 4)) * (tc ** 2).sum(axis=(2, 3))[None]), 1e-12)
        del xc, tc, x_al
        xb = xb * bwin
        xbf = sfft.rfft2(xb, workers=WORKERS)
        cross = xbf * tbf_c[None]
        cross /= (np.abs(cross) + 1e-6)
        ccb = sfft.irfft2(cross * bsm, s=(bs, bs), workers=WORKERS)
        by, bx, bp_ = peak_subpixel(ccb.reshape(-1, bs, bs), BMAX)
        bdy[a:b] = by.reshape(b - a, NB, NB)
        bdx[a:b] = bx.reshape(b - a, NB, NB)
        bpc[a:b] = bp_.reshape(b - a, NB, NB)
        del xb, xbf, cross, ccb, x_bp, x
        if (a // args.chunk) % 20 == 0:
            print(f"[{name}] {b}/{nt} frames  {time.time() - t0:.0f}s", flush=True)

    seg_means = (seg_sum / np.maximum(seg_n, 1)[:, None, None]).astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / (name + ".npz")
    np.savez_compressed(out, name=name, nt=nt, dy=dy, dx=dx, pc=pc, r_raw=r_raw, r_al=r_al,
                        bdy=bdy, bdx=bdx, bpc=bpc, br=br, mean_f=mean_f, e_hi=e_hi, e_mid=e_mid,
                        template=template_raw.astype(np.float32), seg_means=seg_means,
                        seg_n=seg_n)
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
