"""Score AVG5x movies: mc_rule's per-frame metric plus a noise-robust block metric.

    python score_movies.py --out <csv> TIF [TIF ...]
    python score_movies.py --pair OLD NEW          (template shift between two movies)

Read-only. Per movie:
  frame metric  = mc_rule.movie_quality definition (band-pass 2-15 px, Pearson with the
                  movie's own template, centre crop 48 px): quiet = p90 of r, bad < 0.5 quiet,
                  degraded < 0.75 quiet.
  block metric  = same, but on means of 20 consecutive averaged frames (100 raw frames),
                  which averages pixel noise away so misregistration remains: block r
                  median / p5 / min and the share of blocks below 0.8 x their own p90.
  brightness    = mean pixel value (8-bit movie) on 600 sampled frames.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import fft as sfft

M = 48
BLOCK = 20


def make_band(ly, lx):
    fy = np.fft.fftfreq(ly)[:, None].astype(np.float32)
    fx = np.fft.rfftfreq(lx)[None, :].astype(np.float32)

    def gauss(s):
        return np.exp(-2.0 * (np.pi ** 2) * (s ** 2) * (fy ** 2 + fx ** 2)).astype(np.float32)

    return gauss(2.0) - gauss(15.0)


def bandpass(x, band, shape):
    return sfft.irfft2(sfft.rfft2(x, workers=8) * band, s=shape, workers=8).astype(np.float32)


def pearson(a, t):
    ac = a[:, M:-M, M:-M]
    tc = t[M:-M, M:-M]
    ac = ac - ac.mean(axis=(1, 2), keepdims=True)
    tc = tc - tc.mean()
    num = (ac * tc[None]).sum(axis=(1, 2))
    den = np.sqrt((ac ** 2).sum(axis=(1, 2)) * (tc ** 2).sum())
    return (num / np.maximum(den, 1e-12)).astype(np.float32)


def score(path):
    mov = tifffile.memmap(path, mode="r")
    nt, ly, lx = mov.shape
    band = make_band(ly, lx)
    samp = np.unique(np.linspace(0, nt - 1, min(nt, 600)).astype(int))
    raw = np.stack([np.asarray(mov[k], np.float32) for k in samp])
    bright = float(raw.mean())
    stack = bandpass(raw, band, (ly, lx))
    r0 = pearson(stack, stack.mean(axis=0))
    template = stack[r0 >= np.median(r0)].mean(axis=0)
    r = np.zeros(nt, np.float32)
    nb = nt // BLOCK
    rb = np.zeros(nb, np.float32)
    for a in range(0, nb * BLOCK, BLOCK * 16):
        b = min(nb * BLOCK, a + BLOCK * 16)
        x = np.asarray(mov[a:b], dtype=np.float32)
        r[a:b] = pearson(bandpass(x, band, (ly, lx)), template)
        means = x.reshape(-1, BLOCK, ly, lx).mean(axis=1)
        rb[a // BLOCK:b // BLOCK] = pearson(bandpass(means, band, (ly, lx)), template)
    if nb * BLOCK < nt:
        r[nb * BLOCK:] = pearson(bandpass(np.asarray(mov[nb * BLOCK:], np.float32), band, (ly, lx)), template)
    q = float(np.percentile(r, 90))
    qb = float(np.percentile(rb, 90))
    return dict(movie=path.name, folder=str(path.parent), frames=nt, brightness=round(bright, 2),
                quiet_r=round(q, 3), bad_pct=round(100 * float((r < 0.5 * q).mean()), 2),
                degraded_pct=round(100 * float((r < 0.75 * q).mean()), 2),
                block_r_median=round(float(np.median(rb)), 3), block_r_p5=round(float(np.percentile(rb, 5)), 3),
                block_r_min=round(float(rb.min()), 3),
                block_low_pct=round(100 * float((rb < 0.8 * qb).mean()), 2)), template


def template_shift(t1, t2):
    """Integer shift (dy, dx) of t2 relative to t1 by phase correlation, and r after the shift."""
    f = np.fft.fft2(t1) * np.conj(np.fft.fft2(t2))
    cc = np.real(np.fft.ifft2(f / np.maximum(np.abs(f), 1e-12)))
    dy, dx = np.unravel_index(np.argmax(cc), cc.shape)
    ly, lx = t1.shape
    dy = dy - ly if dy > ly // 2 else dy
    dx = dx - lx if dx > lx // 2 else dx
    t2s = np.roll(t2, (dy, dx), axis=(0, 1))
    c = np.corrcoef(t1[M + 20:-M - 20, M + 20:-M - 20].ravel(), t2s[M + 20:-M - 20, M + 20:-M - 20].ravel())[0, 1]
    return int(dy), int(dx), float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pair", nargs=2, type=Path)
    ap.add_argument("tifs", nargs="*", type=Path)
    a = ap.parse_args()
    if a.pair:
        (s1, t1), (s2, t2) = score(a.pair[0]), score(a.pair[1])
        for s in (s1, s2):
            print(s)
        dy, dx, c = template_shift(t1, t2)
        print(f"template shift new vs old: dy {dy:+d} dx {dx:+d}; template r after shift {c:.3f}")
        return
    rows = []
    for p in a.tifs:
        s, _ = score(p)
        rows.append(s)
        print(s, flush=True)
    if a.out:
        with open(a.out, "w", newline="", encoding="ascii") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
