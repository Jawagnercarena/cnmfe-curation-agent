"""Compare two averaged movies of the same session frame by frame (read-only).

    python compare_movies.py <movie A tif> <movie B tif> [--step 10] [--out <npz>]

For every --step-th frame index present in both: are the two frames byte-identical, Pearson r of
the raw pixels (inner crop), and the rigid displacement of A relative to B (unrestricted search on
the band-passed frames) with r on the overlap after undoing it. Used to ask whether a re-run with
unchanged settings reproduces a delivered movie, and how far a re-registered movie sits from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion_metrics as mm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    A = tifffile.memmap(args.a, mode="r")
    B = tifffile.memmap(args.b, mode="r")
    print(f"A {args.a.name}: {A.shape} {A.dtype}\nB {args.b.name}: {B.shape} {B.dtype}")
    n = min(A.shape[0], B.shape[0])
    ly, lx = A.shape[1:]
    fy, fx = mm.freq_grids(ly, lx)
    bp = mm.gauss_tf(fy, fx, 1.5) - mm.gauss_tf(fy, fx, 15.0)
    win = mm.taper(ly, 48)[:, None] * mm.taper(lx, 48)[None, :]

    def bandpass(x):
        return sfft.irfft2(sfft.rfft2(x) * bp, s=(ly, lx)).astype(np.float32)

    sel = np.arange(0, n, args.step)
    same = np.zeros(len(sel), bool)
    r_px = np.zeros(len(sel), np.float32)
    r_al = np.full(len(sel), np.nan, np.float32)
    dy = np.zeros(len(sel), np.int32)
    dx = np.zeros(len(sel), np.int32)
    for i, k in enumerate(sel):
        a = np.asarray(A[k])
        b = np.asarray(B[k])
        same[i] = np.array_equal(a, b)
        af = a[40:-40, 40:-40].astype(np.float64).ravel()
        bf = b[40:-40, 40:-40].astype(np.float64).ravel()
        r_px[i] = np.corrcoef(af, bf)[0, 1]
        ab, bb = bandpass(a.astype(np.float32)), bandpass(b.astype(np.float32))
        cross = sfft.rfft2(ab * win) * np.conj(sfft.rfft2(bb * win))
        cross /= (np.abs(cross) + 1e-6)
        cc = sfft.irfft2(cross * bp, s=(ly, lx))
        py, px = np.unravel_index(cc.argmax(), cc.shape)
        dy[i] = py - ly if py > ly // 2 else py
        dx[i] = px - lx if px > lx // 2 else px
        ya = slice(max(dy[i], 0), ly + min(dy[i], 0)); yb = slice(max(-dy[i], 0), ly + min(-dy[i], 0))
        xa = slice(max(dx[i], 0), lx + min(dx[i], 0)); xb = slice(max(-dx[i], 0), lx + min(-dx[i], 0))
        aa = ab[ya, xa][60:-60, 60:-60]
        b2 = bb[yb, xb][60:-60, 60:-60]
        if aa.size >= 0.2 * ly * lx:
            aa = aa - aa.mean(); b2 = b2 - b2.mean()
            r_al[i] = (aa * b2).sum() / max(np.sqrt((aa ** 2).sum() * (b2 ** 2).sum()), 1e-12)

    print(f"frames compared: {len(sel)} (every {args.step}th of {n}); frame counts A {A.shape[0]}, B {B.shape[0]}")
    print(f"byte-identical frames: {100 * same.mean():.1f}%")
    print(f"pixel r (raw frames): median {np.median(r_px):.4f}  p05 {np.percentile(r_px, 5):.4f}  "
          f"min {r_px.min():.4f}; frames with r > 0.98: {100 * (r_px > 0.98).mean():.1f}%")
    vals, cnt = np.unique(np.stack([dy, dx], 1), axis=0, return_counts=True)
    order = np.argsort(-cnt)[:6]
    print("displacement of A relative to B (dy, dx), most common: "
          + ", ".join(f"({vals[o][0]:+d},{vals[o][1]:+d}) x{cnt[o]}" for o in order))
    zero = (dy == 0) & (dx == 0)
    print(f"zero displacement: {100 * zero.mean():.1f}%; within 1 px of the median "
          f"({np.median(dy):+.0f},{np.median(dx):+.0f}): "
          f"{100 * ((np.abs(dy - np.median(dy)) <= 1) & (np.abs(dx - np.median(dx)) <= 1)).mean():.1f}%; "
          f"r after undoing it: median {np.nanmedian(r_al):.4f}")
    if args.out:
        np.savez_compressed(args.out, sel=sel, same=same, r_px=r_px, r_al=r_al, dy=dy, dx=dx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
