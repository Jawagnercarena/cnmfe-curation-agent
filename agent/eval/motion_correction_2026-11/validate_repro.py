"""Is the reproduced 'prod' registration the delivered movie, up to a constant offset?

For each reproduced 5-frame average and the delivered AVG5x frame with the same global index:
rigid displacement between the two (unrestricted search) and Pearson r on the overlap after
undoing it. Also tries neighbouring delivered frames (index -2..+2) to test the frame mapping.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion_metrics as mm  # noqa: E402

DATA = Path(r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew")
session, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "prod")
rep = tifffile.imread(HERE / "reg2" / session / f"{tag}.tif").astype(np.float32)
gidx = np.load(HERE / "reg2" / session / f"{tag}_s2p.npz")["gidx"]
name = "AVG5x-" + session
deliv = tifffile.memmap(DATA / name / (name + ".tif"), mode="r")
ly, lx = rep.shape[1:]
fy, fx = mm.freq_grids(ly, lx)
bp = mm.gauss_tf(fy, fx, 1.5) - mm.gauss_tf(fy, fx, 15.0)
win = mm.taper(ly, 48)[:, None] * mm.taper(lx, 48)[None, :]


def bandpass(x):
    return sfft.irfft2(sfft.rfft2(x) * bp, s=(ly, lx)).astype(np.float32)


def shift_and_r(a, b):
    """Displacement of a relative to b (unrestricted) and overlap r after undoing it."""
    cross = sfft.rfft2(a * win) * np.conj(sfft.rfft2(b * win))
    cross /= (np.abs(cross) + 1e-6)
    cc = sfft.irfft2(cross * bp, s=(ly, lx))
    py, px = np.unravel_index(cc.argmax(), cc.shape)
    dy = py - ly if py > ly // 2 else py
    dx = px - lx if px > lx // 2 else px
    ys_a = slice(max(dy, 0), ly + min(dy, 0)); ys_b = slice(max(-dy, 0), ly + min(-dy, 0))
    xs_a = slice(max(dx, 0), lx + min(dx, 0)); xs_b = slice(max(-dx, 0), lx + min(-dx, 0))
    aa = a[ys_a, xs_a][60:-60, 60:-60]; bb = b[ys_b, xs_b][60:-60, 60:-60]
    if aa.size < 0.2 * ly * lx:
        return dy, dx, np.nan
    aa = aa - aa.mean(); bb = bb - bb.mean()
    return dy, dx, float((aa * bb).sum() / max(np.sqrt((aa ** 2).sum() * (bb ** 2).sum()), 1e-12))


sel = np.arange(0, len(gidx), 3)
res = {k: [] for k in range(-2, 3)}
DY, DX = [], []
for i in sel:
    a = bandpass(rep[i])
    for k in range(-2, 3):
        j = int(np.clip(gidx[i] + k, 0, deliv.shape[0] - 1))
        dy, dx, r = shift_and_r(a, bandpass(np.asarray(deliv[j], dtype=np.float32)))
        res[k].append(r)
        if k == 0:
            DY.append(dy); DX.append(dx)
DY, DX = np.array(DY), np.array(DX)
print(f"{session} [{tag}] vs delivered: {len(sel)} averaged frames")
for k in range(-2, 3):
    r = np.array(res[k])
    print(f"  delivered index {k:+d}: aligned r median {np.nanmedian(r):.3f}  p10 {np.nanpercentile(r, 10):.3f}  "
          f"p90 {np.nanpercentile(r, 90):.3f}")
vals, cnt = np.unique(np.stack([DY, DX], 1), axis=0, return_counts=True)
order = np.argsort(-cnt)[:6]
print("  displacement reproduced-vs-delivered (dy, dx): most common " +
      ", ".join(f"({vals[o][0]:+d},{vals[o][1]:+d}) x{cnt[o]}" for o in order))
m = (np.abs(DY - np.median(DY)) <= 1) & (np.abs(DX - np.median(DX)) <= 1)
r0 = np.array(res[0])
print(f"  frames within 1 px of the median displacement ({np.median(DY):+.0f},{np.median(DX):+.0f}): {m.mean():.1%}; "
      f"their aligned r median {np.nanmedian(r0[m]):.3f}; the others {np.nanmedian(r0[~m]) if (~m).any() else float('nan'):.3f}")
np.savez(HERE / "reg2" / session / f"{tag}_vs_delivered.npz", sel=sel, dy=DY, dx=DX, r=r0)
