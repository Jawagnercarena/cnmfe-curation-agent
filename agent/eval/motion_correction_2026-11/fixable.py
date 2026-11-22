"""How much of the motion in a raw sample could ANY x-y registration undo?

For every raw 30 Hz frame of the sample trials: find the best rigid shift against the clean
session template with an UNRESTRICTED search (whole frame), then correlate the shifted frame
with the template over their true overlap (no wrap-around). A frame whose best-shift
correlation is still far below the session's quiet-frame level cannot be repaired by x-y
registration (focal-plane change, defocus, within-frame warping).

Scratch only. Needs scratch/raw/<session>/trial*.npy and scratch/reg/<session>/prod.npz.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion_metrics as mm  # noqa: E402

FS = 30.23
session = sys.argv[1]
files = sorted((HERE / "raw" / session).glob("trial*.npy"))
tmpl = np.load(HERE / "reg" / session / "prod.npz")["template"].astype(np.float32)
ly, lx = tmpl.shape
fy, fx = mm.freq_grids(ly, lx)
bp = mm.gauss_tf(fy, fx, 2.5) - mm.gauss_tf(fy, fx, 15.0)
win = mm.taper(ly, 48)[:, None] * mm.taper(lx, 48)[None, :]
t_bp = sfft.irfft2(sfft.rfft2(tmpl) * bp, s=(ly, lx)).astype(np.float32)
tf_c = np.conj(sfft.rfft2(t_bp * win))


def overlap_r(a: np.ndarray, t: np.ndarray, dy: int, dx: int) -> float:
    """Pearson r of frame a (content displaced by dy, dx) and template t on their overlap."""
    ys_a = slice(max(dy, 0), ly + min(dy, 0)); ys_t = slice(max(-dy, 0), ly + min(-dy, 0))
    xs_a = slice(max(dx, 0), lx + min(dx, 0)); xs_t = slice(max(-dx, 0), lx + min(-dx, 0))
    aa = a[ys_a, xs_a]; tt = t[ys_t, xs_t]
    aa = aa[24:-24, 24:-24]; tt = tt[24:-24, 24:-24]
    if aa.size < 0.3 * ly * lx:
        return np.nan
    aa = aa - aa.mean(); tt = tt - tt.mean()
    return float((aa * tt).sum() / max(np.sqrt((aa ** 2).sum() * (tt ** 2).sum()), 1e-12))


DY, DX, R0, RAL, TT, TR = [], [], [], [], [], []
for ti, f in enumerate(files):
    raw = np.load(f).astype(np.float32)
    n = raw.shape[0]
    for a in range(0, n, 96):
        x = raw[a:a + 96]
        x_bp = sfft.irfft2(sfft.rfft2(x, workers=8) * bp, s=(ly, lx), workers=8).astype(np.float32)
        cross = sfft.rfft2(x_bp * win, workers=8) * tf_c
        cross /= (np.abs(cross) + 1e-6)
        cc = sfft.irfft2(cross * bp, s=(ly, lx), workers=8)
        flat = cc.reshape(len(x), -1).argmax(axis=1)
        py, px = np.divmod(flat, lx)
        dy = np.where(py > ly // 2, py - ly, py); dx = np.where(px > lx // 2, px - lx, px)
        for k in range(len(x)):
            R0.append(overlap_r(x_bp[k], t_bp, 0, 0))
            RAL.append(overlap_r(x_bp[k], t_bp, int(dy[k]), int(dx[k])))
        DY.extend(dy.tolist()); DX.extend(dx.tolist())
        TT.extend(((np.arange(a, a + len(x))) / FS).tolist()); TR.extend([ti] * len(x))
DY, DX, R0, RAL, TT, TR = map(np.array, (DY, DX, R0, RAL, TT, TR))
np.savez_compressed(HERE / "reg" / session / "fixable.npz", dy=DY, dx=DX, r0=R0, ral=RAL, t=TT, trial=TR)

ok = np.isfinite(RAL)
q = np.nanpercentile(RAL, 90)                      # quiet-frame level of this session
rel = RAL / q
mdy, mdx = np.median(DY[ok & (rel > 0.8)]), np.median(DX[ok & (rel > 0.8)])
disp = np.hypot(DY - mdy, DX - mdx)                # displacement from the modal (quiet) position
lost = (~ok) | (rel < 0.5)
weak = ok & (rel >= 0.5) & (rel < 0.75)
good = ok & (rel >= 0.75)
print(f"=== {session}: {len(RAL)} raw frames from {len(files)} trials; quiet-frame r (p90) = {q:.3f}")
print(f"  displacement from the modal position (frames whose image is recognisable, rel >= 0.5): "
      f"p50 {np.percentile(disp[~lost], 50):.1f}  p95 {np.percentile(disp[~lost], 95):.1f}  "
      f"p99 {np.percentile(disp[~lost], 99):.1f}  max {disp[~lost].max():.0f} px")
print(f"  frames by best-shift match: good (>=0.75 of quiet) {good.mean():.1%}   "
      f"degraded (0.5-0.75) {weak.mean():.1%}   not recognisable at ANY shift (<0.5) {lost.mean():.1%}")
for lim in (26, 51):
    far = (~lost) & ((np.abs(DY - mdy) > lim) | (np.abs(DX - mdx) > lim))
    print(f"  recognisable but more than {lim} px (one axis) from the modal position: {far.mean():.1%}")
bins = np.arange(0, 19.0, 1.0)
prof = [lost[(TT >= b) & (TT < b + 1)].mean() for b in bins[:-1]]
print("  'not recognisable' fraction by second in trial: " + " ".join(f"{p:.2f}" for p in prof))
prof = [np.median(disp[(~lost) & (TT >= b) & (TT < b + 1)]) for b in bins[:-1]]
print("  median displacement (px) by second in trial:    " + " ".join(f"{p:4.0f}" for p in prof))
