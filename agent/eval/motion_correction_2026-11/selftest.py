"""Self-test of motion_metrics on a synthetic movie with known displacements."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, shift as nd_shift

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_metrics as mm  # noqa: E402

rng = np.random.default_rng(0)
ly = lx = 512
base = np.zeros((ly, lx), np.float32)
for _ in range(400):
    y, x = rng.integers(20, ly - 20), rng.integers(20, lx - 20)
    base[y, x] += rng.uniform(0.5, 2.0)
base = gaussian_filter(base, 4.0)
base = 40 + 150 * base / base.max()

nt = 300
true_dy = np.zeros(nt, np.float32); true_dx = np.zeros(nt, np.float32)
true_dy[100:130] = 3.0; true_dx[100:130] = -2.0          # whole-pixel jump
true_dy[200:230] = 0.4; true_dx[200:230] = 1.6           # sub-pixel
true_dy[250:260] = -11.0; true_dx[250:260] = 7.0         # large
frames = np.empty((nt, ly, lx), np.uint8)
for k in range(nt):
    f = nd_shift(base, (true_dy[k], true_dx[k]), order=1, mode="nearest")
    f = f + rng.normal(0, 6.0, f.shape)
    frames[k] = np.clip(f, 0, 255).astype(np.uint8)

tmp = Path(tempfile.mkdtemp(prefix="mm_selftest_"))
tif = tmp / "synthetic.tif"
tifffile.imwrite(tif, frames, imagej=True)
rc = mm.main([str(tif), "--out", str(tmp / "out")])
z = np.load(tmp / "out" / "synthetic.npz")
err_y = z["dy"] - true_dy
err_x = z["dx"] - true_dx
print("return code", rc)
print("max |err| dy %.3f dx %.3f ; rms dy %.3f dx %.3f" % (
    np.abs(err_y).max(), np.abs(err_x).max(), np.sqrt((err_y ** 2).mean()), np.sqrt((err_x ** 2).mean())))
for lo, hi in ((0, 100), (100, 130), (200, 230), (250, 260)):
    print(f"  frames {lo}-{hi}: true ({true_dy[lo]:+.1f},{true_dx[lo]:+.1f})  "
          f"est ({z['dy'][lo:hi].mean():+.2f},{z['dx'][lo:hi].mean():+.2f})  "
          f"r_raw {z['r_raw'][lo:hi].mean():.3f}  r_al {z['r_al'][lo:hi].mean():.3f}  "
          f"block dy/dx median ({np.median(z['bdy'][lo:hi]):+.2f},{np.median(z['bdx'][lo:hi]):+.2f})")
ok = np.abs(err_y).max() < 0.35 and np.abs(err_x).max() < 0.35
print("SELFTEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
