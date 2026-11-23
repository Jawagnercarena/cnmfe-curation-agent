"""Read-only: does the Cn port depend on movie length?  Compare with Cn.mat."""
import sys, time
import numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi
sys.path.insert(0, sys.argv[1])
import video_compare as vc
sd = vc.OLD_DIR / "AVG5x-TSeries-090826-pnb102-497um-3z-000"
tif = sd / (sd.name + ".tif")
nf, off = vc.layout(tif)
C = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float))
P = np.nan_to_num(np.asarray(loadmat(str(sd / "pnr.mat"))["pnr"], dtype=float))
print("Cn.mat shape", C.shape, "p50/p90 %.3f/%.3f" % (np.percentile(C[20:-20,20:-20],50), np.percentile(C[20:-20,20:-20],90)), "pnr.mat", P.shape)
for n in (1000, 4000, nf):
    st = max(0, nf // 2 - n // 2)
    blk = vc.read_block(tif, off, st, min(n, nf - st))
    t0 = time.time()
    cn, pnr = vc.cn_pnr(blk)
    c = cn[10:-10, 10:-10]
    up = np.kron(cn, np.ones((2, 2)))
    r = np.corrcoef(up[20:-20, 20:-20].ravel(), C[20:-20, 20:-20].ravel())[0, 1] if C.shape == (512, 512) else float("nan")
    print("n=%5d  Cn p50/p90 %.3f/%.3f  %%>0.8 %.1f  PNR p90 %.2f  pixel r vs Cn.mat %.3f  (%.0fs)" % (
        len(blk), np.percentile(c, 50), np.percentile(c, 90), 100 * (c > 0.8).mean(), np.percentile(pnr[10:-10, 10:-10], 90), r, time.time() - t0), flush=True)
    del blk
# motion detector sanity: shift the block mean by known amounts
z = np.load(sys.argv[1] + "/vtest/img/090826-pnb102-497um-3z-000.npz")
ref = z["block_mean"]
for s in (0.3, 0.7, 1.3, 3.0):
    f = ndi.shift(ref, (s, 0), order=3, mode="nearest") + np.random.default_rng(0).normal(0, 12 / np.sqrt(5), ref.shape).astype(np.float32)
    print("true shift %.1f px -> estimated %.2f" % (s, vc.shifts([f], ref)[0]))
