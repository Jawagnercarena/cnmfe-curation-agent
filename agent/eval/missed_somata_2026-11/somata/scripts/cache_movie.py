import sys, time, h5py, numpy as np
from pathlib import Path
mat, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with h5py.File(mat, "r") as h:
    Y = h["Y"]; T = Y.shape[0]
    T2 = T // 2
    ds = np.zeros((T2, 256, 256), np.float32)
    s = np.zeros((512, 512), np.float64); mx = np.zeros((512, 512), np.uint8)
    sat = np.zeros((512, 512), np.int32); mn = np.full((512, 512), 255, np.uint8)
    B = 1024
    for a in range(0, T, B):
        blk = Y[a:min(a+B, T)]                     # (b, d2, d1) -- MATLAB order reversed
        blk = np.ascontiguousarray(blk.transpose(0, 2, 1))   # -> (b, d1, d2)
        s += blk.sum(0, dtype=np.float64); mx = np.maximum(mx, blk.max(0)); mn = np.minimum(mn, blk.min(0))
        sat += (blk == 255).sum(0)
        f = blk.astype(np.float32)
        f = f.reshape(f.shape[0], 256, 2, 256, 2).mean(axis=(2, 4))
        n2 = f.shape[0] // 2
        i0 = a // 2
        ds[i0:i0+n2] = f[:2*n2].reshape(n2, 2, 256, 256).mean(1)
        print(a, round(time.time()-t0, 1), flush=True)
np.save(out / "ds.npy", ds)
np.savez(out / "imgs.npz", mean=s / T, max=mx, min=mn, sat=sat, T=T)
print("done", T, round(time.time()-t0, 1))
