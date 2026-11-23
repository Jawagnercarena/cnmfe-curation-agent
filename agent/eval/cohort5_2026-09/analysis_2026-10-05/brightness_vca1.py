"""Read-only brightness check: 20 evenly spaced raw frames per 6odorDualDiffRew tif
plus the saved per-frame background mean. Nothing is written."""
import os
import numpy as np
from scipy.io import loadmat
import tifffile

ROOT = r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew"
NS = 20
FRAME_BYTES = 512 * 512

print("session | frames | raw mean | raw p50 | raw p99 | %px==255 | Ybg mean | Ybg std")
for d in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, d)
    if not os.path.isdir(sd) or d.startswith("."):
        continue
    tif = os.path.join(sd, d + ".tif")
    if not os.path.exists(tif):
        continue
    ybg_mean = ybg_std = None
    ybg = os.path.join(sd, "Ybg_mean.mat")
    if os.path.exists(ybg):
        v = np.asarray(loadmat(ybg)["mean_Ybg"], dtype=float).ravel()
        ybg_mean, ybg_std = float(v.mean()), float(v.std())
    n = None
    frames = []
    try:
        with tifffile.TiffFile(tif) as t:
            n = int(t.imagej_metadata.get("images", 0)) if t.imagej_metadata else len(t.pages)
            off = t.pages[0].dataoffsets[0]
        idx = np.linspace(0, n - 1, NS).astype(int)
        with open(tif, "rb") as fh:
            for i in idx:
                fh.seek(off + int(i) * FRAME_BYTES)
                frames.append(np.frombuffer(fh.read(FRAME_BYTES), dtype=np.uint8).astype(np.float32))
        fr = np.stack(frames)
        raw_mean, p50, p99 = float(fr.mean()), float(np.percentile(fr, 50)), float(np.percentile(fr, 99))
        sat = float((fr >= 255).mean() * 100)
    except Exception as e:
        raw_mean = p50 = p99 = sat = None
        print("read failed", d, repr(e))
    fmt = lambda v, f: "-" if v is None else format(v, f)
    print(" | ".join([d, fmt(n, "d"), fmt(raw_mean, ".1f"), fmt(p50, ".0f"), fmt(p99, ".0f"),
                      fmt(sat, ".2f"), fmt(ybg_mean, ".1f"), fmt(ybg_std, ".2f")]))
