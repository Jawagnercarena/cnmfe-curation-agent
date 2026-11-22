"""Montage of short raw-frame averages from one trial, to see the displacement by eye."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
session, trial = sys.argv[1], int(sys.argv[2])
starts = [int(s) for s in sys.argv[3:]]
navg = 8
raw = np.load(HERE / "raw" / session / f"trial{trial:03d}.npy", mmap_mode="r")
z = np.load(HERE / "reg" / session / "prod_s2p.npz")
files = sorted((HERE / "raw" / session).glob("trial*.npy"))
idx = [f.name for f in files].index(f"trial{trial:03d}.npy")
edges = np.concatenate([[0], np.cumsum(z["nper"])])
yoff = z["yoff"][edges[idx]:edges[idx + 1]]; xoff = z["xoff"][edges[idx]:edges[idx + 1]]
ncol = len(starts)
fig, ax = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.6))
ref = np.asarray(raw[starts[0]:starts[0] + navg], dtype=np.float32).mean(axis=0)
vmin, vmax = np.percentile(ref, 1), np.percentile(ref, 99.8)
for a, s in zip(np.atleast_1d(ax), starts):
    img = np.asarray(raw[s:s + navg], dtype=np.float32).mean(axis=0)
    a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    a.axhline(256, color="y", lw=0.4); a.axvline(256, color="y", lw=0.4)
    a.set_title(f"frames {s}-{s + navg - 1}\ns2p offset y {np.median(yoff[s:s + navg]):+.0f} x {np.median(xoff[s:s + navg]):+.0f}",
                fontsize=10)
    a.axis("off")
fig.suptitle(f"{session} trial {trial}: raw frames ({navg}-frame means, {navg / 30.2:.2f} s each), fixed crosshair at the centre")
fig.tight_layout()
out = HERE / "fig" / f"{session}_trial{trial:03d}_raw.png"
fig.savefig(out, dpi=75)
print(out)
