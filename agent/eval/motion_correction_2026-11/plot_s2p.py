"""Figure of one suite2p re-registration of a raw sample: offsets, corrXY, reference, mean image."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
session, tag = sys.argv[1], sys.argv[2]
z = np.load(HERE / "reg" / session / f"{tag}_s2p.npz")
yoff, xoff, cxy = z["yoff"], z["xoff"], z["corrXY"]
nper = z["nper"]
edges = np.concatenate([[0], np.cumsum(nper)])
fig, ax = plt.subplots(3, 2, figsize=(16, 11), gridspec_kw=dict(width_ratios=[2.2, 1]))
t = np.arange(len(yoff))
ax[0, 0].plot(t, yoff, lw=0.3, label="yoff"); ax[0, 0].plot(t, xoff, lw=0.3, label="xoff")
for e in edges:
    ax[0, 0].axvline(e, color="k", lw=0.3, alpha=0.4)
ax[0, 0].set_ylabel("suite2p rigid offset (px)"); ax[0, 0].legend(loc="upper right")
ax[0, 0].set_title(f"{session} [{tag}]: raw 30 Hz frames, {len(nper)} sample trials (vertical lines = trial edges)")
ax[1, 0].plot(t, cxy, lw=0.3, color="C3"); ax[1, 0].set_ylabel("corrXY (phase-corr peak)")
a, b = edges[len(nper) // 2], edges[len(nper) // 2 + 1]
ax[2, 0].plot(np.arange(a, b), yoff[a:b], lw=0.6, label="yoff"); ax[2, 0].plot(np.arange(a, b), xoff[a:b], lw=0.6, label="xoff")
ax[2, 0].set_title("one trial, zoomed"); ax[2, 0].set_xlabel("raw frame"); ax[2, 0].legend()
for k, (img, ttl) in enumerate(((z["ref"], "suite2p reference image"), (z["mean_img"], "mean of registered sample"))):
    ax[k, 1].imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
    ax[k, 1].set_title(ttl); ax[k, 1].axis("off")
ax[2, 1].hist2d(xoff, yoff, bins=np.arange(-52, 53, 2), cmap="magma", norm=matplotlib.colors.LogNorm())
ax[2, 1].set_title("offsets (x, y), log counts"); ax[2, 1].set_aspect("equal")
fig.tight_layout()
out = HERE / "fig" / f"{session}_{tag}_s2p.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=70)
print(out)
