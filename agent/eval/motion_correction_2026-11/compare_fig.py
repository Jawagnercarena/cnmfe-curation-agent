"""Before/after figure for one session's raw sample: production settings vs an alternative."""
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
session, alt = sys.argv[1], sys.argv[2]
d = HERE / "reg2" / session
tags = ["prod", alt]
fig, ax = plt.subplots(2, 4, figsize=(21, 10.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1.5]))
zs = {t: np.load(d / f"{t}.npz") for t in tags}
ss = {t: np.load(d / f"{t}_s2p.npz") for t in tags}
movs = {t: tifffile.memmap(d / f"{t}.tif", mode="r") for t in tags}
qs = {t: np.percentile(zs[t]["r_raw"], 90) for t in tags}
# an averaged frame that is bad under production settings (median-bad, not the worst)
bad = np.where(zs["prod"]["r_raw"] < 0.5 * qs["prod"])[0]
pick = int(bad[len(bad) // 2]) if len(bad) else 0
nper = ss["prod"]["nper"]
ntr = len(nper)
per = len(zs["prod"]["r_raw"]) // ntr
for row, t in enumerate(tags):
    z = zs[t]
    tm = z["template"]
    vmin, vmax = np.percentile(tm, 1), np.percentile(tm, 99.7)
    ax[row, 0].imshow(tm, cmap="gray", vmin=vmin, vmax=vmax)
    ax[row, 0].set_title(f"[{t}] clean template of the registered sample", fontsize=10); ax[row, 0].axis("off")
    fr = np.asarray(movs[t][pick], dtype=np.float32)
    ax[row, 1].imshow(fr, cmap="gray", vmin=vmin, vmax=vmax)
    ax[row, 1].set_title(f"[{t}] averaged frame {pick}  (r = {z['r_raw'][pick]:.2f}, quiet level {qs[t]:.2f})", fontsize=10)
    ax[row, 1].axis("off")
    r = z["r_raw"][:per * ntr].reshape(ntr, per) / qs[t]
    im = ax[row, 2].imshow(r, aspect="auto", cmap="viridis", vmin=0, vmax=1.05, extent=[0, per * 5 / 30.23, ntr, 0])
    ax[row, 2].set_title(f"[{t}] frame-template r / quiet level: sample trial x time (s)\n"
                         f"bad (<0.5): {(z['r_raw'] < 0.5 * qs[t]).mean():.1%}   degraded (<0.75): {(z['r_raw'] < 0.75 * qs[t]).mean():.1%}",
                         fontsize=10)
    plt.colorbar(im, ax=ax[row, 2])
    s = ss[t]
    k = np.arange(len(s["xoff"]))
    ax[row, 3].plot(k, s["xoff"], lw=0.3, label="x"); ax[row, 3].plot(k, s["yoff"], lw=0.3, label="y")
    for v in (-51, 51):
        ax[row, 3].axhline(v, color="r", lw=0.6, ls="--")
    ax[row, 3].set_ylim(-110, 110); ax[row, 3].legend(loc="lower right", fontsize=8)
    ax[row, 3].set_title(f"[{t}] suite2p rigid offset per raw frame (px); dashed = +/-51 px limit of the production setting", fontsize=10)
    ax[row, 3].set_xlabel("raw frame of the sample")
fig.suptitle(f"{session}: production settings (top) vs {alt} (bottom), same raw trials, same reference frames")
fig.tight_layout()
out = HERE / "fig" / f"{session}_prod_vs_{alt}.png"
fig.savefig(out, dpi=70)
print(out)
