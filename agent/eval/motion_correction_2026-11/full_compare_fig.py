"""Before/after figure for a full session: the delivered AVG5x movie vs a re-registered one.

    python full_compare_fig.py --delivered <tif> --delivered-npz <motion_metrics npz>
                               --new <tif> --new-npz <motion_metrics npz> --s2p-npz <score_full _s2p.npz>
                               --nfr <xml_frames/<session>_nfr.npy> --out <png> [--title text]

Top row = delivered, bottom row = new. Columns: clean template; one averaged frame that is bad
in the delivered movie (the median one of its bad frames) and the SAME frame index in the new
movie; frame-template r / quiet level as trial x time; suite2p's rigid offsets of the new run
(only available for the new run: the delivered run's ops.npy was deleted). Read-only on inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

FS_RAW = 30.23
FACTOR = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivered", type=Path, required=True)
    ap.add_argument("--delivered-npz", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--new-npz", type=Path, required=True)
    ap.add_argument("--s2p-npz", type=Path, required=True)
    ap.add_argument("--nfr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    tags = ["delivered", "new"]
    zs = {"delivered": np.load(args.delivered_npz), "new": np.load(args.new_npz)}
    movs = {"delivered": tifffile.memmap(args.delivered, mode="r"), "new": tifffile.memmap(args.new, mode="r")}
    qs = {t: float(np.percentile(zs[t]["r_raw"], 90)) for t in tags}
    s2p = np.load(args.s2p_npz)
    nfr = np.load(args.nfr)
    edges = np.concatenate([[0], np.cumsum(nfr)]) / FACTOR
    ntr = len(nfr)
    nb = int(np.median(nfr)) // FACTOR

    bad = np.where(zs["delivered"]["r_raw"] < 0.5 * qs["delivered"])[0]
    pick = int(bad[len(bad) // 2]) if len(bad) else 0
    pick = min(pick, movs["new"].shape[0] - 1)

    fig, ax = plt.subplots(2, 4, figsize=(22, 10.5), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1.5]))
    for row, t in enumerate(tags):
        z = zs[t]
        r_raw = z["r_raw"]
        nt = len(r_raw)
        tm = z["template"]
        vmin, vmax = np.percentile(tm, 1), np.percentile(tm, 99.7)
        ax[row, 0].imshow(tm, cmap="gray", vmin=vmin, vmax=vmax)
        ax[row, 0].set_title(f"[{t}] clean template", fontsize=10)
        ax[row, 0].axis("off")
        fr = np.asarray(movs[t][pick], dtype=np.float32)
        ax[row, 1].imshow(fr, cmap="gray", vmin=vmin, vmax=vmax)
        ax[row, 1].set_title(f"[{t}] averaged frame {pick}  (r = {r_raw[pick]:.2f}, quiet level {qs[t]:.2f})",
                             fontsize=10)
        ax[row, 1].axis("off")
        img = np.full((ntr, nb), np.nan)
        for i in range(ntr):
            a = int(np.ceil(edges[i]))
            b = min(a + nb, nt)
            if b > a:
                img[i, :b - a] = r_raw[a:b] / qs[t]
        im = ax[row, 2].imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=1.05,
                               extent=[0, nb * FACTOR / FS_RAW, ntr, 0])
        ax[row, 2].set_title(f"[{t}] frame-template r / quiet level: trial x time in trial (s)\n"
                             f"bad (<0.5): {(r_raw < 0.5 * qs[t]).mean():.1%}   degraded (<0.75): "
                             f"{(r_raw < 0.75 * qs[t]).mean():.1%}   ({nt} frames)", fontsize=10)
        ax[row, 2].set_ylabel("trial")
        plt.colorbar(im, ax=ax[row, 2])
    ax[0, 3].axis("off")
    ax[0, 3].text(0.02, 0.5, "suite2p shifts of the delivered run are not available\n"
                             "(ops.npy was deleted with the TSeries folder)", fontsize=11, va="center")
    k = np.arange(len(s2p["xoff"]))
    ax[1, 3].plot(k, s2p["xoff"], lw=0.2, label="x")
    ax[1, 3].plot(k, s2p["yoff"], lw=0.2, label="y")
    for v in (-51, 51):
        ax[1, 3].axhline(v, color="r", lw=0.6, ls="--")
    ax[1, 3].set_ylim(-110, 110)
    ax[1, 3].legend(loc="lower right", fontsize=8)
    ax[1, 3].set_title("[new] suite2p rigid offset per raw frame (px); dashed = +/-51 px, the old search limit",
                       fontsize=10)
    ax[1, 3].set_xlabel("raw frame")
    fig.suptitle(args.title or f"{args.delivered.stem}: delivered (top) vs re-registered (bottom)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=70)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
