"""
qc_seedtrace.py -- one panel per GCaMP soma with no candidate that passes both
seeding gates (agent/eval/missed_somata_2026-11, 2026-11-05).

Panel: session-mean crop (60 x 60 full-res px) with the soma disk (yellow), the
final candidates' 0.3-of-max contours (cyan), and -- when the soma was covered at
some stage -- the footprint that last covered it, at the snapshot before the
loss (magenta).  Title: animal, fate code, step where coverage was lost.
READS seedtrace/, somata/, seedtrace_somata.csv, and the 2026-11-04 mean images
(--mean-dir, one <date-animal>/imgs.npz per session).  WRITES only
seedtrace_qc_gatepass.png next to this file.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_seedtrace import COVER_LVL, R_DS, R_FULL, cover_of, disk_index, load_trace

HERE = Path(__file__).resolve().parent
HALF = 30


def col_img(R, k, d):
    v = R[:, k].toarray().ravel()
    img = v.reshape(d, d, order="F")
    if d == 256:
        img = np.kron(img, np.ones((2, 2)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-dir", required=True, help="folder holding <date-animal>/imgs.npz (2026-11-04 cache)")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(HERE / "seedtrace_somata.csv", encoding="utf-8"))
            if r["kind"] == "gcamp" and r["prod_cat"] == "none" and r["prod_gate"] == "True"]
    rows.sort(key=lambda r: (r["fate"], r["session"]))
    n = len(rows)
    nc = 7
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.8 * nr))
    axs = np.atleast_2d(axs)
    cache = {}
    for ax in axs.ravel():
        ax.axis("off")
    for i, r in enumerate(rows):
        s = r["session"]
        tag = "-".join(re.search(r"-(\d{6})-(pnb\d+)-", s).groups())
        if s not in cache:
            snaps, _, _ = load_trace(HERE / "seedtrace" / s / "seedtrace.mat")
            for sn in snaps:
                sn["Rr"] = sn["R"].tocsr()
            cache[s] = (snaps, np.load(Path(a.mean_dir) / tag / "imgs.npz")["mean"])
        snaps, M = cache[s]
        y, x = int(r["r"]), int(r["c"])
        y0, x0 = max(0, y - HALF), max(0, x - HALF)
        crop = (slice(y0, y + HALF), slice(x0, x + HALF))
        ax = axs.ravel()[i]
        ax.imshow(M[crop], cmap="gray")
        fin = snaps[-1]
        idx_f, _ = disk_index(y, x, 512, R_FULL)
        for k in np.unique(fin["Rr"][disk_index(y, x, 512, HALF)[0], :].nonzero()[1]):
            img = col_img(fin["R"], k, 512)[crop]
            if img.max() > 0:
                ax.contour(img, levels=[COVER_LVL * fin["R"][:, k].max()], colors="c", linewidths=0.7)
        lost = r["cover_lost_at"]
        if lost and lost != "never lost":
            t1 = next(t for t, sn in enumerate(snaps) if sn["tag"] == lost)
            sp_ = snaps[t1 - 1]
            idx = disk_index(y / 2.0, x / 2.0, 256, R_DS)[0] if sp_["d"] == 256 else idx_f
            k = int(np.argmax(cover_of(sp_["Rr"], idx)))
            img = col_img(sp_["R"], k, sp_["d"])[crop]
            ax.contour(img, levels=[COVER_LVL * img.max()], colors="m", linewidths=0.9)
        ax.add_patch(plt.Circle((x - x0, y - y0), R_FULL, fill=False, color="y", lw=0.9))
        an = re.search(r"pnb\d+", s).group(0)
        ax.set_title(f"{an} ({y},{x}) {r['fate'][0]}\n{(lost or 'never covered')}", fontsize=7)
    fig.suptitle("GCaMP somata with no candidate, passing both seeding gates (6 FOVA sessions, 090826)\n"
                 "yellow: soma (7 px); cyan: final candidates (0.3 of max); magenta: footprint that last covered the soma, "
                 "before the step named in the title. Fate 1 never covered, 2 merged, 3 deleted, 4 footprint retracted",
                 fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "seedtrace_qc_gatepass.png"
    plt.savefig(out, dpi=110)
    print("wrote", out, "panels", n)


if __name__ == "__main__":
    main()
