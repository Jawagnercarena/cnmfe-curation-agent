"""Read-only.  Within-session stability from 300 frames spread over each tif.

The 8-bit mapping is fixed within a session, so within-session changes are real:
  bright_ratio   median frame mean, last 50 vs first 50 spread frames
  bright_minrel  lowest frame mean / session median (a drop-out shows as << 1)
  n_dim          spread frames with mean < 70% of the session median
  corr_med       median correlation of a frame (DoG, cell scale) with the session mean
  corr_ratio     median correlation, last 50 vs first 50 frames (z-drift / FOV loss < 1)
  corr_min       lowest frame correlation
Writes timecourse.tsv under --out.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_compare as vc   # noqa: E402


def dog(x):
    d = ndi.gaussian_filter(x, 1.5) - ndi.gaussian_filter(x, 8)
    d = d[24:-24, 24:-24]
    d = d - d.mean()
    return d / (np.sqrt((d * d).sum()) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="substrings of session names; appends to the TSV")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ["group", "session", "bright_ratio", "bright_minrel", "n_dim", "corr_med", "corr_ratio", "corr_min"]
    tsv = a.out / "timecourse.tsv"
    append = bool(a.only) and tsv.exists()
    with open(tsv, "a" if append else "w") as fh:
        if not append:
            fh.write("\t".join(cols) + "\n")
        for group, stem, tif, sd in vc.tif_list(a.only):
            nf, off = vc.layout(tif)
            fr = vc.read_idx(tif, off, np.linspace(0, nf - 1, 300).astype(int)).astype(np.float32)
            m = fr.reshape(300, -1).mean(1)
            med = np.median(m)
            ref = dog(fr.mean(0))
            c = np.array([(dog(f) * ref).sum() for f in fr])
            r = [group, stem.replace("AVG5x-TSeries-", ""), np.median(m[-50:]) / np.median(m[:50]), m.min() / med,
                 int((m < 0.7 * med).sum()), np.median(c), np.median(c[-50:]) / np.median(c[:50]), c.min()]
            fh.write("\t".join("%.4f" % v if isinstance(v, (float, np.floating)) else str(v) for v in r) + "\n")
            fh.flush()
            print(*r, flush=True)


if __name__ == "__main__":
    sys.exit(main())
