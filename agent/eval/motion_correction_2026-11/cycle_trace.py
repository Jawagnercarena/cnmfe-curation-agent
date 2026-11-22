"""Per-cycle similarity of a delivered AVG5x movie to its early field.

    python cycle_trace.py <delivered AVG5x tif> <server TSeries folder> --early A B

Template = better-correlated half of the band-passed frames of cycles A..B (which
must be well registered). Prints, for every cycle, the median r of its averaged
frames against that template and the share below 0.5 x the template cycles' p90.
Reads the server folder (XML only) and the tif; writes nothing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from raw_sample_test import FACTOR, bandpass, pearson, xml_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tif", type=Path)
ap.add_argument("src", type=Path)
ap.add_argument("--early", type=int, nargs=2, required=True)
args = ap.parse_args()

nfr = xml_frames(args.src)
gstart = np.concatenate([[0], np.cumsum(nfr)])
mov = tifffile.memmap(args.tif, mode="r")
if mov.shape[0] != int(gstart[-1]) // FACTOR:
    raise SystemExit("frame count does not match the XML")
cyc = np.zeros(mov.shape[0], int)
for t in range(len(nfr)):
    cyc[-(-int(gstart[t]) // FACTOR):int(gstart[t + 1]) // FACTOR] = t + 1
a, b = args.early
early_idx = np.flatnonzero((cyc >= a) & (cyc <= b))
samp = early_idx[np.linspace(0, len(early_idx) - 1, min(600, len(early_idx))).astype(int)]
s = bandpass(np.asarray(mov[samp], dtype=np.float32))
r0 = pearson(s, s.mean(axis=0))
tmpl = s[r0 >= np.median(r0)].mean(axis=0)
del s
r = np.zeros(mov.shape[0], np.float32)
for i in range(0, mov.shape[0], 256):
    r[i:i + 256] = pearson(bandpass(np.asarray(mov[i:i + 256], dtype=np.float32)), tmpl)
q = float(np.percentile(r[early_idx], 90))
print(f"{args.tif.name}: template from cycles {a}-{b}; quiet r {q:.3f}")
rows = []
for t in range(1, len(nfr) + 1):
    m = cyc == t
    if m.any():
        rows.append(f"{t}:{np.median(r[m]) / q:.2f}/{100 * np.mean(r[m] < 0.5 * q):.0f}%")
print("cycle: median r / quiet, share bad")
for i in range(0, len(rows), 10):
    print("  " + "  ".join(rows[i:i + 10]))
