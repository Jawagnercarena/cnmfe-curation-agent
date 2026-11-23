"""Is a session's bad stretch a stable second field (step in z) or ongoing instability?

    python late_stretch_check.py <scratch session dir written by raw_sample_test.py>

Reads <dir>/new_avg.tif and <dir>/scores.npz (registered with the installed rule).
For every sampled cycle: r of its averaged frames against (a) a template from the
early clean cycles and (b) a template from the late cycles, plus mean intensity.
A stable second field shows high r against (b) and low r against (a).
"""
import sys
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from raw_sample_test import bandpass, pearson  # noqa: E402

d = Path(sys.argv[1])
z = np.load(d / "scores.npz")
gcyc = z["gcyc"]
mov = tifffile.imread(d / "new_avg.tif").astype(np.float32)
bp = bandpass(mov)
cycles = sorted(set(gcyc.tolist()))
r_new = z["r_new"]
q = float(np.percentile(r_new, 90))
early = [c for c in cycles if np.mean(r_new[gcyc == c] < 0.5 * q) < 0.05]
late = [c for c in cycles if np.mean(r_new[gcyc == c] < 0.5 * q) > 0.95]


def template(mask):
    s = bp[mask]
    r0 = pearson(s, s.mean(axis=0))
    return s[r0 >= np.median(r0)].mean(axis=0)


ta = template(np.isin(gcyc, early))
tb = template(np.isin(gcyc, late))
ra, rb = pearson(bp, ta), pearson(bp, tb)
print(f"{d.name}: early (good) cycles {early}; late (bad) cycles {late}")
print(f"  early template vs late template: r = {float(pearson(ta[None], tb)[0]):.3f}")
print(f"  {'cycle':>5s} {'r vs early':>10s} {'r vs late':>9s} {'mean F':>7s}")
for c in cycles:
    m = gcyc == c
    print(f"  {c:5d} {np.median(ra[m]):10.3f} {np.median(rb[m]):9.3f} {mov[m].mean():7.1f}")
