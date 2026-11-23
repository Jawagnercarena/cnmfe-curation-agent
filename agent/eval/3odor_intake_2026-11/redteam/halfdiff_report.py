"""Where do the seeds of the half-difference movies sit?  (C2 / C3 / C4, real data)

The half-difference movie D keeps the real noise but also the part of every transient that
changes inside a frame pair, so its seeds are noise OR leftover signal.  To tell them apart
the field is split with the REAL movie's seeding images (cache/<session>.npz):
  quiet   pixels whose real Cn is at or below the interior median (half of the field; no
          cell-like structure there)
  active  pixels whose real Cn is in the top fifth
Per session and min_corr (min_pnr 3): first-pass seeds of D in total, in the quiet half and in
the active fifth; and the number the Gaussian model predicts for a whole field of that length
(mean of the simulated fields of the nearest length).  Also the Cn of D in the quiet half.
If the model describes the real noise, D seeds in the quiet half should stop where the model's do.
Usage: halfdiff_report.py > halfdiff_report.txt
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

CS = (0.52, 0.55, 0.575, 0.60, 0.625, 0.65, 0.70)


def main():
    model = defaultdict(list)
    for f in sorted((rc.CACHE / "null").glob("gauss_T*_s*.npz")):
        z = np.load(f)
        model[int(z["T"])].append((z["cn"].astype(float), z["pnr"].astype(float)))
    Ts = np.array(sorted(t for t in model if len(model[t]) >= 4))
    names = {s[1][:13].rstrip("-"): s for s in rc.sessions()}
    inner = np.zeros((256, 256), bool)
    inner[10:-10, 10:-11] = True
    print("session | group | T | noise_rel | D noise corr x / y / x2 | D Cn in the quiet half p50 / p99 / p99.9 (model p50 / p99 / p99.9) | "
          "then per min_corr: D seeds total / in the quiet half / in the active fifth [model seeds per whole field]")
    rows = []
    for f in sorted((rc.CACHE / "null").glob("halfdiff_*.npz")):
        z = np.load(f)
        key = str(z["tag"])[len("halfdiff_"):]
        if key not in names:
            continue
        g, name, tif, sd, an = names[key]
        cf = rc.CACHE / (name + ".npz")
        if not cf.exists():
            print("  %s: real session not in cache yet" % name)
            continue
        zr = np.load(cf)
        cn_r = zr["cn"].astype(float)
        quiet = inner & (cn_r <= np.median(cn_r[inner]))
        active = inner & (cn_r >= np.percentile(cn_r[inner], 80))
        dcn, dpnr, T = z["cn"].astype(float), z["pnr"].astype(float), int(z["T"])
        Tn = int(Ts[np.argmin(np.abs(Ts - T))])
        mq = np.concatenate([m[0][10:-10, 10:-10].ravel() for m in model[Tn]])
        cells = []
        r = dict(session=name, group=g, T=T, noise_rel=float(zr["noise_rel"]))
        for c in CS:
            m = rc.first_pass(dcn, dpnr, c, 3.0, return_mask=True)
            mm = float(np.mean([rc.first_pass(a, b, c, 3.0) for a, b in model[Tn]]))
            cells.append("%.3f: %d / %d / %d [%.1f]" % (c, int(m.sum()), int((m & quiet).sum()), int((m & active).sum()), mm))
            r["q%.3f" % c] = int((m & quiet).sum())
            r["m%.3f" % c] = mm
        q = dcn[quiet]
        r["cnq50"] = float(np.median(q))
        rows.append(r)
        print("  %s | %s | %d | %.3f | %.3f / %.3f / %.3f | %.3f / %.3f / %.3f (%.3f / %.3f / %.3f) | %s" % (
            name[:22], g, T, r["noise_rel"], float(z["rx1"]), float(z["ry1"]), float(z["rx2"]), np.median(q),
            np.percentile(q, 99), np.percentile(q, 99.9), np.median(mq), np.percentile(mq, 99), np.percentile(mq, 99.9),
            "  ".join(cells)))
    if rows:
        print("\nlowest min_corr on the grid with NO D seed in the quiet half (model: lowest with no seed in any simulated field):")
        for r in rows:
            ok = [c for c in CS if all(r["q%.3f" % c2] == 0 for c2 in CS if c2 >= c)]
            print("  %s (noise_rel %.3f, T %d): %s" % (r["session"][:22], r["noise_rel"], r["T"], ("%.3f" % ok[0]) if ok else "> 0.70"))


if __name__ == "__main__":
    main()
