"""Summary of the noise-only fields in cache/null (noise_null.py, halfdiff_null.py).

Per (variant, T): number of fields, Cn / PNR quantiles, mean first-pass seeds per field
against min_corr at min_pnr 3 / 4 / 5, and two floors:
  floor0   lowest min_corr at which NO field gives a first-pass seed
  floor1   lowest min_corr at which the mean is <= 1 seed per field (the hand-off's criterion)
Usage: null_report.py > null_report.txt
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

SHOW_C = (0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65, 0.70)


def main():
    groups = defaultdict(list)
    for f in sorted((rc.CACHE / "null").glob("*.npz")):
        z = np.load(f)
        groups[(str(z["tag"]), int(z["T"]))].append(z)
    print("variant | T | fields | noise corr x / y / x2 | std/sn | Cn p50 / p99.9 / max (mean of fields; max of fields) | "
          "PNR p50 / p99.9 / max | floor0 at min_pnr 3 / 4 / 5 | floor1 at 3 / 4 / 5")
    detail = []
    for (tag, T), zs in sorted(groups.items()):
        cg, pg = zs[0]["c_grid"], zs[0]["p_grid"]
        G = np.stack([z["grid"] for z in zs]).astype(float)            # fields x c x p
        cn = np.stack([z["cn"][10:-10, 10:-10] for z in zs])
        pn = np.stack([z["pnr"][10:-10, 10:-10] for z in zs])
        f0, f1 = [], []
        for j in range(3):
            any_seed = G[:, :, j].max(axis=0) > 0
            # lowest c from which every higher grid value is also clean
            clean = ~any_seed
            ok = [i for i in range(len(cg)) if clean[i:].all()]
            f0.append(cg[ok[0]] if ok else np.nan)
            m = G[:, :, j].mean(axis=0) <= 1.0
            ok = [i for i in range(len(cg)) if m[i:].all()]
            f1.append(cg[ok[0]] if ok else np.nan)
        print("%s | %d | %d | %.3f / %.3f / %.3f | %.3f | %.3f / %.3f / %.3f; %.3f | %.2f / %.2f / %.2f | "
              "%.2f / %.2f / %.2f | %.2f / %.2f / %.2f" % (
                  tag, T, len(zs), np.mean([z["rx1"] for z in zs]), np.mean([z["ry1"] for z in zs]),
                  np.mean([z["rx2"] for z in zs]), np.mean([z["std_over_sn"] for z in zs]),
                  np.median(cn, axis=(1, 2)).mean(), np.percentile(cn, 99.9, axis=(1, 2)).mean(),
                  cn.max(axis=(1, 2)).mean(), cn.max(),
                  np.median(pn, axis=(1, 2)).mean(), np.percentile(pn, 99.9, axis=(1, 2)).mean(),
                  pn.max(axis=(1, 2)).mean(), *f0, *f1))
        for j in range(3):
            cells = []
            for c in SHOW_C:
                i = int(np.argmin(np.abs(cg - c)))
                cells.append("%.2f: %.1f (max %d)" % (c, G[:, i, j].mean(), G[:, i, j].max()))
            detail.append("%s T %d min_pnr %g | mean seeds per field (max) at min_corr  %s" % (tag, T, pg[j], "  ".join(cells)))
    print()
    print("\n".join(detail))
    mats = sorted((rc.CACHE / "matlab").glob("c4_*.mat"))
    if mats:
        from scipy.io import loadmat
        print("\n== the real greedyROI_endoscope (MATLAB, production copy) on Gaussian noise movies, min_pnr 3 ==")
        print("T | seed | min_corr | components initialised (all greedy passes) | components from the pickNeurons stage | "
              "Python first-pass replay on MATLAB's own Cn0 / PNR0")
        for f in mats:
            z = loadmat(str(f), squeeze_me=True, struct_as_record=False)
            for r in np.atleast_1d(z["res"]):
                kp = "-" if np.isnan(float(r.k_pick)) else "%d" % int(r.k_pick)
                print("  %d | %d | %.3f | %d | %s | %d" % (
                    int(r.T), int(r.seed), float(r.min_corr), int(r.k_init), kp,
                    rc.first_pass(np.asarray(z["Cn0"], float), np.asarray(z["PNR0"], float), float(r.min_corr), float(r.min_pnr))))


if __name__ == "__main__":
    main()
