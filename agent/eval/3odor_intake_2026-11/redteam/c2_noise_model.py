"""C2 / C3.  Is the noise model (Gaussian, white in time, lag-1 spatial 0.18 / 0.13) what the
real sessions' quiet pixels look like?  All 111 sessions, from the cache (pass_all.py).

"Quiet" pixels: PNR at or below the 25th percentile of the interior (border 10 removed).
Per session, medians over the quiet pixels:
  rx1 ry1 rx2 ry2   spatial correlation of the temporal residual at 1 / 2 px
  rt1 rt2           temporal lag-1 / lag-2 autocorrelation of that residual
                    (white noise gives -0.667 / +0.167)
  std/sn mad/sn     spread of the filtered trace over the get_noise_fft sigma; white Gaussian
                    noise gives 1.335 for both (logmexp reads 0.749 of the true sigma)
  psd lo/hi         mean periodogram of the filtered trace in (0, 0.125] and (0.125, 0.25]
                    relative to (0.25, 0.5] (white = 1)
  Cn quiet          median Cn of the quiet pixels (the Gaussian model gives 0.478 at any length)
Usage: c2_noise_model.py > c2_noise_model.txt
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

RULES = rc.HERE.parent / "floor" / "rule_sessions.tsv"


def main():
    cats = {r["session"]: r["cat"] for r in csv.DictReader(open(RULES), delimiter="\t")}
    rows = []
    inner = np.zeros((256, 256), bool)
    inner[10:-10, 10:-10] = True
    for g, name, tif, sd, an in rc.sessions():
        f = rc.CACHE / (name + ".npz")
        if not f.exists():
            continue
        z = np.load(f)
        pnr, cn = z["pnr"].astype(float), z["cn"].astype(float)
        q = inner & (pnr <= np.percentile(pnr[inner], 25))
        hi = inner & (pnr >= np.percentile(pnr[inner], 90))
        sn = z["sn"].astype(float)
        psd = z["psd_hy_low"]
        r = dict(group=g, session=name, animal=an, cat=cats.get(name, "?"), T=int(z["T"]), noise_rel=float(z["noise_rel"]),
                 noise_u8=float(z["noise_u8"]), pct0=float(z["pct0"]), pct255=float(z["pct255"]))
        for k in ("rx1", "ry1", "rx2", "ry2", "rt1", "rt2"):
            r[k] = float(np.nanmedian(z[k][q]))
            r[k + "_all"] = float(np.nanmedian(z[k][inner]))
        r["std_sn"] = float(np.median(z["hy_std"][q] / sn[q]))
        r["mad_sn"] = float(np.median(z["hy_mad"][q] / sn[q]))
        r["std_sn_hi"] = float(np.median(z["hy_std"][hi] / sn[hi]))
        r["psd_lo"] = float(psd[:8].mean())
        r["psd_mid"] = float(psd[8:16].mean())
        r["psd_b0"] = float(psd[0])
        r["cn_quiet"] = float(np.median(cn[q]))
        r["cn_quiet_p90"] = float(np.percentile(cn[q], 90))
        r["cn_p10"] = float(np.percentile(cn[inner], 10))
        r["pnr_p25"] = float(np.percentile(pnr[inner], 25))
        rows.append(r)
    keys = ["rx1", "ry1", "rx2", "ry2", "rt1", "rt2", "std_sn", "mad_sn", "psd_lo", "psd_mid", "cn_quiet", "cn_quiet_p90",
            "cn_p10", "pnr_p25"]
    print("N sessions %d.  Model values: rx1 0.18, ry1 0.13, rx2 = ry2 = 0.008, rt1 -0.667, rt2 0.167, std/sn = mad/sn = 1.335, "
          "psd 1, Cn of noise 0.478" % len(rows))
    print("\n== medians over quiet pixels, by group: median [min-max] ==")
    groups = [("6odor", lambda r: r["group"] == "6odor"), ("3odor all", lambda r: r["group"] == "3odor"),
              ("3odor dim/hazy noted", lambda r: r["cat"] == "dim/hazy noted"),
              ("noise_rel >= 0.19", lambda r: r["noise_rel"] >= 0.19)]
    for nm, sel in groups:
        sub = [r for r in rows if sel(r)]
        print("  %-22s N %3d | %s" % (nm, len(sub), " | ".join(
            "%s %.3f [%.3f-%.3f]" % (k, np.median([r[k] for r in sub]), min(r[k] for r in sub), max(r[k] for r in sub))
            for k in keys)))
    print("\n== Spearman with noise_rel over all sessions ==")
    nr = [r["noise_rel"] for r in rows]
    print("  " + " | ".join("%s %+.2f" % (k, spearmanr([r[k] for r in rows], nr)[0]) for k in keys))
    print("\n== sessions outside the model: rx1 > 0.22 or ry1 > 0.17 or std/sn > 1.45 or Cn quiet > 0.52 ==")
    odd = [r for r in rows if r["rx1"] > 0.22 or r["ry1"] > 0.17 or r["std_sn"] > 1.45 or r["cn_quiet"] > 0.52]
    print("  %d of %d sessions" % (len(odd), len(rows)))
    for r in sorted(odd, key=lambda r: -r["cn_quiet"]):
        print("  %s %s | %s | noise_rel %.3f | rx1 %.3f ry1 %.3f rx2 %.3f ry2 %.3f | rt1 %.3f | std/sn %.3f mad/sn %.3f | "
              "psd lo %.2f mid %.2f | Cn quiet %.3f (p90 %.3f)" % (
                  r["group"], r["session"][:24], r["cat"][:14], r["noise_rel"], r["rx1"], r["ry1"], r["rx2"], r["ry2"],
                  r["rt1"], r["std_sn"], r["mad_sn"], r["psd_lo"], r["psd_mid"], r["cn_quiet"], r["cn_quiet_p90"]))
    print("\n== C3: per animal, quiet-pixel Cn in its brightest vs dimmest sessions (by noise_rel) ==")
    for an in rc.ANIMALS:
        sub = sorted([r for r in rows if r["animal"] == an], key=lambda r: r["noise_rel"])
        if len(sub) < 6:
            continue
        k = max(3, len(sub) // 4)
        lo, hi = sub[:k], sub[-k:]
        print("  %s (N %d): brightest %d sessions noise_rel %.3f-%.3f: Cn quiet median %.3f [%.3f-%.3f] | dimmest %d "
              "noise_rel %.3f-%.3f: Cn quiet median %.3f [%.3f-%.3f] | Spearman(Cn quiet, noise_rel) %+.2f" % (
                  an, len(sub), k, lo[0]["noise_rel"], lo[-1]["noise_rel"], np.median([r["cn_quiet"] for r in lo]),
                  min(r["cn_quiet"] for r in lo), max(r["cn_quiet"] for r in lo), k, hi[0]["noise_rel"], hi[-1]["noise_rel"],
                  np.median([r["cn_quiet"] for r in hi]), min(r["cn_quiet"] for r in hi), max(r["cn_quiet"] for r in hi),
                  spearmanr([r["cn_quiet"] for r in sub], [r["noise_rel"] for r in sub])[0]))
    out = rc.HERE / "c2_sessions.tsv"
    ks = ["group", "session", "animal", "cat", "T", "noise_rel", "noise_u8", "pct0", "pct255"] + keys + ["std_sn_hi", "psd_b0"]
    with open(out, "w") as fh:
        fh.write("\t".join(ks) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
