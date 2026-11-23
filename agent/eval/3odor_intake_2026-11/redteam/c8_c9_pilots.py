"""C8 / C9: what the 12 labelled pilots can already say (the 28-session batch is not back).

Not the test the hand-off asks for (that needs the batch, which ran at the thresholds under
question), and 10 of the 12 pilots are among the 31 sessions f was calibrated on.  It answers
two narrower questions on Taylor's threshold-0 labels:
  C8  per pilot, how many kept cells have (a) a peak Cn below the session's rule value and
      (b) a seeding ceiling below it (ceiling = highest min_corr at which the cell can still
      seed from its own core, given the run's min_pnr).  (b) is the one that costs cells.
  C9  082726-pnb105 ran at min_corr 0.60 (082626-pnb105 at 0.70): kept-rate of the candidates
      whose peak Cn / ceiling lies between 0.60 and the rule value.
Also: kept-rate by ceiling band, pooled, to show where kept cells stop.
Usage: c8_c9_pilots.py > c8_c9_pilots.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import rt_labels as rl  # noqa: E402

F_PREV = 0.444


def floor_prev(T):
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


def main():
    tot = dict(kept=0, pk_below=0, ce_below=0, ce_below_run=0)
    pooled_ce, pooled_pk, pooled_lab, pooled_rule, pooled_run = [], [], [], [], []
    print("session | ran at | pk50 | rule | candidates / kept | kept with peak Cn < rule | kept with ceiling < rule | "
          "kept with ceiling < the run's min_corr | kept ceiling p5 / p10 / p25 | rejected with ceiling >= rule")
    for g, name, tif, sd, an in rl.pilots():
        f = rc.CACHE / (name + ".npz")
        if not f.exists():
            print("  NOT IN CACHE %s" % name)
            continue
        z = np.load(f)
        cn, pnr, T = z["cn"].astype(float), z["pnr"].astype(float), int(z["T"])
        cd = rl.candidates(sd)
        if not cd["join_ok"]:
            print("  LABEL JOIN FAILED %s -- skipped" % name)
            continue
        mc, mp = rl.run_params(sd)
        lab = cd["labels"]
        pk50 = rc.pk(cn, 50)
        rule = round(floor_prev(T) + F_PREV * (pk50 - floor_prev(T)), 2)
        ce, pk = rl.cell_ceiling(cd["core"], cn, pnr, mp)
        k = lab == 1
        tot["kept"] += int(k.sum())
        tot["pk_below"] += int((pk[k] < rule).sum())
        tot["ce_below"] += int((ce[k] < rule).sum())
        tot["ce_below_run"] += int((ce[k] < mc).sum())
        pooled_ce.append(ce)
        pooled_pk.append(pk)
        pooled_lab.append(lab)
        pooled_rule.append(np.full(len(lab), rule))
        pooled_run.append(np.full(len(lab), mc))
        print("  %s | %.2f | %.3f | %.2f | %d / %d | %d | %d | %d | %.3f / %.3f / %.3f | %d of %d" % (
            name[:22], mc, pk50, rule, len(lab), int(k.sum()), int((pk[k] < rule).sum()), int((ce[k] < rule).sum()),
            int((ce[k] < mc).sum()), *np.percentile(ce[k], [5, 10, 25]), int((ce[~k] >= rule).sum()), int((~k).sum())))
        if "pnb105" in name:
            for nm, v in (("peak Cn", pk), ("ceiling", ce)):
                band = (v >= 0.60) & (v < rule)
                below = v < 0.60
                print("      C9 %s by %s: candidates in [0.60, rule %.2f): %d, kept %d | below 0.60: %d, kept %d | "
                      "at or above the rule: %d, kept %d" % (name[:13], nm, rule, int(band.sum()), int(lab[band].sum()),
                                                            int(below.sum()), int(lab[below].sum()),
                                                            int((v >= rule).sum()), int(lab[v >= rule].sum())))
    print("\npooled: %d kept cells; peak Cn below the session's rule value: %d (%.2f%%); ceiling below the rule value: %d (%.2f%%); "
          "ceiling below the run's own min_corr: %d (%.2f%%)" % (
              tot["kept"], tot["pk_below"], 100.0 * tot["pk_below"] / max(tot["kept"], 1), tot["ce_below"],
              100.0 * tot["ce_below"] / max(tot["kept"], 1), tot["ce_below_run"], 100.0 * tot["ce_below_run"] / max(tot["kept"], 1)))
    ce, pk, lab = np.concatenate(pooled_ce), np.concatenate(pooled_pk), np.concatenate(pooled_lab)
    print("\nkept-rate by band, pooled (N %d candidates, %d kept):" % (len(lab), int(lab.sum())))
    for nm, v in (("peak Cn in the core", pk), ("seeding ceiling", ce)):
        print("  by %s:" % nm)
        for lo, hi in ((0, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)):
            s = (v >= lo) & (v < hi)
            print("    %.2f-%.2f: %3d kept of %4d (%.1f%%)" % (lo, hi, int(lab[s].sum()), int(s.sum()),
                                                           100.0 * lab[s].mean() if s.any() else float("nan")))
    rel = ce - np.concatenate(pooled_run)
    print("  by ceiling minus the run's own min_corr (how much margin the kept cells had):")
    for lo, hi in ((-1, -0.05), (-0.05, 0), (0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1)):
        s = (rel >= lo) & (rel < hi)
        print("    %+.2f to %+.2f: %3d kept of %4d (%.1f%%)" % (lo, hi, int(lab[s].sum()), int(s.sum()),
                                                            100.0 * lab[s].mean() if s.any() else float("nan")))


if __name__ == "__main__":
    main()
