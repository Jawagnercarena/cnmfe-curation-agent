"""C7 follow-up: candidate per-session rules scored on the artificially dimmed pilots.

Each rule maps (pk50, T) of a movie to a min_corr.  Scored on the 50 degraded conditions
(5 labelled pilots x 5 noise levels x noise / blur) plus the 5 originals:
  kept   share of the base kept cells (Taylor's kept cells that could seed in the original
         movie) that can still seed from their own core, min_pnr as ACORN would estimate it
  seeds  first-pass seeds at that min_corr (all of them, cells or not), relative to the
         number at the floor in the same movie
Rules:
  0.70          the current value for five of the six animals
  rule f=0.444  floor + 0.444 * (pk50 - floor)                      (the proposal)
  rule f=0.30 / 0.20   same form, smaller constant
  line          clip(a + b * pk50, floor, 0.70), a and b fitted to the min_corr that keeps
                95% (ceiling p5); scored leave-one-session-out
  line - 0.02   the same line lowered by 0.02 (leave-one-session-out)
  pk50 - 0.12   clip(pk50 - 0.12, floor, 0.70)
  floor         floor(T) for every session
Then the same rules applied to the 71 real 3odor sessions (values only; no labels there).
Usage: c7_rules.py > c7_rules.txt
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import rt_labels as rl  # noqa: E402

RULES_PREV = rc.HERE.parent / "floor" / "rule_sessions.tsv"


def floor_prev(T):
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


def main():
    conds = []
    for g, name, tif, sd, an in rl.pilots():
        files = sorted((rc.CACHE / "degrade").glob(name + "__*.npz"))
        if not files:
            continue
        cd = rl.candidates(sd)
        mc_run, mp_run = rl.run_params(sd)
        kc = cd["core"][cd["labels"] == 1]
        zb = np.load(rc.CACHE / (name + ".npz"))
        base_ok = rl.cell_ceiling(kc, zb["cn"].astype(float), zb["pnr"].astype(float), mp_run)[0] >= mc_run
        items = [("orig", 0.0, zb)] + [(str(z["variant"]), float(z["level"]), z) for z in (np.load(f) for f in files)]
        for variant, level, z in items:
            cn, pnr, T, p = z["cn"].astype(float), z["pnr"].astype(float), int(z["T"]), float(z["est_mp"])
            ce = rl.cell_ceiling(kc, cn, pnr, p)[0][base_ok]
            conds.append(dict(session=name, variant=variant, level=level, T=T, pk50=rc.pk(cn, 50), ce=ce, cn=cn, pnr=pnr, p=p,
                              need95=float(np.percentile(ce, 5)), fl=floor_prev(T)))
    sessions = sorted(set(c["session"] for c in conds))

    def line_fit(train):
        x = np.array([c["pk50"] for c in train])
        y = np.array([c["need95"] for c in train])
        A = np.stack([np.ones_like(x), x], axis=1)
        return np.linalg.lstsq(A, y, rcond=None)[0]

    loso = {s: line_fit([c for c in conds if c["session"] != s]) for s in sessions}
    allfit = line_fit(conds)
    rules = {
        "0.70": lambda c: 0.70,
        "rule f=0.444": lambda c: c["fl"] + 0.444 * (c["pk50"] - c["fl"]),
        "rule f=0.30": lambda c: c["fl"] + 0.30 * (c["pk50"] - c["fl"]),
        "rule f=0.20": lambda c: c["fl"] + 0.20 * (c["pk50"] - c["fl"]),
        "line (LOSO)": lambda c: float(np.clip(loso[c["session"]][0] + loso[c["session"]][1] * c["pk50"], c["fl"], 0.70)),
        "line - 0.02 (LOSO)": lambda c: float(np.clip(loso[c["session"]][0] + loso[c["session"]][1] * c["pk50"] - 0.02, c["fl"], 0.70)),
        "pk50 - 0.12": lambda c: float(np.clip(c["pk50"] - 0.12, c["fl"], 0.70)),
        "pk50-0.12, >= fl+.025": lambda c: float(np.clip(c["pk50"] - 0.12, c["fl"] + 0.025, 0.70)),
        "floor + 0.025": lambda c: c["fl"] + 0.025,
        "floor": lambda c: c["fl"],
    }
    print("line fitted on all %d conditions: needed(95%%) = %.3f + %.3f * pk50; leave-one-session-out fits: %s" % (
        len(conds), allfit[0], allfit[1], "; ".join("%.3f + %.3f" % tuple(v) for v in loso.values())))
    deg = [c for c in conds if c["variant"] != "orig"]
    for c in conds:
        c["seeds_floor"] = max(rc.first_pass(c["cn"], c["pnr"], c["fl"], c["p"]), 1)
    print("\n== scored on the %d degraded conditions (%d sessions) ==" % (len(deg), len(sessions)))
    print("rule | conditions keeping >= 95%% / >= 90%% of base kept cells | kept share, median [min] at noise_rel 0.15-0.20 | "
          "at 0.25-0.35 | first-pass seeds relative to the floor, median at 0.25-0.35 | min_corr median [min-max] at 0.25-0.35")
    for nm, fn in rules.items():
        k, lo, hi, sd_, thr = [], [], [], [], []
        for c in deg:
            t = round(fn(c), 2)
            share = float((c["ce"] >= t).mean())
            k.append(share)
            if c["level"] <= 0.20:
                lo.append(share)
            else:
                hi.append(share)
                sd_.append(rc.first_pass(c["cn"], c["pnr"], t, c["p"]) / c["seeds_floor"])
                thr.append(t)
        k = np.array(k)
        print("  %-18s | %2d / %2d of %d | %.2f [%.2f] | %.2f [%.2f] | %.2f | %.2f [%.2f-%.2f]" % (
            nm, int((k >= 0.95).sum()), int((k >= 0.90).sum()), len(k), np.median(lo), min(lo), np.median(hi), min(hi),
            np.median(sd_), np.median(thr), min(thr), max(thr)))
    print("\n== on the %d originals: kept share (all must be 1.00) and min_corr ==" % (len(conds) - len(deg)))
    for nm, fn in rules.items():
        o = [c for c in conds if c["variant"] == "orig"]
        print("  %-18s | kept %s | min_corr %s" % (nm, " ".join("%.2f" % float((c["ce"] >= round(fn(c), 2)).mean()) for c in o),
                                                  " ".join("%.2f" % round(fn(c), 2) for c in o)))

    print("\n== the same rules on the real 3odor sessions (values only) ==")
    cats = {r["session"]: r["cat"] for r in csv.DictReader(open(RULES_PREV), delimiter="\t")}
    real = []
    for g, name, tif, sd, an in rc.sessions():
        f = rc.CACHE / (name + ".npz")
        if g != "3odor" or not f.exists():
            continue
        z = np.load(f)
        real.append(dict(session=name, cat=cats.get(name, "?"), T=int(z["T"]), pk50=rc.pk(z["cn"].astype(float), 50),
                         fl=floor_prev(int(z["T"])), noise_rel=float(z["noise_rel"])))
    full = dict(rules)
    full["line (all)"] = lambda c: float(np.clip(allfit[0] + allfit[1] * c["pk50"], c["fl"], 0.70))
    full["line - 0.02 (all)"] = lambda c: float(np.clip(allfit[0] + allfit[1] * c["pk50"] - 0.02, c["fl"], 0.70))
    for nm, fn in full.items():
        if "LOSO" in nm:
            continue
        v = np.array([round(fn(c), 2) for c in real])
        d = np.array([round(fn(c), 2) for c in real if c["cat"] == "dim/hazy noted"])
        print("  %-18s | all %d: median %.2f [%.2f-%.2f], at the floor in %d, < 0.65 in %d | dim/hazy noted %d: median %.2f [%.2f-%.2f]" % (
            nm, len(v), np.median(v), v.min(), v.max(), int(sum(abs(round(fn(c), 2) - c["fl"]) < 0.006 for c in real)),
            int((v < 0.65).sum()), len(d), np.median(d) if len(d) else np.nan, d.min() if len(d) else np.nan,
            d.max() if len(d) else np.nan))


if __name__ == "__main__":
    main()
