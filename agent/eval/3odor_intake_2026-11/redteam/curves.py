"""The trade-off as curves instead of one rule: what each min_corr keeps and what it costs.

  1. Artificial dimming (cache/degrade): share of the base kept cells that can seed at fixed
     min_corr values, by noise level (pooled over the 5 pilots and both variants), and the
     first-pass seed count relative to 0.70 in the same movie.
  2. Real sessions: the pilots' kept cells tracked into other sessions of the same field
     (as c6_pk50.py part D): share that can seed at each min_corr, bright vs dim targets.
  3. The 71 real 3odor sessions: first-pass seeds at each min_corr with the exact min_pnr
     (no labels: a count of what would be attempted, not of cells).  The real component count
     grows faster than first-pass seeds as min_corr drops (c5_replay.txt).
min_pnr is always what params.py would estimate from the movie's own pre-pass.
Usage: curves.py > curves.txt
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import rt_fov as rf  # noqa: E402
import rt_labels as rl  # noqa: E402

CS = (0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70)
PAIRS = rc.HERE.parent / "fov" / "pairs.tsv"


def FLOOR(T):
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


RULESET = {
    "0.70": lambda pk, fl: 0.70,
    "rule f=0.444": lambda pk, fl: fl + 0.444 * (pk - fl),
    "rule f=0.30": lambda pk, fl: fl + 0.30 * (pk - fl),
    "rule f=0.20": lambda pk, fl: fl + 0.20 * (pk - fl),
    "clip(pk50-0.12, floor, 0.70)": lambda pk, fl: float(np.clip(pk - 0.12, fl, 0.70)),
    "clip(pk50-0.12, floor+.025, .70)": lambda pk, fl: float(np.clip(pk - 0.12, fl + 0.025, 0.70)),
    "floor + 0.025": lambda pk, fl: fl + 0.025,
    "floor": lambda pk, fl: fl,
}
RULES = rc.HERE.parent / "floor" / "rule_sessions.tsv"


def main():
    cats = {r["session"]: r["cat"] for r in csv.DictReader(open(RULES), delimiter="\t")}
    print("== 1. artificial dimming: base kept cells that can seed, cells pooled over 5 pilots x (noise, blur) ==")
    print("noise_rel | cells | " + " | ".join("%.3f" % c for c in CS) + " || first-pass seeds relative to 0.70 (median over movies): " +
          " | ".join("%.3f" % c for c in CS))
    acc = defaultdict(lambda: [[] for _ in CS])
    seeds = defaultdict(lambda: [[] for _ in CS])
    cands = {}
    for g, name, tif, sd, an in rl.pilots():
        cd = rl.candidates(sd)
        cands[name] = cd
        files = sorted((rc.CACHE / "degrade").glob(name + "__*.npz"))
        if not files:
            continue
        mc_run, mp_run = rl.run_params(sd)
        kc = cd["core"][cd["labels"] == 1]
        zb = np.load(rc.CACHE / (name + ".npz"))
        base_ok = rl.cell_ceiling(kc, zb["cn"].astype(float), zb["pnr"].astype(float), mp_run)[0] >= mc_run
        for level, z in [(0.0, zb)] + [(float(z["level"]), z) for z in (np.load(f) for f in files)]:
            cn, pnr, p = z["cn"].astype(float), z["pnr"].astype(float), float(z["est_mp"])
            ce = rl.cell_ceiling(kc, cn, pnr, p)[0][base_ok]
            s70 = max(rc.first_pass(cn, pnr, 0.70, p), 1)
            for i, c in enumerate(CS):
                acc[level][i] += list(ce >= c)
                seeds[level][i].append(rc.first_pass(cn, pnr, c, p) / s70)
    for level in sorted(acc):
        print("  %s | %d | %s || %s" % ("original" if level == 0 else "%.2f" % level, len(acc[level][0]),
                                        " | ".join("%.2f" % np.mean(a) for a in acc[level]),
                                        " | ".join("%.2f" % np.median(a) for a in seeds[level])))

    print("\n== 2. real sessions: the pilots' kept cells tracked into other sessions of the same field ==")
    S = {}
    for g, name, tif, sd, an in rc.sessions():
        z = np.load(rc.CACHE / (name + ".npz"))
        S[name] = dict(group=g, animal=an, T=int(z["T"]), cn=z["cn"].astype(float), pnr=z["pnr"].astype(float),
                       mean=z["mean256"].astype(float), noise_rel=float(z["noise_rel"]), est_mp=float(z["est_mp"]),
                       cat=cats.get(name, "?"))
    prev = {}
    for r in csv.DictReader(open(PAIRS), delimiter="\t"):
        prev[frozenset((r["s1"], r["s2"]))] = float(r["score"])
    st = ndi.generate_binary_structure(2, 2)
    pooled = defaultdict(lambda: [[] for _ in CS])
    pooled3 = defaultdict(lambda: [[] for _ in CS])
    npairs = defaultdict(set)
    byrule = defaultdict(lambda: defaultdict(list))
    thr_rule = defaultdict(lambda: defaultdict(list))
    for pname, cd in cands.items():
        kc = cd["core"][cd["labels"] == 1]
        for b, t in S.items():
            if b == pname or t["animal"] != S[pname]["animal"]:
                continue
            pp = prev.get(frozenset((pname, b)))
            sc, dy, dx = rf.register(S[pname]["mean"], t["mean"])
            if sc < 0.30 or pp is None or pp <= 0.354:
                continue
            m = rf.shift_masks(kc, -dy, -dx)
            inside = np.array([mm.any() and not (mm[:10].any() or mm[-10:].any() or mm[:, :10].any() or mm[:, -11:].any()) for mm in m])
            m = np.array([ndi.binary_dilation(mm, st) for mm in m[inside]])
            if len(m) < 5:
                continue
            key = "dim (noise_rel >= 0.15 or dim/hazy noted)" if (t["noise_rel"] >= 0.15 or t["cat"] == "dim/hazy noted") else \
                ("bright 3odor" if t["group"] == "3odor" else "bright 6odor")
            ce = rl.cell_ceiling(m, t["cn"], t["pnr"], t["est_mp"])[0]
            ce3 = rl.cell_ceiling(m, t["cn"], t["pnr"], 3.0)[0]
            npairs[key].add((pname, b))
            for i, c in enumerate(CS):
                pooled[key][i] += list(ce >= c)
                pooled3[key][i] += list(ce3 >= c)
            fl = FLOOR(t["T"])
            pk50 = rc.pk(t["cn"], 50)
            for rn, fn in RULESET.items():
                th = round(fn(pk50, fl), 2)
                byrule[key][rn] += list(ce >= th)
                thr_rule[key][rn].append(th)
    print("target sessions | pilot-target pairs / target sessions / cell-instances | " + " | ".join("%.3f" % c for c in CS) +
          " || with min_pnr 3: " + " | ".join("%.3f" % c for c in CS))
    for key in sorted(pooled):
        print("  %s | %d / %d / %d | %s || %s" % (key, len(npairs[key]), len(set(b for a, b in npairs[key])), len(pooled[key][0]),
                                               " | ".join("%.2f" % np.mean(a) for a in pooled[key]),
                                               " | ".join("%.2f" % np.mean(a) for a in pooled3[key])))

    print("\n   the same tracked cells under each candidate rule (threshold from the TARGET session's own pk50 and length):")
    print("   rule | " + " | ".join("%s: share (median min_corr)" % k for k in sorted(byrule)))
    for rn in RULESET:
        print("   %-32s | %s" % (rn, " | ".join("%.2f (%.2f)" % (np.mean(byrule[k][rn]), np.median(thr_rule[k][rn]))
                                                 for k in sorted(byrule))))
    print("\n== 3. real 3odor sessions: first-pass seeds at each min_corr (exact min_pnr), median [min-max] ==")
    print("group | N | " + " | ".join("%.3f" % c for c in CS))
    groups = [("dim/hazy noted", lambda t: t["cat"] == "dim/hazy noted"),
              ("noise_rel >= 0.19", lambda t: t["noise_rel"] >= 0.19),
              ("other 3odor (noise_rel < 0.15, no dim note)", lambda t: t["noise_rel"] < 0.15 and t["cat"] != "dim/hazy noted")]
    for nm, sel in groups:
        sub = [t for t in S.values() if t["group"] == "3odor" and sel(t)]
        cells = []
        for c in CS:
            v = [rc.first_pass(t["cn"], t["pnr"], c, t["est_mp"]) for t in sub]
            cells.append("%d [%d-%d]" % (np.median(v), min(v), max(v)))
        print("  %s | %d | %s" % (nm, len(sub), " | ".join(cells)))


if __name__ == "__main__":
    main()
