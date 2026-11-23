"""C10.  Is PNR the gate?  The exact pre-pass min_pnr for every session, and what it does.

  1. 6odor (40): the pre-pass port's estimate vs every "Estimated from Cn/pnr images" line in
     the session's agent_run.log (validates the pre-pass port on what ACORN really chose).
  2. 3odor (71): exact min_pnr (params.py on the pre-pass port) vs the earlier stand-in
     (mp_now in floor/rule_sessions.tsv).
  3. Top-50 Cn peaks at the rule's min_corr, stand-in vs exact min_pnr: share that can seed.
       strict  the peak pixel itself passes and >= 13 of its 25 window pixels pass
       near    some pixel of the peak's 5 x 5 window does (a seed 1-2 px off is the same cell)
       loose   the earlier definition (>= 13 of 25 pass; the peak pixel itself is not checked)
  4. Labelled pilots (12): kept-rate against the candidate's PNR margin over the run's min_pnr,
     and what a higher min_pnr would have cost: share of kept / rejected candidates that could
     no longer seed from their own core at the run's min_corr.
Usage: c10_minpnr.py > c10_minpnr.txt
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import rt_labels as rl  # noqa: E402

PREV = rc.HERE.parent / "floor" / "rule_sessions.tsv"
EST_RE = re.compile(r"Estimated from Cn/pnr images: min_corr=([0-9]+(?:\.[0-9]+)?), min_pnr=([0-9]+(?:\.[0-9]+)?)")


def top50(cn, pnr, c, p, k=50):
    ys, xs, _ = rc.cn_peaks(cn)
    ys, xs = ys[:k], xs[:k]
    if len(ys) == 0:
        return np.nan, np.nan, np.nan
    strict = rc.seedable_at(cn, pnr, ys, xs, c, p)
    ce = rl.ceiling(cn, pnr, p)
    near = ndi.maximum_filter(ce, size=rc.MED)[ys, xs] >= c
    loose = rc.pass_count(cn, pnr, c, p)[ys, xs] >= 13
    return float(strict.mean()), float(near.mean()), float(loose.mean())


def main():
    prev = {r["session"]: r for r in csv.DictReader(open(PREV), delimiter="\t")}
    S = {}
    for g, name, tif, sd, an in rc.sessions():
        f = rc.CACHE / (name + ".npz")
        if f.exists():
            S[name] = (g, sd, an, np.load(f))
    print("sessions in cache: %d (3odor %d, 6odor %d)" % (len(S), sum(v[0] == "3odor" for v in S.values()),
                                                         sum(v[0] == "6odor" for v in S.values())))

    print("\n== 1. 6odor: pre-pass port vs the estimates ACORN logged ==")
    n_ok = n_all = 0
    for name, (g, sd, an, z) in S.items():
        if g != "6odor":
            continue
        logged = [(float(m.group(1)), float(m.group(2))) for m in
                  (EST_RE.search(ln) for ln in open(sd / "agent_run.log", encoding="utf-8", errors="replace")) if m]
        mine = (float(z["est_mc"]), float(z["est_mp"]))
        ok = bool(logged) and all(abs(a - mine[0]) < 1e-9 and abs(b - mine[1]) < 1e-9 for a, b in logged)
        n_all += 1
        n_ok += ok
        if not ok:
            print("  DIFFERS %s: port %.2f / %.1f (raw %.4f / %.3f), logged %s" % (
                name, mine[0], mine[1], float(z["raw_mc"]), float(z["raw_mp"]), sorted(set(logged))))
    print("  identical to every logged estimate in %d of %d sessions" % (n_ok, n_all))

    print("\n== 2. 3odor: exact pre-pass min_pnr vs the stand-in ==")
    rows = []
    for name, (g, sd, an, z) in S.items():
        if g != "3odor" or name not in prev:
            continue
        p = prev[name]
        rows.append(dict(session=name, animal=an, cat=p["cat"], T=int(z["T"]), noise_rel=float(z["noise_rel"]),
                         mp_stand=float(p["mp_now"]), mp_exact=float(z["est_mp"]), mp_raw=float(z["raw_mp"]),
                         mc_est=float(z["est_mc"]), rule=float(p["rule"]), pk50_prev=float(p["pk50"]),
                         cn=z["cn"].astype(float), pnr=z["pnr"].astype(float)))
    d = np.array([r["mp_exact"] - r["mp_stand"] for r in rows])
    print("  N %d: exact - stand-in: median %+.2f, min %+.2f, max %+.2f; exact > stand-in in %d" % (
        len(rows), np.median(d), d.min(), d.max(), int((d > 0).sum())))
    print("  at the 5.0 clamp: stand-in %d, exact %d (raw exact estimate below 5.0 in %d)" % (
        sum(r["mp_stand"] <= 5.0 + 1e-9 for r in rows), sum(r["mp_exact"] <= 5.0 + 1e-9 for r in rows),
        sum(r["mp_raw"] < 5.0 for r in rows)))
    for cat in sorted(set(r["cat"] for r in rows)):
        sub = [r for r in rows if r["cat"] == cat]
        print("  %-24s N %2d  stand-in median %.2f [%.2f-%.2f] | exact median %.2f [%.2f-%.2f]" % (
            cat, len(sub), np.median([r["mp_stand"] for r in sub]), min(r["mp_stand"] for r in sub),
            max(r["mp_stand"] for r in sub), np.median([r["mp_exact"] for r in sub]),
            min(r["mp_exact"] for r in sub), max(r["mp_exact"] for r in sub)))

    print("\n== 3. top-50 Cn peaks at the rule's min_corr: share that can seed, stand-in vs exact min_pnr ==")
    print("session | note | noise_rel | rule | min_pnr stand-in -> exact | strict | near | loose (earlier definition) | "
          "first-pass seeds stand-in -> exact")
    for r in sorted(rows, key=lambda r: r["rule"]):
        a = top50(r["cn"], r["pnr"], r["rule"], r["mp_stand"])
        b = top50(r["cn"], r["pnr"], r["rule"], r["mp_exact"])
        r.update(strict_s=a[0], near_s=a[1], loose_s=a[2], strict_e=b[0], near_e=b[1], loose_e=b[2],
                 seeds_s=rc.first_pass(r["cn"], r["pnr"], r["rule"], r["mp_stand"]),
                 seeds_e=rc.first_pass(r["cn"], r["pnr"], r["rule"], r["mp_exact"]))
        ys, xs, _ = rc.cn_peaks(r["cn"])
        r["pk_pnr_med"] = float(np.median(r["pnr"][ys[:50], xs[:50]]))
        r["pk_pnr_p10"] = float(np.percentile(r["pnr"][ys[:50], xs[:50]], 10))
        if r["cat"] == "dim/hazy noted" or r["noise_rel"] >= 0.19 or r["rule"] < 0.65:
            print("  %s | %s | %.3f | %.2f | %.1f -> %.1f | %.2f -> %.2f | %.2f -> %.2f | %.2f -> %.2f | %d -> %d" % (
                r["session"][:22], r["cat"][:14], r["noise_rel"], r["rule"], r["mp_stand"], r["mp_exact"],
                r["strict_s"], r["strict_e"], r["near_s"], r["near_e"], r["loose_s"], r["loose_e"],
                r["seeds_s"], r["seeds_e"]))
    for label, sel in (("dim/hazy noted", lambda r: r["cat"] == "dim/hazy noted"),
                       ("noise_rel >= 0.19", lambda r: r["noise_rel"] >= 0.19),
                       ("all 3odor", lambda r: True)):
        sub = [r for r in rows if sel(r)]
        for k, nm in (("strict", "strict"), ("near", "near"), ("loose", "loose")):
            e = np.array([r[k + "_e"] for r in sub])
            s = np.array([r[k + "_s"] for r in sub])
            print("  %-18s N %2d %-6s: stand-in median %.2f, sessions >= 0.90: %d | exact median %.2f, sessions >= 0.90: %d" % (
                label, len(sub), nm, np.median(s), int((s >= 0.9).sum()), np.median(e), int((e >= 0.9).sum())))
    sub = [r for r in rows if r["cat"] == "dim/hazy noted"]
    print("  dim/hazy noted: top-50 peak PNR median %.1f [%.1f-%.1f] vs exact min_pnr median %.1f [%.1f-%.1f]" % (
        np.median([r["pk_pnr_med"] for r in sub]), min(r["pk_pnr_med"] for r in sub), max(r["pk_pnr_med"] for r in sub),
        np.median([r["mp_exact"] for r in sub]), min(r["mp_exact"] for r in sub), max(r["mp_exact"] for r in sub)))
    keys = ["session", "animal", "cat", "T", "noise_rel", "rule", "mp_stand", "mp_exact", "mp_raw", "mc_est",
            "strict_s", "strict_e", "near_s", "near_e", "loose_s", "loose_e", "seeds_s", "seeds_e", "pk_pnr_med", "pk_pnr_p10"]
    with open(rc.HERE / "c10_sessions.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

    print("\n== 4. labelled pilots: does min_pnr separate kept from rejected candidates? ==")
    allm, allk = [], []
    extra = {dp: [0, 0, 0, 0] for dp in (0, 1, 2, 3, 4, 6)}       # kept lost, kept, rejected lost, rejected
    for g, name, tif, sd, an in rl.pilots():
        if name not in S:
            continue
        z = S[name][3]
        cn, pnr = z["cn"].astype(float), z["pnr"].astype(float)
        cd = rl.candidates(sd)
        mc, mp = rl.run_params(sd)
        lab = cd["labels"]
        ppk = np.array([pnr[m].max() if m.any() else 0.0 for m in cd["core"]])
        allm.append(ppk - mp)
        allk.append(lab)
        base = rl.cell_ceiling(cd["core"], cn, pnr, mp)[0] >= mc
        for dp in extra:
            ok = rl.cell_ceiling(cd["core"], cn, pnr, mp + dp)[0] >= mc
            e = extra[dp]
            e[0] += int((base & ~ok & (lab == 1)).sum())
            e[1] += int((base & (lab == 1)).sum())
            e[2] += int((base & ~ok & (lab == 0)).sum())
            e[3] += int((base & (lab == 0)).sum())
        print("  %s: N %d kept %d | run min_corr %.2f min_pnr %.1f | label join ok %s (min cosine %.2f) | "
              "candidates seedable from their own core at the run's thresholds: kept %d/%d, rejected %d/%d" % (
                  name[:22], len(lab), int(lab.sum()), mc, mp, cd["join_ok"], cd["join_min_cos"],
                  int((base & (lab == 1)).sum()), int((lab == 1).sum()), int((base & (lab == 0)).sum()), int((lab == 0).sum())))
    m = np.concatenate(allm)
    k = np.concatenate(allk)
    print("  pooled N %d candidates, %d kept" % (len(k), int(k.sum())))
    print("  kept-rate by PNR margin (max PNR in the footprint core minus the run's min_pnr):")
    for lo, hi in ((-99, 0), (0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 999)):
        sel = (m >= lo) & (m < hi)
        print("    margin %4s to %4s: %3d kept of %4d (%.1f%%)" % (
            lo if lo > -99 else "<", hi if hi < 999 else ">", int(k[sel].sum()), int(sel.sum()),
            100.0 * k[sel].mean() if sel.any() else float("nan")))
    print("  raising min_pnr above the run's value (candidates that could seed from their own core at the run's thresholds):")
    for dp, e in extra.items():
        print("    +%d: kept lost %d of %d (%.1f%%), rejected lost %d of %d (%.1f%%)" % (
            dp, e[0], e[1], 100.0 * e[0] / max(e[1], 1), e[2], e[3], 100.0 * e[2] / max(e[3], 1)))


if __name__ == "__main__":
    main()
