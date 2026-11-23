"""C7 report: the reviewer's kept cells under artificial dimming (cache/degrade from c7_degrade.py).

Ground truth: each pilot's candidates and Taylor's labels (rt_labels.candidates).  A cell
"can seed" at (min_corr c, min_pnr p) when some pixel of its footprint core passes and
survives the 5 x 5 median filter there; its "ceiling" is the highest c for which that holds.
Per (session, variant, level):
  pk50 and the rule's min_corr computed on the degraded movie (f 0.444, floor by length)
  min_pnr = params.py's estimate on the degraded pre-pass (what ACORN would use)
  kept cells: peak Cn in the core (p5 / p25 / p50); ceiling p5 and p10 (the min_corr that
              still keeps 95% / 90% of them); share that can seed at 0.70, at the rule, at
              the floor -- over all kept cells and over those that could seed in the
              original movie at the run's own thresholds ("base")
  rejected candidates: share that can seed at the same three thresholds
  first-pass seeds at each threshold: total, inside a kept core, inside a rejected core, elsewhere
Usage: c7_report.py > c7_report.txt
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import rt_labels as rl  # noqa: E402

F_PREV = 0.444


def floor_prev(T):
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


def seeds_where(cn, pnr, c, p, kept_any, rej_any):
    m = rc.first_pass(cn, pnr, c, p, return_mask=True)
    return int(m.sum()), int((m & kept_any).sum()), int((m & rej_any & ~kept_any).sum()), int((m & ~kept_any & ~rej_any).sum())


def main():
    rows = []
    for g, name, tif, sd, an in rl.pilots():
        files = sorted((rc.CACHE / "degrade").glob(name + "__*.npz"))
        base_f = rc.CACHE / (name + ".npz")
        if not files or not base_f.exists():
            continue
        cd = rl.candidates(sd)
        mc_run, mp_run = rl.run_params(sd)
        lab = cd["labels"]
        kc, rj = cd["core"][lab == 1], cd["core"][lab == 0]
        st = ndi.generate_binary_structure(2, 2)
        kept_any = ndi.binary_dilation(kc.any(axis=0), st)
        rej_any = ndi.binary_dilation(rj.any(axis=0), st)
        zb = np.load(base_f)
        base_ok = rl.cell_ceiling(kc, zb["cn"].astype(float), zb["pnr"].astype(float), mp_run)[0] >= mc_run
        conds = [("orig", 0.0, dict(cn=zb["cn"], pnr=zb["pnr"], est_mp=float(zb["est_mp"]), T=int(zb["T"]),
                                    noise_rel_got=float(zb["noise_rel"]), sigma=0.0))]
        for f in files:
            z = np.load(f)
            conds.append((str(z["variant"]), float(z["level"]), dict(cn=z["cn"], pnr=z["pnr"], est_mp=float(z["est_mp"]),
                                                                     T=int(z["T"]), noise_rel_got=float(z["noise_rel_got"]),
                                                                     sigma=float(z["sigma"]))))
        print("== %s: %d candidates, %d kept (label join ok %s); production run min_corr %.2f min_pnr %.1f; "
              "kept cells that could seed from their own core in the original movie: %d ==" % (
                  name, len(lab), int(lab.sum()), cd["join_ok"], mc_run, mp_run, int(base_ok.sum())))
        print("variant level | noise_rel got | pk50 | rule | est min_pnr | kept peak Cn p5/p25/p50 | kept ceiling p5/p10 "
              "(base cells) | base kept cells that can seed at 0.70 / rule / floor | all kept: same | rejected: same | "
              "first-pass seeds total (in kept / in rejected / elsewhere) at 0.70 ; rule ; floor")
        for variant, level, c in conds:
            cn, pnr, T = c["cn"].astype(float), c["pnr"].astype(float), c["T"]
            fl = floor_prev(T)
            pk50 = rc.pk(cn, 50)
            rule = round(fl + F_PREV * (pk50 - fl), 2)
            p = c["est_mp"]
            ce_k, pk_k = rl.cell_ceiling(kc, cn, pnr, p)
            ce_r, _ = rl.cell_ceiling(rj, cn, pnr, p)
            ce_k3, _ = rl.cell_ceiling(kc, cn, pnr, 3.0)
            thr = (0.70, rule, fl)
            sb = [float((ce_k[base_ok] >= t).mean()) for t in thr]
            sa = [float((ce_k >= t).mean()) for t in thr]
            sr = [float((ce_r >= t).mean()) for t in thr]
            sb3 = [float((ce_k3[base_ok] >= t).mean()) for t in thr]
            sw = [seeds_where(cn, pnr, t, p, kept_any, rej_any) for t in thr]
            r = dict(session=name, animal=an, variant=variant, level=level, noise_rel=c["noise_rel_got"], sigma=c["sigma"],
                     T=T, pk50=pk50, pk10=rc.pk(cn, 10), pk25=rc.pk(cn, 25), rule=rule, floor=fl, est_mp=p,
                     n_kept=int(len(kc)), n_base=int(base_ok.sum()),
                     pkc_p5=float(np.percentile(pk_k, 5)), pkc_p25=float(np.percentile(pk_k, 25)), pkc_p50=float(np.median(pk_k)),
                     ceil_p5=float(np.percentile(ce_k[base_ok], 5)), ceil_p10=float(np.percentile(ce_k[base_ok], 10)),
                     ceil3_p5=float(np.percentile(ce_k3[base_ok], 5)),
                     b070=sb[0], brule=sb[1], bfloor=sb[2], a070=sa[0], arule=sa[1], afloor=sa[2],
                     r070=sr[0], rrule=sr[1], rfloor=sr[2], b070_p3=sb3[0], brule_p3=sb3[1], bfloor_p3=sb3[2],
                     seeds070=sw[0][0], seedsrule=sw[1][0], seedsfloor=sw[2][0],
                     else070=sw[0][3], elserule=sw[1][3], elsefloor=sw[2][3])
            rows.append(r)
            print("  %-5s %.2f | %.3f | %.3f | %.2f | %.1f | %.2f/%.2f/%.2f | %.3f/%.3f | %.2f / %.2f / %.2f | "
                  "%.2f / %.2f / %.2f | %.2f / %.2f / %.2f | %d (%d/%d/%d) ; %d (%d/%d/%d) ; %d (%d/%d/%d)" % (
                      variant, level, r["noise_rel"], pk50, rule, p, r["pkc_p5"], r["pkc_p25"], r["pkc_p50"],
                      r["ceil_p5"], r["ceil_p10"], *sb, *sa, *sr, *sw[0], *sw[1], *sw[2]))
        print()
    if not rows:
        print("no degrade outputs yet")
        return
    print("== summary over sessions: base kept cells that can still seed (median over sessions [min-max]) ==")
    by = defaultdict(list)
    for r in rows:
        by[(r["variant"], r["level"])].append(r)
    for (variant, level), sub in sorted(by.items(), key=lambda kv: (kv[0][0] != "orig", kv[0][0], kv[0][1])):
        f = lambda k: "%.2f [%.2f-%.2f]" % (np.median([r[k] for r in sub]), min(r[k] for r in sub), max(r[k] for r in sub))
        print("  %-5s %.2f N %d | noise_rel %s | pk50 %s | rule %s | est min_pnr %s | at 0.70 %s | at the rule %s | at the floor %s | "
              "with min_pnr 3: %s / %s / %s | ceiling p5 %s" % (
                  variant, level, len(sub), f("noise_rel"), f("pk50"), f("rule"), f("est_mp"), f("b070"), f("brule"),
                  f("bfloor"), f("b070_p3"), f("brule_p3"), f("bfloor_p3"), f("ceil_p5")))
    print("\n== sessions x levels where the rule keeps < 95% of base kept cells ==")
    bad = [r for r in rows if r["variant"] != "orig" and r["brule"] < 0.95]
    print("  %d of %d degraded conditions" % (len(bad), sum(r["variant"] != "orig" for r in rows)))
    print("\n== what min_corr would keep 95% (ceiling p5): relation to pk50 ==")
    x = np.array([r["pk50"] for r in rows])
    for key, label in (("ceil_p5", "95% kept, min_pnr as estimated"), ("ceil_p10", "90% kept, min_pnr as estimated"),
                       ("ceil3_p5", "95% kept, min_pnr 3")):
        y = np.array([r[key] for r in rows])
        A = np.stack([np.ones_like(x), x], axis=1)
        co, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ co
        rulev = np.array([r["floor"] + F_PREV * (r["pk50"] - r["floor"]) for r in rows])
        print("  %-32s: needed = %.3f + %.3f * pk50 (rms %.3f, N %d); rule - needed: median %+.3f, max %+.3f; "
              "conditions where the rule is above the needed value: %d" % (
                  label, co[0], co[1], np.sqrt((res ** 2).mean()), len(x), np.median(rulev - y), (rulev - y).max(),
                  int((rulev > y + 1e-9).sum())))
    print("\n== which statistic of the degraded image predicts the needed min_corr (ceiling p5, min_pnr as estimated)? ==")
    y = np.array([r["ceil_p5"] for r in rows])
    for stat in ("pk10", "pk25", "pk50"):
        xs = np.array([r[stat] for r in rows])
        A = np.stack([np.ones_like(xs), xs], axis=1)
        co, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ co
        within = []
        for s in sorted(set(r["session"] for r in rows)):
            i = np.array([r["session"] == s for r in rows])
            if i.sum() >= 3 and np.ptp(xs[i]) > 0:
                within.append(np.polyfit(xs[i], y[i], 1)[0])
        print("  %s: needed = %.3f + %.3f * %s, rms %.3f, max |residual| %.3f; slope inside each session: %s" % (
            stat, co[0], co[1], stat, np.sqrt((res ** 2).mean()), np.abs(res).max(),
            " ".join("%.2f" % w for w in within)))
    keys = list(rows[0].keys())
    with open(rc.HERE / "c7_conditions.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
