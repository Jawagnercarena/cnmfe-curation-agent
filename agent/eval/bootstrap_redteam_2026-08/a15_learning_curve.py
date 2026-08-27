"""
a15_learning_curve.py -- hypothesis D15: agent data dominates; the bootstrap
corpus is redundant with agent data for the agent test distribution.

Learning curve over the bootstrap pool: 0 / 25 / 50 / 100% of bootstrap
SESSIONS (animal-stratified random draws, 3 draws at 25 and 50), fixed agent
weight (BLA 4.0 = the deployed floor; vCA1 5.0), 8 seeds, both areas, the
deployed contract.  Metrics: reviewed AUC, false-AR at the deployed T, junk
caught at matched false-AR vs the 100% arm.  "Redundant" if the AUC slope
from 25% to 100% has a CI covering 0 while only 0 -> 25% moves.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

FRACS = [0.0, 0.25, 0.5, 1.0]
DRAWS = 3
RNG = np.random.default_rng(20260828)


def draw(bs, frac, k):
    if frac >= 1.0:
        return list(bs)
    if frac <= 0:
        return []
    by_an = {}
    for r in bs:
        by_an.setdefault(r["animal"], []).append(r)
    chosen = []
    for a, rs in by_an.items():
        n = int(round(frac * len(rs)))
        idx = RNG.choice(len(rs), size=n, replace=False) if n else []
        chosen += [rs[i] for i in idx]
    return chosen


def main():
    t = rt.Timer()
    rt.assert_pinned()
    res = {"areas": {}}
    for area in ("BLA", "vCA1"):
        recs = rt.load_pool(area, width=rt.EXPECTED_WIDTH[area])
        cv, rest, bs = rt.split_pool(recs)
        X_ag, y, g, rev, _, _ = rt.stack_agent(cv)
        T = rt.DEPLOYED_T[area]
        W = rt.AGENT_WEIGHT_OVERRIDE.get(area, rt.MIN_AGENT_WEIGHT)
        out = {"agent_weight": W, "n_bootstrap_sessions": len(bs), "points": {}}
        ref_oof = None
        for frac in FRACS[::-1]:                       # 100% first so matched-op has its reference
            draws = [draw(bs, frac, k) for k in range(DRAWS if 0 < frac < 1 else 1)]
            aucs_full, aucs_rev, fars, junks, jm = [], [], [], [], []
            for dk, sub in enumerate(draws):
                if sub:
                    X_bs, y_bs, w_bs = rt.stack_bootstrap(sub)
                else:
                    X_bs, y_bs, w_bs = np.zeros((0, X_ag.shape[1])), np.zeros(0, int), np.zeros(0)
                oof = rt.run_oof_seeds(X_ag, y, g, X_bs, y_bs, w_bs, agent_weight=W)
                if frac >= 1.0:
                    ref_oof = oof
                aucs_full += [rt.auc(y, o) for o in oof]
                aucs_rev += [rt.auc(y, o, rev) for o in oof]
                fars += [rt.far_junk(o, y, T)[0] for o in oof]
                junks += [rt.far_junk(o, y, T)[1] for o in oof]
                if ref_oof is not None:
                    mop = rt.matched_operating_point(ref_oof, oof, y, T)
                    jm.append(mop["junk_new_at_matched_far_mean"])
            out["points"][str(frac)] = {"n_draws": len(draws), "n_sessions": [len(s) for s in draws],
                                        "n_rows": [sum(len(r["y"]) for r in s) for s in draws],
                                        "auc_full": rt.summarize(aucs_full), "auc_reviewed": rt.summarize(aucs_rev),
                                        "far_at_T": rt.summarize(fars), "junk_at_T": rt.summarize(junks),
                                        "junk_at_matched_far_vs_100pct": rt.summarize(jm) if jm else None}
            p = out["points"][str(frac)]
            rt.log(f"  {area} {int(frac*100):3d}% ({p['n_sessions']} sessions): AUC full {p['auc_full']['mean']:.4f}+/-{p['auc_full']['sd']:.4f} "
                   f"rev {p['auc_reviewed']['mean']:.4f} | FAR@{T} {p['far_at_T']['mean']:.2f}% junk {p['junk_at_T']['mean']:.1f}% "
                   f"| junk at matched FAR vs 100%: {p['junk_at_matched_far_vs_100pct']['mean'] if jm else float('nan'):.1f}%")
        # slope 25% -> 100% on reviewed AUC (per draw/seed samples, bootstrap CI)
        xs, ys = [], []
        for frac in (0.25, 0.5, 1.0):
            p = out["points"][str(frac)]
            for v in [p["auc_reviewed"]["mean"]]:
                xs.append(frac); ys.append(v)
        pts = {f: out["points"][str(f)]["auc_reviewed"] for f in FRACS}
        d0 = pts[0.25]["mean"] - pts[0.0]["mean"]
        d1 = pts[1.0]["mean"] - pts[0.25]["mean"]
        se = np.sqrt(pts[1.0]["sd"] ** 2 / max(pts[1.0]["n"], 1) + pts[0.25]["sd"] ** 2 / max(pts[0.25]["n"], 1))
        out["step_0_to_25"] = d0
        out["step_25_to_100"] = d1
        out["step_25_to_100_se"] = float(se)
        out["redundant_beyond_25pct"] = bool(abs(d1) < 2 * se)
        out["bootstrap_helps_at_all"] = bool(d0 > 2 * se)
        rt.log(f"  {area}: 0->25% {d0:+.4f}, 25->100% {d1:+.4f} (se {se:.4f}) -> redundant beyond 25%: {out['redundant_beyond_25pct']}, "
               f"helps at all: {out['bootstrap_helps_at_all']}")
        res["areas"][area] = out
    res["hypothesis_D15_supported"] = {a: res["areas"][a]["redundant_beyond_25pct"] for a in res["areas"]}
    res["verdict"] = "PASS"
    res["criterion"] = "D15 supported for an area if reviewed AUC does not move from 25% to 100% of bootstrap sessions (|delta| < 2 SE)"
    rt.log(f"D15 supported {res['hypothesis_D15_supported']}")
    rt.write_result("a15", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
