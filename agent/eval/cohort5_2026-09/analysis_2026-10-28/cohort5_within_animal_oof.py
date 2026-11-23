"""
cohort5_within_animal_oof.py -- false auto-reject of the vCA1 model on the 12
cohort-5 pilot sessions when the OTHER sessions of the same animals are in training
(sessions held out, animals seen): the estimate that applies to the remaining
sessions of these six animals.

Read-only.  Uses the evaluation harness (agent/eval/bootstrap_redteam_2026-08/
rt_lib): the whole labelled vCA1 pool, 35-column features, bootstrap sessions
always in training at their stored weights, agent weight 5.0 (config_vCA1),
StratifiedGroupKFold by session, 5 folds, 3 seeds.  Reports, for the cohort-5
rows only: misses and false auto-reject rate at each threshold (mean and worst
seed), junk caught, per animal and pooled; the rest of the agent pool for
reference.  Writes a JSON next to this script.

Usage: python cohort5_within_animal_oof.py [--thresholds 0.005,0.01,0.02,0.03,0.04] [--seeds 42,43,44]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "bootstrap_redteam_2026-08"))
import rt_lib as rt  # noqa: E402

AREA = "vCA1"
WIDTH = 35
COHORT = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")


def far_block(oofs, y, rows, thresholds):
    out = {}
    for T in thresholds:
        miss = [int(((o[rows] < T) & (y[rows] == 1)).sum()) for o in oofs]
        junk = [float(((o[rows] < T) & (y[rows] == 0)).sum()) / max(int((y[rows] == 0).sum()), 1) for o in oofs]
        n_pos = int((y[rows] == 1).sum())
        out["%.3f" % T] = {"miss_mean": float(np.mean(miss)), "miss_worst": int(max(miss)), "n_pos": n_pos,
                           "far_mean": 100.0 * float(np.mean(miss)) / max(n_pos, 1),
                           "far_worst": 100.0 * max(miss) / max(n_pos, 1),
                           "junk_mean": 100.0 * float(np.mean(junk))}
    return out


def fmt(label, blk, thresholds):
    n_pos = blk["%.3f" % thresholds[0]]["n_pos"]
    parts = []
    for T in thresholds:
        b = blk["%.3f" % T]
        parts.append("T=%.3f: FAR %.2f%% (worst seed %.2f%%, %.1f/%d), junk %.0f%%" % (
            T, b["far_mean"], b["far_worst"], b["miss_mean"], n_pos, b["junk_mean"]))
    return "%-28s kept %4d | %s" % (label, n_pos, " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.005,0.01,0.02,0.03,0.04")
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()
    thresholds = [float(t) for t in a.thresholds.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]

    recs = rt.load_pool(AREA, width=WIDTH)
    cv, rest, bs = rt.split_pool(recs)
    X, y, g, rev, animals, names = rt.stack_agent(cv)
    Xb, yb, wb = rt.stack_bootstrap(bs, width=WIDTH)
    print("pool: %d labelled sessions (%d agent testable, %d agent untestable < %d kept, %d bootstrap); %d agent rows, %d kept" % (
        len(recs), len(cv), len(rest), rt.MIN_POS, len(bs), len(y), int(y.sum())))
    coh_names = [n for n in names if any(("-%s-" % c) in n for c in COHORT)]
    print("cohort-5 sessions in the testable pool: %d" % len(coh_names))
    for n in coh_names:
        print("   " + n)
    untest = [r["name"] for r in rest if any(("-%s-" % c) in r["name"] for c in COHORT)]
    if untest:
        print("cohort-5 sessions NOT testable (< %d kept, training only): %s" % (rt.MIN_POS, untest))

    oofs = [rt.run_oof(X, y, g, Xb, yb, wb, seed=s, agent_weight=rt.AGENT_WEIGHT_OVERRIDE[AREA]) for s in seeds]
    sess_of_row = names[g]
    is_coh = np.array([any(("-%s-" % c) in n for c in COHORT) for n in sess_of_row])
    print()
    print("Within-animal OOF (sessions held out, animals seen; %d seeds). Reviewed rows only are all rows here (threshold-0 reviews)." % len(seeds))
    res = {"thresholds": thresholds, "seeds": seeds, "n_pool_sessions": len(recs), "per_animal": {}, "per_session": {}}
    for an in sorted(set(animals[is_coh]), key=lambda x: int(re.sub(r"\D", "", x))):
        rows = is_coh & (animals == an)
        blk = far_block(oofs, y, rows, thresholds)
        res["per_animal"][an] = blk
        print(fmt("%s (%d sessions)" % (an, len(set(sess_of_row[rows]))), blk, thresholds))
    print()
    for n in coh_names:
        rows = sess_of_row == n
        blk = far_block(oofs, y, rows, thresholds)
        res["per_session"][n] = blk
        print(fmt(n[-40:], blk, thresholds))
    print()
    pooled = far_block(oofs, y, is_coh, thresholds)
    res["cohort5_pooled"] = pooled
    print(fmt("COHORT-5 POOLED", pooled, thresholds))
    other = far_block(oofs, y, ~is_coh, thresholds)
    res["rest_of_pool"] = other
    print(fmt("rest of agent pool", other, thresholds))
    print("AUC (seed %d): cohort-5 rows %.3f, rest %.3f" % (seeds[0], rt.auc(y, oofs[0], is_coh), rt.auc(y, oofs[0], ~is_coh)))
    out = HERE / "cohort5_within_animal_oof.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
