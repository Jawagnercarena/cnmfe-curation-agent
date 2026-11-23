"""
threshold_oof.py -- vCA1 auto-reject threshold on the pool of 2026-11-20
(211 labelled sessions after the cohort-5 returns by Taylor and Yawen).

Read-only.  Evaluation harness of agent/eval/bootstrap_redteam_2026-08/rt_lib:
35-column features, bootstrap sessions always in training at their stored
weights, agent weight 5.0 (config_vCA1), xgboost with threads pinned.

  session-grouped  StratifiedGroupKFold by session, 5 folds: the session is
                   held out, the animal's other sessions are in training.  This
                   is the situation of the remaining sessions of known animals.
  animal-grouped   StratifiedGroupKFold by animal, 5 folds; bootstrap rows of
                   the held-out animals are dropped from training too.  This is
                   the situation of a new animal.

Blocks reported (false auto-reject = kept cells scored below T / kept cells):
  whole agent pool           the basis of the standing rule (largest T with mean
                             <= 0.85% and worst seed <= 1.0% under both groupings)
  cohort-5 pilots            12 sessions reviewed at threshold 0: every candidate
                             was shown, so the false auto-reject is unbiased
  cohort-5 09xx              19 sessions curated at 0.01: cells the curating
                             model scored below 0.01 were never shown, so the
                             figure is a LOWER BOUND; split by reviewer
  rest of the agent pool     older animals (also censored by their own curation)

Usage: python threshold_oof.py [--seeds 8] [--threads 8]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "bootstrap_redteam_2026-08"))
import rt_lib as rt  # noqa: E402

AREA = "vCA1"
WIDTH = 35
COHORT = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
TS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]


def reviewer_of(sd: Path) -> str:
    p = sd / "labels_provenance.txt"
    if p.exists():
        m = re.search(r"reviewer:\s*(\S+)", p.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1)
    return "?"


def block(oofs, y, rows, rev):
    out = {}
    n_pos = int(((y == 1) & rows).sum())
    n_neg = int(((y == 0) & rows).sum())
    n_neg_rev = int(((y == 0) & rows & rev).sum())
    for T in TS:
        miss = [int(((o < T) & (y == 1) & rows).sum()) for o in oofs]
        junk = [100.0 * ((o < T) & (y == 0) & rows).sum() / max(n_neg, 1) for o in oofs]
        junk_r = [100.0 * ((o < T) & (y == 0) & rows & rev).sum() / max(n_neg_rev, 1) for o in oofs]
        out["%.3f" % T] = {"n_pos": n_pos, "miss_mean": float(np.mean(miss)), "miss_worst": int(max(miss)),
                           "far_mean": 100.0 * float(np.mean(miss)) / max(n_pos, 1),
                           "far_worst": 100.0 * max(miss) / max(n_pos, 1),
                           "junk_all": float(np.mean(junk)), "junk_reviewed": float(np.mean(junk_r))}
    return out


def show(label, blk):
    n_pos = blk["%.3f" % TS[0]]["n_pos"]
    print("  %-34s kept %4d" % (label, n_pos))
    print("      %-8s %10s %12s %12s %10s %14s" % ("T", "FAR mean", "worst seed", "misses", "junk all", "junk reviewed"))
    for T in TS:
        b = blk["%.3f" % T]
        print("      %-8.3f %9.2f%% %11.2f%% %6.1f /%4d %9.1f%% %13.1f%%" % (
            T, b["far_mean"], b["far_worst"], b["miss_mean"], n_pos, b["junk_all"], b["junk_reviewed"]))


def rule(blk_by_grouping):
    """largest T with mean <= 0.85% and worst seed <= 1.0% under every grouping given."""
    best = None
    for T in TS:
        k = "%.3f" % T
        if all(b[k]["far_mean"] <= 0.85 and b[k]["far_worst"] <= 1.0 for b in blk_by_grouping):
            best = T
    return best


def oof_animal(X, y, an_idx, Xb, yb, wb, bs_an, an_names, seed, agent_weight, n_jobs):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, an_idx):
        held = set(an_names[np.unique(an_idx[te])].tolist())
        m = np.array([a not in held for a in bs_an])
        agw = rt.fold_agent_weight(int(m.sum()), len(tr), agent_weight)
        X_tr = np.vstack([X[tr], Xb[m]])
        y_tr = np.concatenate([y[tr], yb[m]])
        w_tr = np.concatenate([np.full(len(tr), agw), wb[m]])
        oof[te] = rt.fit_predict(X_tr, y_tr, w_tr, X[te], True, 42, n_jobs=n_jobs)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    seeds = rt.SEEDS[:a.seeds]

    recs = rt.load_pool(AREA, width=WIDTH)
    cv, rest, bs = rt.split_pool(recs)
    X, y, g, rev, animals, names = rt.stack_agent(cv)
    Xb, yb, wb = rt.stack_bootstrap(bs, width=WIDTH)
    bs_an = np.concatenate([[r["animal"] or "?"] * len(r["y"]) for r in bs])
    print("pool: %d labelled sessions (%d agent testable, %d agent untestable < %d kept, %d bootstrap)" % (
        len(recs), len(cv), len(rest), rt.MIN_POS, len(bs)))
    print("agent rows %d, kept %d, reviewed %d; bootstrap rows %d (weight > 0: %d)" % (
        len(y), int(y.sum()), int(rev.sum()), len(yb), int((wb > 0).sum())))
    print("seeds %s; xgboost threads %d; agent weight %s" % (seeds, a.threads, rt.AGENT_WEIGHT_OVERRIDE[AREA]))
    an_names, an_idx = np.unique(animals, return_inverse=True)
    print("animals in the testable agent pool: %d (%s)" % (len(an_names), ", ".join(
        "%s:%d" % (n, len(set(g[an_idx == i]))) for i, n in enumerate(an_names))))
    untest = [r["name"].split("/")[-1].replace("AVG5x-TSeries-", "") for r in rest]
    print("untestable (train only): %s" % untest)

    sess_name = names[g]
    short = np.array([n.split("/")[-1].replace("AVG5x-TSeries-", "") for n in names])[g]
    is_coh = np.array([any(("-%s-" % c) in n for c in COHORT) for n in sess_name])
    is_pilot = is_coh & np.array([s.startswith("08") for s in short])
    is_new = is_coh & ~is_pilot
    rv = {r["name"]: reviewer_of(r["session_dir"]) for r in cv}
    reviewer = np.array([rv[n] for n in sess_name])
    n_auto = {r["name"]: int((~r["reviewed"]).sum()) for r in cv}
    censored = np.array([n_auto[n] > 0 for n in sess_name])
    print("cohort-5 sessions testable: %d pilots (threshold 0), %d 09xx (curated at 0.01): %s" % (
        len(set(sess_name[is_pilot])), len(set(sess_name[is_new])),
        ", ".join("%s %d" % (r, len(set(sess_name[is_new & (reviewer == r)]))) for r in sorted(set(reviewer[is_new])))))
    print("agent sessions reviewed with no auto-reject at all: %d of %d" % (
        len(set(sess_name[~censored])), len(names)))

    res = {"seeds": seeds, "threads": a.threads, "thresholds": TS, "n_pool": len(recs)}
    OOF = {}
    OOF["session"] = [rt.run_oof(X, y, g, Xb, yb, wb, seed=s, agent_weight=rt.AGENT_WEIGHT_OVERRIDE[AREA],
                                 n_jobs=a.threads) for s in seeds]
    OOF["animal"] = [oof_animal(X, y, an_idx, Xb, yb, wb, bs_an, an_names, s, rt.AGENT_WEIGHT_OVERRIDE[AREA],
                                a.threads) for s in seeds]
    allrows = np.ones(len(y), bool)
    for grp in ("session", "animal"):
        oofs = OOF[grp]
        print("\n===== %s-grouped, %d seeds =====" % (grp, len(seeds)))
        print("  AUC seed %d: all rows %.3f, reviewed rows %.3f; cohort-5 reviewed %.3f; rest reviewed %.3f" % (
            seeds[0], rt.auc(y, oofs[0]), rt.auc(y, oofs[0], rev), rt.auc(y, oofs[0], rev & is_coh),
            rt.auc(y, oofs[0], rev & ~is_coh)))
        res[grp] = {}
        blocks = [("WHOLE AGENT POOL (%d sessions)" % len(names), allrows),
                  ("cohort-5 pilots, threshold 0 (12)", is_pilot),
                  ("cohort-5 09xx, curated at 0.01 (19)", is_new),
                  ("  09xx Taylor (pnb99/102/109)", is_new & (reviewer == "Taylor")),
                  ("  09xx Yawen (pnb105)", is_new & (reviewer == "Yawen")),
                  ("rest of the agent pool", ~is_coh),
                  ("ALL sessions with no auto-reject (%d)" % len(set(sess_name[~censored])), ~censored),
                  ("ALL sessions with auto-rejects (%d)" % len(set(sess_name[censored])), censored)]
        for label, rows in blocks:
            blk = block(oofs, y, rows, rev)
            res[grp][label.strip()] = blk
            show(label, blk)
        print("  per cohort-5 animal, pilots only (unbiased) | 09xx (lower bound), FAR mean (misses/kept) at 0.01 / 0.02 / 0.03:")
        for an in sorted(set(animals[is_coh]), key=lambda s: int(re.sub(r"\D", "", s))):
            parts = []
            for tag, rows in (("pilots", is_pilot & (animals == an)), ("09xx", is_new & (animals == an))):
                if not rows.any():
                    parts.append("%s -" % tag)
                    continue
                blk = block(oofs, y, rows, rev)
                parts.append("%s %s" % (tag, " / ".join("%.2f%% (%.1f/%d)" % (
                    blk["%.3f" % T]["far_mean"], blk["%.3f" % T]["miss_mean"], blk["%.3f" % T]["n_pos"]) for T in (0.01, 0.02, 0.03))))
            print("      %-7s %s" % (an, " | ".join(parts)))
    whole = [res[grp]["WHOLE AGENT POOL (%d sessions)" % len(names)] for grp in ("session", "animal")]
    print("\nstanding rule (largest T with mean <= 0.85 pct and worst seed <= 1.0 pct), T grid %s:" % TS)
    print("   whole agent pool: session grouping T = %s; animal grouping T = %s; BOTH T = %s" % (
        rule(whole[:1]), rule(whole[1:]), rule(whole)))
    pil = [res[grp]["cohort-5 pilots, threshold 0 (12)"] for grp in ("session", "animal")]
    print("   12 threshold-0 pilots only (unbiased, 410 kept): session T = %s; animal T = %s; BOTH T = %s" % (
        rule(pil[:1]), rule(pil[1:]), rule(pil)))
    unc = [res[grp]["ALL sessions with no auto-reject (%d)" % len(set(sess_name[~censored]))] for grp in ("session", "animal")]
    print("   all sessions with no auto-reject (unbiased): session T = %s; animal T = %s; BOTH T = %s" % (
        rule(unc[:1]), rule(unc[1:]), rule(unc)))
    out = HERE / "threshold_oof.json"
    out.write_text(json.dumps(res, indent=1), encoding="ascii")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
