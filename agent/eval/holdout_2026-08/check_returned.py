"""
check_returned.py -- how did the two vCA1 sessions returned on 2026-08-27 fare
against the deployed 35-column model?

Why these two are a good test:
  Both were pushed to the reviewer at THRESHOLD 0.00 (review_summary.txt:
  "Auto-rejected: 0", 152 and 167 candidates), so a human labelled EVERY
  candidate.  Nothing is hidden behind an auto-reject, which means the false
  auto-reject rate is directly observable here rather than inferred.

Why the live joblib cannot be used:
  the watcher auto-retrained at 2026-08-28 18:05 and included both sessions
  (136 sessions / 25 agent).  Scoring them with that model would be in-sample.
  Instead this fits the deployed recipe on the OTHER 134 sessions and scores
  the two held-out ones -- a genuine leave-these-out test.

Also runs the same test on the first 13 columns, so the question "did the
35-col upgrade actually help on data it had never seen?" gets a paired answer
on fresh human labels.

Recipe (train_classifier_vCA1 / rt_lib): every labelled session in training,
agent rows at AGENT_WEIGHT_OVERRIDE = 5.0, bootstrap rows at their own weights
(ambiguous + duplicate rows weight 0), StandardScaler, XGB 300/0.05/4/0.8/0.8,
8 model seeds.  Read-only.

Usage:  python check_returned.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RT = HERE.parents[0] / "bootstrap_redteam_2026-08"
sys.path.insert(0, str(RT))
import rt_lib as rt  # noqa: E402

HELD_OUT = [
    "6odorDualDiffRew/AVG5x-TSeries-062226-pnb97-510um-3z-000",
    "6odorDualDiffRew/AVG5x-TSeries-062426-pnb97-612um-24z-000",
]
T = 0.04
WIDTH = 35
OUT = HERE / "holdout_check.json"

# the deployed session-grouped expectation, from agent/eval/pareto_2026-08
EXPECTED = {"far": 0.48, "junk": 48.4, "auc": 0.9081}


def main():
    recs = rt.load_pool("vCA1", width=WIDTH)
    held = [r for r in recs if r["name"] in HELD_OUT]
    train = [r for r in recs if r["name"] not in HELD_OUT]
    if len(held) != len(HELD_OUT):
        sys.exit(f"expected {len(HELD_OUT)} held-out sessions, found {len(held)}")
    rt.log(f"pool {len(recs)} labelled sessions -> train {len(train)} / held out {len(held)}")

    for r in held:
        n = r["n_cand"]
        rt.log(f"  held out {r['name'].split('/')[-1]}: {n} candidates, "
               f"{int(r['auto_rejected'].sum())} auto-rejected, "
               f"{int(r['reviewed'].sum())} reviewed, {int(r['y'].sum())} kept by the human")

    ag = [r for r in train if not r["is_bootstrap"]]
    bs = [r for r in train if r["is_bootstrap"]]
    X_ag = np.vstack([r["X"] for r in ag]); y_ag = np.concatenate([r["y"] for r in ag])
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs, width=WIDTH)
    agw = rt.AGENT_WEIGHT_OVERRIDE["vCA1"]
    rt.log(f"train rows: {len(y_ag)} agent (weight {agw}) + {len(y_bs)} bootstrap "
           f"({int((w_bs == 0).sum())} masked)")

    X_te = np.vstack([r["X"] for r in held])
    y_te = np.concatenate([r["y"] for r in held])
    sess_te = np.concatenate([[r["name"]] * r["n_cand"] for r in held])

    res = {"held_out": HELD_OUT, "T": T, "n_train_sessions": len(train),
           "n_test_rows": int(len(y_te)), "n_test_real": int(y_te.sum()),
           "expected_from_cv": EXPECTED, "widths": {}}

    scores = {}
    for width in (WIDTH, 13):
        S = []
        for seed in rt.SEEDS:
            S.append(rt.fit_predict(
                np.vstack([X_ag[:, :width], X_bs[:, :width]]),
                np.concatenate([y_ag, y_bs]),
                np.concatenate([np.full(len(y_ag), agw), w_bs]),
                X_te[:, :width], xgb_seed=seed))
        S = np.array(S)
        scores[width] = S
        far = [rt.far_junk(s, y_te, T)[0] for s in S]
        junk = [rt.far_junk(s, y_te, T)[1] for s in S]
        aucs = [rt.auc(y_te, s) for s in S]
        entry = {"auc": rt.summarize(aucs), "far_at_T": rt.summarize(far),
                 "junk_at_T": rt.summarize(junk),
                 "reals_lost_at_T": rt.summarize([int((s[y_te == 1] < T).sum()) for s in S]),
                 "junk_caught_at_T": rt.summarize([int((s[y_te == 0] < T).sum()) for s in S]),
                 "per_session": {}}
        for r in held:
            m = sess_te == r["name"]
            entry["per_session"][r["name"]] = {
                "n": int(m.sum()), "n_real": int(y_te[m].sum()),
                "auc": rt.summarize([rt.auc(y_te[m], s[m]) for s in S]),
                "far_at_T": rt.summarize([rt.far_junk(s[m], y_te[m], T)[0] for s in S]),
                "junk_at_T": rt.summarize([rt.far_junk(s[m], y_te[m], T)[1] for s in S]),
                "reals_lost_at_T": rt.summarize([int((s[m][y_te[m] == 1] < T).sum()) for s in S]),
            }
        res["widths"][str(width)] = entry
        rt.log(f"[{width:2d} col] AUC {entry['auc']['mean']:.4f}  "
               f"FAR@{T} {entry['far_at_T']['mean']:.2f}% "
               f"({entry['reals_lost_at_T']['mean']:.1f} of {int(y_te.sum())} real cells)  "
               f"junk {entry['junk_at_T']['mean']:.1f}% "
               f"({entry['junk_caught_at_T']['mean']:.0f} of {int((y_te == 0).sum())})")

    mop = rt.matched_operating_point(scores[13], scores[35], y_te, T)
    res["v2_vs_v1_matched"] = mop
    res["delta_auc_35_minus_13"] = rt.paired_delta(
        [rt.auc(y_te, s) for s in scores[13]], [rt.auc(y_te, s) for s in scores[35]])
    d = res["delta_auc_35_minus_13"]
    rt.log(f"35col - 13col on these two: AUC {d['mean']:+.4f} "
           f"(min {d['min']:+.4f}, {d['n_positive']}/{d['n']} seeds)")
    rt.log(f"  junk at matched false-AR: {mop['junk_ref_mean']:.1f}% -> {mop['junk_new_at_matched_far_mean']:.1f}%")

    # threshold curve on this fresh, fully-labelled data
    res["curve_35col"] = {}
    for t in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15):
        f = [rt.far_junk(s, y_te, t) for s in scores[35]]
        res["curve_35col"][f"{t:.3f}"] = {
            "far": float(np.mean([x[0] for x in f])),
            "far_worst": float(np.max([x[0] for x in f])),
            "junk": float(np.mean([x[1] for x in f])),
            "reals_lost": float(np.mean([(s[y_te == 1] < t).sum() for s in scores[35]])),
        }
    rt.log(f"\n  {'T':>6} {'FAR':>7} {'worst':>7} {'junk':>7} {'cells lost':>11}")
    for t, v in res["curve_35col"].items():
        star = "  <-- deployed" if abs(float(t) - T) < 1e-9 else ""
        rt.log(f"  {float(t):>6.3f} {v['far']:>6.2f}% {v['far_worst']:>6.2f}% "
               f"{v['junk']:>6.1f}% {v['reals_lost']:>10.1f}{star}")

    np.savez_compressed(HERE / "holdout_scores.npz", y=y_te, sess=sess_te,
                        s35=scores[35], s13=scores[13])
    OUT.write_text(json.dumps(res, indent=1, default=rt._jsonable))
    rt.log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
