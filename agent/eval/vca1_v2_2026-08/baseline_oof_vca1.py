"""
Step 3b.1: the vCA1 13-column baseline pin.

8 seeds x StratifiedGroupKFold(5) over the agent CV folds, all bootstrap always
in train at the deployed weighting, agent rows at the FIXED AGENT_WEIGHT_OVERRIDE
(5.0) rather than the BLA sqrt/4.0 recipe that threshold_sweep_v2.run_oof:104
hard-codes.  Methodology otherwise line-for-line
../step2_2026-08/repin_baseline.py / threshold_sweep_v2.run_oof.

This pin serves three later purposes:
  * the paired b13 arm every gate compares rankv2b_35 against;
  * the honest (non-in-sample) hiconf neighbour scores for the CV sessions'
    v2b backfill -- the leak the red team caught in Step 2;
  * the reference operating point (FAR / junk at each threshold) on 13 columns.

Writes PIN/baseline_oof.npz (repin_baseline schema, so
parity_check.load_oof_by_session reads it) + baseline_repin_vca1.json.
Read-only on session dirs.
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import vca1_common as vc


def main():
    vc.configure()
    vc.PIN.mkdir(parents=True, exist_ok=True)

    records = vc.load_pool(v2=False, require_width=13)
    cv, rest, bs = vc.split_pool(records)
    print(f"pool: {len(records)} labeled ({len(cv) + len(rest)} agent, {len(bs)} bootstrap)")
    print(f"  CV folds (>= {vc.MIN_POS} positives): {len(cv)}   train-only agent: {len(rest)}")

    # pool identity: the deployed joblib is the cross-check on the mask
    masked = sum(int((r["w"] == 0).sum()) for r in bs)
    import joblib
    jl = joblib.load(str(vc.JOBLIB_LIVE))
    print(f"  bootstrap masked rows: {masked}  (deployed joblib: {jl.get('n_excluded_ambiguous')})")
    assert masked == jl.get("n_excluded_ambiguous"), "ambiguous+duplicate mask disagrees with the deployed model"

    X_ag, y_ag, g_ag = vc.stack(cv, keys=("X", "y"))
    rev_ag = np.concatenate([r["reviewed"] for r in cv])
    names  = np.concatenate([[r["name"]] * len(r["y"]) for r in cv])
    idx_in = np.concatenate([np.arange(len(r["y"])) for r in cv])
    X_bs, y_bs, _ = vc.stack(bs, keys=("X", "y"))
    w_bs = np.concatenate([r["w"] for r in bs])

    pos, neg = y_ag == 1, y_ag == 0
    print(f"  OOF pool: {len(y_ag)} rows, {pos.sum()} real, {neg.sum()} junk "
          f"({(neg & rev_ag).sum()} reviewed junk)")
    print(f"  agent weight: {vc.AGENT_WEIGHT}  (fixed override, not the sqrt recipe)")

    oof_seeds = np.full((len(vc.SEEDS), len(y_ag)), np.nan)
    for si, seed in enumerate(vc.SEEDS):
        oof_seeds[si] = vc.run_oof_fixed(X_ag, y_ag, g_ag, X_bs, y_bs, w_bs, seed)
        print(f"    seed {seed} AUC {roc_auc_score(y_ag, oof_seeds[si]):.4f}", flush=True)

    res = {"seeds": vc.SEEDS, "agent_weight": vc.AGENT_WEIGHT,
           "n_sessions": len(records), "n_cv_sessions": len(cv),
           "n_pool": int(len(y_ag)), "n_real": int(pos.sum()),
           "n_junk": int(neg.sum()), "n_junk_reviewed": int((neg & rev_ag).sum()),
           "n_bootstrap_masked": masked,
           "cv_sessions": [r["name"] for r in cv]}

    for scope, mask in (("full", np.ones(len(y_ag), bool)), ("reviewed", rev_ag)):
        aucs = [roc_auc_score(y_ag[mask], o[mask]) for o in oof_seeds]
        res[f"auc_{scope}"] = [float(a) for a in aucs]
        print(f"\nAUC [{scope:>8}]: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} "
              f"(min {min(aucs):.4f}, max {max(aucs):.4f})")

    print(f"\n{'T':>5}  {'false-AR %':>20}  {'junk full %':>13}  {'junk reviewed %':>15}")
    table = {}
    for t in vc.THRESHOLDS:
        far = [float((o[pos] < t).sum() / pos.sum() * 100) for o in oof_seeds]
        jf  = [float((o[neg] < t).sum() / neg.sum() * 100) for o in oof_seeds]
        jr  = [float((o[neg & rev_ag] < t).sum() / (neg & rev_ag).sum() * 100) for o in oof_seeds]
        table[f"{t:.2f}"] = {"far": far, "junk_full": jf, "junk_reviewed": jr}
        mark = "  <- deployed" if abs(t - vc.DEPLOYED_T) < 1e-9 else ""
        print(f"{t:5.2f}  {np.mean(far):6.2f} +/- {np.std(far):4.2f} (max {max(far):5.2f})"
              f"  {np.mean(jf):6.1f} +/- {np.std(jf):4.1f}  {np.mean(jr):7.1f}{mark}")
    res["threshold_table"] = table

    np.savez(vc.PIN / "baseline_oof.npz",
             oof_seeds=oof_seeds, session=names, idx_in_session=idx_in,
             y=y_ag, reviewed=rev_ag, seeds=np.array(vc.SEEDS))
    (vc.SP / "baseline_repin_vca1.json").write_text(json.dumps(res, indent=1))
    print(f"\nsaved {vc.PIN / 'baseline_oof.npz'}")
    print(f"saved baseline_repin_vca1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
