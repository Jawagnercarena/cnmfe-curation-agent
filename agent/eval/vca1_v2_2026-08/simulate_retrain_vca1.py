"""
Step 3e: build the retrain-identical 35-column joblib into a REHEARSAL dir.

Mirrors what `train_classifier_vCA1.py --prospective-only --model xgboost
--threshold <T>` would produce after the swap: the 35-column final model plus the
companion 13-column first-pass model, with the same weights, masks and joblib
keys (train_classifier.py:928-944, 1003-1016).

Writes to EXT/_rehearsal/model/classifier.joblib -- NEVER to agent/model/vCA1.
dryrun_curate_vca1.py then curates a real session against it, which is the
end-to-end proof that two-pass 35-column scoring works for this area before any
deploy touches production.
"""
import json
import sys

import joblib
import numpy as np

import vca1_common as vc


def main():
    vc.configure()
    dec = json.loads((vc.SP / "gate_decision.json").read_text())
    arm, T = dec["arm"], dec["chosen_T"]
    W = dec.get("agent_weight") or vc.AGENT_WEIGHT
    print(f"rehearsal retrain: arm {arm}, weight {W:g}, T {T}")

    import train_classifier as tc
    records = vc.load_pool(v2=True, arm=vc.ARM_A, require_width=35)  # v2 files are materialized
    cv, rest, bs = vc.split_pool(records)
    agent = cv + rest
    X = np.vstack([r["X"] for r in records])
    y = np.concatenate([r["y"] for r in records])
    w = np.concatenate([(r["w"] if r["is_bootstrap"] else np.ones(len(r["y"])) * W)
                        for r in records])
    n_masked = int((w == 0).sum())
    keep = w > 0
    print(f"  corpus: {len(records)} sessions ({len(agent)} agent, {len(bs)} bootstrap), "
          f"{len(y)} rows, {n_masked} masked, {int(keep.sum())} active")

    scaler, clf = tc.train_model(X[keep], y[keep], sample_weight=w[keep],
                                 model_type="xgboost")
    fp_scaler, fp_clf = tc.train_model(X[keep][:, :13], y[keep], sample_weight=w[keep],
                                       model_type="xgboost")
    out = {"scaler": scaler, "clf": clf, "model_type": "xgboost",
           "reject_threshold": T, "agent_weight": W,
           "n_features": 35, "n_sessions": len(records),
           "n_training_active": int(keep.sum()),
           "n_excluded_ambiguous": n_masked,
           "first_pass_scaler": fp_scaler, "first_pass_clf": fp_clf,
           "feature_version": 2}
    d = vc.REHEARSAL / "model"
    d.mkdir(parents=True, exist_ok=True)
    assert vc.REHEARSAL != vc.MODEL_DIR and "vCA1\.feature_expansion" in str(d), \
        f"refusing to write outside the rehearsal dir: {d}"
    joblib.dump(out, str(d / "classifier.joblib"))
    print(f"  wrote {d / 'classifier.joblib'}")
    print(f"  keys: 35-col scaler + clf, companion 13-col first-pass, T={T}, "
          f"feature_version=2, n_sessions={len(records)}, masked={n_masked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
