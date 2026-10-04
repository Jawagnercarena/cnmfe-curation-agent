"""
Step 3e: the pre-swap score fixture.

Scores every live 13-column candidate_features.npz with the currently deployed
vCA1 joblib.  swap_vca1.py rehearse then proves the restore path by reproducing
these scores from the BACKUP BYTES with the BACKED-UP joblib -- if that matches
exactly, a rollback provably returns the pipeline to today's behaviour.

Writes preswap_scores.npz ({rel}__scores, reject_threshold).  Read-only.
"""
import sys

import joblib
import numpy as np

import vca1_common as vc


def main():
    vc.configure()
    model = joblib.load(str(vc.JOBLIB_LIVE))
    assert model["scaler"].n_features_in_ == 13, \
        f"deployed joblib is {model['scaler'].n_features_in_}-column, expected 13"
    out = {"reject_threshold": np.array([model.get("reject_threshold", np.nan)])}
    n_rows = 0
    sessions = vc.classify_sessions()
    for sd in sessions[0] + sessions[1] + sessions[2]:
        X = np.load(sd / vc.V1, allow_pickle=True)["feature_matrix"]
        assert X.shape[1] == 13, f"{vc.rel(sd)}: width {X.shape[1]}"
        out[vc.rel(sd) + "__scores"] = model["clf"].predict_proba(
            model["scaler"].transform(X))[:, 1]
        n_rows += len(X)
    np.savez(vc.SP / "preswap_scores.npz", **out)
    n_sess = len(out) - 1
    print(f"deployed model: {model.get('model_type')} @ T={model.get('reject_threshold')}, "
          f"agent_weight {model.get('agent_weight')}, n_sessions {model.get('n_sessions')}")
    print(f"wrote preswap_scores.npz: {n_sess} sessions, {n_rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
