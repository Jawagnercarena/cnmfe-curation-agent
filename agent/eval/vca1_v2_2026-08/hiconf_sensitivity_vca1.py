"""
Follow-on check: does the arm-b0 win depend on WHERE the bootstrap hi-confidence
neighbour scores came from?

The backfill used leave-session-out 13-column fits (honest: no session's own
labels informed its own mask).  Production will instead use the companion
13-column first-pass model in the deployed joblib -- out-of-sample for a NEW
bootstrap session, but a different estimator.  If the b0 result moves when the
source changes, the win is partly an artifact of the offline scoring choice and
must not ship.

This builds a third bootstrap arm, "b0prod", identical to b0 except that the
hi-conf mask comes from the deployed 13-column model (which, for these 111
historical sessions, is IN-SAMPLE -- deliberately the opposite extreme from
leave-session-out).  If b0 and b0prod agree, the feature is insensitive to the
score source across the whole range from in-sample to strictly out-of-sample,
and the shipped LSO-based files are safe.

Writes the arm files to _arms/{key}__b0prod.npz, then gate with:
    python gate_vca1.py --arm b0prod --agent-weight 5.0
Read-only on session dirs.
"""
import sys

import joblib
import numpy as np

import vca1_common as vc


def main():
    vc.configure()
    import features
    model = joblib.load(str(vc.JOBLIB_LIVE))
    assert model["scaler"].n_features_in_ == 13
    hic = np.load(vc.PIN / "hiconf_scores.npz", allow_pickle=True)

    _, bs, _, _ = vc.classify_sessions()
    n_diff_rows = n_rows = 0
    agree = []
    for sd in bs:
        r, k = vc.rel(sd), vc.key(vc.rel(sd))
        npz = np.load(sd / vc.V1, allow_pickle=True)
        X13 = npz["feature_matrix"].astype(float)
        names = [str(x) for x in npz["feature_names"]]
        auto = npz["auto_rejected"].flatten().astype(int)
        n = int(npz["n_candidates"][0])

        hi_prod = model["clf"].predict_proba(model["scaler"].transform(X13))[:, 1] >= features.HICONF_SCORE
        hi_lso = hic[r + "__scores"] >= features.HICONF_SCORE
        agree.append(float((hi_prod == hi_lso).mean()))
        n_diff_rows += int((hi_prod != hi_lso).sum())
        n_rows += n

        z = np.load(sd / "bootstrap_candidates.npz", allow_pickle=True)
        v2b = features.compute_v2b_features(
            z["C_raw"].astype(float), vc.bootstrap_footprints(z), None, hi_prod)
        X35 = features.assemble_v2_matrix(X13, v2b, 1.0)
        assert np.array_equal(X35[:, :13], X13)
        np.savez(vc.ARMS / f"{k}__b0prod.npz",
                 feature_matrix=X35, feature_names=np.array(features.v2_feature_names(names)),
                 auto_rejected=auto, n_candidates=np.array([n]))

    print(f"hi-conf mask agreement between the LSO and deployed sources:")
    print(f"  mean per-session {100 * np.mean(agree):.1f}%  "
          f"(min {100 * min(agree):.1f}%)")
    print(f"  rows differing: {n_diff_rows}/{n_rows} ({100 * n_diff_rows / n_rows:.2f}%)")
    print(f"\nwrote {len(bs)} _arms/*__b0prod.npz")
    print("now: python gate_vca1.py --arm b0prod --agent-weight 5.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
