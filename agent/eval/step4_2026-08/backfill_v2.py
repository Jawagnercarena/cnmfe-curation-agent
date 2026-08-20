"""
Step 4 backfill: build every BLA session's 35-column feature matrix as
candidate_features_v2.npz NEXT TO the v1 file.  Nothing deployed reads the
parallel files; the atomic swap (Step 7, inside the freeze) renames them into
place.  Never modifies candidate_features.npz, labels, or any session output.

Row policy (red-team option (b), brief decision 2):
  labeled agent  : reviewed rows = real v2b (from the .feature_expansion
                   extraction; hiconf neighbors from pinned grouped-OOF
                   scores, deployed scores for the 4 non-OOF sessions) with
                   flag=1;  auto-rejected rows = zeros + flag=0 (their traces
                   were overwritten at finalize and are unrecoverable).
  bootstrap      : 13 + ranks + zeros + flag=0.
  pending        : real v2b + flag=1 for ALL rows (candidate files intact);
                   hiconf from the CURRENT deployed 13-col model (the
                   production-realistic source; the retrained companion model
                   is near-identical).  auto_rejected copied VERBATIM — the
                   review set a reviewer may already hold must not change.

Per-session checks (all hard): width 13 in, 35 out; row count unchanged;
first 13 columns bit-identical to v1; ranks identical on recompute; flag /
zero patterns; labeled sessions additionally allclose (rtol 1e-6) against the
pinned step2_v2b_features.npz where pinned (the backfill-parity invariant,
re-proven on the actual written values).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

SP = Path(__file__).parent
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(SP))

import features                             # shipping module
import train_classifier as tc               # _is_bootstrap_session
from parity_check import (EXT, PIN, DATA_ROOT, load_oof_by_session,
                          footprints_from_sparse)
import joblib

V2_SUFFIX = "candidate_features_v2.npz"


def deployed_scores13(model, X13):
    return model["clf"].predict_proba(model["scaler"].transform(X13))[:, 1]


def classify_sessions():
    labeled_agent, bootstrap, pending, skipped = [], [], [], []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir() or not (sd / "candidate_features.npz").exists():
                continue
            if (sd / "labels.mat").exists():
                (bootstrap if tc._is_bootstrap_session(sd)
                 else labeled_agent).append(sd)
            elif (sd / "ROIs_candidates.jpg").exists():
                pending.append(sd)
            else:
                skipped.append(sd)
    return labeled_agent, bootstrap, pending, skipped


def load_v1(sd):
    npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
    X13 = npz["feature_matrix"]
    assert X13.shape[1] == features.V1_N_FEATURES, \
        f"{sd.name}: v1 width {X13.shape[1]}"
    names = [str(x) for x in npz["feature_names"]]
    auto_rej = npz["auto_rejected"].flatten().astype(int)
    n_cand = int(npz["n_candidates"][0])
    assert len(X13) == n_cand, f"{sd.name}: rows {len(X13)} != n_cand {n_cand}"
    return X13, names, auto_rej, n_cand


def write_v2(sd, X35, names35, auto_rej, n_cand):
    assert X35.shape == (n_cand, 35)
    np.savez(
        sd / V2_SUFFIX,
        feature_matrix=X35,
        feature_names=np.array(names35),
        auto_rejected=np.asarray(auto_rej, dtype=int),
        n_candidates=np.array([n_cand]),
    )


def check_common(sd, X13, X35, flag_expect):
    assert np.array_equal(X35[:, :13], X13), f"{sd.name}: first-13 NOT identical"
    assert np.array_equal(X35[:, 13:26], features.compute_ranks(X13)), \
        f"{sd.name}: ranks not deterministic"
    assert np.array_equal(X35[:, 34], flag_expect), f"{sd.name}: flag pattern"
    z = flag_expect == 0
    assert np.all(X35[np.ix_(z, range(26, 34))] == 0), \
        f"{sd.name}: nonzero v2b under flag=0"


def main():
    labeled_agent, bootstrap, pending, skipped = classify_sessions()
    print(f"sessions: {len(labeled_agent)} labeled agent, {len(bootstrap)} "
          f"bootstrap, {len(pending)} pending, {len(skipped)} skipped")
    for sd in skipped:
        print(f"  SKIPPED (npz but no labels and no ROIs_candidates.jpg): "
              f"{sd.parent.name}/{sd.name}")

    model = joblib.load(AGENT / "model" / "BLA" / "classifier.joblib")
    oof_by_sess = load_oof_by_session()
    pinned = np.load(PIN / "step2_v2b_features.npz", allow_pickle=True)

    n_rows = 0
    n_pin_checked = 0

    # ---- labeled agent ----
    for sd in labeled_agent:
        rel = f"{sd.parent.name}/{sd.name}"
        key = rel.replace("/", "__")
        X13, names, auto_rej, n_cand = load_v1(sd)
        mat_f = EXT / (key + ".mat")
        assert mat_f.exists(), f"{rel}: no extraction mat — repin step missed it"
        m = sio.loadmat(str(mat_f))
        C = m["C_raw"].astype(float)
        fps = footprints_from_sparse(m["A"], int(m["d1"][0][0]),
                                     int(m["d2"][0][0]))
        auto_set = set(auto_rej.tolist())
        review_idx = np.array([i for i in range(n_cand) if i not in auto_set])
        assert len(review_idx) == C.shape[0], f"{rel}: review N mismatch"

        scores = deployed_scores13(model, X13)
        if rel in oof_by_sess:
            for fi, v in oof_by_sess[rel].items():
                scores[fi] = v
        hi = scores[review_idx] >= features.HICONF_SCORE

        v2b_rev = features.compute_v2b_features(C, fps, m["Cn"], hi)
        if key + "__X" in pinned:
            assert (review_idx == pinned[key + "__idx"]).all(), f"{rel}: idx"
            assert np.allclose(v2b_rev, pinned[key + "__X"], rtol=1e-6, atol=0), \
                f"{rel}: pinned-parity FAILED on written values"
            n_pin_checked += 1

        v2b = np.zeros((n_cand, 8))
        v2b[review_idx] = v2b_rev
        flag = np.zeros(n_cand)
        flag[review_idx] = 1.0
        X35 = features.assemble_v2_matrix(X13, v2b, flag)
        check_common(sd, X13, X35, flag)
        write_v2(sd, X35, features.v2_feature_names(names), auto_rej, n_cand)
        n_rows += n_cand

    print(f"labeled agent done: {len(labeled_agent)} sessions, "
          f"{n_pin_checked} re-verified against the pinned values")

    # ---- bootstrap ----
    for sd in bootstrap:
        X13, names, auto_rej, n_cand = load_v1(sd)
        X35 = features.assemble_v2_bootstrap(X13)
        check_common(sd, X13, X35, np.zeros(n_cand))
        write_v2(sd, X35, features.v2_feature_names(names), auto_rej, n_cand)
        n_rows += n_cand
    print(f"bootstrap done: {len(bootstrap)} sessions")

    # ---- pending ----
    for sd in pending:
        rel = f"{sd.parent.name}/{sd.name}"
        X13, names, auto_rej, n_cand = load_v1(sd)
        traces = features.load_traces(sd)
        fps = features.load_spatial(sd)
        Cn = features.load_cn(sd)
        assert traces.shape[0] == n_cand and fps.shape[0] == n_cand, \
            f"{rel}: candidate files do not match npz row count"
        hi = deployed_scores13(model, X13) >= features.HICONF_SCORE
        v2b = features.compute_v2b_features(traces, fps, Cn, hi)
        X35 = features.assemble_v2_matrix(X13, v2b, 1.0)
        check_common(sd, X13, X35, np.ones(n_cand))
        write_v2(sd, X35, features.v2_feature_names(names), auto_rej, n_cand)
        n_rows += n_cand
        print(f"  pending upgraded: {rel} (N={n_cand}, auto_rejected "
              f"kept verbatim: {len(auto_rej)})")

    print(f"\nALL WRITTEN: {len(labeled_agent) + len(bootstrap) + len(pending)}"
          f" candidate_features_v2.npz files, {n_rows} rows total")


if __name__ == "__main__":
    main()
