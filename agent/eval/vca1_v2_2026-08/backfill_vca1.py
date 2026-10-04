"""
Step 3c: build every vCA1 session's 35-column matrix as a PARALLEL file.

Nothing deployed reads these -- the swap (deferred) renames the winning arm into
place.  candidate_features.npz, labels, and every session output are untouched.

Row policy, per class (Step 4's option (b), backfill_v2.py:7-18, plus the vCA1
bootstrap arms):

  labeled agent (23)  candidate_features_v2.npz
                      reviewed rows: real v2b from the .feature_expansion
                      extraction (order='F' footprints), flag=1;
                      auto-rejected rows: zeros + flag=0 (their traces were
                      overwritten at finalize and are unrecoverable).
                      hiconf from PIN/hiconf_scores.npz (grouped-OOF / LOSO).

  bootstrap (111)     candidate_features_v2.npz  = ARM (a): assemble_v2_bootstrap
                      -- real ranks, zero v2b, flag=0.  What
                      bootstrap_preagent.py:402-404 writes today.
                      _arms/{key}__b0.npz = ARM (b0): real v2b from
                      bootstrap_candidates.npz (order='C' footprints), Cn=None so
                      ring_contrast=0 for every session, flag=1.
                      _arms/{key}__b1.npz = ARM (b1): same but with the session
                      Cn.mat where it is the same resolution as the candidates
                      (71 of 111) -- and that Cn is provably the image the
                      bootstrap run itself used (parity phase 3).
                      hiconf from the leave-session-out bootstrap scores.

  pending (29)        candidate_features_v2.npz
                      real v2b for ALL rows via the production loaders, hiconf
                      from the deployed 13-col joblib, flag=1, and auto_rejected
                      copied VERBATIM -- reviewers are holding these review sets
                      right now and the set they see must not change.

Arm files live in EXT/_arms/, never in session dirs: they are intermediates, and
only the winning arm is ever materialized into a session.

Hard checks per file (backfill_v2.check_common + local): width 13 in / 35 out,
row count unchanged, first 13 columns bit-identical to v1, ranks deterministic,
flag/zero patterns, labeled sessions re-verified against the vendored reference,
and the three bootstrap arms identical in columns 0-25.

Writes backfill_report_vca1.json (per-file sha256 -- the swap kit and
verify compare against it).
"""
import hashlib
import json
import sys

import joblib
import numpy as np
import scipy.io as sio

import vca1_common as vc

REPORT = vc.SP / "backfill_report_vca1.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v1(sd):
    npz = np.load(sd / vc.V1, allow_pickle=True)
    X13 = npz["feature_matrix"]
    assert X13.shape[1] == 13, f"{vc.rel(sd)}: v1 width {X13.shape[1]}"
    names = [str(x) for x in npz["feature_names"]]
    auto = npz["auto_rejected"].flatten().astype(int)
    n = int(npz["n_candidates"][0])
    assert len(X13) == n, f"{vc.rel(sd)}: rows {len(X13)} != n_candidates {n}"
    return X13, names, auto, n


def write_npz(path, X35, names35, auto, n):
    assert X35.shape == (n, 35), f"{path.name}: {X35.shape} != ({n}, 35)"
    np.savez(path, feature_matrix=X35, feature_names=np.array(names35),
             auto_rejected=np.asarray(auto, dtype=int),
             n_candidates=np.array([n]))
    return sha256(path)


def check_common(tag, X13, X35, flag_expect):
    import features
    assert np.array_equal(X35[:, :13], X13), f"{tag}: first 13 columns NOT identical to v1"
    assert np.array_equal(X35[:, 13:26], features.compute_ranks(X13)), f"{tag}: ranks not deterministic"
    assert np.array_equal(X35[:, 34], flag_expect), f"{tag}: flag pattern"
    z = flag_expect == 0
    assert np.all(X35[np.ix_(z, range(26, 34))] == 0), f"{tag}: nonzero v2b under flag=0"


def main():
    import features
    import parity_check as pc
    vc.configure()
    vc.ARMS.mkdir(parents=True, exist_ok=True)

    labeled, bootstrap, pending, skipped = vc.classify_sessions()
    print(f"sessions: {len(labeled)} labeled agent, {len(bootstrap)} bootstrap, "
          f"{len(pending)} pending, {len(skipped)} skipped")
    assert not skipped, f"unclassified sessions with an npz: {[vc.rel(s) for s in skipped]}"

    hic = np.load(vc.PIN / "hiconf_scores.npz", allow_pickle=True)
    ref = np.load(vc.PIN / "vca1_v2b_reference.npz", allow_pickle=True)
    cn_stat = {c["rel"]: c for c in json.loads((vc.SP / "vca1_cn_status.json").read_text())}
    model = joblib.load(str(vc.JOBLIB_LIVE))
    assert model["scaler"].n_features_in_ == 13, "deployed vCA1 joblib is not 13-column"

    report = {"labeled_agent": {}, "bootstrap": {}, "pending": {}}
    n_rows = n_ref_checked = 0

    # ---- labeled agent ----
    for sd in labeled:
        r, k = vc.rel(sd), vc.key(vc.rel(sd))
        X13, names, auto, n = load_v1(sd)
        m = sio.loadmat(str(vc.EXT / (k + ".mat")))
        C = m["C_raw"].astype(float)
        fps = pc.footprints_from_sparse(m["A"], int(m["d1"][0][0]), int(m["d2"][0][0]))
        review_idx = np.array([i for i in range(n) if i not in set(auto.tolist())])
        assert len(review_idx) == C.shape[0], f"{r}: review set {len(review_idx)} != C rows {C.shape[0]}"

        hi = hic[r + "__scores"][review_idx] >= features.HICONF_SCORE
        v2b_rev = features.compute_v2b_features(C, fps, m["Cn"], hi)

        assert np.array_equal(review_idx, ref[k + "__idx"]), f"{r}: review_idx != reference idx"
        assert np.allclose(v2b_rev, ref[k + "__X"], rtol=1e-6, atol=0), \
            f"{r}: written v2b does not match the vendored reference"
        n_ref_checked += 1

        v2b = np.zeros((n, 8))
        v2b[review_idx] = v2b_rev
        flag = np.zeros(n)
        flag[review_idx] = 1.0
        X35 = features.assemble_v2_matrix(X13, v2b, flag)
        check_common(r, X13, X35, flag)
        sh = write_npz(sd / vc.V2, X35, features.v2_feature_names(names), auto, n)
        report["labeled_agent"][r] = {"file": vc.V2, "sha256": sh, "n": n,
                                      "n_reviewed": len(review_idx),
                                      "hiconf": int(hi.sum()),
                                      "hiconf_source": str(hic[r + "__source"])}
        n_rows += n
    print(f"labeled agent: {len(labeled)} written, {n_ref_checked} re-verified "
          f"against the vendored reference")

    # ---- bootstrap: arm a in place, arms b0/b1 in _arms/ ----
    n_ring = 0
    for sd in bootstrap:
        r, k = vc.rel(sd), vc.key(vc.rel(sd))
        X13, names, auto, n = load_v1(sd)
        names35 = features.v2_feature_names(names)

        # arm (a): Step 4 behaviour
        Xa = features.assemble_v2_bootstrap(X13)
        check_common(f"{r}[a]", X13, Xa, np.zeros(n))
        sh_a = write_npz(sd / vc.V2, Xa, names35, auto, n)

        # arms (b0)/(b1): real v2b from the persisted candidates
        z = np.load(sd / "bootstrap_candidates.npz", allow_pickle=True)
        C = z["C_raw"].astype(float)
        fps = vc.bootstrap_footprints(z)                    # order='C'
        assert len(C) == n and len(fps) == n, \
            f"{r}: persisted candidates {len(C)}/{len(fps)} != npz rows {n}"
        hi = hic[r + "__scores"] >= features.HICONF_SCORE

        v2b_b0 = features.compute_v2b_features(C, fps, None, hi)     # Cn=None -> ring 0
        X_b0 = features.assemble_v2_matrix(X13, v2b_b0, 1.0)
        check_common(f"{r}[b0]", X13, X_b0, np.ones(n))
        assert np.all(X_b0[:, 33] == 0), f"{r}[b0]: ring_contrast must be 0 with Cn=None"
        sh_b0 = write_npz(vc.ARMS / f"{k}__b0.npz", X_b0, names35, auto, n)

        ring_ok = cn_stat[r]["status"] == "same_res"
        Cn = vc.load_cn_any(sd / "Cn.mat") if ring_ok else None
        if ring_ok:
            v2b_b1 = features.compute_v2b_features(C, fps, Cn, hi)
            n_ring += 1
        else:
            v2b_b1 = v2b_b0
        X_b1 = features.assemble_v2_matrix(X13, v2b_b1, 1.0)
        check_common(f"{r}[b1]", X13, X_b1, np.ones(n))
        # b0 and b1 may differ ONLY in the ring column
        d = np.abs(X_b0 - X_b1)
        d[:, 33] = 0
        assert d.max() == 0, f"{r}: b0 and b1 differ outside ring_contrast"
        if not ring_ok:
            assert np.all(X_b1[:, 33] == 0), f"{r}[b1]: ring must be 0 without a usable Cn"
        sh_b1 = write_npz(vc.ARMS / f"{k}__b1.npz", X_b1, names35, auto, n)

        # the three arms share columns 0-25 exactly
        assert np.array_equal(Xa[:, :26], X_b0[:, :26]), f"{r}: arms differ in columns 0-25"

        report["bootstrap"][r] = {
            "file": vc.V2, "sha256": sh_a, "n": n,
            "b0": {"file": f"_arms/{k}__b0.npz", "sha256": sh_b0},
            "b1": {"file": f"_arms/{k}__b1.npz", "sha256": sh_b1},
            "ring_available": bool(ring_ok), "cn_status": cn_stat[r]["status"],
            "hiconf": int(hi.sum()), "hiconf_source": str(hic[r + "__source"])}
        n_rows += n
    print(f"bootstrap: {len(bootstrap)} x 3 arms written "
          f"(real ring_contrast on {n_ring}, zero on {len(bootstrap) - n_ring})")

    # ---- pending ----
    for sd in pending:
        r = vc.rel(sd)
        X13, names, auto, n = load_v1(sd)
        traces = features.load_traces(sd)
        fps = features.load_spatial(sd)
        Cn = features.load_cn(sd)
        assert traces.shape[0] == n and fps.shape[0] == n, \
            f"{r}: candidate files do not match the npz row count"
        hi = hic[r + "__scores"] >= features.HICONF_SCORE
        v2b = features.compute_v2b_features(traces, fps, Cn, hi)
        X35 = features.assemble_v2_matrix(X13, v2b, 1.0)
        check_common(r, X13, X35, np.ones(n))
        before = np.load(sd / vc.V1, allow_pickle=True)["auto_rejected"]
        sh = write_npz(sd / vc.V2, X35, features.v2_feature_names(names), auto, n)
        after = np.load(sd / vc.V2, allow_pickle=True)["auto_rejected"]
        assert np.array_equal(before, after), f"{r}: auto_rejected changed -- reviewers hold this set"
        report["pending"][r] = {"file": vc.V2, "sha256": sh, "n": n,
                                "n_auto_rejected": len(auto),
                                "hiconf": int(hi.sum())}
        n_rows += n
    print(f"pending: {len(pending)} written, auto_rejected preserved verbatim")

    n_files = len(labeled) + len(bootstrap) + len(pending)
    report["_summary"] = {
        "n_v2_files": n_files, "n_arm_files": 2 * len(bootstrap),
        "n_rows": n_rows, "n_ring_available": n_ring,
        "agent_weight": vc.AGENT_WEIGHT,
        "pool_manifest_sha": sha256(vc.SP / "vca1_pool_manifest.json")}
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"\nALL WRITTEN: {n_files} candidate_features_v2.npz + "
          f"{2 * len(bootstrap)} arm files, {n_rows} rows total")
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
