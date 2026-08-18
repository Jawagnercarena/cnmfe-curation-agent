"""
C0: re-pin the BLA baseline with the CORRECTED harness (real ambiguous mask)
and, in the same process, reproduce the LEGACY harness (mask all-False — the
pre-2026-08-18 bug) so the delta is quantified once.

Methodology mirrors agent/eval/threshold_robustness.py line-for-line:
8 seeds x StratifiedGroupKFold(5) on the agent pool (sessions with >=5 reals),
all bootstrap sessions in every train fold, per-fold agent weight
max(sqrt(n_bs/n_ag_tr), 4.0), StandardScaler, XGB via dm.make_clf.

Saves: baseline_repin.json (metrics) + baseline_oof.npz (per-candidate
seed-wise OOF scores, corrected mode) for the autopsy and experiments.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
import diagnose_model as dm

SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]
T = 0.12


def run_mode(records, agent_weight, tag):
    ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
    bs_recs = [r for r in records if r["is_bootstrap"]]

    X_ag = np.vstack([r["X"] for r in ag_recs])
    y_ag = np.concatenate([r["y"] for r in ag_recs])
    g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag_recs)])
    names = np.concatenate([[r["name"]] * len(r["y"]) for r in ag_recs])
    idx_in_sess = np.concatenate([np.arange(len(r["y"])) for r in ag_recs])

    X_bs = np.vstack([r["X"] for r in bs_recs])
    y_bs = np.concatenate([r["y"] for r in bs_recs])
    w_bs = np.concatenate([r["w"] for r in bs_recs])

    n_masked = int((w_bs == 0).sum())
    pos, neg = y_ag == 1, y_ag == 0
    assert X_ag.shape[1] == 13 and X_bs.shape[1] == 13

    print(f"\n=== {tag} ===")
    print(f"agent sessions={len(ag_recs)} bootstrap={len(bs_recs)} "
          f"agent_weight={agent_weight:.2f} ambiguous-masked bs rows={n_masked}")
    print(f"OOF pool: {len(y_ag)} cands, {pos.sum()} real, {neg.sum()} garbage")

    oof_seeds = np.full((len(SEEDS), len(y_ag)), np.nan)
    aucs, far12, gc12, band, below05 = [], [], [], [], []
    for si, seed in enumerate(SEEDS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(y_ag), np.nan)
        for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
            n_ag_tr = len(tr_idx)
            ag_w = float(max(np.sqrt(len(y_bs) / n_ag_tr), dm.MIN_AGENT_WEIGHT))
            X_trC = np.vstack([X_ag[tr_idx], X_bs])
            y_trC = np.concatenate([y_ag[tr_idx], y_bs])
            w_trC = np.concatenate([np.ones(n_ag_tr) * ag_w, w_bs])
            sc = StandardScaler()
            X_trCs = sc.fit_transform(X_trC)
            X_teCs = sc.transform(X_ag[te_idx])
            spw = dm.compute_spw(y_trC, w_trC)
            clf = dm.make_clf("xgb", spw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_trCs, y_trC, sample_weight=w_trC)
                oof[te_idx] = clf.predict_proba(X_teCs)[:, 1]
        oof_seeds[si] = oof
        aucs.append(roc_auc_score(y_ag, oof))
        far12.append((oof[pos] < T).sum() / pos.sum() * 100)
        gc12.append((oof[neg] < T).sum() / neg.sum() * 100)
        band.append(int(((oof[pos] >= 0.05) & (oof[pos] < T)).sum()))
        below05.append(int((oof[pos] < 0.05).sum()))

    aucs = np.array(aucs)
    res = {
        "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()),
        "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
        "far_at_012_mean": float(np.mean(far12)), "far_at_012_sd": float(np.std(far12)),
        "gc_at_012_mean": float(np.mean(gc12)), "gc_at_012_sd": float(np.std(gc12)),
        "reals_band_005_012_mean": float(np.mean(band)),
        "reals_below_005_mean": float(np.mean(below05)),
        "n_ambiguous_masked_bs_rows": n_masked,
        "n_agent_sessions_eval": len(ag_recs), "n_bootstrap_sessions": len(bs_recs),
        "n_pool": int(len(y_ag)), "n_real": int(pos.sum()), "n_garbage": int(neg.sum()),
        "agent_weight": agent_weight,
    }
    print(f"AUC {res['auc_mean']:.4f} +/- {res['auc_sd']:.4f} "
          f"(range {res['auc_min']:.4f}-{res['auc_max']:.4f})")
    print(f"@0.12: false-AR {res['far_at_012_mean']:.2f}% +/- {res['far_at_012_sd']:.2f}   "
          f"junk {res['gc_at_012_mean']:.1f}% +/- {res['gc_at_012_sd']:.1f}")
    print(f"reals in [0.05,0.12): {res['reals_band_005_012_mean']:.1f} (seed mean)   "
          f"reals <0.05: {res['reals_below_005_mean']:.1f}")
    return res, oof_seeds, names, idx_in_sess, y_ag


def main():
    manifest_util.assert_unchanged()

    joblib_meta = {}
    try:
        import joblib
        d = joblib.load(AGENT / "model" / "BLA" / "classifier.joblib")
        joblib_meta = {k: d[k] for k in
                       ("model_type", "reject_threshold", "n_sessions",
                        "n_excluded_ambiguous") if k in d}
        print(f"deployed joblib: {joblib_meta}")
    except Exception as e:
        print(f"joblib meta unavailable: {e}")

    # CORRECTED mode (dm now aliases the trainer's real ambiguous mask)
    records, agent_weight = dm.load_all_records()
    corrected, oof_seeds, names, idx_in_sess, y_ag = run_mode(
        records, agent_weight, "CORRECTED (real ambiguous mask)")

    # LEGACY mode: reproduce the bug (mask all-False) exactly
    real_mask_fn = dm._get_bootstrap_ambiguous_mask
    dm._get_bootstrap_ambiguous_mask = lambda sd, n: np.zeros(n, dtype=bool)
    try:
        records_l, agent_weight_l = dm.load_all_records()
        legacy, _, _, _, _ = run_mode(
            records_l, agent_weight_l, "LEGACY (bug: ambiguous mask all-False)")
    finally:
        dm._get_bootstrap_ambiguous_mask = real_mask_fn

    out = {"seeds": SEEDS, "threshold": T, "joblib": joblib_meta,
           "corrected": corrected, "legacy": legacy}
    (SP / "baseline_repin.json").write_text(json.dumps(out, indent=1))
    np.savez(SP / "baseline_oof.npz",
             oof_seeds=oof_seeds, session=names, idx_in_session=idx_in_sess,
             y=y_ag, seeds=np.array(SEEDS))
    print(f"\nsaved baseline_repin.json + baseline_oof.npz")
    d_auc = corrected["auc_mean"] - legacy["auc_mean"]
    print(f"corrected - legacy AUC delta: {d_auc:+.4f}")


if __name__ == "__main__":
    main()
