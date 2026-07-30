"""
refresh_features.py — forward-only feature migration + validation helper.

Context: the full candidate trace matrix is overwritten with the final kept set
when a session is finalized, so trace-derived features (population coherence,
kinetics) cannot be recomputed for reviewed sessions. We therefore roll new
trace features out FORWARD-ONLY: new sessions get real values; historical rows
carry a "not available" indicator and neutral zeros.

One new feature IS backfillable for the whole history: the motion missing-
indicator (motion_data_present), because it depends only on whether Ybg_mean.mat
exists — not on the traces. motion_correlation is already stored in each npz.

Modes:
  --validate-motion   (default) Measure the effect of adding motion_data_present
                      to the CURRENT feature set: grouped OOF AUC + threshold
                      sweep, 13-col baseline vs 14-col. Writes nothing.

  --write-forward     Migrate every candidate_features.npz to the forward-only
                      schema: keep existing columns, insert motion_data_present
                      (from Ybg_mean.mat existence), append pop_coherence /
                      pop_sync_frac / pop_data_present as neutral zeros with
                      present=0. Run ONLY with the watcher stopped, as part of
                      the coordinated swap.

The canonical forward-only column order MUST match
features.compute_session_features. It is asserted below against
features.FORWARD_FEATURE_NAMES when that constant is available.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))

import train_classifier as tc
from config import DATA_ROOT


def log(m=""):
    print(m, flush=True)


# Columns appended for the forward-only schema, in order, after the existing
# 13 features. motion_data_present is INSERTED after motion_correlation at write
# time; the three pop_* columns are appended at the end.
POP_COLS = ["pop_coherence", "pop_sync_frac", "pop_data_present"]


# ---------------------------------------------------------------------------
# Session iteration
# ---------------------------------------------------------------------------

def labeled_sessions():
    out = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for sd in sorted(task_dir.iterdir()):
            if sd.is_dir() and (sd / "candidate_features.npz").exists() \
                    and (sd / "labels.mat").exists():
                out.append(sd)
    return out


def all_npz_sessions():
    out = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for sd in sorted(task_dir.iterdir()):
            if sd.is_dir() and (sd / "candidate_features.npz").exists():
                out.append(sd)
    return out


def has_motion_data(sd):
    """1.0 if a valid background signal was available at curation time."""
    return 1.0 if (sd / "Ybg_mean.mat").exists() else 0.0


def reconstruct_full_labels(sd, N):
    """Mirror train_classifier.load_prospective_session label reconstruction."""
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    y_review      = sio.loadmat(str(sd / "labels.mat"))["labels"].flatten().astype(float)
    auto_rejected = npz["auto_rejected"].flatten().astype(int)
    if N == len(y_review):
        return y_review
    expected_review = N - len(auto_rejected)
    if len(y_review) != expected_review:
        return None
    auto_set = set(auto_rejected.tolist())
    review_indices = [i for i in range(N) if i not in auto_set]
    y_full = np.zeros(N, dtype=float)
    for k, idx in enumerate(review_indices):
        y_full[idx] = y_review[k]
    return y_full


# ---------------------------------------------------------------------------
# Weighting (mirrors train_classifier.main)
# ---------------------------------------------------------------------------

def build_records():
    recs = []
    for sd in labeled_sessions():
        npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
        X   = npz["feature_matrix"]
        names = list(npz["feature_names"])
        y = reconstruct_full_labels(sd, len(X))
        if y is None:
            log(f"  [WARN] {sd.name}: label size mismatch — skipping.")
            continue
        recs.append({
            "X": X, "names": names, "y": y,
            "is_bootstrap": (sd / "bootstrap_match_stats.json").exists(),
            "motion_present": has_motion_data(sd),
            "session_dir": sd,
        })
    return recs


def apply_weights(recs):
    n_bs = sum(len(r["y"]) for r in recs if r["is_bootstrap"])
    n_ag = sum(len(r["y"]) for r in recs if not r["is_bootstrap"])
    agent_weight = (float(max(np.sqrt(n_bs / n_ag), 4.0)) if n_ag and n_bs else 4.0)
    for gi, r in enumerate(recs):
        w = np.ones(len(r["y"]), dtype=float)
        if not r["is_bootstrap"]:
            w *= agent_weight
        else:
            rec = tc._get_bootstrap_recovery(r["session_dir"])
            if rec is not None and rec < tc.BAD_SESSION_RECOVERY_THRESHOLD:
                w *= tc.BAD_SESSION_WEIGHT
            ambig = tc._get_bootstrap_ambiguous_mask(r["session_dir"], len(r["y"]))
            w[ambig] = 0.0
        r["w"] = w
        r["group"] = gi
    return agent_weight


# ---------------------------------------------------------------------------
# Feature-matrix variants
# ---------------------------------------------------------------------------

def matrix_baseline(r):
    return r["X"]


def matrix_with_motion_indicator(r):
    """Insert motion_data_present right after motion_correlation."""
    X = r["X"]
    names = r["names"]
    mc_idx = names.index("motion_correlation")
    col = np.full((len(X), 1), r["motion_present"], dtype=float)
    return np.hstack([X[:, :mc_idx + 1], col, X[:, mc_idx + 1:]])


# ---------------------------------------------------------------------------
# Grouped OOF CV
# ---------------------------------------------------------------------------

def oof_proba(recs, matrix_fn, n_splits=5):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    X = np.vstack([matrix_fn(r) for r in recs])
    y = np.concatenate([r["y"] for r in recs])
    w = np.concatenate([r["w"] for r in recs])
    g = np.concatenate([[r["group"]] * len(r["y"]) for r in recs])
    is_bs = np.concatenate([[r["is_bootstrap"]] * len(r["y"]) for r in recs])

    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in cv.split(X, y, g):
            wt = w[tr]; m = wt > 0; ytr = y[tr]
            ep = float(wt[(ytr == 1) & m].sum())
            en = float(wt[(ytr == 0) & m].sum())
            spw = en / ep if ep > 0 else 1.0
            clf = tc._make_clf("xgboost", scale_pos_weight=spw)
            clf.fit(X[tr], ytr, sample_weight=wt)
            oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof, y, is_bs, roc_auc_score(y, oof), X


def sweep(oof, y, is_bs, thresholds):
    ag = ~is_bs
    reals   = ag & (y == 1)
    garbage = ag & (y == 0)
    out = {}
    for t in thresholds:
        far = float(np.mean(oof[reals]   < t)) if reals.sum()   else float("nan")
        grb = float(np.mean(oof[garbage] < t)) if garbage.sum() else float("nan")
        out[t] = (far, grb)
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def validate_motion(recs):
    thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    aw = apply_weights(recs)
    n_ag = sum(1 for r in recs if not r["is_bootstrap"])
    n_bs = sum(1 for r in recs if r["is_bootstrap"])
    n_present = sum(1 for r in recs if r["motion_present"] > 0)

    log("=" * 70)
    log("VALIDATE #1 — motion missing-indicator (motion_data_present)")
    log("=" * 70)
    log(f"Labeled sessions in CV: {len(recs)}  ({n_ag} agent, {n_bs} bootstrap)")
    log(f"  with Ybg_mean present (indicator=1): {n_present} / {len(recs)}")
    log(f"  agent up-weight: {aw:.2f}x")

    oof_b, y_b, bs_b, auc_b, _  = oof_proba(recs, matrix_baseline)
    oof_m, y_m, bs_m, auc_m, Xm = oof_proba(recs, matrix_with_motion_indicator)

    log(f"\nGrouped OOF AUC (5-fold, split by session):")
    log(f"  baseline (13 feats):          {auc_b:.4f}")
    log(f"  + motion_data_present (14):   {auc_m:.4f}   (delta {auc_m - auc_b:+.4f})")

    sb = sweep(oof_b, y_b, bs_b, thresholds)
    sm = sweep(oof_m, y_m, bs_m, thresholds)
    log(f"\nThreshold sweep on AGENT sessions (clean labels):")
    log(f"  {'thr':>5}  {'BASE false-AR':>13} {'BASE garbage':>12}   "
        f"{'+IND false-AR':>13} {'+IND garbage':>12}")
    for t in thresholds:
        fb, gb = sb[t]; fm, gm = sm[t]
        log(f"  {t:>5.2f}  {fb:>12.1%} {gb:>11.1%}   {fm:>12.1%} {gm:>11.1%}")

    # Importance of the motion features in a full 14-col fit
    y = np.concatenate([r["y"] for r in recs])
    w = np.concatenate([r["w"] for r in recs])
    m = w > 0
    ep = float(w[(y == 1) & m].sum()); en = float(w[(y == 0) & m].sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = tc._make_clf("xgboost", scale_pos_weight=(en / ep if ep else 1.0))
        clf.fit(Xm, y, sample_weight=w)
    names14 = (recs[0]["names"][:recs[0]["names"].index("motion_correlation") + 1]
               + ["motion_data_present"]
               + recs[0]["names"][recs[0]["names"].index("motion_correlation") + 1:])
    imp = clf.feature_importances_
    log(f"\nFeature importance (14-col full fit) — motion features:")
    for nm in ("motion_correlation", "motion_data_present"):
        i = names14.index(nm)
        log(f"  {nm:<22} {imp[i]:.4f}  (rank {int((imp > imp[i]).sum()) + 1}/{len(imp)})")


def write_forward():
    """Migrate all npz to the forward-only 17-col schema (watcher must be down)."""
    try:
        import features as feat
        canonical = getattr(feat, "FORWARD_FEATURE_NAMES", None)
    except Exception:
        canonical = None

    n_ok = 0
    sessions = all_npz_sessions()
    for sd in sessions:
        npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
        X = npz["feature_matrix"]; names = list(npz["feature_names"])
        if "motion_data_present" in names:
            log(f"  [SKIP] {sd.parent.name}/{sd.name}: already migrated.")
            continue
        mc = names.index("motion_correlation")
        ind = np.full((len(X), 1), has_motion_data(sd), dtype=float)
        pop = np.zeros((len(X), len(POP_COLS)), dtype=float)   # neutral + present=0
        new_X = np.hstack([X[:, :mc + 1], ind, X[:, mc + 1:], pop])
        new_names = (names[:mc + 1] + ["motion_data_present"]
                     + names[mc + 1:] + POP_COLS)
        if canonical is not None and new_names != list(canonical):
            log(f"  [ERR] {sd.name}: column order != features.FORWARD_FEATURE_NAMES")
            log(f"        got:      {new_names}")
            log(f"        expected: {list(canonical)}")
            return
        np.savez(
            sd / "candidate_features.npz",
            feature_matrix=new_X,
            feature_names=np.array(new_names),
            auto_rejected=npz["auto_rejected"],
            n_candidates=npz["n_candidates"],
        )
        n_ok += 1
    log(f"\nMigrated {n_ok}/{len(sessions)} npz to the forward-only schema.")
    log("Next: python train_classifier.py --prospective-only --model xgboost")
    log("Then re-sweep the threshold and restart the watcher.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-forward", action="store_true",
                    help="Migrate npz to forward-only schema (watcher must be down).")
    args = ap.parse_args()

    if args.write_forward:
        write_forward()
    else:
        recs = build_records()
        validate_motion(recs)


if __name__ == "__main__":
    main()
