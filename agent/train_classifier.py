"""
train_classifier.py
Trains (or retrains) the neuron quality binary classifier.

Run this after completing one or more sessions of MATLAB review so the
classifier learns your keep/delete decisions.

For each completed agent-pipeline session the script handles two paths:

  Prospective (labels.mat + candidate_features.npz both exist):
    These files are written automatically by CNMFe_final_save.m and
    curator.py going forward.  No extra work needed.

  Retrospective (review_neuron.mat + neuron.mat exist, no labels.mat yet):
    Used for the sessions reviewed before this labelling infrastructure was
    in place.  MATLAB is called to extract plain A / C_raw arrays from
    the Sources2D objects, Python then extracts features and derives labels
    by matching footprints between the review set and the final curated set.
    The result is written as candidate_features.npz + labels.mat so the
    session is treated as prospective in all future runs.

Usage:
    python train_classifier.py             # train and save model
    python train_classifier.py --dry-run   # report stats, do not save
    python train_classifier.py --retro-only  # only process retrospective sessions
"""

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT_DIR = Path(__file__).parent
import config
from config import DATA_ROOT, MODEL_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(AGENT_DIR))
import features as feat_module
import run_cnmfe

# Bootstrap sessions with recovery below this are treated as severely noisy
# (extreme CNMFe merging) and their candidates are down-weighted in training.
BAD_SESSION_RECOVERY_THRESHOLD = 0.40
BAD_SESSION_WEIGHT = 0.4

# Floor for the dynamic agent up-weight (max(sqrt(n_bootstrap/n_agent), floor)).
# Agent labels are unconditionally higher quality than bootstrap labels (~41%
# noisy negatives), so the floor should not decay below a meaningful minimum
# regardless of session counts. Empirically validated via 5-fold OOF sweep on
# 13 BLA agent sessions (2026-03-30): performance plateau at 3–5x; floor=4.0
# gives 0.5% false-AR vs 0.8% at 3.13x with no AUC cost.
MIN_AGENT_WEIGHT = 4.0

# Cosine-similarity threshold for matching review candidates to final neurons.
# Neurons kept by the user will still overlap strongly with their updated
# footprint after spatial re-estimation; deleted neurons will not match anything.
SPATIAL_MATCH_THRESHOLD = 0.60

from local_config import MATLAB_EXE, REPO_ROOT as _REPO_ROOT
REPO_ROOT = str(_REPO_ROOT)


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_prospective_sessions() -> list[Path]:
    """Sessions that already have both candidate_features.npz and labels.mat."""
    sessions = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if ((session_dir / "candidate_features.npz").exists() and
                    (session_dir / "labels.mat").exists()):
                sessions.append(session_dir)
    return sessions


def find_retro_sessions() -> list[Path]:
    """
    Agent-pipeline sessions that were reviewed (ROIs.jpg exists) but are
    missing labels.mat, candidate_features.npz, or both.

    This covers two cases:
      - Sessions reviewed before the labelling infrastructure existed
        (no labels.mat and no candidate_features.npz)
      - Sessions where candidate_features.npz was deleted for a feature
        refresh (labels.mat exists but candidate_features.npz does not)
    In either case the retro path re-extracts everything from MATLAB and
    overwrites/creates both files.
    """
    sessions = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            is_agent   = (session_dir / "ROIs_candidates.jpg").exists()
            is_done    = (session_dir / "ROIs.jpg").exists()
            has_revnm  = (session_dir / "review_neuron.mat").exists()
            has_neuron = (session_dir / "neuron.mat").exists()
            has_labels   = (session_dir / "labels.mat").exists()
            has_features = (session_dir / "candidate_features.npz").exists()
            # Include if reviewed but either file is missing
            if is_agent and is_done and has_revnm and has_neuron \
                    and not (has_labels and has_features):
                sessions.append(session_dir)
    return sessions


# ---------------------------------------------------------------------------
# Retrospective label extraction
# ---------------------------------------------------------------------------

def _extract_retro_arrays(session_dir: Path) -> bool:
    """
    Call MATLAB to extract plain double arrays from the Sources2D objects in
    review_neuron.mat and neuron.mat.  Saves two temporary files:
      retro_review.mat  — A_review, C_review, d1, d2
      retro_final.mat   — A_final
    Returns True on success.
    """
    sd   = str(session_dir).replace("\\", "/")
    repo = REPO_ROOT.replace("\\", "/")

    # NOTE: _run_matlab collapses the script to one line, so MATLAB % comments
    # would terminate the rest of the script. No comments allowed here.
    script = (
        f"cd('{repo}');"
        f" run('cnmfe_setup.m');"
        f" load('{sd}/review_neuron.mat');"
        f" A_review = full(neuron.A);"
        f" C_review = neuron.C_raw;"
        f" d1 = neuron.options.d1;"
        f" d2 = neuron.options.d2;"
        f" save('{sd}/retro_review.mat', 'A_review', 'C_review', 'd1', 'd2', '-v7');"
        f" fprintf('review_neuron extracted: %d neurons\\n', size(A_review, 2));"
        f" clear neuron A_review C_review;"
        f" load('{sd}/neuron.mat');"
        f" A_final = full(neuron.A);"
        f" save('{sd}/retro_final.mat', 'A_final', '-v7');"
        f" fprintf('final neuron extracted: %d neurons\\n', size(A_final, 2));"
    )
    return run_cnmfe._run_matlab(script, log, timeout_hours=0.25)


def _retro_label_session(session_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the temp arrays written by _extract_retro_arrays, derive labels by
    spatial matching, extract features, and persist candidate_features.npz +
    labels.mat so this session is treated as prospective in all future runs.

    Returns (feature_matrix, labels) or None on failure.
    Cleans up the temp .mat files either way.
    """
    review_file = session_dir / "retro_review.mat"
    final_file  = session_dir / "retro_final.mat"

    try:
        if not review_file.exists() or not final_file.exists():
            log(f"  [RETRO] Temp files missing for {session_dir.name}.")
            return None

        rdata    = sio.loadmat(str(review_file))
        A_review = rdata["A_review"]          # pixels × N_review
        C_review = rdata["C_review"]          # N_review × T
        d1       = int(rdata["d1"].flat[0])
        d2       = int(rdata["d2"].flat[0])

        fdata    = sio.loadmat(str(final_file))
        A_final  = fdata["A_final"]           # pixels × N_kept

        N_review = A_review.shape[1]
        N_kept   = A_final.shape[1]

        # --- One-to-one spatial matching via Hungarian algorithm ---
        # Each final neuron is assigned to at most one review candidate.
        # Without one-to-one constraint, spatially overlapping deleted neurons
        # can spuriously match a kept neighbor and get mislabelled as keep=1.
        from scipy.optimize import linear_sum_assignment
        nr       = np.sqrt((A_review ** 2).sum(axis=0)) + 1e-12   # (N_review,)
        nf       = np.sqrt((A_final  ** 2).sum(axis=0)) + 1e-12   # (N_kept,)
        corr_mat = (A_review / nr).T @ (A_final / nf)             # N_review × N_kept
        # Minimise negative correlation → maximise correlation
        row_ind, col_ind = linear_sum_assignment(-corr_mat)
        labels = np.zeros(N_review, dtype=float)
        for r, c in zip(row_ind, col_ind):
            if corr_mat[r, c] > SPATIAL_MATCH_THRESHOLD:
                labels[r] = 1

        n_kept_matched = int(labels.sum())
        log(f"  [RETRO] {session_dir.parent.name}/{session_dir.name}:")
        log(f"    {N_review} review candidates → "
            f"{n_kept_matched} kept, {N_review - n_kept_matched} deleted "
            f"(matched against {N_kept} final neurons, "
            f"threshold={SPATIAL_MATCH_THRESHOLD})")

        if n_kept_matched == 0:
            log(f"  [RETRO] WARNING: 0 kept neurons matched. "
                f"Check that review_neuron.mat and neuron.mat are from the same session.")

        # --- Feature extraction from review candidates ---
        footprints = A_review.T.reshape(N_review, d1, d2)   # (N, H, W)
        bg_signal  = feat_module.load_background(session_dir)
        Cn         = feat_module.load_cn(session_dir)

        rows = []
        for i in range(N_review):
            sf = feat_module.spatial_features(footprints[i])
            tf = feat_module.temporal_features(C_review[i])
            mf = feat_module.motion_features(C_review[i], bg_signal)
            cf = feat_module.cn_features(footprints[i], Cn)
            rows.append({**sf, **tf, **mf, **cf})

        feature_names  = list(rows[0].keys())
        feature_matrix = np.array([[r[k] for k in feature_names] for r in rows])

        # --- Persist so this session is prospective from now on ---
        np.savez(
            session_dir / "candidate_features.npz",
            feature_matrix=feature_matrix,
            feature_names=np.array(feature_names),
            # auto_rejected unknown for retro sessions — leave empty
            auto_rejected=np.array([], dtype=int),
            n_candidates=np.array([N_review]),
        )
        sio.savemat(
            str(session_dir / "labels.mat"),
            {"labels": labels.reshape(-1, 1)},
        )
        log(f"  [RETRO] candidate_features.npz + labels.mat saved for "
            f"{session_dir.name}.")

        return feature_matrix, labels

    except Exception as e:
        log(f"  [RETRO] Error processing {session_dir.name}: {e}")
        log(traceback.format_exc())
        return None
    finally:
        review_file.unlink(missing_ok=True)
        final_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Load prospective session data
# ---------------------------------------------------------------------------

def load_prospective_session(session_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (feature_matrix, labels) from a session with both required files.

    candidate_features.npz stores features for ALL N candidates (including
    auto-rejected ones).  labels.mat is written by CNMFe_final_save.m and
    covers only the non-auto-rejected subset shown during MATLAB review.
    When sizes differ we reconstruct the full label vector by assigning
    label=0 to auto-rejected candidates and filling in the user's decisions
    for the rest, in the same order as review_indices in curator.py.
    """
    try:
        npz    = np.load(str(session_dir / "candidate_features.npz"), allow_pickle=True)
        ldata  = sio.loadmat(str(session_dir / "labels.mat"))

        X            = npz["feature_matrix"]
        y_review     = ldata["labels"].flatten().astype(float)
        auto_rejected = npz["auto_rejected"].flatten().astype(int)

        if len(X) == len(y_review):
            # Sizes match — either no auto-rejects or retro session (all candidates labeled)
            return X, y_review

        expected_review = len(X) - len(auto_rejected)
        if len(y_review) != expected_review:
            log(f"  [WARN] {session_dir.name}: unexpected size mismatch "
                f"(features={len(X)}, labels={len(y_review)}, "
                f"auto_rejected={len(auto_rejected)}, expected_review={expected_review}), "
                f"skipping.")
            return None

        # Reconstruct full label vector.
        # review_indices preserves original candidate order with auto-rejected removed
        # (mirrors curator.py: review_indices = [i for i in range(N) if i not in auto_rejected])
        auto_set = set(auto_rejected.tolist())
        review_indices = [i for i in range(len(X)) if i not in auto_set]

        y_full = np.zeros(len(X), dtype=float)   # auto-rejected -> label 0 (delete)
        for k, idx in enumerate(review_indices):
            y_full[idx] = y_review[k]

        log(f"  [INFO] {session_dir.name}: reconstructed full labels "
            f"({len(auto_rejected)} auto-rejected assigned label=0, "
            f"{int(y_review.sum())} kept by user of {len(y_review)} reviewed)")

        return X, y_full

    except Exception as e:
        log(f"  [WARN] Could not load {session_dir.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Bootstrap weighting helpers
# ---------------------------------------------------------------------------

def _is_bootstrap_session(session_dir: Path) -> bool:
    """True if this is a pre-agent (spatial-matched) bootstrap session.

    The definitive marker is bootstrap_match_stats.json, written by
    bootstrap_preagent.py.  A handful of early pre-agent sessions were matched
    before that file existed, so they carry candidate_features.npz + labels.mat +
    a curated neuron.mat but no stats JSON.  Without this fallback they were
    misread as agent-reviewed and trained at full agent weight (4.0x) instead of
    base bootstrap weight — they have no agent-review artifacts, so detect them by
    provenance: a curated neuron.mat present and none of the agent-review files.
    """
    if (session_dir / "bootstrap_match_stats.json").exists():
        return True
    agent_artifacts = ("ROIs_candidates.jpg", "agent_run.log", "review_report.pdf",
                       "review_assigned.txt", "review_summary.txt")
    if any((session_dir / a).exists() for a in agent_artifacts):
        return False
    return (session_dir / "neuron.mat").exists()


def _get_bootstrap_recovery(session_dir: Path) -> float | None:
    """Return the bootstrap recovery rate for a session, or None if not bootstrap."""
    stats_file = session_dir / "bootstrap_match_stats.json"
    if not stats_file.exists():
        return None
    with open(stats_file) as f:
        bs = json.load(f)
    n_curated = bs.get("n_curated", 0)
    return bs["n_matched"] / n_curated if n_curated > 0 else 0.0


def _get_bootstrap_ambiguous_mask(session_dir: Path, n_candidates: int) -> np.ndarray:
    """
    Return a boolean mask (length n_candidates) marking bootstrap candidates that
    were the Hungarian best-match to an UNMATCHED curated neuron (similarity < threshold).

    These are excluded from training (weight=0) because their true label is unknown:
    they are likely merged/distorted versions of real neurons that CNMFe failed to
    cleanly re-detect, so treating them as confident negatives introduces false-negative
    label noise.

    Candidates NOT in this mask that are still labeled 0 are genuine hard negatives —
    they were never the closest match to any real neuron in the Hungarian assignment.

    How: bootstrap_match_stats.json stores candidate_indices sorted descending by
    similarity.  The first n_matched entries are the positive pairs (above threshold).
    Everything after that is a candidate matched to a real neuron but below threshold.
    """
    stats_file = session_dir / "bootstrap_match_stats.json"
    if not stats_file.exists():
        return np.zeros(n_candidates, dtype=bool)

    with open(stats_file) as f:
        bs = json.load(f)

    n_matched    = bs.get("n_matched", 0)
    cand_indices = bs.get("candidate_indices", [])

    # Pairs are sorted best-first; everything after n_matched is an unmatched pair.
    unmatched_cand_idxs = cand_indices[n_matched:]

    mask = np.zeros(n_candidates, dtype=bool)
    for idx in unmatched_cand_idxs:
        if 0 <= idx < n_candidates:
            mask[idx] = True

    return mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_clf(model_type: str, scale_pos_weight: float = 1.0):
    """Instantiate a fresh untrained classifier of the requested type."""
    if model_type == "lr":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    elif model_type == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost not installed in this environment")
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        if not _LGBM_AVAILABLE:
            raise RuntimeError("lightgbm not installed in this environment")
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def train_model(X: np.ndarray, y: np.ndarray,
                sample_weight: np.ndarray | None = None,
                model_type: str = "xgboost"):
    """Fit StandardScaler + classifier. Returns (scaler, clf)."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute scale_pos_weight from effective weighted class counts.
    # This is the XGBoost/LightGBM equivalent of sklearn's class_weight="balanced".
    # For LR, class_weight="balanced" handles this automatically; we pass spw=1.0.
    if sample_weight is not None and model_type != "lr":
        mask = sample_weight > 0
        eff_pos = float(sample_weight[(y == 1) & mask].sum())
        eff_neg = float(sample_weight[(y == 0) & mask].sum())
        scale_pos_weight = eff_neg / eff_pos if eff_pos > 0 else 1.0
    else:
        scale_pos_weight = 1.0

    clf = _make_clf(model_type, scale_pos_weight=scale_pos_weight)
    clf.fit(X_scaled, y, sample_weight=sample_weight)
    return scaler, clf


def _grouped_cv(X_scaled: np.ndarray, y: np.ndarray,
                groups: np.ndarray, sample_weight: np.ndarray,
                n_splits: int = 5, model_type: str = "xgboost") -> float:
    """
    Grouped k-fold cross-validation (split by session, not by candidate).
    Measures between-session generalisation rather than within-session discrimination.
    Returns mean ROC-AUC across folds.

    Note: bootstrap test folds have noisy labels (~41% of real neurons labeled 0
    due to CNMFe merging during re-detection).  The reported AUC is therefore a
    conservative lower bound.  Agent-session-only AUC is the clean upper estimate.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in cv.split(X_scaled, y, groups):
            w_tr = sample_weight[train_idx]
            if model_type != "lr":
                mask = w_tr > 0
                y_tr = y[train_idx]
                eff_pos = float(w_tr[(y_tr == 1) & mask].sum())
                eff_neg = float(w_tr[(y_tr == 0) & mask].sum())
                spw = eff_neg / eff_pos if eff_pos > 0 else 1.0
            else:
                spw = 1.0
            clf_cv = _make_clf(model_type, scale_pos_weight=spw)
            clf_cv.fit(X_scaled[train_idx], y[train_idx], sample_weight=w_tr)
            y_te = y[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            proba = clf_cv.predict_proba(X_scaled[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Bootstrap ROI evaluation
# ---------------------------------------------------------------------------

def _bootstrap_roi_eval(records: list[dict], n_splits: int = 5,
                        model_types: list[str] | None = None,
                        min_pos: int = 5) -> None:
    """
    Compare training on agent-only vs agent+bootstrap, evaluated on held-out
    agent sessions (clean human labels).  Answers: did bootstrap help?

    A near-zero delta on agent folds is expected: the benefit of bootstrap
    appears on out-of-distribution sessions (CTA, WSE, later-lifecycle animals
    with lower SNR and fewer cells) which are not represented in the agent test
    folds.  Feature regime shifts printed below confirm bootstrap covers
    qualitatively different recording conditions.
    """
    ag_recs = [r for r in records if not r["is_bootstrap"]]
    bs_recs = [r for r in records if r["is_bootstrap"]]

    usable  = [r for r in ag_recs if int(r["y"].sum()) >= min_pos]
    skipped = [r for r in ag_recs if int(r["y"].sum()) < min_pos]

    log(f"\n{'=' * 65}")
    log(f"Bootstrap ROI Evaluation (--eval)")
    log(f"{'=' * 65}")
    log(f"  Agent sessions usable for CV (>= {min_pos} positives): {len(usable)}")
    if skipped:
        log(f"  Excluded from CV (< {min_pos} positives): {len(skipped)}")
        for r in skipped:
            log(f"    {r['session_dir'].parent.name}/{r['session_dir'].name}: "
                f"{int(r['y'].sum())} pos")

    if len(usable) < 3:
        log("  Too few agent sessions for meaningful comparison. Skipping.")
        return

    # Fit scaler on ALL data so feature scaling is consistent across conditions
    X_all_data = np.vstack([r["X"] for r in records])
    scaler = StandardScaler()
    scaler.fit(X_all_data)

    # Bootstrap pool — always in train, never in test
    if bs_recs:
        X_bs_sc   = scaler.transform(np.vstack([r["X"] for r in bs_recs]))
        y_bs_pool = np.concatenate([r["y"] for r in bs_recs])
        w_bs_pool = np.concatenate([r["w"] for r in bs_recs])
    else:
        X_bs_sc = y_bs_pool = w_bs_pool = None

    X_ag    = np.vstack([r["X"] for r in usable])
    y_ag    = np.concatenate([r["y"] for r in usable])
    g_ag    = np.concatenate([[r["sess_idx"]] * len(r["y"]) for r in usable])
    X_ag_sc = scaler.transform(X_ag)

    n_pos_ag = int(y_ag.sum())
    log(f"\n  Agent CV pool: {len(X_ag)} candidates | "
        f"{n_pos_ag} keep ({n_pos_ag/len(y_ag):.1%}) | "
        f"{len(y_ag)-n_pos_ag} delete")
    if y_bs_pool is not None:
        log(f"  Bootstrap pool: {len(y_bs_pool)} candidates "
            f"(always in train, never in test)")

    if model_types is None:
        model_types = ["lr"]
        if _XGBOOST_AVAILABLE:
            model_types.append("xgboost")
        if _LGBM_AVAILABLE:
            model_types.append("lightgbm")

    n_cv = min(n_splits, len(usable))
    log(f"\n  {n_cv}-fold grouped CV — test fold = held-out agent sessions")
    log(f"  Cond A: agent-only  |  Cond B: agent + ALL bootstrap")
    log(f"\n  {'Model':<12}  {'Agent-only AUC':>16}  {'+ Bootstrap AUC':>16}  {'Delta':>7}")
    log(f"  {'-'*12}  {'-'*16}  {'-'*16}  {'-'*7}")

    cv = StratifiedGroupKFold(n_splits=n_cv, shuffle=True, random_state=42)

    for mtype in model_types:
        aucs_A, aucs_B = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train_idx, test_idx in cv.split(X_ag_sc, y_ag, g_ag):
                y_te = y_ag[test_idx]
                if len(np.unique(y_te)) < 2:
                    continue
                X_te = X_ag_sc[test_idx]

                # Condition A: agent train only — uniform weight, clean labels
                y_tr_A = y_ag[train_idx]
                w_tr_A = np.ones(len(train_idx))
                spw_A  = (float((y_tr_A == 0).sum()) / max(float((y_tr_A == 1).sum()), 1)
                          if mtype != "lr" else 1.0)
                clf_A  = _make_clf(mtype, spw_A)
                clf_A.fit(X_ag_sc[train_idx], y_tr_A, sample_weight=w_tr_A)
                aucs_A.append(roc_auc_score(y_te, clf_A.predict_proba(X_te)[:, 1]))

                # Condition B: agent train + ALL bootstrap
                if X_bs_sc is not None:
                    n_ag_tr = len(train_idx)
                    ag_w_B  = float(max(np.sqrt(len(y_bs_pool) / n_ag_tr), MIN_AGENT_WEIGHT)) if n_ag_tr > 0 else MIN_AGENT_WEIGHT
                    w_ag_B  = np.ones(n_ag_tr) * ag_w_B
                    X_tr_B  = np.vstack([X_ag_sc[train_idx], X_bs_sc])
                    y_tr_B  = np.concatenate([y_ag[train_idx], y_bs_pool])
                    w_tr_B  = np.concatenate([w_ag_B, w_bs_pool])
                    if mtype != "lr":
                        mask  = w_tr_B > 0
                        ep    = float(w_tr_B[(y_tr_B == 1) & mask].sum())
                        en    = float(w_tr_B[(y_tr_B == 0) & mask].sum())
                        spw_B = en / ep if ep > 0 else 1.0
                    else:
                        spw_B = 1.0
                    clf_B = _make_clf(mtype, spw_B)
                    clf_B.fit(X_tr_B, y_tr_B, sample_weight=w_tr_B)
                    aucs_B.append(roc_auc_score(y_te, clf_B.predict_proba(X_te)[:, 1]))
                else:
                    aucs_B.append(aucs_A[-1])

        mA = float(np.mean(aucs_A)) if aucs_A else float("nan")
        mB = float(np.mean(aucs_B)) if aucs_B else float("nan")
        sA = float(np.std(aucs_A))  if len(aucs_A) > 1 else 0.0
        sB = float(np.std(aucs_B))  if len(aucs_B) > 1 else 0.0
        d  = mB - mA
        log(f"  {mtype:<12}  {mA:.3f} ± {sA:.3f}       {mB:.3f} ± {sB:.3f}    {d:+.3f}")

    # Feature regime shift — evidence that bootstrap covers different territory
    ag_pos = np.vstack([r["X"][r["y"] == 1] for r in ag_recs]) if ag_recs else None
    bs_pos = np.vstack([r["X"][r["y"] == 1] for r in bs_recs]) if bs_recs else None
    if ag_pos is not None and bs_pos is not None and len(bs_pos) > 0:
        try:
            fn = list(np.load(str(usable[0]["session_dir"] / "candidate_features.npz"),
                              allow_pickle=True)["feature_names"])
        except Exception:
            fn = [f"f{i}" for i in range(ag_pos.shape[1])]
        notable = [(name, ag_pos[:, i].mean(), bs_pos[:, i].mean())
                   for i, name in enumerate(fn)
                   if abs(ag_pos[:, i].mean()) > 1e-9 and
                   abs(100 * (bs_pos[:, i].mean() - ag_pos[:, i].mean()) /
                       ag_pos[:, i].mean()) > 15]
        if notable:
            log(f"\n  Feature regime shift (|delta| > 15%  —  bootstrap covers different territory):")
            log(f"  {'Feature':<22}  {'Agent pos':>10}  {'BS pos':>10}  {'Delta':>8}")
            for name, a_, b_ in notable:
                pct = 100 * (b_ - a_) / abs(a_)
                log(f"  {name:<22}  {a_:10.3f}  {b_:10.3f}  {pct:+7.1f}%")
            log(f"\n  Bootstrap positives: lower SNR, less circular, lower skewness.")
            log(f"  This matches later-lifecycle animals (CTA/WSE) with weaker signals.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and report labels but do not save the model.")
    parser.add_argument("--retro-only", action="store_true",
                        help="Only run retrospective extraction (no training).")
    parser.add_argument("--prospective-only", action="store_true",
                        help="Skip retrospective extraction (no MATLAB calls). "
                             "Used by the watcher for fast auto-retraining.")
    parser.add_argument(
        "--model", choices=["lr", "xgboost", "lightgbm", "auto"], default="auto",
        help="Classifier to use. 'auto' runs all available models and picks the best by "
             "CV AUC. (default: auto)")
    parser.add_argument("--eval", action="store_true",
                        help="Run bootstrap ROI evaluation: compare agent-only vs "
                             "agent+bootstrap training on held-out agent test folds. "
                             "Combines with --dry-run to skip saving the model.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the auto-calibrated reject_threshold stored in the "
                             "model file. Use when the area-specific threshold differs from "
                             "the default (e.g. --threshold 0.04 for vCA1 bootstrap-only).")
    args = parser.parse_args()

    log("=" * 65)
    log("CNMFe Neuron Quality Classifier — Training")
    log("=" * 65)

    # Records collected before weighting so agent_weight can be computed
    # dynamically from the actual session counts.
    # Each entry: {'X', 'y', 'is_bootstrap', 'session_dir', 'sess_idx'}
    records: list[dict] = []
    sess_idx = 0

    # ------------------------------------------------------------------
    # Prospective sessions
    # ------------------------------------------------------------------
    prospective = find_prospective_sessions()
    log(f"\nProspective sessions (candidate_features.npz + labels.mat): "
        f"{len(prospective)}")
    for session_dir in prospective:
        result = load_prospective_session(session_dir)
        if result is not None:
            X, y = result
            is_bs = _is_bootstrap_session(session_dir)
            log(f"  {'[BS]' if is_bs else '[AG]'} "
                f"{session_dir.parent.name}/{session_dir.name}: "
                f"{int(y.sum())} keep, {int((y == 0).sum())} delete  "
                f"({len(y)} total)")
            records.append({"X": X, "y": y, "is_bootstrap": is_bs,
                            "session_dir": session_dir, "sess_idx": sess_idx})
            sess_idx += 1

    # ------------------------------------------------------------------
    # Retrospective sessions
    # ------------------------------------------------------------------
    retro = find_retro_sessions()
    if args.prospective_only:
        log(f"\nRetrospective sessions skipped (--prospective-only): {len(retro)}")
        retro = []
    elif retro and getattr(config, "FEATURE_VERSION", 1) >= 2:
        # The retro path re-extracts 13-column rows.  Under the 35-column v2
        # contract that would silently mix widths in the training corpus, so
        # refuse and leave these sessions out (same effect as --prospective-only)
        # rather than corrupting the pool.
        log(f"\nRetrospective sessions found: {len(retro)} — REFUSED, not extracted.")
        log("  The retro path emits 13-column rows but this area's feature contract")
        log("  is 35-column v2 (config.FEATURE_VERSION >= 2).  Regenerate these")
        log("  sessions with the Step 4 backfill tooling (agent/eval/step4_2026-08/)")
        log("  instead:")
        for _sd in retro:
            log(f"    {_sd.parent.name}/{_sd.name}")
        retro = []
    else:
        log(f"\nRetrospective sessions (no labels.mat yet): {len(retro)}")
    for session_dir in retro:
        log(f"\n  Extracting: {session_dir.parent.name}/{session_dir.name}")
        ok = _extract_retro_arrays(session_dir)
        if not ok:
            log(f"  [RETRO] MATLAB extraction failed — skipping.")
            continue
        result = _retro_label_session(session_dir)
        if result is not None:
            X, y = result
            records.append({"X": X, "y": y, "is_bootstrap": False,
                            "session_dir": session_dir, "sess_idx": sess_idx})
            sess_idx += 1

    # ------------------------------------------------------------------
    # Summary and weighting
    # ------------------------------------------------------------------
    if not records:
        log("\nNo labeled data found. Run CNMFe_final_save.m for at least one "
            "session to generate labels.")
        return

    # Contract-width guard: scoring is positional, so a mixed-width corpus
    # (candidate_features.npz regeneration interrupted mid-swap) must never
    # train.  Fail loudly with session names instead.
    widths = sorted({r["X"].shape[1] for r in records})
    _fv = getattr(config, "FEATURE_VERSION", 1)
    expected_width = 35 if _fv >= 2 else None
    if len(widths) > 1 or (expected_width is not None
                           and widths != [expected_width]):
        by_w: dict[int, list[str]] = {}
        for r in records:
            by_w.setdefault(r["X"].shape[1], []).append(
                f"{r['session_dir'].parent.name}/{r['session_dir'].name}")
        detail = ";  ".join(
            f"{w} cols: {len(s)} session(s), e.g. {s[0]}"
            for w, s in sorted(by_w.items()))
        raise RuntimeError(
            f"Feature-width mismatch in training corpus (FEATURE_VERSION="
            f"{_fv}, expected {expected_width or 'uniform'} cols): {detail}. "
            f"This is a half-swapped corpus — complete or roll back the "
            f"feature-contract swap before training.")

    n_bs_examples = sum(len(r["y"]) for r in records if r["is_bootstrap"])
    n_ag_examples = sum(len(r["y"]) for r in records if not r["is_bootstrap"])

    # Dynamic agent up-weight: counteracts bootstrap volume dominance.
    # Formula: max(sqrt(n_bootstrap / n_agent), MIN_AGENT_WEIGHT) — see the
    # module-level MIN_AGENT_WEIGHT for the floor's rationale/validation.
    if n_ag_examples > 0 and n_bs_examples > 0:
        agent_weight = float(max(np.sqrt(n_bs_examples / n_ag_examples), MIN_AGENT_WEIGHT))
    else:
        agent_weight = MIN_AGENT_WEIGHT

    log(f"\n{'=' * 65}")
    log(f"Sessions: {len(records)} total  "
        f"({sum(1 for r in records if not r['is_bootstrap'])} agent, "
        f"{sum(1 for r in records if r['is_bootstrap'])} bootstrap)")
    log(f"Agent up-weight: max(sqrt({n_bs_examples}/{n_ag_examples}), {MIN_AGENT_WEIGHT}) = {agent_weight:.2f}x")
    log(f"Bad-session down-weight: {BAD_SESSION_WEIGHT}x  "
        f"(recovery < {BAD_SESSION_RECOVERY_THRESHOLD:.0%})")

    all_X, all_y, all_w, all_groups = [], [], [], []
    n_excluded = 0

    for r in records:
        X, y = r["X"], r["y"]
        n = len(y)
        w = np.ones(n, dtype=float)

        if not r["is_bootstrap"]:
            w *= agent_weight
        else:
            recovery = _get_bootstrap_recovery(r["session_dir"])
            if recovery is not None and recovery < BAD_SESSION_RECOVERY_THRESHOLD:
                w *= BAD_SESSION_WEIGHT

            # Zero-weight ambiguous bootstrap negatives: candidates that were the
            # Hungarian best-match to an unmatched curated neuron (similarity <
            # threshold).  Their true label is unknown — likely merged versions of
            # real neurons — so including them as label=0 introduces false-negative
            # noise.  Setting weight=0 excludes them without altering labels.mat.
            ambig = _get_bootstrap_ambiguous_mask(r["session_dir"], n)
            w[ambig] = 0.0
            n_excluded += int(ambig.sum())

        r["w"] = w   # store for _bootstrap_roi_eval
        all_X.append(X)
        all_y.append(y)
        all_w.append(w)
        all_groups.extend([r["sess_idx"]] * n)

    X_all  = np.vstack(all_X)
    y_all  = np.concatenate(all_y)
    w_all  = np.concatenate(all_w)
    groups = np.array(all_groups)

    n_total  = len(y_all)
    n_pos    = int(y_all.sum())
    n_neg    = n_total - n_pos
    n_active = int((w_all > 0).sum())

    log(f"\nTotal candidates: {n_total}  ({n_pos} keep, {n_neg} delete)")
    log(f"  Excluded (ambiguous bootstrap negatives, weight=0): {n_excluded}")
    log(f"  Active in training (weight > 0): {n_active}")

    if args.retro_only:
        log("\n[--retro-only] Skipping training.")
        return

    # ------------------------------------------------------------------
    # Bootstrap ROI evaluation (optional — adds ~1 min to run time)
    # ------------------------------------------------------------------
    if args.eval:
        model_types = (["lr", "xgboost", "lightgbm"]
                       if args.model == "auto"
                       else [args.model])
        _bootstrap_roi_eval(records, model_types=model_types)

    if len(np.unique(y_all)) < 2:
        log("\nOnly one class present — cannot train a binary classifier. "
            "Need both keep and delete examples.")
        return

    # ------------------------------------------------------------------
    # 3-way model comparison via grouped cross-validation
    # ------------------------------------------------------------------
    model_arg = args.model
    candidates = ["lr"]
    if _XGBOOST_AVAILABLE:
        candidates.append("xgboost")
    else:
        log("  [INFO] XGBoost not available in this environment — skipping.")
    if _LGBM_AVAILABLE:
        candidates.append("lightgbm")

    # If user specified a model explicitly, only evaluate that one
    if model_arg != "auto":
        candidates = [model_arg]

    cv_results: dict[str, float] = {}
    min_class = min(n_pos, n_neg)
    n_sessions = len(records)
    if min_class >= 10 and n_sessions >= 5:
        n_splits = min(5, n_sessions)
        scaler_cv = StandardScaler()
        X_scaled_cv = scaler_cv.fit_transform(X_all)
        log(f"\n  Grouped {n_splits}-fold CV (split by session):")
        for mtype in candidates:
            try:
                auc = _grouped_cv(X_scaled_cv, y_all, groups, w_all,
                                  n_splits=n_splits, model_type=mtype)
                cv_results[mtype] = auc
                quality = "good" if auc > 0.80 else "moderate" if auc > 0.65 else "poor"
                log(f"    {mtype:12s}  AUC={auc:.3f}  ({quality})")
            except Exception as e:
                log(f"    {mtype:12s}  FAILED: {e}")
        log(f"  (Bootstrap test folds are noisy: ~41% of real neurons labeled 0 "
            f"-> AUC is a conservative lower bound)")
    else:
        log(f"\n  Skipping CV (fewer than 10 samples in minority class or fewer than 5 sessions).")

    # Pick the best model by CV AUC; fall back sensibly if CV was skipped
    if cv_results:
        best_model = max(cv_results, key=cv_results.get)
        log(f"\n  Selected model: {best_model}  (CV AUC={cv_results[best_model]:.3f})")
    elif candidates:
        best_model = "xgboost" if "xgboost" in candidates else candidates[0]
        log(f"\n  No CV results — defaulting to {best_model}.")
    else:
        best_model = "lr"

    if args.dry_run:
        log("\n[--dry-run] Model not saved.")
        return

    # ------------------------------------------------------------------
    # Final model fit on all data
    # ------------------------------------------------------------------
    scaler, clf = train_model(X_all, y_all, sample_weight=w_all, model_type=best_model)

    # v2 contract: also fit the companion 13-column first-pass model on the
    # same corpus/weights and store it in the SAME joblib.  curator.py uses it
    # to pick high-confidence neighbors (nb_corr_max) before the 35 columns
    # exist; one file keeps the deploy swap atomic.
    _v2_extra = {}
    if getattr(config, "FEATURE_VERSION", 1) >= 2:
        _n_base = feat_module.V1_N_FEATURES
        fp_scaler, fp_clf = train_model(
            X_all[:, :_n_base], y_all, sample_weight=w_all,
            model_type=best_model)
        _v2_extra = {"first_pass_scaler": fp_scaler,
                     "first_pass_clf": fp_clf,
                     "feature_version": 2}
        log(f"\nCompanion first-pass model fit on the first {_n_base} columns "
            f"(stored in the same joblib for two-pass curation).")

    # Calibrated reject_threshold per model type.
    # Derived via 5-fold grouped OOF sweep on 19 agent sessions with real weights
    # (agent_weight, bad-session 0.4x, ambiguous=0) — see diagnose_model.py Section 3.
    #
    # LR@0.10:   0.22 false-AR/session,  5.7 garbage/session  (old baseline)
    # XGB@0.10:  0.16 false-AR/session,  8.3 garbage/session  (strictly better on both)
    # XGB@0.12:  0.22 false-AR/session, 11.2 garbage/session  (matched safety, 2x garbage)
    # XGB@0.15:  0.36 false-AR/session, 17.0 garbage/session  (3x garbage, costs 0.14 neurons/session)
    # XGB@0.19:  efficiency knee — do not exceed
    #
    # 0.11 chosen: step from 0.10->0.11 has infinite efficiency (no real neurons cross
    # the boundary — false-AR stays at 0.8% while catching 7.1% garbage vs 5.9% at 0.10).
    # 0.12 is the first step that costs false-AR (+0.3pp). Revisit as sessions are added.
    # NOTE: requires bootstrap training — agent-only XGB is miscalibrated at any threshold.
    # UPDATE 2026-06-23 (BLA, 32 agent sessions): re-ran the OOF Pareto sweep
    # (diagnose_model.py) + weight re-sweep (sweep_weights.py). The model sharpened:
    # at 0.11 false-AR fell to 0.3% (from 0.8%), the free-garbage runway now extends
    # to ~0.09 and the efficiency knee moved to 0.20. Raised XGBoost 0.11 -> 0.15
    # (0.9% false-AR, 15.6% garbage caught, ~1.7x the garbage at 0.11). floor=4.0 and
    # bad_w=0.4 re-confirmed optimal (both sit at the false-AR minimum). vCA1 overrides
    # this default to 0.05 via train_classifier_vCA1.py.
    # UPDATE 2026-07-12 (BLA, 39 agent sessions after recovering 5 stranded bla12
    # returns): re-ran the OOF threshold sweep + sweep_weights.py. Model sharpened
    # further (OOF AUC 0.859 -> 0.874), but the added bla12 sessions have some
    # low-scoring real neurons, so 0.15 now costs 1.1% false-AR (was 0.7%). Lowered
    # XGBoost 0.15 -> 0.14 to hold the sub-1% false-AR posture: 0.8% false-AR,
    # 19.2% garbage caught (vs 21.4% at 0.15) — smooth curve, knee still at 0.18.
    # floor=4.0 / bad_w=0.4 left unchanged (AUC flat across floors 1-8; the false-AR
    # wiggles are noise from a single 5-session, single-animal batch — not re-tuned).
    # UPDATE 2026-08-09 (BLA, 64 agent / 91 bootstrap after fixing the pre-agent
    # bootstrap misclassification: 9 legacy pre-agent sessions lacked
    # bootstrap_match_stats.json and were mis-trained as agent @4.0x — see
    # _is_bootstrap_session). The cleaner model (OOF AUC 0.897 -> 0.910) reopened
    # headroom: 0.12 dropped to 0.68% false-AR. Raised xgboost 0.12 -> 0.13:
    # 0.85% false-AR (still sub-1%; worst of 8 seeds 1.0%), 27.1% garbage caught
    # (was 24.4%). 0.14 rejected (0.99% mean, seed tail to 1.5% — too much for BLA).
    # UPDATE 2026-08-18 (BLA, 75 agent / 91 bootstrap): reverted xgboost 0.13 ->
    # 0.12. The pre-agreed trigger fired: each returned batch adds a few dim-but-
    # real cells near the boundary, so fixed-threshold false-AR drifts up while
    # AUC stays flat (~0.910 since the fix) — 0.13 walked 0.85% -> 0.93% -> 1.02%
    # across 61 -> 69 -> 75 sessions. 0.12 restores 0.83% false-AR / 27.9%
    # garbage. The extra ~2.8pts of junk at 0.13 was bought with false-AR, not
    # skill (at matched false-AR the junk-caught is flat) — not worth it for BLA.
    _THRESHOLD_BY_MODEL = {"lr": 0.10, "xgboost": 0.12, "lightgbm": 0.11}
    reject_threshold = _THRESHOLD_BY_MODEL.get(best_model, 0.10)
    if args.threshold is not None:
        log(f"\n  Overriding reject_threshold: {reject_threshold:.2f} -> {args.threshold:.2f} (--threshold flag)")
        reject_threshold = args.threshold

    model_path = MODEL_DIR / "classifier.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "clf": clf,
            "model_type": best_model,
            "reject_threshold": reject_threshold,
            "agent_weight": agent_weight,
            "n_features": int(X_all.shape[1]),
            "n_sessions": len(records),
            "n_training_active": n_active,
            "n_excluded_ambiguous": n_excluded,
            "cv_results": cv_results,
            **_v2_extra,
        },
        str(model_path),
    )
    log(f"\nModel saved -> {model_path}  (type: {best_model})")
    log(f"Sessions used: {len(records)}")
    log("The next session processed by the watcher will use this classifier.\n")


if __name__ == "__main__":
    main()
