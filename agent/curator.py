"""
curator.py
Auto-curation pipeline:
  1. Load trained classifier (or fall back to one-class anomaly detection)
  2. Score all candidate neurons
  3. Detect merge candidates (split cells)
  4. Build review_neuron.mat for MATLAB final review
  5. Generate PDF report and review_summary.txt

Called by watcher.py after each headless run.
After the user completes their MATLAB review, run train_classifier.py to
retrain the model on the new labeled examples.
"""

import json
import shutil
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import features as feat_module
import config
from config import DATA_ROOT, MODEL_DIR

AGENT_DIR     = Path(__file__).parent
LABELS_DIR    = AGENT_DIR / "labels"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(exist_ok=True)

# Confidence thresholds for binary classifier (when available)
REJECT_THRESHOLD = 0.10   # score below this → auto-reject (high confidence bad)
REVIEW_THRESHOLD = 0.70   # score above this → tentatively keep, still shown in review

# Per-area override of the reject threshold, set by an area wrapper before
# watcher.main() (see watcher_DG_AL.py).  None = use the model's calibrated value.
#
# A new area starts with no classifier, so score_neurons falls back to an
# IsolationForest fitted on whatever A.txt files exist under DATA_ROOT.  On a
# cold start the only A.txt is the session's own *unreviewed* candidates
# (CNMFe_Biane_headless.m writes it; CNMFe_final_save.m overwrites it with the
# curated set later), so that model characterises the junk it is meant to
# exclude, and its min-max-normalised scores put at least one neuron below any
# positive cutoff.  Setting this to 0.0 auto-rejects nothing and sends every
# candidate to a human, which is the only defensible behaviour until enough
# reviewed sessions exist to measure a false-auto-reject rate.
THRESHOLD_OVERRIDE = None

# Merge candidate thresholds
MERGE_OVERLAP_MIN  = 0.30   # spatial overlap fraction to flag as possible merge
MERGE_CORR_MIN     = 0.70   # temporal correlation to flag as possible merge

# Motion artifact: neurons with this background correlation get flagged
MOTION_CORR_FLAG   = 0.55


# ---- Model persistence ----

def _model_path():
    return MODEL_DIR / "classifier.joblib"


def _load_model():
    """
    Load trained classifier state (scaler + weights).
    Returns (scaler, clf, model_type, reject_threshold) or None if no model exists.
    Uses joblib — the standard sklearn serialization format.

    reject_threshold is stored in the model file because the calibrated threshold
    differs by model type: LR with class_weight="balanced" uses 0.10, while
    XGBoost/LightGBM with scale_pos_weight=12x needs ~0.03 to achieve equivalent
    false-auto-reject safety (see calibration analysis in agent_cv_eval history).
    """
    mp = _model_path()
    if not mp.exists():
        return None
    data = joblib.load(str(mp))
    threshold = data.get("reject_threshold", REJECT_THRESHOLD)
    return data["scaler"], data["clf"], data["model_type"], threshold


def _save_model(scaler, clf, model_type: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf, "model_type": model_type},
                str(_model_path()))


def _load_first_pass():
    """
    Companion 13-column first-pass model, stored in the SAME joblib as the
    final classifier by train_classifier.py under the v2 (35-column) feature
    contract.  Used only to pick high-confidence neighbors for the nb_corr_max
    feature before the full 35 columns exist.  Returns (scaler, clf) or None
    (v1-contract joblibs — vCA1/DG_AL — simply lack the keys).
    """
    mp = _model_path()
    if not mp.exists():
        return None
    data = joblib.load(str(mp))
    if "first_pass_scaler" in data and "first_pass_clf" in data:
        return data["first_pass_scaler"], data["first_pass_clf"]
    return None


def _check_arity(scaler, feature_matrix: np.ndarray, what: str):
    """
    The joblib stores no feature names and scoring is positional, so a feature
    contract / model mismatch (half-swapped deploy state) would otherwise
    produce silently wrong scores.  Fail loudly instead.
    """
    n_expected = getattr(scaler, "n_features_in_", None)
    if n_expected is not None and feature_matrix.shape[1] != n_expected:
        raise ValueError(
            f"Feature-arity mismatch: matrix has {feature_matrix.shape[1]} "
            f"columns but the {what} model expects {n_expected}. The feature "
            f"contract and the deployed model are out of sync — do not trust "
            f"any score produced in this state; restore a consistent deploy "
            f"(see the Step 4 rollback) before curating.")


# ---- One-class fallback (before labeled data exists) ----

def _build_oneclass_model(log) -> tuple:
    """
    Train an Isolation Forest on all accepted neurons from completed sessions.
    This characterises the 'good neuron' space without needing rejection labels.
    """
    log("  [CURATOR] Building one-class anomaly detector from historical accepted neurons...")

    data_root = DATA_ROOT
    all_features = []

    for task_dir in data_root.iterdir():
        if not task_dir.is_dir():
            continue
        for session_dir in task_dir.iterdir():
            if not session_dir.is_dir():
                continue
            a_file = session_dir / "A.txt"
            c_file = session_dir / "C_raw.txt"
            if not a_file.exists() or not c_file.exists():
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fm, _, _ = feat_module.extract_all(session_dir, lambda m: None)
                    # Under the v2 contract the fallback must match the
                    # 35-column session matrix.  There is no model in this
                    # path, so the high-confidence neighbor set is empty.
                    if getattr(config, "FEATURE_VERSION", 1) >= 2:
                        traces = feat_module.load_traces(session_dir)
                        fps    = feat_module.load_spatial(session_dir)
                        Cn     = feat_module.load_cn(session_dir)
                        v2b    = feat_module.compute_v2b_features(
                            traces, fps, Cn, np.zeros(len(fm), dtype=bool))
                        fm = feat_module.assemble_v2_matrix(fm, v2b, 1.0)
                all_features.append(fm)
            except Exception as e:
                log(f"  [CURATOR] Skipping {session_dir.name} for training: {e}")

    if not all_features:
        return None, None, "none"

    X = np.vstack(all_features)
    log(f"  [CURATOR] One-class training set: {X.shape[0]} accepted neurons "
        f"from historical sessions.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(X_scaled)

    _save_model(scaler, clf, "isolation_forest")
    log("  [CURATOR] One-class model trained and saved.")
    return scaler, clf, "isolation_forest"


# ---- Scoring ----

def score_neurons(feature_matrix: np.ndarray, log) -> tuple[np.ndarray, str, float]:
    """
    Score each neuron. Returns (scores, model_type, reject_threshold).
    scores: (N,) — higher = more likely to be a real neuron [0..1].
    reject_threshold: the calibrated auto-reject cutoff for this model.
    """
    model = _load_model()

    if model is None:
        log("  [CURATOR] No model found. Training one-class detector...")
        scaler, clf, model_type = _build_oneclass_model(log)
        if scaler is None:
            log("  [CURATOR] Cannot build model — no historical data. Passing all to review.")
            return np.ones(len(feature_matrix)), "none", REJECT_THRESHOLD
        model = (scaler, clf, model_type, REJECT_THRESHOLD)

    scaler, clf, model_type, reject_threshold = model
    _check_arity(scaler, feature_matrix, f"deployed ({model_type})")
    X_scaled = scaler.transform(feature_matrix)

    if model_type in ("logistic_regression", "lr", "xgboost", "lightgbm"):
        # Binary classifier: probability of being a good neuron
        scores = clf.predict_proba(X_scaled)[:, 1]
    elif model_type == "isolation_forest":
        # Isolation Forest: decision_function → map to [0, 1]
        raw = clf.decision_function(X_scaled)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        scores = np.ones(len(feature_matrix))

    log(f"  [CURATOR] Scored {len(scores)} neurons with model '{model_type}' "
        f"(reject threshold: {reject_threshold:.2f}). "
        f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    return scores, model_type, reject_threshold


# ---- Merge detection ----

def find_merge_candidates(
    overlap_matrix: np.ndarray,
    traces: np.ndarray,
    log,
) -> list[tuple[int, int, float, float]]:
    """
    Flag neuron pairs as possible split cells.
    Returns list of (i, j, overlap_ratio, temporal_corr).
    """
    N = len(traces)
    candidates = []

    for i in range(N):
        for j in range(i + 1, N):
            ov = overlap_matrix[i, j]
            if ov < MERGE_OVERLAP_MIN:
                continue
            # Check temporal correlation
            ti, tj = traces[i], traces[j]
            if ti.std() < 1e-9 or tj.std() < 1e-9:
                continue
            corr = float(np.corrcoef(ti, tj)[0, 1])
            if corr >= MERGE_CORR_MIN:
                candidates.append((i, j, float(ov), corr))

    if candidates:
        log(f"  [CURATOR] {len(candidates)} merge candidate pair(s) detected.")
    else:
        log("  [CURATOR] No merge candidates detected.")

    return candidates


# ---- PDF report ----

def _generate_pdf(
    session_dir: Path,
    session_name: str,
    feature_matrix: np.ndarray,
    feature_names: list,
    scores: np.ndarray,
    traces: np.ndarray,
    merge_candidates: list,
    auto_rejected: list[int],
    motion_flagged: list[int],
    log,
    reject_threshold: float = REJECT_THRESHOLD,
):
    """
    Generate a PDF report showing flagged neurons for pre-review inspection.
    Requires matplotlib (included in the valence conda env).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless rendering
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import scipy.io as sio
    except ImportError:
        log("  [CURATOR] matplotlib not available — skipping PDF generation.")
        return

    pdf_path = session_dir / "review_report.pdf"
    sf_file  = session_dir / "spatial_footprints.mat"
    cn_file  = session_dir / "Cn.mat"

    footprints = feat_module.load_spatial(session_dir)
    Cn = sio.loadmat(str(cn_file))["Cn"] if cn_file.exists() else None

    N = len(scores)
    review_idx   = [i for i in range(N) if i not in auto_rejected]
    motion_set   = set(motion_flagged)
    merge_pairs  = {(i, j) for i, j, *_ in merge_candidates}

    with PdfPages(str(pdf_path)) as pdf:

        # ---- Page 1: Summary ----
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        n_uncertain   = sum(1 for i in review_idx
                            if reject_threshold <= scores[i] <= REVIEW_THRESHOLD)
        n_likely_keep = sum(1 for i in review_idx if scores[i] > REVIEW_THRESHOLD)
        lines = [
            f"CNMFe Agent — Review Report",
            f"Session: {session_name}",
            f"",
            f"Total candidates:       {N}",
            f"",
            f"  Auto-rejected:        {len(auto_rejected)}  "
            f"(score < {reject_threshold:.2f})  — audit section below",
            f"  Motion suspects:      {len(motion_flagged)}",
            f"  Merge candidates:     {len(merge_candidates)} pairs",
            f"  Uncertain (review):   {n_uncertain}  "
            f"(score {reject_threshold:.2f}-{REVIEW_THRESHOLD:.2f})",
            f"  Likely keep:          {n_likely_keep}  "
            f"(score > {REVIEW_THRESHOLD:.2f})",
            f"",
            f"Awaiting MATLAB review: {len(review_idx)} neurons",
            f"  (all sections except auto-rejected)",
            f"",
            f"Score thresholds:",
            f"  Auto-reject  < {reject_threshold:.2f}",
            f"  Review zone    {reject_threshold:.2f} - {REVIEW_THRESHOLD:.2f}",
            f"  Likely keep  > {REVIEW_THRESHOLD:.2f}",
            f"",
            f"Merge candidates flagged when:",
            f"  Spatial overlap  ≥ {MERGE_OVERLAP_MIN:.0%}  AND",
            f"  Temporal corr.   ≥ {MERGE_CORR_MIN:.2f}",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---- Load model for per-neuron feature contribution analysis ----
        _model_data = None
        try:
            _mp = MODEL_DIR / "classifier.joblib"
            if _mp.exists():
                _model_data = joblib.load(str(_mp))
        except Exception:
            pass

        def _rejection_reasons(idx):
            """Return a one-line string of the top 3 features driving rejection."""
            if _model_data is None:
                return None
            mtype  = _model_data.get("model_type", "")
            scaler = _model_data["scaler"]
            clf    = _model_data["clf"]
            x      = feature_matrix[idx]
            x_sc   = (x - scaler.mean_) / scaler.scale_

            if mtype in ("logistic_regression", "lr"):
                # Exact linear contribution: positive = toward keep
                contribs = clf.coef_[0] * x_sc
                worst_fi = np.argsort(contribs)[:3]
                parts = [
                    f"{feature_names[fi]}={x[fi]:.3f} (Δ{contribs[fi]:+.2f})"
                    for fi in worst_fi
                ]
            elif mtype in ("xgboost", "lightgbm"):
                # Approximate: feature importance × scaled value magnitude.
                # High-importance features with extreme scaled values drive rejection.
                # Negative x_sc → feature is below the mean (bad direction for most features).
                importances = clf.feature_importances_
                contribs = importances * x_sc   # positive = above mean on important feature
                worst_fi = np.argsort(contribs)[:3]
                parts = [
                    f"{feature_names[fi]}={x[fi]:.3f} (imp={importances[fi]:.3f}, sc={x_sc[fi]:+.2f})"
                    for fi in worst_fi
                ]
            else:
                return None
            return "Top rejection drivers:  " + "   |   ".join(parts)

        # ---- Pages: Auto-rejected neurons ----
        if auto_rejected:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.set_title(
                f"Auto-rejected — {len(auto_rejected)} neurons  "
                f"(classifier score < {reject_threshold:.2f})",
                fontsize=13, fontweight="bold", color="crimson")
            ax.text(0.05, 0.82,
                    "The classifier was confident these are not real neurons.\n"
                    "Shown here for audit only — they do NOT appear in MATLAB review.\n"
                    "Sorted worst-first (lowest score at top).\n"
                    "Each page shows top 3 features driving rejection (Δ = log-odds contribution).",
                    transform=ax.transAxes, fontsize=11, verticalalignment="top",
                    fontfamily="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for idx in sorted(auto_rejected, key=lambda i: scores[i]):   # worst first
                _neuron_page(pdf, idx, footprints, traces, scores, Cn,
                             label=f"Neuron {idx+1} — AUTO-REJECTED  "
                                   f"(score={scores[idx]:.2f})",
                             color="crimson",
                             extra_text=_rejection_reasons(idx))

        # ---- Pages: Motion artifact suspects ----
        if motion_flagged:
            fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
            axes.axis("off")
            axes.set_title("Motion Artifact Suspects", fontsize=14, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            for idx in motion_flagged:
                _neuron_page(pdf, idx, footprints, traces, scores, Cn,
                             label=f"Neuron {idx+1} — MOTION SUSPECT "
                                   f"(bg_corr={feature_matrix[idx, feature_names.index('motion_correlation')]:.2f})",
                             color="red")

        # ---- Pages: Merge candidates ----
        if merge_candidates:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.set_title("Merge Candidates (possible split cells)", fontsize=14, fontweight="bold")
            pairs_text = "\n".join(
                f"  Neurons {i+1} & {j+1}:  overlap={ov:.0%},  temporal_corr={corr:.2f}"
                for i, j, ov, corr in merge_candidates
            )
            ax.text(0.05, 0.90, pairs_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment="top", fontfamily="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            shown = set()
            for i, j, ov, corr in merge_candidates:
                for idx in (i, j):
                    if idx not in shown:
                        _neuron_page(pdf, idx, footprints, traces, scores, Cn,
                                     label=f"Neuron {idx+1} — MERGE CANDIDATE "
                                           f"(overlap={ov:.0%}, corr={corr:.2f})",
                                     color="orange")
                        shown.add(idx)

        # ---- Pages: Uncertain neurons (review zone) ----
        uncertain = [i for i in review_idx
                     if reject_threshold <= scores[i] <= REVIEW_THRESHOLD
                     and i not in motion_set
                     and i not in {k for pair in merge_candidates for k in pair[:2]}]
        if uncertain:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.set_title(f"Uncertain Neurons (score {reject_threshold:.2f}-{REVIEW_THRESHOLD:.2f})",
                         fontsize=14, fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for idx in uncertain:
                _neuron_page(pdf, idx, footprints, traces, scores, Cn,
                             label=f"Neuron {idx+1} — UNCERTAIN (score={scores[idx]:.2f})",
                             color="steelblue")

        # ---- Pages: Likely-keep neurons ----
        merge_shown = {k for pair in merge_candidates for k in pair[:2]}
        likely_keep = [i for i in review_idx
                       if scores[i] > REVIEW_THRESHOLD
                       and i not in motion_set
                       and i not in merge_shown]
        if likely_keep:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.set_title(
                f"Likely keep — {len(likely_keep)} neurons  "
                f"(classifier score > {REVIEW_THRESHOLD:.2f})",
                fontsize=13, fontweight="bold", color="darkgreen")
            ax.text(0.05, 0.82,
                    "These neurons scored above the review threshold.\n"
                    "They appear in MATLAB review — usually need only a quick scan.\n"
                    "Sorted best-first (highest score at top).",
                    transform=ax.transAxes, fontsize=11, verticalalignment="top",
                    fontfamily="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for idx in sorted(likely_keep, key=lambda i: scores[i], reverse=True):  # best first
                _neuron_page(pdf, idx, footprints, traces, scores, Cn,
                             label=f"Neuron {idx+1} — LIKELY KEEP  "
                                   f"(score={scores[idx]:.2f})",
                             color="darkgreen")

    log(f"  [CURATOR] PDF report saved: {pdf_path.name}")


def _neuron_page(pdf, idx, footprints, traces, scores, Cn, label, color,
                 extra_text=None):
    """Render one neuron (footprint + trace) to a PDF page."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(label, color=color, fontsize=12, fontweight="bold")

    # Spatial footprint
    ax = axes[0]
    fp = footprints[idx]
    ax.imshow(fp, cmap="hot", interpolation="nearest")
    ax.set_title("Spatial footprint")
    ax.axis("off")

    # Temporal trace
    ax = axes[1]
    ax.plot(traces[idx], linewidth=0.6, color="black")
    ax.set_title(f"C_raw trace  (score={scores[idx]:.2f})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fluorescence (a.u.)")

    if extra_text:
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig.text(0.5, 0.02, extra_text, ha="center", fontsize=8,
                 style="italic", color="dimgray")
    else:
        plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---- Write review_neuron.mat ----

def _write_review_mat(session_dir: Path, review_indices: list[int], log):
    """
    Write review_neuron.mat — a copy of neuron.mat with only the review-set
    neurons. Cn is included for viewNeurons display.

    Because neuron.mat contains a MATLAB class object that Python cannot
    reconstruct, we generate a small MATLAB script that MATLAB itself runs
    to build review_neuron.mat. The script is written to the session folder
    and called automatically via run_cnmfe._run_matlab().
    """
    from local_config import REPO_ROOT
    repo = str(REPO_ROOT)
    sd   = str(session_dir).replace("\\", "/")

    # Write the index list (1-based for MATLAB)
    indices_str = ", ".join(str(i + 1) for i in sorted(review_indices))

    script_content = f"""
cd('{repo}');
run('cnmfe_setup.m');
load('{sd}/neuron.mat');
load('{sd}/Cn.mat');
keep_idx = [{indices_str}];
all_idx = 1:size(neuron.A, 2);
delete_idx = setdiff(all_idx, keep_idx);
neuron.delete(delete_idx);
save('{sd}/review_neuron.mat', 'neuron', 'Cn');
fprintf('review_neuron.mat written: %d neurons.\\n', length(keep_idx));
"""
    script_file = session_dir / "_write_review_mat.m"
    script_file.write_text(script_content)

    # Run it
    import run_cnmfe
    ok = run_cnmfe._run_matlab(script_content, log, timeout_hours=0.25)
    if ok:
        script_file.unlink(missing_ok=True)
        log(f"  [CURATOR] review_neuron.mat written with {len(review_indices)} neurons.")
    else:
        log("  [CURATOR] WARNING: Could not write review_neuron.mat via MATLAB. "
            "Falling back to full neuron.mat copy.")
        shutil.copy(str(session_dir / "neuron.mat"),
                    str(session_dir / "review_neuron.mat"))


# ---- v2 (35-column) feature contract ----

def _extend_features_v2(session_dir: Path, X13: np.ndarray, base_names: list,
                        traces: np.ndarray, log):
    """
    Build the 35-column v2 matrix for the FULL candidate set:
    13 base | 13 within-session percentile ranks | 8 v2b | v2_present=1.

    Two-pass scoring: the nb_corr_max feature needs high-confidence neighbor
    scores before the 35-column model can run, so pass 1 scores the 13 base
    columns with the companion first-pass model stored in the same joblib.
    Without any model (cold one-class fallback) the high-confidence set is
    empty and nb_corr_max is 0 for every candidate.
    """
    assert X13.shape[1] == feat_module.V1_N_FEATURES, \
        f"v2 extension expects the {feat_module.V1_N_FEATURES}-column base " \
        f"matrix, got {X13.shape[1]}"

    fp = _load_first_pass()
    if fp is not None:
        fp_scaler, fp_clf = fp
        _check_arity(fp_scaler, X13, "first-pass (13-col)")
        s1 = fp_clf.predict_proba(fp_scaler.transform(X13))[:, 1]
        hiconf = s1 >= feat_module.HICONF_SCORE
        log(f"  [CURATOR] v2 pass 1: {int(hiconf.sum())}/{len(s1)} candidates "
            f"are high-confidence neighbors (score >= "
            f"{feat_module.HICONF_SCORE}).")
    else:
        hiconf = np.zeros(len(X13), dtype=bool)
        log("  [CURATOR] v2 pass 1: no first-pass model — high-confidence "
            "neighbor set empty (nb_corr_max = 0).")

    footprints = feat_module.load_spatial(session_dir)
    Cn         = feat_module.load_cn(session_dir)
    v2b = feat_module.compute_v2b_features(traces, footprints, Cn, hiconf)
    X35 = feat_module.assemble_v2_matrix(X13, v2b, 1.0)
    names35 = feat_module.v2_feature_names(list(base_names))
    log(f"  [CURATOR] v2 feature matrix assembled: {X35.shape[1]} columns "
        f"(flag=1, real v2b for all {len(X35)} candidates).")
    return X35, names35


# ---- Main entry point ----

def prepare_review_package(session_name: str, session_dir: Path, log):
    """
    Called by watcher.py after a headless run completes.
    Produces: review_neuron.mat, review_summary.txt, review_report.pdf
    """
    log(f"\n[CURATOR] Preparing review package for {session_name}...")

    # Extract features
    feature_matrix, feature_names, overlap_matrix = feat_module.extract_all(session_dir, log)
    N = len(feature_matrix)

    traces = feat_module.load_traces(session_dir)

    # v2 contract areas: extend to the 35-column matrix before scoring
    # (pass 2 below then scores all 35 columns with the deployed model).
    if getattr(config, "FEATURE_VERSION", 1) >= 2:
        feature_matrix, feature_names = _extend_features_v2(
            session_dir, feature_matrix, feature_names, traces, log)

    # Score neurons
    scores, model_type, reject_threshold = score_neurons(feature_matrix, log)

    if THRESHOLD_OVERRIDE is not None:
        log(f"  [CURATOR] Threshold override: {THRESHOLD_OVERRIDE:.3f} "
            f"(model's calibrated value was {reject_threshold:.3f})")
        reject_threshold = THRESHOLD_OVERRIDE

    # Auto-rejected (high confidence bad)
    auto_rejected = [i for i in range(N) if scores[i] < reject_threshold]
    log(f"  [CURATOR] Auto-rejected: {len(auto_rejected)}/{N} "
        f"(score < {reject_threshold:.2f})")

    # Save candidate features for classifier training.
    # labels.mat is written by CNMFe_final_save.m after the user's review.
    # train_classifier.py reads both files together to build labeled training data.
    np.savez(
        session_dir / "candidate_features.npz",
        feature_matrix=feature_matrix,
        feature_names=np.array(feature_names),
        auto_rejected=np.array(auto_rejected, dtype=int),
        n_candidates=np.array([N]),
    )
    log(f"  [CURATOR] candidate_features.npz saved "
        f"({N} candidates, {len(feature_names)} features).")

    # Motion artifact suspects
    mc_idx = feature_names.index("motion_correlation")
    motion_flagged = [
        i for i in range(N)
        if i not in auto_rejected
        and feature_matrix[i, mc_idx] >= MOTION_CORR_FLAG
    ]
    log(f"  [CURATOR] Motion artifact suspects: {len(motion_flagged)}")

    # Merge candidates (among non-rejected neurons)
    keep_mask = [i for i in range(N) if i not in auto_rejected]
    sub_overlap = overlap_matrix[np.ix_(keep_mask, keep_mask)]
    sub_traces  = traces[keep_mask]
    raw_merges  = find_merge_candidates(sub_overlap, sub_traces, log)
    # Map sub-indices back to global indices
    merge_candidates = [
        (keep_mask[i], keep_mask[j], ov, corr)
        for i, j, ov, corr in raw_merges
    ]

    # Review set: everything that isn't auto-rejected
    review_indices = [i for i in range(N) if i not in auto_rejected]

    # Generate PDF
    _generate_pdf(
        session_dir, session_name,
        feature_matrix, feature_names, scores, traces,
        merge_candidates, auto_rejected, motion_flagged, log,
        reject_threshold=reject_threshold,
    )

    # Write review_neuron.mat via MATLAB
    _write_review_mat(session_dir, review_indices, log)

    # Write summary
    merge_summary = "\n".join(
        f"  Neurons {i+1} & {j+1}:  overlap={ov:.0%},  temporal_corr={corr:.2f}"
        for i, j, ov, corr in merge_candidates
    ) or "  None"

    motion_summary = (
        ", ".join(f"Neuron {i+1}" for i in motion_flagged)
        if motion_flagged else "None"
    )

    summary = (
        f"Session: {session_name}\n"
        f"{'='*55}\n"
        f"Total candidates:       {N}\n"
        f"Auto-rejected:          {len(auto_rejected)}  "
        f"(model: {model_type}, threshold: {reject_threshold:.2f})\n"
        f"Motion artifact flags:  {len(motion_flagged)}\n"
        f"  {motion_summary}\n"
        f"Merge candidates:       {len(merge_candidates)} pairs\n"
        f"{merge_summary}\n"
        f"Awaiting your review:   {len(review_indices)} neurons\n"
        f"\n"
        f"Open MATLAB and run:  run_final_review.m\n"
        f"See PDF report:       review_report.pdf\n"
    )
    (session_dir / "review_summary.txt").write_text(summary)
    log(f"  [CURATOR] review_summary.txt written.")
    log(f"[CURATOR] Review package complete: {len(review_indices)} neurons to review "
        f"(from {N} candidates).")
