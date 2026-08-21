"""
bootstrap_preagent.py
Generates training labels for pre-agent sessions by re-running headless CNMFe
on the original .tif files, then matching the candidate pool against the curated
neuron.mat via Hungarian algorithm.

For each eligible pre-agent session:
  1. Extract A_final from the curated neuron.mat via a quick MATLAB call
  2. Estimate CNMFe parameters from animal_params.json + Cn.mat
  3. Run full headless CNMFe into a _bootstrap/ subfolder
     (dir_nm_override in CNMFe_Biane_headless.m redirects all outputs there,
     so the curated neuron.mat, A.txt, etc. are never overwritten)
  4. Load candidate footprints + traces from _bootstrap/ in Python
  5. Hungarian-match candidates to A_final -> keep/delete labels
  6. Extract features, save candidate_features.npz + labels.mat to session_dir
  7. Delete _bootstrap/

The script is resumable: sessions that already have both candidate_features.npz
and labels.mat are skipped automatically.

After this script completes, retrain the classifier:
    python train_classifier.py --prospective-only

Usage:
    python bootstrap_preagent.py                         # process all eligible sessions
    python bootstrap_preagent.py --dry-run               # list what would be processed
    python bootstrap_preagent.py --sessions-file f.txt --keep-candidates
        # explicit re-run list (overwrites npz/labels/JSON) + persist candidates

    # Parallel workers — split sessions across N prompts (0-indexed):
    python bootstrap_preagent.py --worker 0 --num-workers 4
    python bootstrap_preagent.py --worker 1 --num-workers 4
    python bootstrap_preagent.py --worker 2 --num-workers 4
    python bootstrap_preagent.py --worker 3 --num-workers 4
    # Each worker takes every Nth session (interleaved), so long sessions are
    # spread evenly rather than all landing in one worker's batch.
"""

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

AGENT_DIR = Path(__file__).parent
import config
from config import DATA_ROOT
from local_config import REPO_ROOT as _REPO_ROOT
REPO_ROOT = str(_REPO_ROOT)

# Cosine-similarity threshold for matching bootstrap candidates to final neurons.
SPATIAL_MATCH_THRESHOLD = 0.45

# Threshold sweep recorded in bootstrap_match_stats.json (schema_version 2).
RECOVERY_SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _reorder_Fcols_to_C(A: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """
    Reindex a MATLAB-linearized (pixels, N) matrix (column-major pixel order,
    p = row + col*d1, as produced by full(neuron.A)) into numpy C-order pixel
    rows (p = row*d2 + col) on the same (d1, d2) grid.

    Both sides of a cosine similarity must share one pixel order: candidates
    loaded from the (N, d1, d2) spatial_footprints.mat stack are flattened
    C-order, while A_final from neuron.mat is F-order. Comparing them without
    this reindex scores each candidate against the TRANSPOSED image of the
    curated footprint — the bug that scrambled all bootstrap labels until
    2026-08 (see agent/eval/bootstrap_matching_2026-08/AUDIT.md).
    """
    return (A.T.reshape(-1, d2, d1)      # (N, d2, d1) = [n, col, row]
              .transpose(0, 2, 1)        # (N, d1, d2) = [n, row, col]
              .reshape(-1, d1 * d2).T)   # (pixels, N), C-order rows

# NOTE: temporal confirmation (cross-run Pearson r on S or C_raw) was tested and
# removed.  CNMFe temporal traces are demixed outputs of a non-convex factorisation —
# they are not stable across independent runs even for the same physical neuron.
# S fails because sparse spike trains give r ~ 0 for any timing disagreement.
# C_raw fails because demixing depends on the full background model + convergence
# path, which differs between runs.  Spatial cosine similarity (threshold 0.45) is
# the only reliable cross-run identifier and has been validated separately.

sys.path.insert(0, str(AGENT_DIR))
import features as feat_module
import params as params_module
import run_cnmfe


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_bootstrap_sessions() -> list[tuple[str, Path, Path]]:
    """
    Pre-agent sessions with .tif + neuron.mat that have not yet been bootstrapped.
    Returns list of (task_name, session_dir, tif_path).
    """
    sessions = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # Skip agent-pipeline sessions
            if (session_dir / "ROIs_candidates.jpg").exists():
                continue
            # Must have curated outputs
            has_neuron  = (session_dir / "neuron.mat").exists()
            has_outputs = ((session_dir / "A.txt").exists() or
                           (session_dir / "spatial_footprints.mat").exists())
            if not has_neuron or not has_outputs:
                continue
            # Already bootstrapped — skip
            if ((session_dir / "candidate_features.npz").exists() and
                    (session_dir / "labels.mat").exists()):
                continue
            # Needs a .tif to run headless
            tifs = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
            if not tifs:
                continue
            sessions.append((task_dir.name, session_dir, tifs[0]))
    return sessions


def find_missing_stats_sessions() -> list[tuple[str, Path, Path]]:
    """
    Already-bootstrapped pre-agent sessions that are MISSING bootstrap_match_stats.json.

    These are legacy sessions matched before that file was written: they have
    candidate_features.npz + labels.mat + neuron.mat but no stats JSON, so
    train_classifier's provenance fallback still trains them as bootstrap, but
    without the recovery rate (bad-session 0.4x) and ambiguous-candidate mask that
    the JSON supplies.  Reprocessing re-runs the match and regenerates the JSON
    (and refreshes npz/labels) so they get full, correct bootstrap weighting.
    Used by --refresh-missing-stats.  Mirrors find_bootstrap_sessions but inverts
    the "already bootstrapped" test: require npz+labels present and JSON absent.
    """
    sessions = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # Skip agent-pipeline sessions
            if (session_dir / "ROIs_candidates.jpg").exists():
                continue
            # Must already be a labeled bootstrap session (npz + labels present) ...
            if not ((session_dir / "candidate_features.npz").exists() and
                    (session_dir / "labels.mat").exists()):
                continue
            # ... but MISSING the stats JSON (that's the whole point)
            if (session_dir / "bootstrap_match_stats.json").exists():
                continue
            # Must have curated outputs + a .tif to re-run headless matching
            has_neuron  = (session_dir / "neuron.mat").exists()
            has_outputs = ((session_dir / "A.txt").exists() or
                           (session_dir / "spatial_footprints.mat").exists())
            if not has_neuron or not has_outputs:
                continue
            tifs = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
            if not tifs:
                continue
            sessions.append((task_dir.name, session_dir, tifs[0]))
    return sessions


# ---------------------------------------------------------------------------
# Step 1 — Extract A_final from curated neuron.mat
# ---------------------------------------------------------------------------

def _extract_final_footprints(session_dir: Path,
                              work_dir: Path | None = None
                              ) -> tuple[np.ndarray, int, int] | None:
    """
    Call MATLAB to extract A_final and the original gSig/gSiz from neuron.mat.
    Saves a temp retro_final.mat (in work_dir if given, so diagnostic runs can
    keep session_dir strictly read-only), loads it in Python, then removes it.
    Returns (A_final, gSig, gSiz) or None on failure.
      A_final : (pixels, N_kept)  — MATLAB column-major pixel rows
      gSig    : scalar, Gaussian half-width used in the original run
      gSiz    : scalar, Gaussian full-width used in the original run
    """
    wd   = work_dir or session_dir
    sd   = str(session_dir).replace("\\", "/")
    wds  = str(wd).replace("\\", "/")
    repo = REPO_ROOT.replace("\\", "/")
    script = (
        f"cd('{repo}');"
        f" run('cnmfe_setup.m');"
        f" load('{sd}/neuron.mat');"
        f" A_final = full(neuron.A);"
        f" gSig_orig = neuron.options.gSig;"
        f" gSiz_orig = neuron.options.gSiz;"
        f" save('{wds}/retro_final.mat', 'A_final', 'gSig_orig', 'gSiz_orig', '-v7');"
        f" fprintf('Extracted %d final neurons (gSig=%d gSiz=%d)\\n',"
        f" size(A_final, 2), gSig_orig, gSiz_orig);"
    )
    ok = run_cnmfe._run_matlab(script, log, timeout_hours=0.25)
    if not ok:
        return None
    try:
        data = sio.loadmat(str(wd / "retro_final.mat"))
        A_final   = data["A_final"]
        gSig_orig = int(np.asarray(data["gSig_orig"]).flat[0])
        gSiz_orig = int(np.asarray(data["gSiz_orig"]).flat[0])
        return A_final, gSig_orig, gSiz_orig
    except Exception as e:
        log(f"  [BOOTSTRAP] Could not load retro_final.mat: {e}")
        return None
    finally:
        (wd / "retro_final.mat").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 3 — Run headless CNMFe into bootstrap_dir
# ---------------------------------------------------------------------------

def _run_headless_into_bootstrap(tif_path: Path, bootstrap_dir: Path,
                                  p: dict) -> bool:
    """
    Run CNMFe_Biane_headless.m with dir_nm_override set to bootstrap_dir so all
    outputs (neuron.mat, A.txt, spatial_footprints.mat, etc.) are saved there
    instead of the session folder, preserving the curated files.
    """
    nam  = str(tif_path).replace("\\", "/")
    bdir = str(bootstrap_dir).replace("\\", "/") + "/"
    repo = REPO_ROOT.replace("\\", "/")

    script = (
        f"cd('{repo}');"
        f" run('cnmfe_setup.m');"
        f" nam = '{nam}';"
        f" gSig = {p['gSig']}; gSiz = {p['gSiz']};"
        f" min_corr = {p['min_corr']}; min_pnr = {p['min_pnr']}; bd = {p['bd']};"
        f" dir_nm_override = '{bdir}';"
        f" run('CNMFe_Biane_headless.m');"
    )
    return run_cnmfe._run_matlab(script, log, timeout_hours=4.0)


# ---------------------------------------------------------------------------
# Steps 4-6 — Load candidates, match, extract features, save labels
# ---------------------------------------------------------------------------

def _load_candidates(bootstrap_dir: Path) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """
    Load spatial footprints and raw traces from the bootstrap output directory.
    Returns (A_review, C_raw, d1, d2) or None on failure.
      A_review : (pixels, N_candidates)
      C_raw    : (N_candidates, T)
    """
    try:
        sf_file = bootstrap_dir / "spatial_footprints.mat"
        if not sf_file.exists():
            log("  [BOOTSTRAP] spatial_footprints.mat not found in _bootstrap/.")
            return None
        data = sio.loadmat(str(sf_file))
        if "spatial_footprints" in data:
            fp3d = data["spatial_footprints"]   # (N, H, W)
            N, H, W = fp3d.shape
            A_review = fp3d.reshape(N, H * W).T  # (pixels, N)
            d1, d2 = H, W
        else:
            # MATLAB refuses to save variables >2GB in v7 format (it warns
            # "not saved" and writes a stub) — happens when a permissive run
            # yields >~1000 candidates. Rebuild from A.txt, which the headless
            # run always writes: its columns are MATLAB F-order pixels, so
            # reindex to the C-order convention this loader promises.
            a_file = bootstrap_dir / "A.txt"
            cn_file = bootstrap_dir / "Cn.mat"
            if not a_file.exists() or not cn_file.exists():
                log("  [BOOTSTRAP] spatial_footprints.mat is a >2GB save stub "
                    "and A.txt/Cn.mat fallback is unavailable.")
                return None
            log("  [BOOTSTRAP] spatial_footprints.mat is a >2GB save stub — "
                "rebuilding candidates from A.txt (slow, ~minutes)...")
            d1, d2 = sio.loadmat(str(cn_file))["Cn"].shape
            A_txt = np.loadtxt(str(a_file))          # (pixels, N), F-order rows
            if A_txt.ndim == 1:
                A_txt = A_txt[:, np.newaxis]
            if A_txt.shape[0] != d1 * d2:
                log(f"  [BOOTSTRAP] A.txt has {A_txt.shape[0]} pixel rows, "
                    f"expected {d1}x{d2}={d1 * d2}.")
                return None
            A_review = _reorder_Fcols_to_C(A_txt, d1, d2)

        c_raw_file = bootstrap_dir / "C_raw.txt"
        if not c_raw_file.exists():
            log("  [BOOTSTRAP] C_raw.txt not found in _bootstrap/.")
            return None
        C_raw = np.loadtxt(str(c_raw_file))  # (N, T)
        if C_raw.ndim == 1:
            C_raw = C_raw[np.newaxis, :]

        return A_review, C_raw, d1, d2

    except Exception as e:
        log(f"  [BOOTSTRAP] Error loading candidates from _bootstrap/: {e}")
        log(traceback.format_exc())
        return None


def _match_and_save(session_dir: Path, bootstrap_dir: Path,
                    A_final: np.ndarray, out_dir: Path | None = None,
                    keep_candidates: bool = False,
                    run_params: dict | None = None) -> bool:
    """
    Load candidates from bootstrap_dir, Hungarian-match against A_final,
    extract features, and save candidate_features.npz + labels.mat +
    bootstrap_match_stats.json (schema_version 2) to out_dir (defaults to
    session_dir). With keep_candidates, additionally persists the candidate
    footprints, traces and the full similarity matrix as
    bootstrap_candidates.npz so matching can be redone offline forever after.
    """
    out = out_dir or session_dir
    result = _load_candidates(bootstrap_dir)
    if result is None:
        return False
    A_review, C_raw, d1, d2 = result
    N_review = A_review.shape[1]

    if A_final.shape[0] != d1 * d2:
        log(f"  [BOOTSTRAP] ERROR: A_final has {A_final.shape[0]} pixels but "
            f"candidates are {d1}x{d2}={d1 * d2}. Cannot match.")
        return False
    # Pixel-order alignment (2026-08 fix): A_review rows are C-order, A_final
    # rows are MATLAB F-order. Reindex A_final so the cosine compares
    # like-for-like; see _reorder_Fcols_to_C.
    A_final = _reorder_Fcols_to_C(np.asarray(A_final, dtype=float), d1, d2)
    N_kept  = A_final.shape[1]

    # Diagnostic: log shapes so mismatches are immediately visible
    log(f"  [BOOTSTRAP] A_final:  {A_final.shape}  (pixels x N_kept)")
    log(f"  [BOOTSTRAP] A_review: {A_review.shape}  (pixels x N_candidates)")

    # One-to-one Hungarian matching (same logic as train_classifier retro path)
    nr       = np.sqrt((A_review ** 2).sum(axis=0)) + 1e-12   # (N_review,)
    nf       = np.sqrt((A_final  ** 2).sum(axis=0)) + 1e-12   # (N_kept,)
    corr_mat = (A_review / nr).T @ (A_final / nf)             # (N_review, N_kept)
    row_ind, col_ind = linear_sum_assignment(-corr_mat)

    # Diagnostic: show per-pair similarities so we can see how far off they are
    pair_sims = sorted(
        [(corr_mat[r, c], r, c) for r, c in zip(row_ind, col_ind)],
        reverse=True,
    )
    sim_strs = [f"{s:.3f}" for s, _, _ in pair_sims]
    log(f"  [BOOTSTRAP] Hungarian pair similarities (best first): {sim_strs}")

    labels = np.zeros(N_review, dtype=float)
    for r, c in zip(row_ind, col_ind):
        if corr_mat[r, c] > SPATIAL_MATCH_THRESHOLD:
            labels[r] = 1

    n_keep = int(labels.sum())
    log(f"  [BOOTSTRAP] Matched: {n_keep} keep, {N_review - n_keep} delete "
        f"({N_review} candidates vs {N_kept} final neurons, "
        f"threshold={SPATIAL_MATCH_THRESHOLD})")
    if n_keep == 0:
        log("  [BOOTSTRAP] WARNING: 0 neurons matched. Check params or .tif integrity.")

    # Feature extraction
    # Cn: prefer bootstrap_dir — the headless run saves it at full resolution
    # (d1 x d2, after imresize in CNMFe_Biane_headless.m), guaranteed to match
    # the spatial footprints.  session_dir/Cn.mat may be at a different resolution
    # (e.g. half-res from the original pre-pass without the upsample step).
    Cn        = feat_module.load_cn(bootstrap_dir)
    if Cn is None:
        Cn = feat_module.load_cn(session_dir)
    # Ybg_mean: headless saves it to bootstrap_dir — use it if present
    bg_signal = feat_module.load_background(bootstrap_dir)

    footprints = A_review.T.reshape(N_review, d1, d2)   # (N, H, W)
    rows = []
    for i in range(N_review):
        sf = feat_module.spatial_features(footprints[i])
        tf = feat_module.temporal_features(C_raw[i])
        mf = feat_module.motion_features(C_raw[i], bg_signal)
        cf = feat_module.cn_features(footprints[i], Cn)
        rows.append({**sf, **tf, **mf, **cf})

    feature_names  = list(rows[0].keys())
    feature_matrix = np.array([[r[k] for k in feature_names] for r in rows])

    # Under the 35-column v2 contract (BLA), bootstrap rows carry real ranks
    # but zero-filled v2b + v2_present=0: their candidate traces come from a
    # re-run, not the reviewed recording, so v2b values would not be
    # label-faithful.  This keeps this writer in lockstep with curator.py.
    if getattr(config, "FEATURE_VERSION", 1) >= 2:
        feature_matrix = feat_module.assemble_v2_bootstrap(feature_matrix)
        feature_names  = feat_module.v2_feature_names(feature_names)

    np.savez(
        out / "candidate_features.npz",
        feature_matrix=feature_matrix,
        feature_names=np.array(feature_names),
        auto_rejected=np.array([], dtype=int),   # bootstrap always shows all candidates
        n_candidates=np.array([N_review]),
    )
    sio.savemat(
        str(out / "labels.mat"),
        {"labels": labels.reshape(-1, 1)},
    )

    # Save per-pair similarity scores and matched indices so we can audit
    # match quality and re-apply any threshold in Python without re-running
    # MATLAB.  Includes all N_kept pairs (one per curated neuron), sorted
    # best-first, with the threshold used.
    # candidate_indices: row index into the N_review candidate array
    # curated_indices:   column index into the N_kept curated array
    #
    # schema_version 2 additions (legacy keys and their semantics unchanged):
    #   ambiguous_candidate_indices — the Hungarian partner of each UNMATCHED
    #     curated neuron (pair at/below threshold): true label unknown, mask
    #     from training. Previously only derivable as candidate_indices[n_matched:].
    #   duplicate_candidate_indices — unassigned candidates whose best
    #     similarity to ANY curated neuron clears the threshold: same-cell
    #     re-detections that would otherwise be full-weight negatives.
    ambiguous  = [int(r) for s, r, c in pair_sims if s <= SPATIAL_MATCH_THRESHOLD]
    assigned   = {int(r) for _, r, _ in pair_sims}
    best_per_cand = corr_mat.max(axis=1)
    duplicates = [int(j) for j in range(N_review)
                  if j not in assigned
                  and best_per_cand[j] > SPATIAL_MATCH_THRESHOLD]
    match_stats = {
        "threshold": SPATIAL_MATCH_THRESHOLD,
        "n_candidates": N_review,
        "n_curated": N_kept,
        "n_matched": n_keep,
        "pair_similarities": [round(float(s), 4) for s, _, _ in pair_sims],
        "candidate_indices": [int(r) for s, r, c in pair_sims],
        "curated_indices":   [int(c) for s, r, c in pair_sims],
        "schema_version": 2,
        "matcher": "hungarian_cosine_pixelorder_fixed",
        "session": session_dir.name,
        "d1": d1,
        "d2": d2,
        "cnmfe_params": {k: float(v) for k, v in (run_params or {}).items()
                         if isinstance(v, (int, float, np.integer, np.floating))},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "recovery_by_threshold": {
            f"{t:.2f}": int(sum(1 for s, _, _ in pair_sims if s > t))
            for t in RECOVERY_SWEEP_THRESHOLDS},
        "ambiguous_candidate_indices": ambiguous,
        "duplicate_candidate_indices": duplicates,
        "per_curated_best_similarity": [round(float(s), 4)
                                        for s in corr_mat.max(axis=0)],
    }
    with open(out / "bootstrap_match_stats.json", "w") as f:
        json.dump(match_stats, f, indent=2)

    if keep_candidates:
        from scipy import sparse
        A_csr = sparse.csr_matrix(np.asarray(A_review.T, dtype=np.float32))
        np.savez_compressed(
            out / "bootstrap_candidates.npz",
            # candidate footprints, (N_review, d1*d2) CSR, C-order pixel columns
            A_data=A_csr.data, A_indices=A_csr.indices,
            A_indptr=A_csr.indptr, A_shape=np.array(A_csr.shape),
            C_raw=C_raw.astype(np.float32),
            sim_matrix=corr_mat.astype(np.float32),
            d1=np.array([d1]), d2=np.array([d2]),
        )
        log(f"  [BOOTSTRAP] Saved bootstrap_candidates.npz "
            f"({A_csr.nnz} nonzero footprint px, full sim matrix).")

    log(f"  [BOOTSTRAP] Saved: {N_review} candidates, {n_keep} kept, "
        f"{len(ambiguous)} ambiguous, {len(duplicates)} duplicates, "
        f"{feature_matrix.shape[1]} features.")
    return True


# ---------------------------------------------------------------------------
# Main per-session pipeline
# ---------------------------------------------------------------------------

def bootstrap_session(task_name: str, session_dir: Path, tif_path: Path,
                      out_dir: Path | None = None,
                      keep_candidates: bool = False,
                      params_override: dict | None = None) -> bool:
    """
    Full bootstrap pipeline for one session.
    Returns True if candidate_features.npz + labels.mat were successfully saved.

    out_dir (diagnostic mode): redirect ALL outputs — _bootstrap/, npz, labels,
    JSON, temp files, clamp_warning.txt — to out_dir, leaving session_dir
    strictly read-only. _bootstrap/ is kept for inspection in this mode.
    keep_candidates: also persist bootstrap_candidates.npz (footprints, traces,
    full similarity matrix).
    params_override: dict merged over the estimated params (after the
    gSig/gSiz restore), e.g. {"min_corr": 0.30} for permissive re-runs.
    """
    target = out_dir or session_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir = target / "_bootstrap"
    bootstrap_dir.mkdir(exist_ok=True)
    _succeeded = False

    try:
        log(f"\n{'=' * 65}")
        log(f"  {task_name}/{session_dir.name}"
            + ("  [diag mode]" if out_dir is not None else ""))
        log(f"{'=' * 65}")

        # Step 1: extract A_final + original gSig/gSiz from curated neuron.mat
        log("  Step 1: Extracting curated neuron footprints (MATLAB ~15s)...")
        result1 = _extract_final_footprints(session_dir, work_dir=target)
        if result1 is None:
            log("  FAILED at step 1.")
            return False
        A_final, gSig_orig, gSiz_orig = result1
        log(f"  Step 1 done: {A_final.shape[1]} curated neurons "
            f"(original gSig={gSig_orig}, gSiz={gSiz_orig}).")

        # Step 2: estimate parameters.
        # Pre-agent sessions lack pnr.mat (that file was added by the agent pipeline).
        # Run a quick pre-pass into _bootstrap/ to get Cn + pnr at the correct
        # resolution, then use those for parameter estimation.
        # Use bootstrap_mode=True for more permissive min_corr: pre-agent sessions
        # were originally run with min_corr=0.3 and human review as the filter.
        # Use the original gSig/gSiz from neuron.mat for the pre-pass so Cn is
        # computed with the correct kernel size.
        log("  Step 2: Estimating CNMFe parameters...")
        session_name = session_dir.name
        params_dir = session_dir
        if out_dir is not None:
            # Diag mode: estimation reads (and clamp_warning.txt writes) happen
            # in the shadow dir; seed it with copies of the session's Cn/pnr.
            for fname in ("Cn.mat", "pnr.mat"):
                src = session_dir / fname
                if src.exists() and not (out_dir / fname).exists():
                    shutil.copy2(str(src), str(out_dir / fname))
            params_dir = out_dir
        image_dir = params_dir
        if not (params_dir / "pnr.mat").exists():
            log(f"  Step 2a: No pnr.mat found — running pre-pass "
                f"(gSig={gSig_orig}, gSiz={gSiz_orig} from original run)...")
            pre_ok = run_cnmfe.run_pre_pass_to_dir(
                tif_path, bootstrap_dir, gSig_orig, gSiz_orig, log)
            if pre_ok:
                image_dir = bootstrap_dir
            else:
                log("  Step 2a: Pre-pass failed — will use defaults for min_corr/min_pnr.")
        p = params_module.estimate_all_params(session_name, params_dir, log,
                                              image_dir=image_dir,
                                              bootstrap_mode=True)
        # Override gSig/gSiz with values from the original run
        p["gSig"] = gSig_orig
        p["gSiz"] = gSiz_orig
        if params_override:
            p.update(params_override)
            log(f"  Step 2: params_override applied: {params_override}")
        log(f"  Step 2 done: {p}")

        # Step 3: run headless CNMFe, all outputs go to _bootstrap/
        log(f"  Step 3: Running headless CNMFe into _bootstrap/ (30-120 min)...")
        ok = _run_headless_into_bootstrap(tif_path, bootstrap_dir, p)
        if not ok:
            log("  FAILED at step 3.")
            return False

        # Steps 4-6: match candidates, extract features, save labels
        log("  Step 4: Matching candidates to ground truth, extracting features...")
        success = _match_and_save(session_dir, bootstrap_dir, A_final,
                                  out_dir=out_dir,
                                  keep_candidates=keep_candidates,
                                  run_params=p)
        if not success:
            log("  FAILED at step 4.")
            return False

        _succeeded = True
        return True

    except Exception as e:
        log(f"  [BOOTSTRAP] Unexpected error: {e}")
        log(traceback.format_exc())
        return False

    finally:
        if out_dir is not None:
            log(f"  Diag mode: all outputs (incl. _bootstrap/) kept in {target}")
        elif bootstrap_dir.exists():
            if _succeeded:
                shutil.rmtree(str(bootstrap_dir), ignore_errors=True)
                log("  _bootstrap/ cleaned up.")
            else:
                log("  _bootstrap/ preserved for diagnostics (run failed).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="List eligible sessions without running anything.")
    parser.add_argument("--refresh-missing-stats", action="store_true",
                        help="Reprocess already-bootstrapped pre-agent sessions that are "
                             "MISSING bootstrap_match_stats.json (legacy sessions matched "
                             "before that file existed). Regenerates the JSON and refreshes "
                             "npz/labels so they get proper bootstrap weighting. NOTE: this "
                             "overwrites those sessions' npz/labels with a fresh match.")
    parser.add_argument("--sessions-file", type=Path, default=None,
                        help="Text file of explicit session dir paths (one per "
                             "line, # comments allowed). Processes exactly these "
                             "sessions even if already bootstrapped — i.e. a "
                             "re-run that OVERWRITES their npz/labels/JSON. "
                             "Used for the post-fix corpus re-runs.")
    parser.add_argument("--keep-candidates", action="store_true",
                        help="Persist bootstrap_candidates.npz (candidate "
                             "footprints, traces, full similarity matrix) so "
                             "matching can be redone offline. Recommended for "
                             "all re-runs.")
    parser.add_argument("--worker", type=int, default=0,
                        help="Worker index (0-based). Use with --num-workers.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers. Each worker "
                             "processes every Nth session (interleaved), so "
                             "long sessions are spread evenly across workers.")
    args = parser.parse_args()

    if args.worker < 0 or args.worker >= args.num_workers:
        parser.error(f"--worker must be in [0, num-workers). "
                     f"Got {args.worker} with --num-workers {args.num_workers}.")
    if args.sessions_file and args.refresh_missing_stats:
        parser.error("--sessions-file and --refresh-missing-stats are mutually "
                     "exclusive.")

    if args.sessions_file:
        all_sessions = []
        for line in args.sessions_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sd = Path(line)
            if not sd.is_dir():
                parser.error(f"--sessions-file entry is not a directory: {sd}")
            if not (sd / "neuron.mat").exists():
                parser.error(f"--sessions-file entry has no neuron.mat: {sd}")
            tifs = list(sd.glob("*.tif")) + list(sd.glob("*.tiff"))
            if not tifs:
                parser.error(f"--sessions-file entry has no .tif: {sd}")
            all_sessions.append((sd.parent.name, sd, tifs[0]))
    else:
        all_sessions = (find_missing_stats_sessions()
                        if args.refresh_missing_stats
                        else find_bootstrap_sessions())
    # Interleaved split: worker W takes indices W, W+N, W+2N, ...
    sessions = all_sessions[args.worker::args.num_workers]

    worker_tag = (f" [worker {args.worker + 1}/{args.num_workers}]"
                  if args.num_workers > 1 else "")
    log(f"Pre-agent sessions eligible for bootstrap: {len(all_sessions)} total, "
        f"{len(sessions)} assigned to this worker{worker_tag}.")
    if not sessions:
        log("Nothing to do.")
        return

    if args.dry_run:
        log("")
        for task_name, session_dir, tif_path in sessions:
            has_cn = (session_dir / "Cn.mat").exists()
            log(f"  {task_name}/{session_dir.name}")
            log(f"    tif: {tif_path.name}  Cn: {'present' if has_cn else 'MISSING (will regenerate)'}")
        return

    n_ok = n_fail = 0
    for i, (task_name, session_dir, tif_path) in enumerate(sessions, 1):
        ok = bootstrap_session(task_name, session_dir, tif_path,
                               keep_candidates=args.keep_candidates)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        log(f"\n  Progress{worker_tag}: {i}/{len(sessions)}  ({n_ok} ok, {n_fail} failed)\n")

    log(f"\n{'=' * 65}")
    log(f"Bootstrap complete{worker_tag}: {n_ok} succeeded, {n_fail} failed.")
    if n_ok > 0:
        log("Next step:  python train_classifier.py --prospective-only")
    log(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
