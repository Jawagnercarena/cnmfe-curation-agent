"""
recurate_sessions.py
Re-run curator scoring on specific sessions using the current trained model,
without re-extracting features or re-running CNMFe.

Use this when the model has been retrained and you want to update the
review package (auto-rejected set, review_neuron.mat, PDF report) for
sessions that haven't been reviewed in MATLAB yet.

Usage:
    python recurate_sessions.py                   # processes all pending sessions
    python recurate_sessions.py <session_path> ... # process specific sessions
    python recurate_sessions.py <session_path> --threshold 0   # review everything

--threshold overrides the model's calibrated cutoff. Use it when the model is
scoring data outside its training distribution (a new viral strategy, indicator,
or objective), where the calibrated cutoff is extrapolation rather than
calibration; 0 auto-rejects nothing so a human labels every candidate.

This module operates on the BLA data root. For vCA1 use recurate_sessions_vCA1.py.

A session is "pending" if it has:
  - candidate_features.npz  (features already extracted by watcher)
  - review_neuron.mat        (old review package exists)
  - NO labels.mat            (not yet reviewed in MATLAB)
  - ROIs_candidates.jpg      (agent pipeline, not bootstrap)
"""
import argparse
import sys
import numpy as np
from pathlib import Path

AGENT_DIR  = Path(__file__).parent
from config import DATA_ROOT

sys.path.insert(0, str(AGENT_DIR))
import curator
import features as feat_module


def log(msg):
    print(msg, flush=True)


def find_pending_sessions() -> list[Path]:
    """Sessions with existing review packages that haven't been reviewed yet."""
    pending = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."): continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir(): continue
            if ((sd / "candidate_features.npz").exists() and
                    (sd / "review_neuron.mat").exists() and
                    (sd / "ROIs_candidates.jpg").exists() and
                    not (sd / "labels.mat").exists() and
                    not (sd / "bootstrap_match_stats.json").exists()):
                pending.append(sd)
    return pending


def recurate(session_dir: Path, threshold_override: float | None = None):
    """
    Re-score candidates in session_dir with the current model,
    update candidate_features.npz (auto_rejected only),
    regenerate review_neuron.mat via MATLAB, and regenerate the PDF.

    threshold_override replaces the model's calibrated reject_threshold. Use it
    when the model is scoring data outside its training distribution, where the
    calibrated cutoff carries no meaning -- 0.0 auto-rejects nothing and sends
    every candidate to human review.
    """
    session_name = f"{session_dir.parent.name}/{session_dir.name}"
    log(f"\n{'='*60}")
    log(f"Re-curating: {session_name}")
    log(f"{'='*60}")

    feat_file = session_dir / "candidate_features.npz"
    if not feat_file.exists():
        log("  [ERROR] candidate_features.npz not found — run watcher first.")
        return

    # Load saved features (do NOT re-extract — they're already correct)
    npz = np.load(str(feat_file), allow_pickle=True)
    feature_matrix = npz["feature_matrix"]
    feature_names  = list(npz["feature_names"])
    N = len(feature_matrix)

    old_auto_rejected = set(npz["auto_rejected"].flatten().astype(int).tolist())
    log(f"  Loaded {N} candidates ({len(feature_names)} features).")
    log(f"  Old auto-rejected: {len(old_auto_rejected)}")

    # Score with current model
    scores, model_type, reject_threshold = curator.score_neurons(feature_matrix, log)

    if threshold_override is not None:
        log(f"  Threshold override: {threshold_override:.3f} "
            f"(model's calibrated value was {reject_threshold:.3f})")
        reject_threshold = threshold_override

    # New auto-rejected set (use calibrated threshold stored in model metadata)
    new_auto_rejected = [i for i in range(N) if scores[i] < reject_threshold]
    new_auto_set = set(new_auto_rejected)
    log(f"  New auto-rejected: {len(new_auto_rejected)}")

    newly_rejected = new_auto_set - old_auto_rejected
    newly_rescued  = old_auto_rejected - new_auto_set
    log(f"  Newly rejected (removed from review): {len(newly_rejected)}")
    log(f"  Rescued from auto-reject (added to review): {len(newly_rescued)}")

    # Update candidate_features.npz with new auto_rejected (other arrays unchanged)
    np.savez(
        feat_file,
        feature_matrix=feature_matrix,
        feature_names=np.array(feature_names),
        auto_rejected=np.array(new_auto_rejected, dtype=int),
        n_candidates=np.array([N]),
    )
    log(f"  candidate_features.npz updated (auto_rejected refreshed).")

    # Load traces and overlap for PDF / merge detection
    try:
        traces = feat_module.load_traces(session_dir)
    except Exception as e:
        log(f"  [WARN] Could not load traces: {e}. PDF will be incomplete.")
        traces = np.zeros((N, 1))

    try:
        _, _, overlap_matrix = feat_module.extract_all(session_dir, lambda m: None)
    except Exception:
        overlap_matrix = np.zeros((N, N))

    # Motion suspects
    try:
        mc_idx = feature_names.index("motion_correlation")
        motion_flagged = [i for i in range(N)
                          if i not in new_auto_set
                          and feature_matrix[i, mc_idx] >= curator.MOTION_CORR_FLAG]
    except ValueError:
        motion_flagged = []
    log(f"  Motion suspects: {len(motion_flagged)}")

    # Merge candidates (among non-rejected)
    keep_mask = [i for i in range(N) if i not in new_auto_set]
    if len(keep_mask) >= 2:
        sub_overlap = overlap_matrix[np.ix_(keep_mask, keep_mask)]
        sub_traces  = traces[keep_mask]
        raw_merges  = curator.find_merge_candidates(sub_overlap, sub_traces, log)
        merge_candidates = [(keep_mask[i], keep_mask[j], ov, corr)
                            for i, j, ov, corr in raw_merges]
    else:
        merge_candidates = []

    review_indices = [i for i in range(N) if i not in new_auto_set]
    log(f"  Review set: {len(review_indices)} neurons (was "
        f"{N - len(old_auto_rejected)} with old model)")

    # Regenerate PDF
    curator._generate_pdf(
        session_dir, session_name,
        feature_matrix, feature_names, scores, traces,
        merge_candidates, new_auto_rejected, motion_flagged, log,
        reject_threshold=reject_threshold,
    )

    # Regenerate review_neuron.mat via MATLAB
    curator._write_review_mat(session_dir, review_indices, log)

    # Regenerate summary text
    merge_summary = "\n".join(
        f"  Neurons {i+1} & {j+1}:  overlap={ov:.0%},  temporal_corr={corr:.2f}"
        for i, j, ov, corr in merge_candidates
    ) or "  None"
    motion_summary = (
        ", ".join(f"Neuron {i+1}" for i in motion_flagged) if motion_flagged else "None"
    )
    summary = (
        f"Session: {session_name}\n"
        f"{'='*55}\n"
        f"Total candidates:       {N}\n"
        f"Auto-rejected:          {len(new_auto_rejected)}  "
        f"(model: {model_type}, threshold: {reject_threshold:.2f})\n"
        f"  [prev model rejected: {len(old_auto_rejected)}]\n"
        f"  Newly rejected vs prev: {len(newly_rejected)}  "
        f"Rescued from prev auto-reject: {len(newly_rescued)}\n"
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
    log(f"  review_summary.txt updated.")
    log(f"  Done. {len(review_indices)} neurons to review in MATLAB.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*",
                    help="session dirs to re-curate (default: all pending sessions)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the model's calibrated reject threshold; "
                         "0 auto-rejects nothing and sends every candidate to review")
    args = ap.parse_args()

    if args.sessions:
        sessions = [Path(p) for p in args.sessions]
    else:
        sessions = find_pending_sessions()
        if not sessions:
            log("No pending sessions found (review_neuron.mat exists, no labels.mat).")
            return
        log(f"Found {len(sessions)} pending session(s):")
        for s in sessions:
            log(f"  {s.parent.name}/{s.name}")

    for session_dir in sessions:
        if not session_dir.is_dir():
            log(f"\n[SKIP] Not a directory: {session_dir}")
            continue
        recurate(session_dir, threshold_override=args.threshold)

    log("\nAll done. Open MATLAB and run run_final_review.m for each session.")


if __name__ == "__main__":
    main()
