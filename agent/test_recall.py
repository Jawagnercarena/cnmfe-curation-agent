"""
test_recall.py
Tests classifier recall on pre-agent sessions where neuron.mat contains
only confirmed-good neurons (no candidates, just the final accepted set).

These sessions have no rejected candidates, so this is a one-sided test:
  - True positives: accepted neurons the classifier scores >= REJECT_THRESHOLD
  - False negatives: accepted neurons the classifier wrongly rejects (score < REJECT_THRESHOLD)

A false-negative rate > 5% suggests the model is too aggressive on the
types of neurons seen in these sessions.

Usage:
    cd C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent
    python test_recall.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio

AGENT_DIR = Path(__file__).parent
from config import DATA_ROOT, MODEL_DIR

sys.path.insert(0, str(AGENT_DIR))
import features as feat_module
from curator import REJECT_THRESHOLD


def log(msg):
    print(msg, flush=True)


def find_preagent_sessions() -> list[Path]:
    """Sessions with A.txt + neuron.mat but NOT ROIs_candidates.jpg."""
    sessions = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            is_agent = (session_dir / "ROIs_candidates.jpg").exists()
            has_outputs = ((session_dir / "A.txt").exists() or
                           (session_dir / "spatial_footprints.mat").exists())
            if not is_agent and has_outputs:
                sessions.append(session_dir)
    return sessions


def load_model():
    model_path = MODEL_DIR / "classifier.joblib"
    if not model_path.exists():
        log("No classifier.joblib found. Run train_classifier.py first.")
        return None, None
    m = joblib.load(str(model_path))
    return m["scaler"], m["clf"]


def main():
    scaler, clf = load_model()
    if clf is None:
        return

    sessions = find_preagent_sessions()
    log(f"Found {len(sessions)} pre-agent sessions to test.\n")

    total_neurons = 0
    total_rejected = 0
    session_results = []

    for session_dir in sessions:
        try:
            feature_matrix, feature_names, _ = feat_module.extract_all(session_dir, log)
            N = len(feature_matrix)

            X_sc = scaler.transform(feature_matrix)
            probs = clf.predict_proba(X_sc)[:, 1]

            n_rejected = int((probs < REJECT_THRESHOLD).sum())
            session_results.append((session_dir, N, n_rejected, probs))
            total_neurons += N
            total_rejected += n_rejected

            pct = 100 * n_rejected / N if N > 0 else 0
            flag = "  *** CONCERN ***" if pct > 5 else ""
            log(f"  {session_dir.parent.name}/{session_dir.name}:  "
                f"{N} neurons,  {n_rejected} wrongly rejected  "
                f"({pct:.1f}%){flag}")

        except Exception as e:
            log(f"  ERROR in {session_dir.name}: {e}")

    if total_neurons == 0:
        log("No neurons found.")
        return

    overall_pct = 100 * total_rejected / total_neurons
    log(f"\n{'=' * 60}")
    log(f"Overall recall test:")
    log(f"  Confirmed-good neurons:  {total_neurons}")
    log(f"  Wrongly rejected:        {total_rejected}  ({overall_pct:.1f}%)")
    log(f"  Recall (kept):           {total_neurons - total_rejected}  "
        f"({100 - overall_pct:.1f}%)")
    log(f"  REJECT_THRESHOLD used:   {REJECT_THRESHOLD:.2f}")

    if overall_pct <= 2:
        log(f"\n  Result: EXCELLENT — model rarely rejects confirmed neurons.")
    elif overall_pct <= 5:
        log(f"\n  Result: GOOD — acceptable false-negative rate.")
    elif overall_pct <= 10:
        log(f"\n  Result: MODERATE — consider lowering REJECT_THRESHOLD slightly.")
    else:
        log(f"\n  Result: CONCERN — model is rejecting too many confirmed neurons. "
            f"Check feature consistency between pre-agent and agent sessions.")

    # Show score distribution for all confirmed neurons
    all_scores = np.concatenate([res[3] for res in session_results])
    percentiles = [1, 5, 10, 25, 50]
    log(f"\n  Score distribution for {total_neurons} confirmed-good neurons:")
    for p in percentiles:
        log(f"    {p:3d}th percentile: {np.percentile(all_scores, p):.3f}")


if __name__ == "__main__":
    main()
