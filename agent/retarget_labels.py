"""
retarget_labels.py
Re-derives labels.mat from bootstrap_match_stats.json at any spatial
similarity threshold WITHOUT re-running MATLAB.

Requires that bootstrap_match_stats.json was saved with candidate_indices
and curated_indices (written by bootstrap_preagent.py after 2026-03-22).

Usage:
    python retarget_labels.py --threshold 0.40
    python retarget_labels.py --threshold 0.40 --dry-run

    # Preview only (no writes):
    python retarget_labels.py --threshold 0.40 --dry-run

What it does:
  For each session that has a bootstrap_match_stats.json with full indices:
    1. Load the N_review × 1 candidate labels (all zeros initially).
    2. For each (candidate_idx, curated_idx, similarity) triple:
       - If similarity > new threshold → set labels[candidate_idx] = 1
    3. Overwrite labels.mat with the new labels.
    4. Update the "threshold" field in bootstrap_match_stats.json to the new
       value so check_match_stats.py shows correct colouring.

Sessions whose JSON was saved without indices (old runs) are skipped with
a warning — those sessions must be re-run with bootstrap_preagent.py to
get the indices saved.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from config import DATA_ROOT


def retarget_session(json_path: Path, new_thresh: float,
                     dry_run: bool) -> dict:
    """
    Re-apply threshold for one session.  Returns a summary dict.
    """
    with open(json_path) as f:
        s = json.load(f)

    if "candidate_indices" not in s:
        return {"status": "skip_no_indices", "path": str(json_path)}

    sims         = s["pair_similarities"]   # sorted best-first
    cand_idxs    = s["candidate_indices"]
    n_candidates = s["n_candidates"]

    # Capture old values BEFORE mutating s
    old_thresh    = s.get("threshold", "?")
    old_n_matched = s.get("n_matched", "?")

    labels = np.zeros(n_candidates, dtype=np.float64)
    n_matched = 0
    for sim, cidx in zip(sims, cand_idxs):
        if sim > new_thresh:
            labels[cidx] = 1
            n_matched += 1

    session_dir = json_path.parent
    labels_path = session_dir / "labels.mat"

    if not dry_run:
        sio.savemat(str(labels_path), {"labels": labels.reshape(-1, 1)})

        # Update threshold in JSON so check_match_stats.py is consistent
        s["threshold"] = new_thresh
        s["n_matched"] = n_matched
        with open(json_path, "w") as f:
            json.dump(s, f, indent=2)
    task    = json_path.parent.parent.name
    session = json_path.parent.name

    return {
        "status":        "dry_run" if dry_run else "updated",
        "session":       f"{task}/{session}",
        "n_candidates":  n_candidates,
        "n_curated":     s["n_curated"],
        "old_threshold": old_thresh,
        "old_matched":   old_n_matched,
        "new_threshold": new_thresh,
        "new_matched":   n_matched,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-apply spatial match threshold to bootstrap labels.mat files."
    )
    parser.add_argument("--threshold", type=float, required=True,
                        help="New spatial similarity threshold (e.g. 0.40)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing files.")
    args = parser.parse_args()

    files = sorted(DATA_ROOT.rglob("bootstrap_match_stats.json"))
    if not files:
        print("No bootstrap_match_stats.json files found.")
        return

    skipped  = []
    results  = []

    for f in files:
        r = retarget_session(f, args.threshold, args.dry_run)
        if r["status"] == "skip_no_indices":
            skipped.append(r["path"])
        else:
            results.append(r)

    # Report
    print(f"\n{'Session':<50}  {'Curated':>7}  {'OldMatch':>8}  "
          f"{'NewMatch':>8}  {'Change':>6}")
    print("-" * 90)
    total_old = total_new = 0
    for r in results:
        delta = r["new_matched"] - r["old_matched"]
        sign  = "+" if delta >= 0 else ""
        print(f"{r['session'][-50:]:<50}  {r['n_curated']:>7}  "
              f"{r['old_matched']:>8}  {r['new_matched']:>8}  "
              f"{sign}{delta:>5}")
        total_old += r["old_matched"]
        total_new += r["new_matched"]

    print()
    delta_total = total_new - total_old
    sign = "+" if delta_total >= 0 else ""
    print(f"TOTAL: {total_old} → {total_new} matched  ({sign}{delta_total})")
    print(f"Threshold: {results[0]['old_threshold'] if results else '?'} → {args.threshold}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        print(f"\nUpdated {len(results)} labels.mat files.")

    if skipped:
        print(f"\n[SKIPPED] {len(skipped)} sessions without candidate_indices "
              f"(old bootstrap runs — must re-run bootstrap_preagent.py):")
        for p in skipped:
            print(f"  {p}")


if __name__ == "__main__":
    main()
