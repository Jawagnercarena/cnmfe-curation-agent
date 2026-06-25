"""
check_match_stats.py
Reads bootstrap_match_stats.json from all completed bootstrap sessions
and shows the per-pair similarity distribution so we can audit whether
the SPATIAL_MATCH_THRESHOLD is well-calibrated.

Usage:
    python check_match_stats.py
"""

import json
from pathlib import Path

from config import DATA_ROOT


def main():
    files = sorted(DATA_ROOT.rglob("bootstrap_match_stats.json"))
    if not files:
        print("No bootstrap_match_stats.json files found yet.")
        print("These are written by bootstrap_preagent.py after the code update.")
        return

    all_matched   = []   # similarities of pairs ABOVE threshold (kept)
    all_unmatched = []   # similarities of pairs BELOW threshold (not kept)

    print(f"{'Session':<50}  {'N_cand':>6}  {'Match':>5}  {'Curated':>7}  "
          f"{'Recovery':>9}  Similarities (best→worst)")
    print("-" * 120)

    for f in files:
        task        = f.parent.parent.name
        session     = f.parent.name
        with open(f) as fh:
            s = json.load(fh)

        thresh    = s["threshold"]
        sims      = s["pair_similarities"]   # sorted best-first
        n_matched = s["n_matched"]
        n_curated = s["n_curated"]
        n_cands   = s["n_candidates"]
        recovery  = n_matched / n_curated * 100 if n_curated else 0

        matched_sims   = [x for x in sims if x > thresh]
        unmatched_sims = [x for x in sims if x <= thresh]
        all_matched.extend(matched_sims)
        all_unmatched.extend(unmatched_sims)

        sim_str = "  ".join(
            f"\033[32m{x:.3f}\033[0m" if x > thresh else f"\033[31m{x:.3f}\033[0m"
            for x in sims
        )
        short = f"{task}/{session}"[-50:]
        print(f"{short:<50}  {n_cands:>6}  {n_matched:>5}  {n_curated:>7}  "
              f"{recovery:>8.0f}%  {sim_str}")

    # Summary stats
    print()
    print("=" * 60)
    import statistics
    if all_matched:
        print(f"MATCHED pairs  (n={len(all_matched)}):  "
              f"min={min(all_matched):.3f}  "
              f"median={statistics.median(all_matched):.3f}  "
              f"max={max(all_matched):.3f}")
    if all_unmatched:
        near_thresh = [x for x in all_unmatched if x > thresh - 0.15]
        print(f"UNMATCHED pairs (n={len(all_unmatched)}):  "
              f"min={min(all_unmatched):.3f}  "
              f"median={statistics.median(all_unmatched):.3f}  "
              f"max={max(all_unmatched):.3f}")
        if near_thresh:
            print(f"  → {len(near_thresh)} unmatched pairs within 0.15 of threshold "
                  f"(potential missed matches): {sorted(near_thresh, reverse=True)}")

    total_matched   = len(all_matched)
    total_unmatched = len(all_unmatched)
    if total_matched + total_unmatched > 0:
        overall = total_matched / (total_matched + total_unmatched) * 100
        print(f"\nOverall recovery: {total_matched}/{total_matched + total_unmatched} "
              f"curated neurons matched ({overall:.1f}%)")


if __name__ == "__main__":
    main()
