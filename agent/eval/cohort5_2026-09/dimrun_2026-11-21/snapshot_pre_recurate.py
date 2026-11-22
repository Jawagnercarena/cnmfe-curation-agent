"""
snapshot_pre_recurate.py -- keep the watcher's curation of a labelled-dim-run
session before recurate_sessions_vCA1.py --threshold 0 rewrites it.

recurate rewrites candidate_features.npz with an empty auto_rejected array, so
the set the deployed threshold would have auto-rejected is lost unless it is
kept here.  With the threshold-0 labels that set gives the false auto-reject
rate of the deployed threshold on dim fields, which has not been measured.

For each session folder given: reads candidate_features.npz (read-only), scores
the feature matrix with the current vCA1 model exactly as recurate does
(curator.score_neurons), and writes <out>/<session>_curated_by_watcher.npz with
the original arrays plus scores, reject_threshold, model_type and the model
file's modification time.  Write-once: an existing snapshot is never
overwritten.  Nothing in the session folder is written.

Usage:
    python snapshot_pre_recurate.py <session_dir> [<session_dir> ...] [--out DIR]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

AGENT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AGENT))
import config_vCA1                      # noqa: E402
sys.modules["config"] = config_vCA1
import curator                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    if not a.out.is_dir():
        sys.exit(f"output folder missing: {a.out}")

    model_file = Path(curator._model_path())
    model_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(model_file.stat().st_mtime))
    print(f"model: {model_file}  (modified {model_mtime})")

    bad = 0
    for sd in a.sessions:
        src = sd / "candidate_features.npz"
        dst = a.out / (sd.name + "_curated_by_watcher.npz")
        if not src.is_file():
            print(f"  !! no candidate_features.npz in {sd}")
            bad += 1
            continue
        if (sd / "labels.mat").exists():
            print(f"  !! {sd.name} is labelled; not a pre-review snapshot, skipped")
            bad += 1
            continue
        if dst.exists():
            print(f"  snapshot exists, not overwritten: {dst.name}")
            continue
        d = np.load(str(src), allow_pickle=True)
        fm = d["feature_matrix"]
        ar = np.sort(d["auto_rejected"].ravel().astype(int))
        n = int(d["n_candidates"].ravel()[0])
        scores, model_type, thr = curator.score_neurons(fm, lambda m: None)
        scores = np.asarray(scores, dtype=float)
        below = np.flatnonzero(scores < thr)
        same = np.array_equal(ar, below)
        np.savez(
            dst,
            feature_matrix=fm,
            feature_names=d["feature_names"],
            auto_rejected=ar,
            n_candidates=np.array([n]),
            scores=scores,
            reject_threshold=np.array([float(thr)]),
            model_type=np.array([str(model_type)]),
            model_mtime=np.array([model_mtime]),
        )
        print(f"  {sd.name}: N {n}, auto-rejected in the npz {ar.size}, "
              f"scores < {thr:g} now {below.size}, same set: {same}, "
              f"scores {scores.min():.3f}-{scores.max():.3f} -> {dst.name}")
        if not same or fm.shape[0] != n:
            print("  !! the npz's auto-rejected set is not what the current model gives at its "
                  "threshold; look before re-curating")
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
