"""
pilot_far.py -- out-of-sample false auto-reject rate of a frozen model on
sessions that were reviewed at threshold 0 (every candidate carries a human
label, so misses below any threshold are visible).

Read-only.  For each session the stored 35-column feature matrix is scored
with the named joblib exactly as the curator scores it (scaler.transform, then
clf.predict_proba[:, 1]; agent/curator.py:204-208, no thread pinning, same as
production) and joined with the full-length labels through rt_lib.load_record.
Prints per session, per animal and pooled: kept cells, misses at each
threshold, junk caught, and the rule-of-three 95% upper bound when a group has
zero misses.  Writes nothing unless --out is given.

The model must be one that never saw these sessions (e.g. the
classifier_prepilot_<date>.joblib snapshot taken before their labels were
ingested); the script prints the joblib's mtime, md5 and n_sessions so the
report can say which model was measured.

Usage:
    python pilot_far.py --model <joblib> --area vCA1 --sessions <dir> [<dir> ...]
                        [--thresholds 0.01,0.02,0.03,0.04] [--out result.json]
                        [--allow-auto-rejected]
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bootstrap_redteam_2026-08"))
import rt_lib as rt  # noqa: E402


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def score_like_curator(model: dict, X: np.ndarray) -> np.ndarray:
    scaler, clf = model["scaler"], model["clf"]
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is not None and n_in != X.shape[1]:
        raise ValueError(f"model expects {n_in} columns, session has {X.shape[1]}")
    return clf.predict_proba(scaler.transform(X))[:, 1]


def summarise(rows: list[dict], thresholds: list[float]) -> dict:
    """rows: per-session dicts with 'scores', 'y'.  Returns per-T pooled stats."""
    s = np.concatenate([r["scores"] for r in rows])
    y = np.concatenate([r["y"] for r in rows])
    kept = int((y == 1).sum())
    out = {"n_sessions": len(rows), "n_cand": int(len(y)), "kept": kept, "per_T": {}}
    for T in thresholds:
        far, junk, _ = rt.far_junk(s, y, T)
        miss = int((s[y == 1] < T).sum())
        entry = {"misses": miss, "far_pct": far, "junk_caught_pct": junk}
        if miss == 0 and kept > 0:
            entry["far_upper95_pct_rule_of_three"] = 300.0 / kept
        out["per_T"][f"{T:.2f}"] = entry
    return out


def fmt_block(label: str, block: dict, thresholds: list[float]) -> str:
    cells = []
    for T in thresholds:
        e = block["per_T"][f"{T:.2f}"]
        ub = f" (<= {e['far_upper95_pct_rule_of_three']:.1f}% at 95%)" if "far_upper95_pct_rule_of_three" in e else ""
        cells.append(f"T={T:.2f}: {e['misses']:3d} miss = {e['far_pct']:5.2f}%{ub}, junk {e['junk_caught_pct']:5.1f}%")
    return f"{label:44s} kept {block['kept']:5d} of {block['n_cand']:5d} | " + " | ".join(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="joblib that never saw these sessions")
    ap.add_argument("--area", default="vCA1", help="area token for rt_lib (animal parsing)")
    ap.add_argument("--sessions", nargs="+", required=True, type=Path, help="session directories")
    ap.add_argument("--thresholds", default="0.01,0.02,0.03,0.04")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON result path")
    ap.add_argument("--allow-auto-rejected", action="store_true",
                    help="also accept sessions whose npz has auto-rejected candidates "
                         "(their misses below the curation threshold are NOT measurable)")
    a = ap.parse_args()
    thresholds = [float(t) for t in a.thresholds.split(",") if t.strip()]

    model = joblib.load(str(a.model))
    print(f"model: {a.model}")
    print(f"  mtime {dt.datetime.fromtimestamp(a.model.stat().st_mtime):%Y-%m-%d %H:%M:%S}  md5 {md5_of(a.model)}")
    print("  " + ", ".join(f"{k}={model.get(k)}" for k in
                           ("model_type", "n_features", "feature_version", "n_sessions", "reject_threshold", "agent_weight")))

    rows, skipped = [], []
    for sd in a.sessions:
        sd = sd.resolve()
        if not (sd / "candidate_features.npz").exists() or not (sd / "labels.mat").exists():
            skipped.append((sd.name, "no candidate_features.npz + labels.mat"))
            continue
        rec = rt.load_record(sd, a.area)
        if rec["auto_rejected"].size > 0 and not a.allow_auto_rejected:
            skipped.append((sd.name, f"{rec['auto_rejected'].size} auto-rejected candidates (not threshold 0)"))
            continue
        scores = score_like_curator(model, rec["X"])
        rows.append({"name": rec["name"], "animal": rec["animal"], "scores": scores,
                     "y": np.asarray(rec["y"]), "reviewed": np.asarray(rec["reviewed"]),
                     "n_auto_rejected": int(rec["auto_rejected"].size)})

    if not rows:
        sys.exit("no usable session")

    print()
    print("per session")
    for r in rows:
        print(fmt_block(r["name"][-44:], summarise([r], thresholds), thresholds))
    print()
    print("per animal")
    result = {"model": str(a.model), "model_md5": md5_of(a.model), "thresholds": thresholds,
              "sessions": [r["name"] for r in rows], "per_animal": {}, "pooled": None, "skipped": skipped}
    for an in sorted({r["animal"] for r in rows}):
        block = summarise([r for r in rows if r["animal"] == an], thresholds)
        result["per_animal"][an] = block
        print(fmt_block(f"{an} ({block['n_sessions']} sessions)", block, thresholds))
    print()
    pooled = summarise(rows, thresholds)
    result["pooled"] = pooled
    print(fmt_block(f"POOLED ({pooled['n_sessions']} sessions)", pooled, thresholds))
    if skipped:
        print()
        print("skipped:")
        for name, why in skipped:
            print(f"  {name}: {why}")
    if a.out:
        a.out.write_text(json.dumps(result, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
