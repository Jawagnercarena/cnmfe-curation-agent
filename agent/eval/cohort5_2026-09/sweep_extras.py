"""
sweep_extras.py -- characterise the candidates a gSig sweep arm finds that the
session's live run has no counterpart for ("missed by reference"), so the
operator can judge them with numbers beside the PDF.

Read-only.  Re-does the sweep's footprint match (max cosine similarity of the
arm's footprints against the live run's, threshold 0.45 as in
agent/sweep_gsig.py), extracts the 13 base features on the arm directory with
agent/features.py extract_all, and scores every arm candidate with the
13-column companion first-pass model stored in the area's joblib
(first_pass_scaler / first_pass_clf; the same model the curator uses for the
neighbour mask).  Prints, per unmatched candidate: first-pass score, area,
circularity, peak SNR, skewness, best cosine to the live run; then group
medians for the unmatched arm candidates, the matched arm candidates and the
live run's own candidates (with the live candidates split by the curator's
auto-reject set when candidate_features.npz is present).

Usage:
    python sweep_extras.py --area vCA1 --session <live session dir> --arm gSig07
                           [--model agent/model/vCA1/classifier.joblib] [--out json]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent.parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT  # noqa: E402
import features as feat  # noqa: E402

SPATIAL_MATCH_THRESHOLD = 0.45
SHOW = ["first_pass", "area", "circularity", "peak_snr", "skewness", "cn_correlation", "best_cos"]


def quiet(msg=""):
    pass


def footprints_matrix(session_dir: Path):
    fps = feat.load_spatial(session_dir)
    N = len(fps)
    A = fps.reshape(N, -1).T.astype(np.float64)
    return A / (np.sqrt((A ** 2).sum(axis=0)) + 1e-12), N


def first_pass_scores(model: dict, X13: np.ndarray, names: list[str]) -> np.ndarray:
    sc, clf = model.get("first_pass_scaler"), model.get("first_pass_clf")
    if sc is None or clf is None:
        raise SystemExit("joblib has no companion first-pass model (first_pass_scaler / first_pass_clf)")
    n_in = getattr(sc, "n_features_in_", X13.shape[1])
    if n_in != X13.shape[1]:
        raise SystemExit(f"first-pass model expects {n_in} columns, extract_all gave {X13.shape[1]}: {names}")
    return clf.predict_proba(sc.transform(X13))[:, 1]


def table(rows: np.ndarray, names: list[str], scores: np.ndarray, best: np.ndarray, idx: np.ndarray) -> list[dict]:
    out = []
    for i in idx:
        d = {"cand": int(i) + 1, "first_pass": float(scores[i]), "best_cos": float(best[i])}
        for k in ("area", "circularity", "peak_snr", "skewness", "cn_correlation"):
            if k in names:
                d[k] = float(rows[i, names.index(k)])
        out.append(d)
    return out


def medians(rows: np.ndarray, names: list[str], scores: np.ndarray, idx: np.ndarray) -> dict:
    if len(idx) == 0:
        return {}
    d = {"n": int(len(idx)), "first_pass": float(np.median(scores[idx])),
         "first_pass_frac_ge_0.02": float((scores[idx] >= 0.02).mean())}
    for k in ("area", "circularity", "peak_snr", "skewness", "cn_correlation"):
        if k in names:
            d[k] = float(np.median(rows[idx, names.index(k)]))
    return d


def fmt(d: dict) -> str:
    return "  ".join(f"{k} {v:7.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="vCA1")
    ap.add_argument("--session", required=True, type=Path, help="the live session directory")
    ap.add_argument("--arm", required=True, help="arm folder name under <AREA>/.sweep/<session>/, e.g. gSig07")
    ap.add_argument("--model", type=Path, default=None, help="joblib with the first-pass model (default: the area's deployed one)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    live = a.session.resolve()
    arm = DATA_PARENT / a.area / ".sweep" / live.name / a.arm
    model_path = a.model or (AGENT_DIR / "model" / a.area / "classifier.joblib")
    model = joblib.load(str(model_path))

    A_arm, n_arm = footprints_matrix(arm)
    A_live, n_live = footprints_matrix(live)
    best = (A_arm.T @ A_live).max(axis=1)                    # (n_arm,)
    unmatched = np.nonzero(best < SPATIAL_MATCH_THRESHOLD)[0]
    matched = np.nonzero(best >= SPATIAL_MATCH_THRESHOLD)[0]

    X_arm, names, _ = feat.extract_all(arm, quiet)
    X_live, names_l, _ = feat.extract_all(live, quiet)
    if names != names_l:
        raise SystemExit("feature name mismatch between arm and live extraction")
    s_arm = first_pass_scores(model, X_arm, names)
    s_live = first_pass_scores(model, X_live, names)

    auto = np.array([], dtype=int)
    npz = live / "candidate_features.npz"
    if npz.exists():
        auto = np.load(npz, allow_pickle=True)["auto_rejected"].astype(int).ravel()
    live_kept = np.setdiff1d(np.arange(n_live), auto)

    print(f"session {live.name}  arm {a.arm}  model {model_path.name}")
    print(f"arm candidates {n_arm}: matched {len(matched)}, unmatched (missed by the live run) {len(unmatched)}; "
          f"live candidates {n_live} ({len(auto)} auto-rejected by the curator)")
    print()
    print(f"unmatched arm candidates (cosine < {SPATIAL_MATCH_THRESHOLD} to every live footprint):")
    rows = table(X_arm, names, s_arm, best, unmatched)
    for r in sorted(rows, key=lambda r: -r["first_pass"]):
        print("  " + fmt(r))
    print()
    print("group medians:")
    groups = {
        "arm unmatched": medians(X_arm, names, s_arm, unmatched),
        "arm matched": medians(X_arm, names, s_arm, matched),
        "live all": medians(X_live, names, s_live, np.arange(n_live)),
        "live not auto-rejected": medians(X_live, names, s_live, live_kept),
        "live auto-rejected": medians(X_live, names, s_live, auto),
    }
    for g, d in groups.items():
        print(f"  {g:24s} " + (fmt(d) if d else "(none)"))
    if a.out:
        a.out.write_text(json.dumps({"session": live.name, "arm": a.arm, "model": str(model_path),
                                     "unmatched": rows, "groups": groups}, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
