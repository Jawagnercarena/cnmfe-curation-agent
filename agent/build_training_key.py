"""
build_training_key.py -- build the answer key a trainee is scored against.

For every pool session (review_neuron.mat + labels.mat + candidate_features.npz
+ labels_provenance.txt, BLA and vCA1 only) this writes

    DATA_PARENT/.training/keys/<area>/<task>/<session>/training_key.mat

holding, per review candidate (one per column of review_neuron.mat, in that
order): the reference reviewer's keep/delete and motion tag, the deployed
classifier's score (IN-SAMPLE: the model was trained on these labels), a
"contested" flag where reviewer and model disagree, the full 35-column feature
row with within-session percentile ranks, and a plain-language hint that names
the evidence behind the reference decision.  It also writes
.training/curriculum.csv, one row per key, with a difficulty measure.

The key never touches the production session folder.  Scores are positional
(no feature names in the joblib), so the npz width and names are checked
against the v2 contract and the deployed model before scoring.

Usage (central machine, valence env):
  python agent/build_training_key.py BLA\\6odorDualDiffRew\\AVG5x-...-000
  python agent/build_training_key.py --area vCA1 --reviewer Taylor
  python agent/build_training_key.py --all [--force] [--no-matlab] [--dry-run]

--no-matlab skips the spatial-overlap augment (one headless MATLAB launch that
reads each review_neuron.mat; needed for the duplicate hint).  All cut-offs are
named parameters with ASSUMED defaults (see PARAM_DEFAULTS); they are stored in
the key and printed in every trainee report.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, REPO_ROOT, PYTHON_EXE  # noqa: F401
import training_common as tc
import features as feat_module
from ingest_returns import read_provenance

KEY_SCHEMA_VERSION = 1
EXPECTED_NAMES = feat_module.v2_feature_names(tc.BASE13)   # the 35-column contract

# ASSUMED defaults (2026-10-01).  Every value is overridable on the CLI, is
# written into key.params, and appears in the footer of every trainee report.
PARAM_DEFAULTS = {
    "CONTEST_SCORE": 0.5,    # model keep/delete midpoint; NOT the joblib reject_threshold
    "CLEAR_MARGIN": 0.25,    # clear-cut = model at least this far from CONTEST_SCORE
    "HINT_LOW_PCT": 0.2,     # within-session rank at or below -> "low"
    "HINT_HIGH_PCT": 0.8,    # within-session rank at or above -> "high"
    "DUP_OVERLAP": 0.5,      # cosine footprint overlap -> duplicate hint (needs augment)
}

HINT_LEGEND = [
    "0: no feature-based hint",
    "1: motion -- the reference reviewer deleted this as a motion artefact (m)",
    "2: low SNR / noise -- weak transients relative to the baseline noise",
    "3: diffuse / neuropil-like -- large non-circular footprint, or little contrast with its surround",
    "4: duplicate / split -- overlaps another candidate, or its trace tracks a confident neighbour",
    "5: few plausible transients -- too few events, or events without a calcium-like shape",
    "6: Cn mismatch -- the footprint does not line up with a blob in the correlation image",
    "7: dim but real -- the reference kept it despite a low SNR",
    "9: contested -- reference and model disagree; not counted against the trainee",
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_session_inputs(sd: Path) -> dict | None:
    """npz + labels.mat + provenance -> dict, or None (reason logged) if the
    session cannot be keyed."""
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = np.asarray(npz["feature_matrix"], dtype=float)
    names = [str(x) for x in np.asarray(npz["feature_names"]).ravel()]
    auto = np.asarray(npz["auto_rejected"]).ravel().astype(int)
    N = int(X.shape[0])

    if X.shape[1] != len(EXPECTED_NAMES) or names != EXPECTED_NAMES:
        log(f"  SKIP {sd.name}: feature file is {X.shape[1]} columns, not the "
            f"{len(EXPECTED_NAMES)}-column v2 contract (or the names differ). "
            f"The training program supports v2 sessions only.")
        return None

    ldata = sio.loadmat(str(sd / "labels.mat"))
    labels = np.asarray(ldata["labels"]).ravel().astype(int)
    has_motion = "motion_delete" in ldata
    motion = (np.asarray(ldata["motion_delete"]).ravel().astype(int)
              if has_motion else np.zeros(len(labels), dtype=int))
    n_review = len(labels)

    # Review-column -> npz-row mapping (train_classifier.load_prospective_session).
    if n_review == N:
        row0 = np.arange(N)
    elif n_review == N - len(auto):
        auto_set = set(auto.tolist())
        row0 = np.array([i for i in range(N) if i not in auto_set], dtype=int)
    else:
        log(f"  SKIP {sd.name}: unexpected size mismatch (features={N}, "
            f"labels={n_review}, auto_rejected={len(auto)}, "
            f"expected_review={N - len(auto)}).")
        return None

    reviewer = read_provenance(sd) or ""
    return {"X": X, "names": names, "auto": auto, "N": N, "labels": labels,
            "motion": motion, "has_motion": has_motion, "row0": row0,
            "n_review": n_review, "reviewer": reviewer}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_dir: Path) -> dict:
    """Deployed joblib + metadata; {'available': False, 'why': ...} if unusable."""
    import joblib
    mp = Path(model_dir) / "classifier.joblib"
    if not mp.exists():
        return {"available": False, "why": f"no model at {mp}", "path": str(mp)}
    d = joblib.load(str(mp))
    scaler, clf = d.get("scaler"), d.get("clf")
    if scaler is None or clf is None:
        return {"available": False, "why": "joblib lacks scaler/clf", "path": str(mp)}
    return {
        "available": True, "path": str(mp), "scaler": scaler, "clf": clf,
        "model_type": str(d.get("model_type", "")),
        "n_sessions": float(d.get("n_sessions", np.nan)),
        "n_features": float(d.get("n_features", getattr(scaler, "n_features_in_", np.nan))),
        "feature_version": float(d.get("feature_version", 0)),
        "reject_threshold": float(d.get("reject_threshold", np.nan)),
    }


def score_matrix(model: dict, X: np.ndarray) -> tuple[np.ndarray, str]:
    """(scores over all N rows, score_kind).  Positional scoring, so the width
    is checked the way curator._check_arity does; a mismatch yields NaNs."""
    if not model.get("available"):
        return np.full(len(X), np.nan), "unavailable"
    n_exp = getattr(model["scaler"], "n_features_in_", None)
    if n_exp is not None and X.shape[1] != n_exp:
        log(f"  WARN feature-arity mismatch: matrix has {X.shape[1]} columns but the "
            f"deployed model expects {n_exp}; scores marked unavailable.")
        return np.full(len(X), np.nan), "unavailable"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs = model["scaler"].transform(X)
        s = model["clf"].predict_proba(Xs)[:, 1]
    return np.asarray(s, dtype=float), "insample"


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

def _pct_text(r: float) -> str:
    """Within-session percentile phrase for a rank r in (0, 1]."""
    if r <= 0.5:
        return f"bottom {max(1, int(round(100 * r)))}% of this session"
    return f"top {max(1, int(round(100 * (1 - r))))}% of this session"


def compute_hints(key: dict, P: dict) -> tuple[np.ndarray, list[str]]:
    """Per-item (hint_code, hint text).  Rank-based rules only; columns are
    looked up by name.  Requires key['features'], key['ranks'],
    key['feature_names'], key['ref_keep'], key['ref_motion'], key['contested'],
    key['model_score'] and, when present, key['overlap_max'] /
    key['overlap_partner'] / key['spatial_ok']."""
    names = list(key["feature_names"])
    col = {n: i for i, n in enumerate(names)}
    F, R = key["features"], key["ranks"]
    keep = key["ref_keep"].astype(bool)
    mot = key["ref_motion"].astype(bool)
    con = key["contested"].astype(bool)
    sc = key["model_score"]
    lo, hi = P["HINT_LOW_PCT"], P["HINT_HIGH_PCT"]
    n = len(keep)
    has_sp = bool(key.get("spatial_ok", 0)) and "overlap_max" in key

    def val(name, i):
        j = col[name]
        return f"{name} {F[i, j]:.3g} ({_pct_text(R[i, j])})"

    codes = np.zeros(n, dtype=float)
    texts = []
    for i in range(n):
        parts = []     # (code, text) in priority order
        if mot[i]:
            parts.append((1, "reference deleted this as a motion artefact (m): the transient "
                             "coincides with the frame moving or changing plane"))
        if not keep[i]:
            dup = []
            if has_sp and key["overlap_max"][i] >= P["DUP_OVERLAP"]:
                dup.append(f"footprint overlaps review candidate #{int(key['overlap_partner'][i])} "
                           f"(cosine {key['overlap_max'][i]:.2f})")
            if R[i, col["nb_corr_max"]] >= hi:
                dup.append("trace tracks a confident neighbour: " + val("nb_corr_max", i))
            if dup:
                parts.append((4, "likely duplicate / split: " + "; ".join(dup)))
            diff = []
            if R[i, col["area"]] >= hi and R[i, col["circularity"]] <= lo:
                diff.append(val("area", i) + ", " + val("circularity", i))
            if R[i, col["ring_contrast"]] <= lo:
                diff.append("little contrast with its surround: " + val("ring_contrast", i))
            if diff:
                parts.append((3, "diffuse / neuropil-like: " + "; ".join(diff)))
            snr = []
            if R[i, col["peak_snr"]] <= lo:
                snr.append(val("peak_snr", i))
            if R[i, col["ev_snr"]] <= lo:
                snr.append(val("ev_snr", i))
            if snr:
                parts.append((2, "low SNR / noise: " + ", ".join(snr)))
            ev = []
            if R[i, col["events_per_min"]] <= lo:
                ev.append(val("events_per_min", i))
            if R[i, col["ev_rate"]] <= lo:
                ev.append(val("ev_rate", i))
            if R[i, col["ev_frac_plausible"]] <= lo:
                ev.append(val("ev_frac_plausible", i))
            if ev:
                parts.append((5, "few plausible transients: " + ", ".join(ev)))
            if R[i, col["cn_correlation"]] <= lo:
                parts.append((6, "footprint does not match the correlation image: "
                                 + val("cn_correlation", i)))
        else:
            dim = []
            if R[i, col["peak_snr"]] <= lo:
                dim.append(val("peak_snr", i))
            if R[i, col["ev_snr"]] <= lo:
                dim.append(val("ev_snr", i))
            if dim:
                parts.append((7, "dim but real -- the reference kept it: " + ", ".join(dim)))

        ref_word = "keep" if keep[i] else "delete"
        if con[i]:
            code = 9
            s_txt = "n/a" if np.isnan(sc[i]) else f"{sc[i]:.2f}"
            head = (f"contested -- reference says {ref_word}, model score {s_txt}; "
                    f"not counted against you")
            rest = [t for _, t in parts]
            text = head + ("; " + "; ".join(rest) if rest else "")
        elif parts:
            code = parts[0][0]
            text = f"reference {ref_word}: " + "; ".join(t for _, t in parts)
        else:
            code = 0
            text = (f"reference {ref_word}; no feature-based hint"
                    if not keep[i] else "reference keep; model agrees")
        codes[i] = code
        texts.append(text)
    return codes, texts


# ---------------------------------------------------------------------------
# Key assembly / save
# ---------------------------------------------------------------------------

def make_key(inp: dict, model: dict, P: dict, area: str, task: str, session: str) -> dict:
    X, row0, n = inp["X"], inp["row0"], inp["n_review"]
    scores_all, kind = score_matrix(model, X)
    ranks_all = feat_module.compute_ranks(X)
    # The npz already carries ranks of the 13 base columns (cols 14-26); they
    # were computed over the same full candidate set, so they must agree.
    nb = len(tc.BASE13)
    ranks_match = bool(np.allclose(ranks_all[:, :nb], X[:, nb:2 * nb], atol=1e-9))
    if not ranks_match:
        log(f"  WARN {session}: recomputed base ranks differ from npz columns 14-26 "
            f"(max abs diff {np.max(np.abs(ranks_all[:, :nb] - X[:, nb:2 * nb])):.3g}); "
            f"hints use the recomputed ranks.")

    score = scores_all[row0]
    avail = kind == "insample"
    model_keep = (score >= P["CONTEST_SCORE"]).astype(np.uint8) if avail else np.zeros(n, np.uint8)
    ref_keep = inp["labels"].astype(np.uint8)
    contested = (ref_keep != model_keep).astype(np.uint8) if avail else np.zeros(n, np.uint8)
    clear = np.zeros(n, np.uint8)
    if avail:
        clear = ((contested == 0) & (np.abs(score - P["CONTEST_SCORE"]) >= P["CLEAR_MARGIN"])).astype(np.uint8)

    key = {
        "schema_version": float(KEY_SCHEMA_VERSION),
        "area": area, "task": task, "session": session,
        "reference_reviewer": inp["reviewer"], "key_built_at": tc.now_str(),
        "n_candidates": float(inp["N"]), "n_review": float(n),
        "n_auto_rejected": float(len(inp["auto"])),
        "review_col": np.arange(1, n + 1, dtype=float),
        "npz_row0": row0.astype(float),
        "ref_keep": ref_keep, "ref_motion": inp["motion"].astype(np.uint8),
        "has_motion_field": float(inp["has_motion"]),
        "model_score": score.astype(float), "model_keep": model_keep,
        "score_kind": kind,
        "model_path": str(model.get("path", "")), "model_type": str(model.get("model_type", "")),
        "model_n_sessions": float(model.get("n_sessions", np.nan)),
        "model_n_features": float(model.get("n_features", np.nan)),
        "model_feature_version": float(model.get("feature_version", 0)),
        "model_reject_threshold": float(model.get("reject_threshold", np.nan)),
        "contested": contested, "clear_cut": clear,
        "params": {k: float(v) for k, v in P.items()},
        "n_features": float(X.shape[1]),
        "feature_names": list(inp["names"]),
        "features": X[row0, :].astype(float),
        "ranks": ranks_all[row0, :].astype(float),
        "ranks_match_npz": float(ranks_match),
        "hint_legend": list(HINT_LEGEND),
    }
    key["hint_code"], key["hint"] = compute_hints(key, P)
    return key


def _col(v, dtype=float):
    a = np.asarray(v, dtype=dtype).reshape(-1, 1)
    return a


def _cell_col(strings):
    c = np.empty((len(strings), 1), dtype=object)
    for i, s in enumerate(strings):
        c[i, 0] = str(s) if str(s) else "-"
    return c


def _cell_row(strings):
    c = np.empty((1, len(strings)), dtype=object)
    for i, s in enumerate(strings):
        c[0, i] = str(s)
    return c


def key_to_mat(key: dict) -> dict:
    """MATLAB-friendly dict: column vectors, uint8 flags, cell arrays for
    strings, a struct for params."""
    m = {}
    for k, v in key.items():
        if k in ("ref_keep", "ref_motion", "model_keep", "contested", "clear_cut"):
            m[k] = _col(v, np.uint8)
        elif k in ("review_col", "npz_row0", "model_score", "hint_code",
                   "overlap_max", "overlap_partner", "n_pixels"):
            m[k] = _col(v, float)
        elif k in ("features", "ranks", "centroid_yx"):
            m[k] = np.asarray(v, dtype=float)
        elif k == "feature_names":
            m[k] = _cell_row(v)
        elif k in ("hint", "hint_legend"):
            m[k] = _cell_col(v)
        elif k == "params":
            m[k] = {kk: float(vv) for kk, vv in v.items()}
        elif isinstance(v, str):
            m[k] = v if v else "-"
        else:
            m[k] = float(v)
    return m


def save_key(key: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), key_to_mat(key), format="5", do_compression=True, oned_as="column")


def merge_spatial(key: dict, spatial_path: Path) -> bool:
    """Fold the MATLAB augment output into the key.  Returns True if merged."""
    if not Path(spatial_path).exists():
        return False
    s = sio.loadmat(str(spatial_path))
    n = int(key["n_review"])
    ok = float(np.asarray(s.get("spatial_ok", 0)).ravel()[0])
    if ok != 1 or len(np.asarray(s["overlap_max"]).ravel()) != n:
        key["spatial_ok"] = 0.0
        key["augmented_at"] = str(np.asarray(s.get("augmented_at", "")).ravel()[0]) if "augmented_at" in s else ""
        return True
    key["overlap_max"] = np.asarray(s["overlap_max"], dtype=float).ravel()
    key["overlap_partner"] = np.asarray(s["overlap_partner"], dtype=float).ravel()
    key["centroid_yx"] = np.asarray(s["centroid_yx"], dtype=float).reshape(n, 2)
    key["n_pixels"] = np.asarray(s["n_pixels"], dtype=float).ravel()
    key["spatial_ok"] = 1.0
    key["augmented_at"] = str(np.asarray(s["augmented_at"]).ravel()[0]) if "augmented_at" in s else ""
    return True


def curriculum_row(key: dict, sd: Path, kpath: Path) -> dict:
    sc = key["model_score"]
    avail = key["score_kind"] == "insample"
    P = key["params"]
    near = float(np.mean(np.abs(sc - P["CONTEST_SCORE"]) < P["CLEAR_MARGIN"])) if avail else float("nan")
    return {
        "area": key["area"], "task": key["task"], "session": key["session"],
        "reviewer": key["reference_reviewer"],
        "n_candidates": int(key["n_candidates"]), "n_review": int(key["n_review"]),
        "n_keep": int(key["ref_keep"].sum()), "n_motion": int(key["ref_motion"].sum()),
        "has_motion_field": int(key["has_motion_field"]),
        "frac_contested": f"{float(key['contested'].mean()):.4f}",
        "frac_clear": f"{float(key['clear_cut'].mean()):.4f}",
        "frac_near_boundary": f"{near:.4f}",
        "has_video": int((sd / f"{sd.name}.mat").exists()),
        "score_kind": key["score_kind"],
        "spatial_ok": int(key.get("spatial_ok", 0)) if "spatial_ok" in key else "",
        "key_path": str(kpath), "built_at": key["key_built_at"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_one(sd: Path, area: str, task: str, session: str, out_root: Path,
              model: dict, P: dict) -> tuple[dict, Path] | None:
    """Build (not yet augmented) key for one session and save it."""
    inp = load_session_inputs(sd)
    if inp is None:
        return None
    key = make_key(inp, model, P, area, task, session)
    kpath = tc.key_path(area, task, session, out_root)
    save_key(key, kpath)
    return key, kpath


def run_augment(manifest: Path, repo_root: Path) -> bool:
    """One headless MATLAB launch over every key in the manifest.  The inline
    script is joined onto one line by _run_matlab, so: semicolons only, no '%'."""
    import run_cnmfe
    repo = str(repo_root).replace("\\", "/")
    man = str(manifest).replace("\\", "/")
    script = (f"addpath(genpath('{repo}/ca_source_extraction')); "
              f"addpath('{repo}/training'); "
              f"acorn_training_key_augment('{man}');")
    return run_cnmfe._run_matlab(script, log, timeout_hours=2.0)


def update_curriculum(root: Path, rows: list[dict]) -> Path:
    path = root / "curriculum.csv"
    existing = {(r["area"], r["task"], r["session"]): r for r in tc.read_csv_rows(path)}
    for r in rows:
        existing[(r["area"], r["task"], r["session"])] = r
    ordered = sorted(existing.values(), key=lambda r: (r["area"], r["task"], r["session"]))
    tc.write_csv_rows(path, tc.CURRICULUM_COLUMNS, ordered)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="area\\task\\session (relative to DATA_PARENT) or absolute")
    ap.add_argument("--area", help="all pool sessions of this area (BLA or vCA1)")
    ap.add_argument("--all", action="store_true", help="all pool sessions of BLA and vCA1")
    ap.add_argument("--reviewer", help="only sessions whose provenance names this reviewer")
    ap.add_argument("--no-matlab", action="store_true", help="skip the spatial-overlap augment")
    ap.add_argument("--force", action="store_true", help="rebuild keys that already exist")
    ap.add_argument("--dry-run", action="store_true", help="list what would be built; write nothing")
    for k, v in PARAM_DEFAULTS.items():
        ap.add_argument("--" + k.lower().replace("_", "-"), type=float, default=v,
                        help=f"{k} (ASSUMED default {v})")
    args = ap.parse_args(argv)
    P = {k: float(getattr(args, k.lower())) for k in PARAM_DEFAULTS}

    if args.all:
        sessions = tc.pool_sessions(None, args.reviewer)
    elif args.area:
        sessions = tc.pool_sessions(args.area, args.reviewer)
    elif args.sessions:
        sessions = []
        for s in args.sessions:
            sd = tc.resolve_session(s)
            if not sd.is_dir():
                sys.exit(f"ERROR: session folder not found: {sd}")
            if not tc.is_pool_session(sd):
                sys.exit(f"ERROR: {sd} lacks one of {tc.POOL_FILES}; not a pool session")
            sessions.append(sd)
    else:
        sys.exit("ERROR: give sessions, --area <A>, or --all")
    if not sessions:
        print("No pool sessions matched.")
        return 0

    root = tc.training_root()
    print(f"{'DRY RUN: ' if args.dry_run else ''}{len(sessions)} session(s); keys under {root / 'keys'}")
    print("params (ASSUMED defaults unless overridden): " + ", ".join(f"{k}={v}" for k, v in P.items()))

    models = {}
    built, skipped, rows, manifest_lines = [], [], [], []
    for sd in sessions:
        area, task, session = tc.session_parts(sd)
        try:
            area = tc.check_area(area)
        except ValueError as e:
            print(f"  SKIP {session}: {e}")
            skipped.append((sd, str(e)))
            continue
        kpath = tc.key_path(area, task, session, root)
        if kpath.exists() and not args.force:
            print(f"  exists, skip (use --force): {area}/{task}/{session}")
            continue
        if args.dry_run:
            print(f"  would build {area}/{task}/{session} -> {kpath}")
            continue
        if area not in models:
            models[area] = load_model(tc.area_config(area).MODEL_DIR)
            m = models[area]
            if m["available"]:
                print(f"  [{area}] model {m['path']}: {m['model_type']}, n_features={m['n_features']:.0f}, "
                      f"feature_version={m['feature_version']:.0f}, n_sessions={m['n_sessions']:.0f}, "
                      f"reject_threshold={m['reject_threshold']}")
            else:
                print(f"  [{area}] model unavailable: {m['why']} -- keys get NaN scores")
        print(f"[{area}/{task}/{session}]")
        res = build_one(sd, area, task, session, root, models[area], P)
        if res is None:
            skipped.append((sd, "see SKIP line above"))
            continue
        key, kpath = res
        built.append((sd, key, kpath))
        manifest_lines.append(f"{kpath}\t{sd}")
        print(f"  n_review={int(key['n_review'])} keep={int(key['ref_keep'].sum())} "
              f"motion={int(key['ref_motion'].sum())} contested={int(key['contested'].sum())} "
              f"clear={int(key['clear_cut'].sum())} score_kind={key['score_kind']}")

    if args.dry_run:
        print("DRY RUN: nothing written.")
        return 0

    if built and not args.no_matlab:
        manifest = root / "keys" / "_augment_manifest.txt"
        manifest.write_text("\n".join(manifest_lines) + "\n", encoding="ascii")
        print(f"\nRunning the MATLAB spatial augment over {len(built)} key(s) ...")
        ok = run_augment(manifest, REPO_ROOT)
        if not ok:
            print("  WARN: MATLAB augment did not complete; keys keep spatial_ok=0 "
                  "(duplicate hints use trace correlation only).")
        for sd, key, kpath in built:
            merged = merge_spatial(key, kpath.parent / tc.SPATIAL_NAME)
            if not merged:
                key["spatial_ok"] = 0.0
            key["hint_code"], key["hint"] = compute_hints(key, key["params"])
            save_key(key, kpath)
    else:
        for sd, key, kpath in built:
            key["spatial_ok"] = 0.0
            save_key(key, kpath)

    for sd, key, kpath in built:
        rows.append(curriculum_row(key, sd, kpath))
    if rows:
        cpath = update_curriculum(root, rows)
        print(f"\ncurriculum: {cpath}")
    print(f"\nDone: built {len(built)} key(s), skipped {len(skipped)}.")
    for sd, why in skipped:
        print(f"  - {sd.name}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
