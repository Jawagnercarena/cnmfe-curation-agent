"""
rt_lib.py -- independent evaluator for the bootstrap-fix red team
(docs/BOOTSTRAP_REDTEAM_BRIEF.md).  Written from the spec, not from the harness.

Deliberately does NOT import train_classifier, diagnose_model, sweep_weights,
bmlib, vca1_common, threshold_sweep_v2, gate_vca1, backfill_v2, parity_check,
redteam_lib or bootstrap_preagent.  Every loading, labelling, weighting, CV,
metric and matching step is re-implemented here from first principles so that
agreement with the claimed numbers is evidence, not circularity.

Imported on purpose (declared): agent/features.py feature MATH
(compute_v2b_features, compute_ranks, assemble_v2_matrix, the four v1 row
functions).  That arithmetic is production-defined and was parity-checked
bit-for-bit by two earlier projects; what this red team tests is labels,
orientation, weighting, CV structure and decisions.  features.py LOADERS are
never used here (they are an audit target in attack #2).

Spec re-implemented (from the brief + code reading):
- Pool: every session dir under <DATA_PARENT>/<area>/<task>/ (dot-prefixed
  task dirs skipped) that has candidate_features.npz + labels.mat.
  Bootstrap = bootstrap_match_stats.json present, else (no agent-review
  artifact and a curated neuron.mat).
- Labels: labels.mat covers the review set; auto-rejected rows -> 0; if the
  lengths already match, labels apply to all rows.  y = (labels == 1).
- Bootstrap weights: 1.0; x0.4 if n_matched/n_curated < 0.40; 0.0 for
  ambiguous_candidate_indices + duplicate_candidate_indices (schema 2) or
  candidate_indices[n_matched:] (legacy).
- CV: agent sessions with >= 5 positives are the eval pool (FULL candidate
  set, auto-rejected rows included as label 0); StratifiedGroupKFold(5,
  shuffle, random_state=seed) grouped by session; bootstrap always in train,
  all rows present (weight-0 rows included in the scaler fit and the fit call,
  exactly as the harnesses do); per-fold agent weight = fixed override or
  max(sqrt(n_bootstrap_rows / n_agent_train_rows), 4.0); StandardScaler on
  the training block; XGBClassifier(300, lr .05, depth 4, subsample .8,
  colsample .8, spw = weighted neg/pos over w>0, random_state 42, n_jobs -1).
  Training rows are stacked agent-train-fold first, then bootstrap in the
  sorted-walk order -- row order matters for xgboost's subsampling.
- Seeds: [42, 1, 7, 13, 100, 2024, 31337, 9]; 3-seed claims used [42, 43, 44].
- Threshold rule (Step 5): largest T in [0.03, 0.10] with mean false-AR
  <= 0.85% and worst-seed false-AR <= 1.0%; gate junk_full >= 30%.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SP = Path(__file__).resolve().parent
AGENT = SP.parents[1]                      # ...\CNMF_E_LEGACY_BIANE_CLAUDE\agent
RESULTS = Path(os.environ["RT_RESULTS_DIR"]) if os.environ.get("RT_RESULTS_DIR") else SP / "results"
FIXTURES = SP / "fixtures"
SHEETS = SP / "contact_sheets"
PIN_FILE = SP / "pin_manifest.json"

DATA_PARENT = Path(r"D:\Julian_CNMFe")
AREAS = ("BLA", "vCA1", "DG_AL")
DATA_ROOT = {a: DATA_PARENT / a for a in AREAS}
MODEL_DIR = {a: AGENT / "model" / a for a in AREAS}
EXT = {a: DATA_ROOT[a] / ".feature_expansion" for a in AREAS}

SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]
SEEDS3 = [42, 43, 44]
THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
MIN_POS = 5
MIN_AGENT_WEIGHT = 4.0
BAD_RECOVERY = 0.40
BAD_WEIGHT = 0.4
MATCH_THR = 0.45                      # bootstrap Hungarian threshold
FINAL_SAVE_THR = 0.60                 # CNMFe_final_save.m / retro Hungarian threshold
AGENT_WEIGHT_OVERRIDE = {"vCA1": 5.0}  # config_vCA1.AGENT_WEIGHT_OVERRIDE; BLA/DG_AL: none
DEPLOYED_T = {"BLA": 0.04, "vCA1": 0.05, "DG_AL": 0.0}
EXPECTED_WIDTH = {"BLA": 35, "vCA1": 13, "DG_AL": 13}
V1_NAMES = ["area", "circularity", "eccentricity", "compactness", "max_weight",
            "weight_spread", "peak_snr", "transient_freq", "events_per_min",
            "baseline_stability", "skewness", "motion_correlation", "cn_correlation"]
CN_COL = 12

AGENT_ARTIFACTS = ("ROIs_candidates.jpg", "agent_run.log", "review_report.pdf",
                   "review_assigned.txt", "review_summary.txt")

# Cross-check against the machine-local config without depending on it for logic.
try:
    sys.path.insert(0, str(AGENT))
    import local_config as _lc            # noqa: E402
    assert Path(_lc.DATA_PARENT) == DATA_PARENT, \
        f"DATA_PARENT mismatch: {_lc.DATA_PARENT} vs {DATA_PARENT}"
except ImportError:
    pass

# Feature MATH imported on purpose (see module docstring).
import features as F                      # noqa: E402

_ANIMAL_KEYS = None


def animal_keys() -> set:
    global _ANIMAL_KEYS
    if _ANIMAL_KEYS is None:
        _ANIMAL_KEYS = set(json.loads(
            (AGENT / "config" / "animal_params.json").read_text()).keys())
    return _ANIMAL_KEYS


def log(msg: str = ""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def iter_sessions(area: str, root: Path | None = None):
    """Sorted walk of <root>/<task>/<session>, dot-prefixed task dirs skipped
    (the same visibility rule every production scanner uses)."""
    root = DATA_ROOT[area] if root is None else root
    for td in sorted(root.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if sd.is_dir():
                yield sd


def rel(sd: Path) -> str:
    return f"{sd.parent.name}/{sd.name}"


def key(rel_str: str) -> str:
    return rel_str.replace("/", "__")


def is_bootstrap(sd: Path) -> bool:
    if (sd / "bootstrap_match_stats.json").exists():
        return True
    if any((sd / a).exists() for a in AGENT_ARTIFACTS):
        return False
    return (sd / "neuron.mat").exists()


def labeled_sessions(area: str, root: Path | None = None) -> list[Path]:
    return [sd for sd in iter_sessions(area, root)
            if (sd / "candidate_features.npz").exists() and (sd / "labels.mat").exists()]


def pending_sessions(area: str) -> list[Path]:
    return [sd for sd in iter_sessions(area)
            if (sd / "candidate_features.npz").exists() and not (sd / "labels.mat").exists()]


# ---------------------------------------------------------------------------
# Names: animals and dates
# ---------------------------------------------------------------------------

def _tokens_after_tseries(name: str) -> list[str]:
    parts = name.split("-")
    try:
        i = parts.index("TSeries")
    except ValueError:
        return []
    return parts[i + 1:]


def date_of(name: str):
    """Recording date from the token after 'TSeries' (MMDDYY or MMDDYYYY)."""
    toks = _tokens_after_tseries(name)
    if not toks:
        return None
    d = toks[0]
    try:
        if len(d) == 6:
            return _dt.date(2000 + int(d[4:6]), int(d[0:2]), int(d[2:4]))
        if len(d) == 8:
            return _dt.date(int(d[4:8]), int(d[0:2]), int(d[2:4]))
    except ValueError:
        return None
    return None


def animal_of(area: str, name: str) -> str | None:
    """Animal id.  BLA: bla<n>; vCA1 agent: pnb<n>; DG_AL: DG6D/DG6E; vCA1
    bootstrap: the first post-date numeric token that is a key of
    animal_params.json (a following bare number is a depth, e.g. 921-880)."""
    for pat in (r"-(bla\d+)-", r"-(pnb\d+)-", r"-(DG6[A-Z])-"):
        m = re.search(pat, name)
        if m:
            return m.group(1)
    toks = _tokens_after_tseries(name)
    keys = animal_keys()
    hits = [t for t in toks[1:] if t.isdigit() and t in keys]
    if len(hits) >= 1:
        return hits[0]
    return None


# ---------------------------------------------------------------------------
# Labels, weights, records
# ---------------------------------------------------------------------------

def load_labels(sd: Path) -> np.ndarray:
    return sio.loadmat(str(sd / "labels.mat"))["labels"].flatten().astype(float)


def load_json(sd: Path) -> dict | None:
    f = sd / "bootstrap_match_stats.json"
    return json.loads(f.read_text()) if f.exists() else None


def reconstruct_labels(y_rev: np.ndarray, n: int, auto_rejected: np.ndarray):
    """Full-length label vector + reviewed mask (spec: review set = all rows
    minus auto-rejected, in candidate order)."""
    reviewed = np.ones(n, dtype=bool)
    reviewed[np.asarray(auto_rejected, dtype=int)] = False
    if len(y_rev) == n:
        return y_rev.copy(), reviewed
    if len(y_rev) != int(reviewed.sum()):
        raise ValueError(f"labels {len(y_rev)} != review set {int(reviewed.sum())} (n={n})")
    y = np.zeros(n, dtype=float)
    y[reviewed] = y_rev
    return y, reviewed


def bootstrap_weights(sd: Path, n: int) -> np.ndarray:
    w = np.ones(n, dtype=float)
    js = load_json(sd)
    if js is None:
        return w
    n_cur = js.get("n_curated", 0)
    recovery = js["n_matched"] / n_cur if n_cur > 0 else 0.0
    if recovery < BAD_RECOVERY:
        w *= BAD_WEIGHT
    if js.get("schema_version", 1) >= 2:
        excl = list(js.get("ambiguous_candidate_indices", [])) + \
            list(js.get("duplicate_candidate_indices", []))
    else:
        excl = list(js.get("candidate_indices", []))[js.get("n_matched", 0):]
    for i in excl:
        if 0 <= i < n:
            w[i] = 0.0
    return w


def load_record(sd: Path, area: str, feature_file: Path | None = None,
                width: int | None = None) -> dict:
    """One session -> record.  feature_file overrides candidate_features.npz
    (parallel v2 files, arm files) -- labels/auto_rejected still come from the
    session dir's live npz unless the override carries them."""
    live = np.load(sd / "candidate_features.npz", allow_pickle=True)
    src = live if feature_file is None else np.load(feature_file, allow_pickle=True)
    X = np.asarray(src["feature_matrix"], dtype=float)
    auto = np.asarray(src["auto_rejected"], dtype=int).ravel() if "auto_rejected" in src.files \
        else np.asarray(live["auto_rejected"], dtype=int).ravel()
    n = len(X)
    if width is not None and X.shape[1] != width:
        raise ValueError(f"{rel(sd)}: width {X.shape[1]} != {width}")
    y_rev = load_labels(sd)
    y, reviewed = reconstruct_labels(y_rev, n, auto)
    bs = is_bootstrap(sd)
    return {
        "name": rel(sd), "session_dir": sd, "area": area,
        "X": X, "y": (y == 1).astype(int), "reviewed": reviewed,
        "auto_rejected": auto, "n_cand": n, "is_bootstrap": bs,
        "w": bootstrap_weights(sd, n) if bs else np.ones(n),
        "animal": animal_of(area, sd.name), "date": date_of(sd.name),
    }


def load_pool(area: str, width: int | None = None, root: Path | None = None,
              feature_file_of=None, extra_dirs: list[Path] | None = None,
              exclude: set[str] | None = None) -> list[dict]:
    """The labeled pool in sorted-walk order (the order the harnesses use).
    feature_file_of(rec_session_dir) -> Path | None lets attacks substitute
    parallel/arm files.  extra_dirs: explicit session dirs to append (e.g. the
    parked .excluded session when reconstructing the 08-24 pool)."""
    recs = []
    dirs = labeled_sessions(area, root)
    if extra_dirs:
        dirs = sorted(dirs + list(extra_dirs), key=lambda p: (p.parent.name, p.name))
    for sd in dirs:
        if exclude and (rel(sd) in exclude or sd.name in exclude):
            continue
        ff = feature_file_of(sd) if feature_file_of else None
        recs.append(load_record(sd, area, ff, width))
    return recs


def split_pool(recs: list[dict], min_pos: int = MIN_POS):
    ag = [r for r in recs if not r["is_bootstrap"]]
    bs = [r for r in recs if r["is_bootstrap"]]
    cv = [r for r in ag if int(r["y"].sum()) >= min_pos]
    rest = [r for r in ag if int(r["y"].sum()) < min_pos]
    return cv, rest, bs


def stack_agent(cv: list[dict], cols: slice | None = None):
    X = np.vstack([r["X"] for r in cv])
    if cols is not None:
        X = X[:, cols]
    y = np.concatenate([r["y"] for r in cv])
    g = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(cv)])
    rev = np.concatenate([r["reviewed"] for r in cv])
    animals = np.concatenate([[r["animal"] or "?"] * len(r["y"]) for r in cv])
    names = np.array([r["name"] for r in cv])
    return X, y, g, rev, animals, names


def stack_bootstrap(bs: list[dict], cols: slice | None = None, width: int | None = None):
    if not bs:
        w = width if width is not None else 0
        if cols is not None and w:
            w = len(range(*cols.indices(w)))
        return np.zeros((0, w)), np.zeros(0, int), np.zeros(0)
    X = np.vstack([r["X"] for r in bs])
    if cols is not None:
        X = X[:, cols]
    y = np.concatenate([r["y"] for r in bs])
    w = np.concatenate([r["w"] for r in bs])
    return X, y, w


# ---------------------------------------------------------------------------
# Model factory and CV harness
# ---------------------------------------------------------------------------

def make_xgb(spw: float, random_state: int = 42, n_jobs: int = -1) -> XGBClassifier:
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, eval_metric="auc",
                         verbosity=0, random_state=random_state, n_jobs=n_jobs)


def spw_of(y: np.ndarray, w: np.ndarray) -> float:
    m = w > 0
    pos = float(w[(y == 1) & m].sum())
    neg = float(w[(y == 0) & m].sum())
    return neg / pos if pos > 0 else 1.0


def fold_agent_weight(n_bs_rows: int, n_ag_tr: int, override: float | None) -> float:
    if override is not None:
        return float(override)
    if n_bs_rows > 0 and n_ag_tr > 0:
        return float(max(np.sqrt(n_bs_rows / n_ag_tr), MIN_AGENT_WEIGHT))
    return MIN_AGENT_WEIGHT


def fit_predict(X_tr, y_tr, w_tr, X_te, use_scaler=True, xgb_seed=42,
                drop_zero_weight=False, n_jobs=-1):
    """One fit.  drop_zero_weight=False keeps weight-0 rows in the scaler and
    the fit call (the harness convention); True removes them (gate_vca1.loao)."""
    if drop_zero_weight:
        keep = w_tr > 0
        X_tr, y_tr, w_tr = X_tr[keep], y_tr[keep], w_tr[keep]
    if use_scaler:
        sc = StandardScaler()
        X_trs = sc.fit_transform(X_tr)
        X_tes = sc.transform(X_te)
    else:
        X_trs, X_tes = X_tr, X_te
    clf = make_xgb(spw_of(y_tr, w_tr), random_state=xgb_seed, n_jobs=n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_trs, y_tr, sample_weight=w_tr)
        return clf.predict_proba(X_tes)[:, 1]


def run_oof(X_ag, y_ag, g_ag, X_bs, y_bs, w_bs, seed, agent_weight=None,
            use_scaler=True, n_splits=5, xgb_seed=42, bs_train_mask=None,
            n_jobs=-1):
    """Grouped OOF on the agent pool; bootstrap always in train.
    bs_train_mask: optional (n_folds-invariant) callable(te_idx) -> bool mask
    over bootstrap rows to include for that fold (animal-level leakage drop)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y_ag), np.nan)
    for tr, te in cv.split(X_ag, y_ag, g_ag):
        if bs_train_mask is not None:
            m = bs_train_mask(te)
            Xb, yb, wb = X_bs[m], y_bs[m], w_bs[m]
        else:
            Xb, yb, wb = X_bs, y_bs, w_bs
        agw = fold_agent_weight(len(yb), len(tr), agent_weight)
        X_tr = np.vstack([X_ag[tr], Xb]) if len(yb) else X_ag[tr]
        y_tr = np.concatenate([y_ag[tr], yb])
        w_tr = np.concatenate([np.full(len(tr), agw), wb])
        oof[te] = fit_predict(X_tr, y_tr, w_tr, X_ag[te], use_scaler, xgb_seed,
                              n_jobs=n_jobs)
    return oof


def run_oof_seeds(X_ag, y_ag, g_ag, X_bs, y_bs, w_bs, seeds=SEEDS, **kw):
    return np.array([run_oof(X_ag, y_ag, g_ag, X_bs, y_bs, w_bs, s, **kw) for s in seeds])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc(y, s, mask=None):
    if mask is not None:
        y, s = y[mask], s[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def far_junk(oof, y, T, reviewed=None):
    """(false-AR %, junk-caught % on all junk, junk-caught % on reviewed junk)."""
    pos, neg = y == 1, y == 0
    far = float((oof[pos] < T).sum() / max(pos.sum(), 1) * 100)
    jf = float((oof[neg] < T).sum() / max(neg.sum(), 1) * 100)
    if reviewed is None:
        return far, jf, float("nan")
    nr = neg & reviewed
    jr = float((oof[nr] < T).sum() / max(nr.sum(), 1) * 100)
    return far, jf, jr


def threshold_table(oof_seeds, y, reviewed=None, thresholds=THRESHOLDS):
    tab = {}
    for t in thresholds:
        rows = [far_junk(o, y, t, reviewed) for o in oof_seeds]
        tab[f"{t:.3f}"] = {
            "far": [r[0] for r in rows], "junk_full": [r[1] for r in rows],
            "junk_reviewed": [r[2] for r in rows]}
    return tab


def rule_T(table: dict, lo=0.03, hi=0.10, mean_max=0.85, worst_max=1.0, junk_min=30.0):
    """Step-5 rule.  Returns (chosen_T or None, gate_pass, detail)."""
    chosen = None
    for k in sorted(table, key=float):
        t = float(k)
        if t < lo - 1e-9 or t > hi + 1e-9:
            continue
        far = table[k]["far"]
        if np.mean(far) <= mean_max and max(far) <= worst_max:
            chosen = t
    if chosen is None:
        return None, False, {}
    row = next(v for k, v in table.items() if abs(float(k) - chosen) < 1e-9)
    gate = bool(np.mean(row["far"]) <= 1.0 and np.mean(row["junk_full"]) >= junk_min)
    return chosen, gate, {"far_mean": float(np.mean(row["far"])),
                          "far_max": float(max(row["far"])),
                          "junk_full": float(np.mean(row["junk_full"])),
                          "junk_reviewed": float(np.nanmean(row["junk_reviewed"]))}


def matched_operating_point(oof_ref, oof_new, y, t_ref, grid=None):
    """PER SEED: (a) false-AR of `new` at the threshold matching `ref`'s
    junk-caught at t_ref; (b) junk-caught of `new` at the threshold matching
    `ref`'s false-AR at t_ref.  Returns dict of per-seed lists + means."""
    grid = np.arange(0.001, 0.5005, 0.001) if grid is None else grid
    pos, neg = y == 1, y == 0
    out = {"far_ref": [], "junk_ref": [], "far_new_at_matched_junk": [],
           "junk_new_at_matched_far": [], "t_new_junk": [], "t_new_far": []}
    for r, n in zip(np.atleast_2d(oof_ref), np.atleast_2d(oof_new)):
        far_r = float((r[pos] < t_ref).sum() / pos.sum() * 100)
        junk_r = float((r[neg] < t_ref).sum() / neg.sum() * 100)
        out["far_ref"].append(far_r)
        out["junk_ref"].append(junk_r)
        junk_n = np.array([(n[neg] < t).sum() / neg.sum() * 100 for t in grid])
        far_n = np.array([(n[pos] < t).sum() / pos.sum() * 100 for t in grid])
        i = np.argmax(junk_n >= junk_r) if (junk_n >= junk_r).any() else None
        out["far_new_at_matched_junk"].append(float(far_n[i]) if i is not None else float("nan"))
        out["t_new_junk"].append(float(grid[i]) if i is not None else float("nan"))
        ok = np.where(far_n <= far_r)[0]
        j = ok[-1] if len(ok) else None
        out["junk_new_at_matched_far"].append(float(junk_n[j]) if j is not None else float("nan"))
        out["t_new_far"].append(float(grid[j]) if j is not None else float("nan"))
    for k in list(out):
        out[k + "_mean"] = float(np.nanmean(out[k]))
    return out


def paired_delta(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    return {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "n_positive": int((d > 0).sum()), "n": int(len(d)),
            "all_positive": bool((d > 0).all())}


# ---------------------------------------------------------------------------
# Footprint / trace loaders (all orientations explicit)
# ---------------------------------------------------------------------------

def load_mat_var(path: Path, var: str, h5_transpose="matlab"):
    """scipy first; MATLAB v7.3 via h5py with the dims reversed back to
    MATLAB order (h5py stores MATLAB arrays with reversed axes)."""
    try:
        return sio.loadmat(str(path))[var]
    except NotImplementedError:
        import h5py
        with h5py.File(str(path), "r") as f:
            a = np.array(f[var][()])
        return a.transpose(tuple(range(a.ndim))[::-1]) if h5_transpose else a


def load_stack(path: Path) -> np.ndarray:
    """spatial_footprints.mat -> (N, H, W) image stack."""
    return np.asarray(load_mat_var(path, "spatial_footprints"), dtype=float)


def load_curated_stack(sd: Path) -> np.ndarray:
    return load_stack(sd / "spatial_footprints.mat")


def load_cn(path: Path):
    if not Path(path).exists():
        return None
    try:
        return np.asarray(load_mat_var(Path(path), "Cn"), dtype=float)
    except Exception:
        return None


def fcols_to_images(A: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """(pixels x N) MATLAB-linearized columns -> (N, d1, d2).  order='F'."""
    A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    return np.asarray([A[:, k].reshape((d1, d2), order="F") for k in range(A.shape[1])])


def ccols_to_images(A: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """The WRONG reshape for MATLAB columns (numpy C-order) -- the control."""
    A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    return np.asarray([A[:, k].reshape((d1, d2), order="C") for k in range(A.shape[1])])


def stack_to_rows(stack: np.ndarray, order: str) -> np.ndarray:
    """(N, H, W) -> (N, pixels) flattened in the given order."""
    n = stack.shape[0]
    return np.asarray([stack[k].flatten(order=order) for k in range(n)])


def load_A_txt(sd: Path) -> np.ndarray:
    return np.loadtxt(str(sd / "A.txt"))


def load_bootstrap_candidates(sd: Path) -> dict:
    """bootstrap_candidates.npz: CSR footprints (rows = candidates, pixel
    order C), C_raw, sim_matrix under the fixed metric, d1, d2."""
    from scipy import sparse
    z = np.load(sd / "bootstrap_candidates.npz", allow_pickle=True)
    A = sparse.csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]),
                          shape=tuple(z["A_shape"]))
    d1, d2 = int(z["d1"][0]), int(z["d2"][0])
    return {"A_rows_C": A, "d1": d1, "d2": d2,
            "C_raw": np.asarray(z["C_raw"], dtype=float),
            "sim_matrix": np.asarray(z["sim_matrix"], dtype=float)}


def bootstrap_images(bc: dict) -> np.ndarray:
    dense = np.asarray(bc["A_rows_C"].todense(), dtype=float)
    return dense.reshape(len(dense), bc["d1"], bc["d2"])          # order='C'


def load_extraction(area: str, rel_str: str) -> dict | None:
    """.feature_expansion/<task>__<session>.mat: A (pixels x N, F-order
    columns), C_raw (N x T), Cn (d1 x d2), d1, d2."""
    p = EXT[area] / f"{key(rel_str)}.mat"
    if not p.exists():
        return None
    m = sio.loadmat(str(p))
    d1, d2 = int(np.asarray(m["d1"]).flat[0]), int(np.asarray(m["d2"]).flat[0])
    A = m["A"]
    A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)      # saved sparse by extract_*.m
    return {"A": np.asarray(A, dtype=float), "C_raw": np.asarray(m["C_raw"], dtype=float),
            "Cn": np.asarray(m["Cn"], dtype=float) if "Cn" in m else None,
            "d1": d1, "d2": d2, "path": p}


# ---------------------------------------------------------------------------
# Matching / geometry
# ---------------------------------------------------------------------------

def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return (a / na[:, None]) @ (b / nb[:, None]).T


def hungarian_pairs(sim: np.ndarray, thr: float):
    """1:1 assignment maximising similarity; returns list of (row, col, sim)
    above thr, sorted best-first."""
    if sim.size == 0:
        return []
    r, c = linear_sum_assignment(-sim)
    pairs = [(int(i), int(j), float(sim[i, j])) for i, j in zip(r, c) if sim[i, j] > thr]
    return sorted(pairs, key=lambda p: -p[2])


def greedy_per_final(sim_review_by_final: np.ndarray, thr: float) -> np.ndarray:
    """CNMFe_final_save.m rule: each FINAL neuron (column) claims its best
    still-unclaimed review candidate (row) if corr > thr.  Returns labels."""
    n_rev, n_fin = sim_review_by_final.shape
    labels = np.zeros(n_rev, dtype=int)
    used = np.zeros(n_rev, dtype=bool)
    for jf in range(n_fin):
        col = sim_review_by_final[:, jf].copy()
        col[used] = -np.inf
        bi = int(np.argmax(col))
        if col[bi] > thr:
            labels[bi] = 1
            used[bi] = True
    return labels


def centroids(stack: np.ndarray) -> np.ndarray:
    n, d1, d2 = stack.shape
    ys, xs = np.mgrid[0:d1, 0:d2]
    out = np.full((n, 2), np.nan)
    for k in range(n):
        w = stack[k].sum()
        if w > 0:
            out[k] = ((ys * stack[k]).sum() / w, (xs * stack[k]).sum() / w)
    return out


def binary_mask(img: np.ndarray, frac: float = 0.2) -> np.ndarray:
    return img > frac * img.max() if img.max() > 0 else img > 0


def iou(m1: np.ndarray, m2: np.ndarray) -> float:
    u = (m1 | m2).sum()
    return float((m1 & m2).sum() / u) if u else 0.0


# ---------------------------------------------------------------------------
# Pin (what the corpus looked like when a number was computed)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_pin(areas=AREAS) -> dict:
    pin = {"areas": {}}
    for area in areas:
        rows = []
        for sd in iter_sessions(area):
            npz = sd / "candidate_features.npz"
            lab = sd / "labels.mat"
            js = sd / "bootstrap_match_stats.json"
            if not npz.exists():
                continue
            row = {"rel": rel(sd), "has_labels": lab.exists(),
                   "is_bootstrap": is_bootstrap(sd) if lab.exists() else None,
                   "npz_size": npz.stat().st_size,
                   "npz_mtime": npz.stat().st_mtime}
            if lab.exists():
                row.update({"labels_mtime": lab.stat().st_mtime,
                            "labels_size": lab.stat().st_size,
                            "labels_sha256": _sha256(lab)})
            if js.exists():
                row["json_sha256"] = _sha256(js)
            rows.append(row)
        jl = MODEL_DIR[area] / "classifier.joblib"
        pin["areas"][area] = {
            "sessions": rows,
            "n_labeled": sum(1 for r in rows if r["has_labels"]),
            "n_agent": sum(1 for r in rows if r["has_labels"] and not r["is_bootstrap"]),
            "n_bootstrap": sum(1 for r in rows if r["has_labels"] and r["is_bootstrap"]),
            "n_pending": sum(1 for r in rows if not r["has_labels"]),
            "joblib": {"path": str(jl), "md5": _md5(jl) if jl.exists() else None,
                       "size": jl.stat().st_size if jl.exists() else None,
                       "mtime": jl.stat().st_mtime if jl.exists() else None},
        }
    pin["pin_hash"] = hashlib.sha256(
        json.dumps(pin["areas"], sort_keys=True, default=str).encode()).hexdigest()
    return pin


def load_pin() -> dict:
    return json.loads(PIN_FILE.read_text())


def pin_hash() -> str:
    return load_pin()["pin_hash"]


def diff_pin(old: dict, new: dict) -> dict:
    out = {}
    for area in old["areas"]:
        o = {r["rel"]: r for r in old["areas"][area]["sessions"]}
        n = {r["rel"]: r for r in new["areas"][area]["sessions"]}
        changed = [k for k in o.keys() & n.keys() if o[k] != n[k]]
        out[area] = {"new": sorted(n.keys() - o.keys()), "gone": sorted(o.keys() - n.keys()),
                     "changed": sorted(changed),
                     "joblib_changed": old["areas"][area]["joblib"]["md5"] != new["areas"][area]["joblib"]["md5"]}
    return out


def assert_pinned(areas=AREAS):
    """Refuse to run if the corpus moved since pin_manifest.json was written."""
    if not PIN_FILE.exists():
        raise RuntimeError("pin_manifest.json missing -- run rt_pin.py first")
    old = load_pin()
    new = compute_pin(areas)
    d = diff_pin(old, new)
    moved = {a: v for a, v in d.items() if v["new"] or v["gone"] or v["changed"] or v["joblib_changed"]}
    if moved:
        raise RuntimeError(f"corpus moved since the pin: {json.dumps(moved, indent=1)[:2000]}")
    return old["pin_hash"]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.t0 = time.time()

    def s(self) -> float:
        return round(time.time() - self.t0, 1)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    raise TypeError(f"not jsonable: {type(o)}")


def write_result(name: str, payload: dict, script: str | Path, timer: Timer | None = None):
    RESULTS.mkdir(exist_ok=True)
    payload = dict(payload)
    payload.setdefault("attack", name)
    payload["pin_hash"] = pin_hash() if PIN_FILE.exists() else None
    payload["script"] = Path(script).name
    payload["script_sha256"] = _sha256(Path(script))
    payload["written_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    if timer is not None:
        payload["runtime_s"] = timer.s()
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1, default=_jsonable))
    log(f"wrote {out.relative_to(SP)}")
    return out


STANDING_RESULTS = SP / "results"


def read_result(name: str) -> dict:
    """Standing results (never the refuter scratch dir)."""
    return json.loads((STANDING_RESULTS / f"{name}.json").read_text())


def summarize(vals) -> dict:
    v = np.asarray(vals, float)
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "n": int(len(v))}
