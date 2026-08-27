"""
Shared constants and helpers for the vCA1 35-column (v2) contract project.

Every script in this directory imports this module FIRST.  Importing it
injects config_vCA1 as `config` before any shared pipeline module is loaded,
which is what makes train_classifier / diagnose_model / curator / features /
manifest_util resolve vCA1 paths (the wrapper pattern of
train_classifier_vCA1.py:23-24 and watcher_vCA1.py:18-19).

The Step 4 tooling in ../step4_2026-08/ is the BLA deploy record and is never
modified.  We import its helpers and re-point their module globals with
configure().  That is delicate for two reasons:

  1. swap_v2.MANIFEST is computed at import time as `BK / "backup_manifest.json"`
     (swap_v2.py:44).  Overriding only BK would leave MANIFEST pointing at
     D:\\Julian_CNMFe\\BLA\\.feature_expansion\\_v1_backup\\backup_manifest.json --
     the sole index of BLA's rollback, which is still armed.  do_backup() writes
     that path (swap_v2.py:106).  So configure() must set MANIFEST explicitly.

  2. backfill_v2 does `from parity_check import EXT, PIN, DATA_ROOT`, which binds
     the VALUES into backfill_v2's own namespace.  Re-pointing parity_check alone
     would not move them.  So configure() sets the names on every module that
     holds a copy.

After re-pointing, assert_no_bla() walks the module globals and refuses if any
Path still mentions BLA.  Run it before anything that writes.
"""
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[2]          # ...\CNMF_E_LEGACY_BIANE_CLAUDE\agent
STEP4 = AGENT / "eval" / "step4_2026-08"
SP    = Path(__file__).resolve().parent              # this directory

for _p in (str(AGENT), str(STEP4), str(SP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- area injection: must happen before any shared pipeline import ----
import config_vCA1                                    # noqa: E402
sys.modules["config"] = config_vCA1

import numpy as np                                    # noqa: E402
import scipy.io as sio                                # noqa: E402

# ---- area constants ----
AREA        = config_vCA1.AREA                        # "vCA1"
DATA_ROOT   = config_vCA1.DATA_ROOT                   # D:\Julian_CNMFe\vCA1
MODEL_DIR   = config_vCA1.MODEL_DIR                   # agent\model\vCA1
JOBLIB_LIVE = MODEL_DIR / "classifier.joblib"

EXT       = DATA_ROOT / ".feature_expansion"          # dot prefix = scanner-invisible
PIN       = EXT / "_pinned"                           # baseline OOF, hiconf, v2b reference
BK        = EXT / "_v1_backup"                        # pre-swap v1 npz + joblib
ARMS      = EXT / "_arms"                             # bootstrap arm files (b0/b1), never in session dirs
REHEARSAL = EXT / "_rehearsal"                        # dry-run model + session copy

JOBLIB_BK_NAME = "classifier_v1_2026-08-24.joblib"    # the deployed joblib's date

V1     = "candidate_features.npz"
V2     = "candidate_features_v2.npz"
ARM_A  = "a"
ARM_B0 = "b0"
ARM_B1 = "b1"
ARM_B0P = "b0prod"   # b0 with production-style hiconf (sensitivity check)

# Fixed agent up-weight for this area (config_vCA1.py:25).  The trainer honours
# AGENT_WEIGHT_OVERRIDE in main() (train_classifier.py:792-794) but the eval
# harnesses do not -- threshold_sweep_v2.run_oof:104 hard-codes the BLA
# sqrt/4.0 recipe, which resolves to ~7x here.  run_oof_fixed() below is the
# corrected copy.
AGENT_WEIGHT = float(getattr(config_vCA1, "AGENT_WEIGHT_OVERRIDE", 5.0))
DEPLOYED_T   = 0.05                                   # current vCA1 reject_threshold

SEEDS      = [42, 1, 7, 13, 100, 2024, 31337, 9]      # == threshold_sweep_v2.SEEDS
THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
MIN_POS    = 5                                        # positives needed to be a CV test fold
HICONF     = 0.5                                      # features.HICONF_SCORE

PARKED = "AVG5x-TSeries-030426-pnb88-187um-35z-000"   # Step 0b, must never reappear


def rel(sd: Path) -> str:
    """Session dir -> 'task/session'."""
    return f"{sd.parent.name}/{sd.name}"


def key(rel_str: str) -> str:
    """'task/session' -> the extraction-mat basename 'task__session'."""
    return rel_str.replace("/", "__")


# ---------------------------------------------------------------------------
# Re-pointing the Step 4 modules
# ---------------------------------------------------------------------------

def configure():
    """Point every imported Step 4 module at vCA1 and prove no BLA path survives."""
    import parity_check as pc
    import backfill_v2 as bf
    import swap_v2 as sw

    for m in (pc, bf):
        m.EXT = EXT
        m.PIN = PIN
        m.DATA_ROOT = DATA_ROOT

    sw.DATA_ROOT      = DATA_ROOT
    sw.BK             = BK
    sw.MANIFEST       = BK / "backup_manifest.json"    # see module docstring, reason 1
    sw.JOBLIB_LIVE    = JOBLIB_LIVE
    sw.JOBLIB_BK_NAME = JOBLIB_BK_NAME
    sw.SP             = SP
    sw.SMOKE_REL      = None                           # vCA1 has no autopsy session

    assert_no_bla(pc, bf, sw)
    return pc, bf, sw


def assert_no_bla(*modules):
    """Refuse if any module global still points into the BLA tree or model dir."""
    bad = []
    for m in modules:
        for name, val in vars(m).items():
            if isinstance(val, Path) and "BLA" in str(val):
                bad.append(f"{m.__name__}.{name} = {val}")
    if bad:
        raise RuntimeError(
            "BLA paths survived configure() -- refusing to run.  A write through "
            "one of these would touch the BLA corpus or its armed rollback:\n  "
            + "\n  ".join(bad))
    return True


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cn_any(path: Path):
    """
    Cn from a .mat, tolerating MATLAB v7.3.  EVAL-ONLY -- features.load_cn
    (features.py:47-53) is scipy-only and is deliberately NOT modified: every
    agent-session Cn.mat is v5, so production never needs this.  12 of the 111
    vCA1 bootstrap sessions have v7.3 Cn.mat files.

    h5py stores MATLAB arrays transposed, hence the .T (bmlib.load_stack:52-59).
    Returns None if the file is missing or unreadable.
    """
    if path is None or not Path(path).exists():
        return None
    try:
        return sio.loadmat(str(path))["Cn"]
    except NotImplementedError:
        import h5py
        with h5py.File(str(path), "r") as f:
            return np.array(f["Cn"][()]).T
    except Exception:
        return None


def bootstrap_footprints(npz) -> np.ndarray:
    """
    Candidate footprints from bootstrap_candidates.npz -> (N, d1, d2).

    ORDER='C' is mandatory here.  bootstrap_preagent.py:467 stores
    `A_review.T` where A_review came from a C-order reshape
    (bootstrap_preagent.py:274), and the npz comment at :470 says so.  The
    review_neuron.mat extractions are the opposite convention and go through
    parity_check.footprints_from_sparse (order='F').  Mixing them silently
    transposes every footprint -- the exact class of bug that scrambled the
    bootstrap labels for the life of the pipeline.
    """
    from scipy import sparse
    A = sparse.csr_matrix(
        (npz["A_data"], npz["A_indices"], npz["A_indptr"]),
        shape=tuple(npz["A_shape"]))
    d1, d2 = int(npz["d1"][0]), int(npz["d2"][0])
    dense = np.asarray(A.todense(), dtype=float)       # (N, d1*d2)
    return dense.reshape(len(dense), d1, d2)           # order='C' (numpy default)


# ---------------------------------------------------------------------------
# CV harness
# ---------------------------------------------------------------------------

def run_oof_fixed(X_ag, y_ag, g_ag, X_bs, y_bs, w_bs, seed,
                  agent_weight=None):
    """
    threshold_sweep_v2.run_oof (:99-116) with ONE change: the per-fold agent
    weight is this area's fixed AGENT_WEIGHT_OVERRIDE instead of the BLA
    sqrt/4.0 recipe at :104 (which resolves to ~7x on vCA1 and is not what the
    trainer deploys).  Everything else -- StratifiedGroupKFold(5, shuffle,
    seed), bootstrap always in train, StandardScaler, dm.make_clf("xgb",
    compute_spw) -- is identical so the numbers stay comparable to Step 4's.
    """
    import warnings
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    import diagnose_model as dm

    w = AGENT_WEIGHT if agent_weight is None else float(agent_weight)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y_ag), np.nan)
    for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
        X_tr = np.vstack([X_ag[tr_idx], X_bs])
        y_tr = np.concatenate([y_ag[tr_idx], y_bs])
        w_tr = np.concatenate([np.ones(len(tr_idx)) * w, w_bs])
        sc = StandardScaler()
        X_trs = sc.fit_transform(X_tr)
        X_tes = sc.transform(X_ag[te_idx])
        clf = dm.make_clf("xgb", dm.compute_spw(y_tr, w_tr))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_trs, y_tr, sample_weight=w_tr)
            oof[te_idx] = clf.predict_proba(X_tes)[:, 1]
    return oof


def bootstrap_weights(sd: Path, n: int) -> np.ndarray:
    """Deployed bootstrap row weights: 0.4x bad-session, 0 for ambiguous+duplicate."""
    import train_classifier as tc
    w = np.ones(n, dtype=float)
    recovery = tc._get_bootstrap_recovery(sd)
    if recovery is not None and recovery < tc.BAD_SESSION_RECOVERY_THRESHOLD:
        w *= tc.BAD_SESSION_WEIGHT
    w[tc._get_bootstrap_ambiguous_mask(sd, n)] = 0.0
    return w


def classify_sessions():
    """
    vCA1 sessions by role.  Same rule as backfill_v2.classify_sessions:50-65 but
    against the vCA1 root (that function closes over parity_check.DATA_ROOT).
    Dot-prefixed task dirs are skipped, so .excluded / .feature_expansion are
    invisible exactly as they are to every production scanner.
    """
    import train_classifier as tc
    labeled_agent, bootstrap, pending, skipped = [], [], [], []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir() or not (sd / V1).exists():
                continue
            if (sd / "labels.mat").exists():
                (bootstrap if tc._is_bootstrap_session(sd) else labeled_agent).append(sd)
            elif (sd / "ROIs_candidates.jpg").exists():
                pending.append(sd)
            else:
                skipped.append(sd)
    assert not any(sd.name == PARKED for sd in labeled_agent + bootstrap + pending), \
        f"{PARKED} is back in the pool -- it was parked in Step 0b"
    return labeled_agent, bootstrap, pending, skipped


ANIMAL_RE = r"-(pnb\d+)-"        # vCA1 agent animals; bootstrap use numeric ids


def animal_of(sd: Path) -> str:
    """Animal id from a session name, or '?' if it does not match the pnb form."""
    import re
    m = re.search(ANIMAL_RE, sd.name)
    return m.group(1) if m else "?"


def load_pool(v2: bool = False, arm: str = ARM_A, require_width=None):
    """
    The labeled pool as records, used identically by the baseline and the gates
    so their b13 numbers are directly comparable.

    v2=False : the live 13-column candidate_features.npz.
    v2=True  : the parallel candidate_features_v2.npz; for BOOTSTRAP sessions,
               arm 'b0'/'b1' instead reads ARMS/{key}__{arm}.npz.  Agent and
               pending rows are identical across arms by construction.

    Per record: name, X, y (full-length, auto-rejected reconstructed as 0),
    reviewed mask, is_bootstrap, w (deployed bootstrap weighting), animal.

    Label reconstruction mirrors train_classifier.load_prospective_session:299-325
    (labels.mat covers only the review set unless sizes already match).
    """
    import train_classifier as tc
    labeled_agent, bootstrap, _, _ = classify_sessions()
    records = []
    for sd in labeled_agent + bootstrap:
        is_bs = sd in bootstrap
        if v2 and is_bs and arm in (ARM_B0, ARM_B1, ARM_B0P):
            f = ARMS / f"{key(rel(sd))}__{arm}.npz"
        else:
            f = sd / (V2 if v2 else V1)
        if not f.exists():
            raise FileNotFoundError(f"{rel(sd)}: missing {f.name}")
        npz = np.load(f, allow_pickle=True)
        X = npz["feature_matrix"].astype(float)
        if require_width is not None and X.shape[1] != require_width:
            raise ValueError(f"{rel(sd)}: width {X.shape[1]} != {require_width}")
        auto = npz["auto_rejected"].flatten().astype(int)
        reviewed = np.ones(len(X), dtype=bool)
        reviewed[auto] = False
        y_rev = sio.loadmat(str(sd / "labels.mat"))["labels"].flatten().astype(float)
        if len(y_rev) == len(X):
            y = y_rev
        else:
            assert len(y_rev) == int(reviewed.sum()), \
                f"{rel(sd)}: labels {len(y_rev)} != review set {int(reviewed.sum())}"
            y = np.zeros(len(X))
            y[reviewed] = y_rev
        records.append({
            "name": rel(sd), "session_dir": sd, "X": X,
            "y": (y == 1).astype(int), "reviewed": reviewed,
            "is_bootstrap": is_bs, "animal": animal_of(sd),
            "w": bootstrap_weights(sd, len(X)) if is_bs else np.ones(len(X)),
        })
    return records


def split_pool(records):
    """(cv_agent, train_only_agent, bootstrap) -- CV folds need >= MIN_POS positives."""
    ag = [r for r in records if not r["is_bootstrap"]]
    bs = [r for r in records if r["is_bootstrap"]]
    cv = [r for r in ag if r["y"].sum() >= MIN_POS]
    rest = [r for r in ag if r["y"].sum() < MIN_POS]
    return cv, rest, bs


def stack(recs, keys=("X", "y")):
    """Concatenate records into pooled arrays plus a group vector."""
    out = [np.vstack([r[k] for r in recs]) if recs[0][k].ndim > 1
           else np.concatenate([r[k] for r in recs]) for k in keys]
    groups = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(recs)])
    return (*out, groups)
