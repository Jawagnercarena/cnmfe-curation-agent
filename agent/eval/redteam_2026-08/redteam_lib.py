"""Red-team evaluator, written FROM SCRATCH against the spec in the brief.
Deliberately does NOT import harness.py / diagnose_model / train_classifier —
every loading, weighting, masking, CV, and metric step is re-implemented here
from first principles so agreement with the claimed numbers is evidence, not
circularity.

Spec (from the brief + deployed-code reading, re-implemented independently):
- Pool: pinned pool_manifest.json (170 sessions). Agent sessions: full
  candidate X from candidate_features.npz; labels.mat covers the review set;
  auto-rejected rows get label 0. Reviewed-only pool = rows at review_indices.
- Bootstrap weights: 1.0; *0.4 if match recovery < 0.40; 0.0 for ambiguous
  rows (bootstrap_match_stats.json candidate_indices[n_matched:]).
- Eval pool: agent sessions with >=5 reals. Bootstrap = train-only.
- CV: StratifiedGroupKFold(5, shuffle, random_state=seed) on agent rows
  grouped by session; per-fold agent weight = max(sqrt(n_bs_rows/n_ag_tr), 4.0).
- Model: StandardScaler + XGBClassifier(300, lr .05, depth 4, subsample .8,
  colsample .8, spw = weighted neg/pos of the train fold, random_state=42).
- Seeds: [42, 1, 7, 13, 100, 2024, 31337, 9].
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE = Path(r"D:\Julian_CNMFe\BLA")
PIN = BASE / ".feature_expansion" / "_pinned"
SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]
T = 0.12
MIN_AGENT_WEIGHT = 4.0
BAD_RECOVERY = 0.40
BAD_WEIGHT = 0.4

V2_NAMES = ["ev_rate", "ev_snr", "ev_template_corr", "ev_asym",
            "ev_frac_plausible", "nb_corr_max", "nb_corr_any", "ring_contrast"]


def load_pool(verbose=True):
    """Load the pinned 170-session pool. Returns list of dicts."""
    manifest = json.loads((PIN / "pool_manifest.json").read_text())
    v2 = np.load(PIN / "step2_v2_features.npz", allow_pickle=True)
    v2b = np.load(PIN / "step2_v2b_features.npz", allow_pickle=True)

    records = []
    n_masked_total = 0
    for m in manifest:
        sd = BASE / m["rel"]
        npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
        X = npz["feature_matrix"].astype(float)
        n_cand = int(npz["n_candidates"][0])
        assert len(X) == n_cand, m["rel"]
        auto_rej = set(int(i) for i in npz["auto_rejected"].flatten())
        lab = sio.loadmat(str(sd / "labels.mat"))["labels"].flatten().astype(float)

        if len(lab) == n_cand:
            y_full = (lab == 1).astype(int)
        else:
            review_indices = [i for i in range(n_cand) if i not in auto_rej]
            assert len(lab) == len(review_indices), m["rel"]
            y_full = np.zeros(n_cand, dtype=int)
            for k, idx in enumerate(review_indices):
                y_full[idx] = 1 if lab[k] == 1 else 0

        rec = {"name": m["rel"], "is_bootstrap": m["is_bootstrap"],
               "X13": X, "y": y_full, "n_cand": n_cand,
               "auto_rej": np.array(sorted(auto_rej), dtype=int)}

        if m["is_bootstrap"]:
            w = np.ones(n_cand)
            stats_f = sd / "bootstrap_match_stats.json"
            if stats_f.exists():
                bs = json.loads(stats_f.read_text())
                n_curated = bs.get("n_curated", 0)
                recovery = bs["n_matched"] / n_curated if n_curated > 0 else 0.0
                if recovery < BAD_RECOVERY:
                    w *= BAD_WEIGHT
                for idx in bs.get("candidate_indices", [])[bs.get("n_matched", 0):]:
                    if 0 <= idx < n_cand:
                        w[idx] = 0.0
            n_masked_total += int((w == 0).sum())
            rec["w"] = w
        else:
            key = m["rel"].replace("/", "__")
            rec["ridx"] = v2[key + "__idx"]
            rec["Xv2"] = v2[key + "__X"]
            rec["Xv2b"] = v2b[key + "__X"]
        records.append(rec)

    if verbose:
        n_ag = sum(1 for r in records if not r["is_bootstrap"])
        print(f"loaded {len(records)} sessions ({n_ag} agent, "
              f"{len(records) - n_ag} bootstrap); "
              f"ambiguous-masked bootstrap rows: {n_masked_total}")
    return records


def variant_X(rec, variant, v2_cols=None, full=False):
    """Build the design matrix for one record. v2_cols: subset of 0..7 to keep
    (None = all 8). Variants: b13, rank26, v2_22, rankv2_35, v2b_22,
    rankv2b_35; suffix '_noflag' drops the present flag. full=True keeps the
    whole candidate set for agent records (b13/rank26 only)."""
    noflag = variant.endswith("_noflag")
    v = variant.replace("_noflag", "")
    X13 = rec["X13"]
    ranks = rankdata(X13, axis=0, method="average") / len(X13)
    X26 = np.hstack([X13, ranks])
    cols = list(range(8)) if v2_cols is None else list(v2_cols)

    if not rec["is_bootstrap"] and full:
        assert v in ("b13", "rank26")
        return {"b13": X13, "rank26": X26}[v]
    if not rec["is_bootstrap"]:
        ridx = rec["ridx"]
        parts = {"b13": [X13[ridx]], "rank26": [X26[ridx]],
                 "v2_22": [X13[ridx], rec["Xv2"][:, cols]],
                 "rankv2_35": [X26[ridx], rec["Xv2"][:, cols]],
                 "v2b_22": [X13[ridx], rec["Xv2b"][:, cols]],
                 "rankv2b_35": [X26[ridx], rec["Xv2b"][:, cols]]}[v]
        if v != "b13" and v != "rank26" and not noflag:
            parts.append(np.ones((len(ridx), 1)))
        return np.hstack(parts) if len(parts) > 1 else parts[0]
    else:
        z = np.zeros((len(X13), len(cols)))
        parts = {"b13": [X13], "rank26": [X26],
                 "v2_22": [X13, z], "rankv2_35": [X26, z],
                 "v2b_22": [X13, z], "rankv2b_35": [X26, z]}[v]
        if v != "b13" and v != "rank26" and not noflag:
            parts.append(np.zeros((len(X13), 1)))
        return np.hstack(parts) if len(parts) > 1 else parts[0]


def make_xgb(spw, random_state=42):
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, eval_metric="auc",
                         verbosity=0, random_state=random_state, n_jobs=-1)


def spw_of(y, w):
    pos = float(w[y == 1].sum())
    neg = float(w[y == 0].sum())
    return neg / pos if pos > 0 else 1.0


def run_oof(records, variant, reviewed_only=True, agent_only=False,
            seeds=SEEDS, v2_cols=None, xgb_seed=42, drop_eval_rows=None):
    """8-seed grouped OOF. Returns (oof_seeds, y, session_names, idx_in_sess).
    reviewed_only=False -> agent rows are the FULL candidate set (repro of the
    0.9099 baseline pool). drop_eval_rows: set of (session, full_idx) removed
    from the eval pool (C5)."""
    ag, bs = [], []
    for rec in records:
        if rec["is_bootstrap"]:
            if not agent_only:
                bs.append(rec)
            continue
        if reviewed_only:
            Xv = variant_X(rec, variant, v2_cols)
            y = rec["y"][rec["ridx"]]
            fidx = rec["ridx"]
        else:
            # full pool only defined for b13/rank26 (no v2 for auto-rejected)
            Xv = variant_X(rec, variant, v2_cols, full=True)
            y = rec["y"]
            fidx = np.arange(rec["n_cand"])
        if y.sum() >= 5:
            ag.append((rec["name"], Xv, y, fidx))

    X_ag = np.vstack([a[1] for a in ag])
    y_ag = np.concatenate([a[2] for a in ag])
    g_ag = np.concatenate([[i] * len(a[2]) for i, a in enumerate(ag)])
    names = np.concatenate([[a[0]] * len(a[2]) for a in ag])
    fidx_all = np.concatenate([a[3] for a in ag])

    if bs:
        X_bs = np.vstack([variant_X(r, variant, v2_cols) for r in bs])
        y_bs = np.concatenate([r["y"] for r in bs])
        w_bs = np.concatenate([r["w"] for r in bs])
    else:
        X_bs = np.empty((0, X_ag.shape[1]))
        y_bs = np.empty(0, dtype=int)
        w_bs = np.empty(0)

    keep_eval = np.ones(len(y_ag), dtype=bool)
    if drop_eval_rows:
        for i in range(len(y_ag)):
            if (names[i], int(fidx_all[i])) in drop_eval_rows:
                keep_eval[i] = False

    oof_seeds = np.full((len(seeds), len(y_ag)), np.nan)
    for si, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
            n_ag_tr = len(tr_idx)
            ag_w = max(np.sqrt(len(y_bs) / n_ag_tr), MIN_AGENT_WEIGHT) \
                if len(y_bs) else MIN_AGENT_WEIGHT
            X_tr = np.vstack([X_ag[tr_idx], X_bs])
            y_tr = np.concatenate([y_ag[tr_idx], y_bs])
            w_tr = np.concatenate([np.full(n_ag_tr, ag_w), w_bs])
            sc = StandardScaler()
            X_trs = sc.fit_transform(X_tr)
            X_tes = sc.transform(X_ag[te_idx])
            clf = make_xgb(spw_of(y_tr, w_tr), random_state=xgb_seed)
            clf.fit(X_trs, y_tr, sample_weight=w_tr)
            oof_seeds[si, te_idx] = clf.predict_proba(X_tes)[:, 1]

    return (oof_seeds[:, keep_eval], y_ag[keep_eval], names[keep_eval],
            fidx_all[keep_eval])


def holdout_eval(records, variant, test_names, v2_cols=None, xgb_seed=42):
    """Deterministic holdout: train on everything except the named agent
    sessions (bootstrap always in train), test pooled on them. Mirrors the
    step2 gate protocol; xgb_seed varies the only stochastic element."""
    Xtr, ytr, wtr = [], [], []
    Xte, yte = [], []
    n_ag_tr = n_bs_tr = 0
    for rec in records:
        Xv = variant_X(rec, variant, v2_cols)
        if rec["is_bootstrap"]:
            Xtr.append(Xv); ytr.append(rec["y"]); wtr.append(rec["w"])
            n_bs_tr += len(rec["y"])
        elif rec["name"] in test_names:
            Xte.append(Xv); yte.append(rec["y"][rec["ridx"]])
        else:
            y = rec["y"][rec["ridx"]]
            Xtr.append(Xv); ytr.append(y); wtr.append(np.full(len(y), np.nan))
            n_ag_tr += len(y)
    ag_w = max(np.sqrt(n_bs_tr / n_ag_tr), MIN_AGENT_WEIGHT) \
        if n_ag_tr and n_bs_tr else MIN_AGENT_WEIGHT
    w_tr = np.concatenate([np.where(np.isnan(w), ag_w, w) for w in wtr])
    X_tr, y_tr = np.vstack(Xtr), np.concatenate(ytr)
    X_te, y_te = np.vstack(Xte), np.concatenate(yte)
    sc = StandardScaler()
    clf = make_xgb(spw_of(y_tr, w_tr), random_state=xgb_seed)
    clf.fit(sc.fit_transform(X_tr), y_tr, sample_weight=w_tr)
    return clf.predict_proba(sc.transform(X_te))[:, 1], y_te


def summarize(oof_seeds, y, base_oof_seeds, base_y=None):
    """AUC + matched-operating-point metrics per seed vs a baseline OOF."""
    if base_y is None:
        base_y = y
    pos, neg = y == 1, y == 0
    bpos, bneg = base_y == 1, base_y == 0
    aucs, far_mj, gc_mf, band = [], [], [], []
    for s in range(oof_seeds.shape[0]):
        oof, bo = oof_seeds[s], base_oof_seeds[s]
        aucs.append(roc_auc_score(y, oof))
        gc_b = (bo[bneg] < T).mean()
        far_b = (bo[bpos] < T).mean()
        far_mj.append((oof[pos] < np.quantile(oof[neg], gc_b)).mean() * 100)
        gc_mf.append((oof[neg] < np.quantile(oof[pos], far_b)).mean() * 100)
        band.append(int(((oof[pos] >= 0.05) & (oof[pos] < T)).sum()))
    return {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "aucs": [float(a) for a in aucs],
            "far_mj": float(np.mean(far_mj)), "far_mj_sd": float(np.std(far_mj)),
            "gc_mf": float(np.mean(gc_mf)), "gc_mf_sd": float(np.std(gc_mf)),
            "band": float(np.mean(band))}


def fmt(tag, s):
    return (f"{tag:<22} AUC {s['auc_mean']:.4f} +/- {s['auc_sd']:.4f}  "
            f"far@mj {s['far_mj']:.2f}%  junk@mf {s['gc_mf']:.1f}%  "
            f"band {s['band']:.1f}")
