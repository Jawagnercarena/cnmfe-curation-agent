"""
c3_vca1_weight_sweep.py — G4 for vCA1: sweep FIXED agent-weight multipliers.

The production formula agent_weight = max(sqrt(n_bs_cand/n_agent_cand), 4.0)
resolves to 7.01x for vCA1 (50,370 bootstrap vs 1,026 agent candidates), so
sweep_weights.py's floor sweep {1..8} is inert — every floor below 7 yields
7.01. With clean bootstrap labels the question is whether agent weight should
drop BELOW the sqrt ratio, so this sweeps fixed multipliers instead.

5-fold StratifiedGroupKFold, test folds = agent sessions >=5 positives,
bootstrap always in train with the v2 ambiguous+duplicate mask applied,
bad-session 0.4x (now binds 0 sessions). 3 seeds. Reports OOF AUC and
false-AR% / garbage% at T=0.05 (deployed) and 0.07.

Run (background, ~minutes): valence python c3_vca1_weight_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AGENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_DIR))

import config_vCA1
sys.modules["config"] = config_vCA1

import train_classifier as tc

WEIGHTS = [1.0, 2.0, 3.5, 5.0, 7.01]
SEEDS = [42, 43, 44]
THRESHOLDS = [0.05, 0.07]
MIN_POS = 5


def load_pool():
    recs = []
    for task_dir in sorted(config_vCA1.DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for sd in sorted(task_dir.iterdir()):
            if not sd.is_dir():
                continue
            if not ((sd / "candidate_features.npz").exists()
                    and (sd / "labels.mat").exists()):
                continue
            out = tc.load_prospective_session(sd)
            if out is None:
                continue
            X, y = out
            is_bs = tc._is_bootstrap_session(sd)
            ambig = (tc._get_bootstrap_ambiguous_mask(sd, len(y))
                     if is_bs else np.zeros(len(y), bool))
            rec = tc._get_bootstrap_recovery(sd) if is_bs else None
            bad = rec is not None and rec < tc.BAD_SESSION_RECOVERY_THRESHOLD
            recs.append((sd.name, X, y, is_bs, ambig, bad))
    return recs


def main():
    recs = load_pool()
    n_ag = sum(1 for r in recs if not r[3])
    n_bs = sum(1 for r in recs if r[3])
    print(f"pool: {len(recs)} sessions ({n_ag} agent, {n_bs} bootstrap)")

    X_all = np.vstack([r[1] for r in recs])
    y_all = np.concatenate([r[2] for r in recs])
    groups = np.concatenate([[i] * len(r[2]) for i, r in enumerate(recs)])
    is_bs_row = np.concatenate([[r[3]] * len(r[2]) for r in recs])
    ambig_row = np.concatenate([r[4] for r in recs])
    bad_row = np.concatenate([[r[5]] * len(r[2]) for r in recs])

    # eligible agent test sessions
    ag_ok = {i for i, r in enumerate(recs)
             if not r[3] and r[2].sum() >= MIN_POS}
    print(f"agent sessions usable as test folds: {len(ag_ok)}")

    for w_agent in WEIGHTS:
        aucs, fars, gcs = {t: [] for t in THRESHOLDS}, {t: [] for t in THRESHOLDS}, {t: [] for t in THRESHOLDS}
        auc_list = []
        for seed in SEEDS:
            test_scores = np.full(len(y_all), np.nan)
            ag_sessions = sorted(ag_ok)
            ag_mask_all = np.isin(groups, ag_sessions)
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            ag_y = y_all[ag_mask_all]
            ag_groups = groups[ag_mask_all]
            for tr_idx, te_idx in skf.split(np.zeros(ag_mask_all.sum()), ag_y, ag_groups):
                te_sessions = np.unique(ag_groups[te_idx])
                train_mask = ~np.isin(groups, te_sessions)
                w = np.ones(len(y_all))
                w[is_bs_row & bad_row] = tc.BAD_SESSION_WEIGHT
                w[is_bs_row & ambig_row] = 0.0
                w[~is_bs_row] = w_agent
                w[~train_mask] = 0.0
                pos = (w > 0) & (y_all == 1)
                neg = (w > 0) & (y_all == 0)
                spw = w[neg].sum() / max(w[pos].sum(), 1e-9)
                clf = tc._make_clf("xgboost", scale_pos_weight=spw)
                m = train_mask & (w > 0)
                clf.fit(X_all[m], y_all[m], sample_weight=w[m])
                te_mask = np.isin(groups, te_sessions)
                test_scores[te_mask] = clf.predict_proba(X_all[te_mask])[:, 1]
            sc_mask = ~np.isnan(test_scores)
            auc = roc_auc_score(y_all[sc_mask], test_scores[sc_mask])
            auc_list.append(auc)
            for t in THRESHOLDS:
                rej = test_scores[sc_mask] < t
                yv = y_all[sc_mask]
                fars[t].append(100 * (rej & (yv == 1)).sum() / max((yv == 1).sum(), 1))
                gcs[t].append(100 * (rej & (yv == 0)).sum() / max((yv == 0).sum(), 1))
        msg = (f"agent_w={w_agent:5.2f}  AUC={np.mean(auc_list):.4f}"
               f"±{np.std(auc_list):.4f}")
        for t in THRESHOLDS:
            msg += (f" | T={t}: FAR {np.mean(fars[t]):.1f}%"
                    f" gc {np.mean(gcs[t]):.1f}%")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
