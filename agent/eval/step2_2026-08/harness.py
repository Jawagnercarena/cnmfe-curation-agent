"""Shared 8-seed OOF harness for the zero-cost experiments. Mirrors
repin_baseline.py / agent/eval/threshold_robustness.py exactly; the only
degree of freedom is a per-record feature transform."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))
import diagnose_model as dm

SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]
T = 0.12


def run_variant(records, transform, seeds=SEEDS):
    """transform(X) -> X' applied per record (session). Returns per-seed OOF
    matrix over the agent eval pool plus summary metrics."""
    ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
    bs_recs = [r for r in records if r["is_bootstrap"]]

    X_ag = np.vstack([transform(r["X"]) for r in ag_recs])
    y_ag = np.concatenate([r["y"] for r in ag_recs])
    g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag_recs)])
    if bs_recs:
        X_bs = np.vstack([transform(r["X"]) for r in bs_recs])
        y_bs = np.concatenate([r["y"] for r in bs_recs])
        w_bs = np.concatenate([r["w"] for r in bs_recs])
    else:
        X_bs = np.empty((0, X_ag.shape[1]))
        y_bs = np.empty(0, dtype=int)
        w_bs = np.empty(0)

    oof_seeds = np.full((len(seeds), len(y_ag)), np.nan)
    for si, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
            n_ag_tr = len(tr_idx)
            ag_w = float(max(np.sqrt(len(y_bs) / n_ag_tr), dm.MIN_AGENT_WEIGHT))
            X_trC = np.vstack([X_ag[tr_idx], X_bs]) if len(y_bs) else X_ag[tr_idx]
            y_trC = np.concatenate([y_ag[tr_idx], y_bs])
            w_trC = np.concatenate([np.ones(n_ag_tr) * ag_w, w_bs])
            sc = StandardScaler()
            X_trCs = sc.fit_transform(X_trC)
            X_teCs = sc.transform(X_ag[te_idx])
            spw = dm.compute_spw(y_trC, w_trC)
            clf = dm.make_clf("xgb", spw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_trCs, y_trC, sample_weight=w_trC)
                oof_seeds[si, te_idx] = clf.predict_proba(X_teCs)[:, 1]
    return oof_seeds, y_ag


def summarize(oof_seeds, y, base_oof_seeds):
    """Per-seed paired metrics vs the baseline OOF (same seeds/splits):
    AUC, false-AR at the threshold matching baseline junk-caught@0.12, and
    junk-caught at the threshold matching baseline false-AR@0.12."""
    pos, neg = y == 1, y == 0
    aucs, far_mj, gc_mf, band = [], [], [], []
    for s in range(oof_seeds.shape[0]):
        oof, base = oof_seeds[s], base_oof_seeds[s]
        aucs.append(roc_auc_score(y, oof))
        gc_b = (base[neg] < T).mean()          # baseline junk-caught @0.12
        far_b = (base[pos] < T).mean()         # baseline false-AR  @0.12
        t_mj = np.quantile(oof[neg], gc_b)     # variant threshold at matched junk
        far_mj.append((oof[pos] < t_mj).mean() * 100)
        t_mf = np.quantile(oof[pos], far_b)    # variant threshold at matched FAR
        gc_mf.append((oof[neg] < t_mf).mean() * 100)
        band.append(int(((oof[pos] >= 0.05) & (oof[pos] < T)).sum()))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
        "far_at_matched_junk": float(np.mean(far_mj)),
        "far_at_matched_junk_sd": float(np.std(far_mj)),
        "gc_at_matched_far": float(np.mean(gc_mf)),
        "gc_at_matched_far_sd": float(np.std(gc_mf)),
        "band_005_012": float(np.mean(band)),
    }
