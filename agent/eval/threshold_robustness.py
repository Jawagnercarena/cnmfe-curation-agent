"""
Robustness check for the BLA threshold decision (0.14 -> 0.12).

Reproduces diagnose_model.py's Section-3 OOF sweep (real weights) but repeats it
over multiple CV seeds so we can see whether false-AR at each threshold is stable
or just single-seed noise. Reports mean +/- std across seeds.
"""
import sys, warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))

import diagnose_model as dm

records, agent_weight = dm.load_all_records()
ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
bs_recs = [r for r in records if r["is_bootstrap"]]

X_ag = np.vstack([r["X"] for r in ag_recs])
y_ag = np.concatenate([r["y"] for r in ag_recs])
g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag_recs)])

X_bs = np.vstack([r["X"] for r in bs_recs])
y_bs = np.concatenate([r["y"] for r in bs_recs])
w_bs = np.concatenate([r["w"] for r in bs_recs])

pos_mask = y_ag == 1
neg_mask = y_ag == 0
n_pos, n_neg = pos_mask.sum(), neg_mask.sum()

THRESHOLDS = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]

print(f"agent sessions={len(ag_recs)}  bootstrap={len(bs_recs)}  "
      f"agent_weight={agent_weight:.2f}")
print(f"OOF pool: {len(y_ag)} cands, {n_pos} real ({n_pos/len(y_ag):.1%}), {n_neg} garbage")
print(f"Seeds: {SEEDS}\n")

far_by_t = {t: [] for t in THRESHOLDS}
gc_by_t  = {t: [] for t in THRESHOLDS}
aucs = []

for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y_ag), np.nan)
    for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
        y_tr, X_tr, X_te = y_ag[tr_idx], X_ag[tr_idx], X_ag[te_idx]
        n_ag_tr, n_bs_tr = len(tr_idx), len(y_bs)
        ag_w = float(max(np.sqrt(n_bs_tr / n_ag_tr), dm.MIN_AGENT_WEIGHT))
        w_ag = np.ones(n_ag_tr) * ag_w
        X_trC = np.vstack([X_tr, X_bs])
        y_trC = np.concatenate([y_tr, y_bs])
        w_trC = np.concatenate([w_ag, w_bs])
        sc = StandardScaler()
        X_trCs = sc.fit_transform(X_trC)
        X_teCs = sc.transform(X_te)
        spw = dm.compute_spw(y_trC, w_trC)
        clf = dm.make_clf("xgb", spw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_trCs, y_trC, sample_weight=w_trC)
            oof[te_idx] = clf.predict_proba(X_teCs)[:, 1]
    aucs.append(roc_auc_score(y_ag, oof))
    for t in THRESHOLDS:
        far_by_t[t].append((oof[pos_mask] < t).sum() / n_pos * 100)
        gc_by_t[t].append((oof[neg_mask] < t).sum() / n_neg * 100)

print(f"OOF AUC across seeds: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  "
      f"(range {min(aucs):.3f}-{max(aucs):.3f})\n")
print(f"{'T':>5}  {'false-AR mean+/-sd':>22}  {'garbage mean+/-sd':>22}")
print(f"{'-'*5}  {'-'*22}  {'-'*22}")
for t in THRESHOLDS:
    far = np.array(far_by_t[t]); gc = np.array(gc_by_t[t])
    mark = "  <-- deployed 0.14" if abs(t-0.14)<1e-9 else ("  <-- proposed 0.12" if abs(t-0.12)<1e-9 else "")
    print(f"{t:5.2f}  {far.mean():6.2f}% +/- {far.std():4.2f}  ({far.min():.1f}-{far.max():.1f})"
          f"   {gc.mean():6.2f}% +/- {gc.std():4.2f}{mark}")
