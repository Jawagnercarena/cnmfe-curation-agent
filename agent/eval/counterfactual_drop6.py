"""
Counterfactual: what does the BLA model look like with the 6 incoherent-label
early sessions removed from BOTH training and the OOF test pool?

If removing pure noise from training makes the model *better* on the remaining
good sessions (higher AUC, lower false-AR on THEM), that confirms they're
net-harmful, not just inflating the metric. 8-seed OOF, real weights.
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

SUSPECTS = ["bla2-695", "bla3-667", "bla4-751", "bla5-527", "bla3-665", "bla4-755"]
THRESHOLDS = [0.10, 0.11, 0.12, 0.13, 0.14]
SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]

records, aw = dm.load_all_records()
bs_recs = [r for r in records if r["is_bootstrap"]]
X_bs = np.vstack([r["X"] for r in bs_recs]); y_bs = np.concatenate([r["y"] for r in bs_recs])
w_bs = np.concatenate([r["w"] for r in bs_recs])

def sweep(ag_recs, label):
    X = np.vstack([r["X"] for r in ag_recs]); y = np.concatenate([r["y"] for r in ag_recs])
    g = np.concatenate([[i]*len(r["y"]) for i, r in enumerate(ag_recs)])
    pos, neg = y == 1, y == 0
    far = {t: [] for t in THRESHOLDS}; gc = {t: [] for t in THRESHOLDS}; aucs = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, g):
            ag_w = float(max(np.sqrt(len(y_bs)/len(tr)), dm.MIN_AGENT_WEIGHT))
            Xtr = np.vstack([X[tr], X_bs]); ytr = np.concatenate([y[tr], y_bs])
            wtr = np.concatenate([np.ones(len(tr))*ag_w, w_bs])
            sc = StandardScaler()
            clf = dm.make_clf("xgb", dm.compute_spw(ytr, wtr))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(sc.fit_transform(Xtr), ytr, sample_weight=wtr)
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, oof))
        for t in THRESHOLDS:
            far[t].append((oof[pos] < t).sum()/pos.sum()*100)
            gc[t].append((oof[neg] < t).sum()/neg.sum()*100)
    print(f"\n=== {label} ===")
    print(f"agent sessions={len(ag_recs)}  reals={int(pos.sum())}  "
          f"AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"{'T':>5}  {'false-AR':>18}  {'garbage':>10}")
    for t in THRESHOLDS:
        f = np.array(far[t]); c = np.array(gc[t])
        mk = "  <-- deployed" if abs(t-0.12) < 1e-9 else ""
        print(f"{t:5.2f}  {f.mean():5.2f}% +/-{f.std():4.2f}  {c.mean():6.2f}%{mk}")

ag_all = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
ag_clean = [r for r in ag_all if not any(s in r["name"] for s in SUSPECTS)]
sweep(ag_all,   f"FULL POOL  ({len(ag_all)} agent sessions)")
sweep(ag_clean, f"DROP 6 SUSPECTS  ({len(ag_clean)} agent sessions)")
