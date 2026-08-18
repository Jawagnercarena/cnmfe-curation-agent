"""
Counterfactual for the REAL fix: reclassify the 9 misclassified bootstrap-
provenance sessions (no stats-json, no agent artifacts, has neuron.mat) from
agent -> bootstrap. This drops them from 4.0x agent weight to 1.0x base bootstrap
weight AND moves them out of the OOF eval pool (bootstrap is training-only), while
KEEPING them as low-weight training signal.

Compare three worlds, 8-seed OOF:
  (1) CURRENT   : 9 as agent (4.0x, in eval pool)  -- the deployed misclassification
  (2) RECLASSIFY: 9 as bootstrap (1.0x, training-only)  -- the proposed fix
  (3) DROP      : 9 removed entirely  -- the earlier exclusion idea, for reference
"""
import sys, warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
from config import DATA_ROOT
import diagnose_model as dm

AGENT_ARTIFACTS = ["ROIs_candidates.jpg", "agent_run.log", "review_report.pdf",
                   "review_assigned.txt", "review_summary.txt"]
THRESHOLDS = [0.10, 0.11, 0.12, 0.13]
SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]

def is_misclassified(sd: Path) -> bool:
    return ((not (sd / "bootstrap_match_stats.json").exists())
            and (not any((sd / a).exists() for a in AGENT_ARTIFACTS))
            and (sd / "neuron.mat").exists())

records, _ = dm.load_all_records()
mis = [r for r in records if is_misclassified(r["session_dir"])]
print(f"misclassified sessions detected: {len(mis)}")

def run(mode):
    recs = []
    for r in records:
        r2 = dict(r)
        if is_misclassified(r["session_dir"]):
            if mode == "drop":
                continue
            if mode == "reclassify":
                r2["is_bootstrap"] = True
                r2["w"] = np.ones(len(r["y"]), dtype=float)   # base bootstrap weight, no json -> flat 1.0x
        recs.append(r2)
    ag = [r for r in recs if not r["is_bootstrap"] and r["y"].sum() >= 5]
    bs = [r for r in recs if r["is_bootstrap"]]
    X = np.vstack([r["X"] for r in ag]); y = np.concatenate([r["y"] for r in ag])
    g = np.concatenate([[i]*len(r["y"]) for i, r in enumerate(ag)])
    Xb = np.vstack([r["X"] for r in bs]); yb = np.concatenate([r["y"] for r in bs])
    wb = np.concatenate([r["w"] for r in bs])
    pos, neg = y == 1, y == 0
    far = {t: [] for t in THRESHOLDS}; gc = {t: [] for t in THRESHOLDS}; aucs = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, g):
            aw = float(max(np.sqrt(len(yb)/len(tr)), dm.MIN_AGENT_WEIGHT))
            Xtr = np.vstack([X[tr], Xb]); ytr = np.concatenate([y[tr], yb])
            wtr = np.concatenate([np.ones(len(tr))*aw, wb])
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
    print(f"\n=== {mode.upper():10} ===  agent={len(ag)} (reals {int(pos.sum())})  bootstrap={len(bs)}"
          f"  AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    for t in THRESHOLDS:
        f = np.array(far[t]); c = np.array(gc[t])
        mk = "  <-- 0.12 deployed" if abs(t-0.12) < 1e-9 else ""
        print(f"   T={t:.2f}   false-AR {f.mean():5.2f}% +/-{f.std():4.2f}   garbage {c.mean():6.2f}%{mk}")

run("current")
run("reclassify")
run("drop")
