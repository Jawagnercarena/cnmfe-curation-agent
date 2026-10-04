"""
c3_assumption_checks.py — pre-deploy audit of BLA training assumptions.

  --mode weights   For each fixed agent weight, run the OOF (agent test folds,
                   bootstrap train-only, current masking) at 3 seeds, sweep
                   T=0.02..0.12 and apply the Step-5 rule (largest T with mean
                   false-AR <= 0.85% and worst seed <= 1.0%). Reports the
                   rule-chosen T and junk-caught AT THAT T per weight, i.e.
                   junk at matched false-AR — the fair weight comparison.
  --mode dups      At the deployed weight, compare three treatments of the
                   duplicate rows (same-cell re-detections above threshold to a
                   matched curated neuron): masked (weight 0, current),
                   label 0 (old behavior), label 1 (treat as the real cell).

Both modes: 5-fold StratifiedGroupKFold over agent sessions with >= 5
positives, same model factory/weights as the trainer. Read-only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AGENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_DIR))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["weights", "dups"], required=True)
ap.add_argument("--area", choices=["BLA", "vCA1"], default="BLA")
ap.add_argument("--seeds", type=int, default=3)
ARGS = ap.parse_args()

if ARGS.area == "vCA1":
    import config_vCA1 as cfg
    sys.modules["config"] = cfg
else:
    import config as cfg

import train_classifier as tc

SEEDS = [42 + i for i in range(ARGS.seeds)]
THRESHOLDS = [round(t, 2) for t in np.arange(0.02, 0.1201, 0.01)]
MIN_POS = 5
DEPLOYED_W = 5.0 if ARGS.area == "vCA1" else 4.0


def load_pool():
    recs = []
    for task_dir in sorted(cfg.DATA_ROOT.iterdir()):
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
            n = len(y)
            ambig = np.zeros(n, bool)
            dup = np.zeros(n, bool)
            if is_bs:
                jp = sd / "bootstrap_match_stats.json"
                if jp.exists():
                    s = json.load(open(jp))
                    for i in s.get("ambiguous_candidate_indices", []):
                        if 0 <= i < n:
                            ambig[i] = True
                    for i in s.get("duplicate_candidate_indices", []):
                        if 0 <= i < n:
                            dup[i] = True
            recs.append(dict(name=sd.name, X=X, y=y.astype(int), is_bs=is_bs,
                             ambig=ambig, dup=dup))
    return recs


def run_condition(recs, w_agent, dup_mode):
    """Return per-seed dict of (auc, far_by_T, junk_by_T) on agent test folds."""
    X_all = np.vstack([r["X"] for r in recs])
    y_all = np.concatenate([r["y"] for r in recs]).astype(int)
    groups = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(recs)])
    is_bs = np.concatenate([[r["is_bs"]] * len(r["y"]) for r in recs])
    ambig = np.concatenate([r["ambig"] for r in recs])
    dup = np.concatenate([r["dup"] for r in recs])

    y_train = y_all.copy()
    base_w = np.ones(len(y_all))
    base_w[is_bs & ambig] = 0.0
    if dup_mode == "masked":
        base_w[is_bs & dup] = 0.0
    elif dup_mode == "label1":
        y_train[is_bs & dup] = 1
    # dup_mode == "label0": leave label 0, weight 1

    ag_ok = sorted({i for i, r in enumerate(recs)
                    if not r["is_bs"] and r["y"].sum() >= MIN_POS})
    ag_mask = np.isin(groups, ag_ok)
    results = []
    for seed in SEEDS:
        scores = np.full(len(y_all), np.nan)
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for _, te in skf.split(np.zeros(ag_mask.sum()), y_all[ag_mask], groups[ag_mask]):
            te_sessions = np.unique(groups[ag_mask][te])
            train = ~np.isin(groups, te_sessions)
            w = base_w.copy()
            w[~is_bs] = w_agent
            w[~train] = 0.0
            m = w > 0
            pos = m & (y_train == 1)
            neg = m & (y_train == 0)
            spw = w[neg].sum() / max(w[pos].sum(), 1e-9)
            clf = tc._make_clf("xgboost", scale_pos_weight=spw)
            clf.fit(X_all[m], y_train[m], sample_weight=w[m])
            te_mask = np.isin(groups, te_sessions)
            scores[te_mask] = clf.predict_proba(X_all[te_mask])[:, 1]
        sm = ~np.isnan(scores)
        yv, sv = y_all[sm], scores[sm]
        far = {t: 100 * ((sv < t) & (yv == 1)).sum() / (yv == 1).sum() for t in THRESHOLDS}
        junk = {t: 100 * ((sv < t) & (yv == 0)).sum() / (yv == 0).sum() for t in THRESHOLDS}
        results.append((roc_auc_score(yv, sv), far, junk))
    return results


def summarize(label, results):
    aucs = [r[0] for r in results]
    chosen = None
    for t in THRESHOLDS:
        fars = [r[1][t] for r in results]
        if np.mean(fars) <= 0.85 and max(fars) <= 1.0:
            chosen = t
    if chosen is None:
        print(f"{label:28s} AUC {np.mean(aucs):.4f}+/-{np.std(aucs):.4f} | no T meets the rule")
        return
    fars = [r[1][chosen] for r in results]
    junks = [r[2][chosen] for r in results]
    print(f"{label:28s} AUC {np.mean(aucs):.4f}+/-{np.std(aucs):.4f} | rule T={chosen:.2f}: "
          f"FAR {np.mean(fars):.2f}% (max {max(fars):.2f}) junk {np.mean(junks):.1f}%", flush=True)


if __name__ == "__main__":
    recs = load_pool()
    n_ag = sum(1 for r in recs if not r["is_bs"])
    n_dup = sum(int(r["dup"].sum()) for r in recs)
    print(f"[{ARGS.area}] {len(recs)} sessions ({n_ag} agent), {n_dup} duplicate rows, "
          f"seeds {SEEDS}, features {recs[0]['X'].shape[1]}")
    if ARGS.mode == "weights":
        for w in (1.0, 2.0, 3.0, 4.0, 5.0, 7.0):
            summarize(f"agent_w={w:.1f} (dups masked)", run_condition(recs, w, "masked"))
    else:
        for mode in ("masked", "label0", "label1"):
            summarize(f"w={DEPLOYED_W:.1f} dups={mode}", run_condition(recs, DEPLOYED_W, mode))
