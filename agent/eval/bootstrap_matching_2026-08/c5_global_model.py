"""
c5_global_model.py — does a pooled cross-area ("global GRIN-lens") model help,
now that bootstrap labels are correct?

For each TARGET area (vCA1, BLA, DG_AL): held-out test = target agent sessions
(>= MIN_POS positives), 5-fold grouped CV (fewer folds if fewer sessions).
  Cond A: target-only training (target bootstrap + target agent train folds),
          target's deployed weighting (agent weight = config override or 4.0;
          bootstrap masked per the trainer's helper).
  Cond B: A + ALL other areas' rows (agent + bootstrap, their own masks) at a
          neutral per-row weight SOURCE_W.
Features: the shared first 13 columns (bit-identical across the v1 and v2
contracts) so areas on different contracts can pool. Reports AUC A/B and
false-AR / junk-caught at the target's deployed threshold. 3 seeds.

Usage: valence python c5_global_model.py [--source-w 1.0] [--seeds 3]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

AGENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_DIR))

ap = argparse.ArgumentParser()
ap.add_argument("--source-w", type=float, default=1.0)
ap.add_argument("--seeds", type=int, default=3)
ARGS = ap.parse_args()
SEEDS = [42 + i for i in range(ARGS.seeds)]
MIN_POS = 5

import config as cfg_bla
import config_vCA1 as cfg_vca1
import config_DG_AL as cfg_dg
import train_classifier as tc

AREAS = {"BLA": cfg_bla, "vCA1": cfg_vca1, "DG_AL": cfg_dg}
DEPLOYED_T = {"BLA": 0.04, "vCA1": 0.05, "DG_AL": 0.05}


def load_area(cfg):
    recs = []
    for td in sorted(cfg.DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            if not ((sd / "candidate_features.npz").exists()
                    and (sd / "labels.mat").exists()):
                continue
            out = tc.load_prospective_session(sd)
            if out is None:
                continue
            X, y = out
            X = np.asarray(X, dtype=float)[:, :13]
            y = (np.asarray(y) == 1).astype(int)
            is_bs = tc._is_bootstrap_session(sd)
            w = np.ones(len(y))
            if is_bs:
                rec = tc._get_bootstrap_recovery(sd)
                if rec is not None and rec < tc.BAD_SESSION_RECOVERY_THRESHOLD:
                    w *= tc.BAD_SESSION_WEIGHT
                w[tc._get_bootstrap_ambiguous_mask(sd, len(y))] = 0.0
            recs.append(dict(X=X, y=y, is_bs=is_bs, w=w, name=sd.name))
    return recs


def make_clf(spw):
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                         eval_metric="auc", verbosity=0, random_state=42, n_jobs=-1)


def evaluate(target, pools):
    cfg = AREAS[target]
    recs = pools[target]
    ag = [r for r in recs if not r["is_bs"] and r["y"].sum() >= MIN_POS]
    bs = [r for r in recs if r["is_bs"]]
    if len(ag) < 3:
        print(f"[{target}] only {len(ag)} usable agent sessions — skipped")
        return
    aw_override = getattr(cfg, "AGENT_WEIGHT_OVERRIDE", None)
    X_ag = np.vstack([r["X"] for r in ag]); y_ag = np.concatenate([r["y"] for r in ag])
    g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag)])
    if bs:
        X_bs = np.vstack([r["X"] for r in bs]); y_bs = np.concatenate([r["y"] for r in bs])
        w_bs = np.concatenate([r["w"] for r in bs])
    else:
        X_bs = np.zeros((0, 13)); y_bs = np.zeros(0, int); w_bs = np.zeros(0)
    others = [r for a, rs in pools.items() if a != target for r in rs]
    X_ot = np.vstack([r["X"] for r in others]); y_ot = np.concatenate([r["y"] for r in others])
    w_ot = np.concatenate([r["w"] for r in others]) * ARGS.source_w
    T = DEPLOYED_T[target]
    n_splits = min(5, len(ag))
    print(f"\n[{target}] test pool: {len(ag)} agent sessions, {len(y_ag)} rows, "
          f"{int(y_ag.sum())} real | own bootstrap rows {len(y_bs)} | other-area rows {len(y_ot)} "
          f"(source_w={ARGS.source_w}) | T={T}")
    out = {"A": [], "B": []}
    for seed in SEEDS:
        scores = {"A": np.full(len(y_ag), np.nan), "B": np.full(len(y_ag), np.nan)}
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in cv.split(X_ag, y_ag, g_ag):
            if len(np.unique(y_ag[te])) < 2:
                continue
            agw = float(aw_override) if aw_override is not None else \
                float(max(np.sqrt(max(len(y_bs), 1) / len(tr)), 4.0))
            base_X = [X_ag[tr], X_bs]; base_y = [y_ag[tr], y_bs]
            base_w = [np.ones(len(tr)) * agw, w_bs]
            for cond in ("A", "B"):
                Xs, ys, ws = list(base_X), list(base_y), list(base_w)
                if cond == "B":
                    Xs.append(X_ot); ys.append(y_ot); ws.append(w_ot)
                Xt = np.vstack(Xs); yt = np.concatenate(ys); wt = np.concatenate(ws)
                m = wt > 0
                spw = wt[m & (yt == 0)].sum() / max(wt[m & (yt == 1)].sum(), 1e-9)
                sc = StandardScaler().fit(Xt[m])
                clf = make_clf(spw)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(sc.transform(Xt[m]), yt[m], sample_weight=wt[m])
                scores[cond][te] = clf.predict_proba(sc.transform(X_ag[te]))[:, 1]
        for cond in ("A", "B"):
            s = scores[cond]; ok = ~np.isnan(s)
            yv, sv = y_ag[ok], s[ok]
            far = 100 * ((sv < T) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
            junk = 100 * ((sv < T) & (yv == 0)).sum() / max((yv == 0).sum(), 1)
            out[cond].append((roc_auc_score(yv, sv), far, junk))
    for cond, label in (("A", f"{target}-only"), ("B", f"{target} + other areas")):
        a = np.array(out[cond])
        print(f"  {label:22s} AUC {a[:,0].mean():.4f}+/-{a[:,0].std():.4f} | "
              f"T={T}: FAR {a[:,1].mean():.2f}% junk {a[:,2].mean():.1f}%")
    d = np.array(out["B"])[:, 0] - np.array(out["A"])[:, 0]
    print(f"  paired AUC delta (B-A): {d.mean():+.4f} (min {d.min():+.4f}, max {d.max():+.4f})", flush=True)


if __name__ == "__main__":
    pools = {a: load_area(c) for a, c in AREAS.items()}
    for a, rs in pools.items():
        n_ag = sum(1 for r in rs if not r["is_bs"])
        print(f"{a}: {len(rs)} sessions ({n_ag} agent, {len(rs) - n_ag} bootstrap), "
              f"{sum(len(r['y']) for r in rs)} rows")
    for target in ("vCA1", "BLA", "DG_AL"):
        evaluate(target, pools)
