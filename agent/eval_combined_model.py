"""
eval_combined_model.py
Does adding BLA training data improve the vCA1 classifier?

Held-out test = vCA1 agent sessions (clean human labels).  Two conditions,
evaluated with 5-fold grouped CV (split by session):

  A (vCA1-only):  train on vCA1 bootstrap + vCA1 agent train-folds
  B (combined):   condition A  +  ALL BLA data (bootstrap + agent)

vCA1 weighting is identical in both conditions (agent up-weight floored at 4.0,
bad-session 0.4x, ambiguous=0).  BLA candidates are appended at a neutral weight
(default 1.0) so this purely measures whether cross-area data helps vCA1 — it is
NOT the deployed weighting scheme.

    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe eval_combined_model.py
    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe eval_combined_model.py --bla-weight 2.0
"""
import argparse
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

import config            # BLA
import config_vCA1       # vCA1
import train_classifier as tc

MIN_AGENT_WEIGHT = 4.0


def load_area(data_root):
    """Return list of {X, y, is_bootstrap, w_raw, session_dir} for one area.

    w_raw is the per-candidate weight WITHOUT the agent up-weight (that is applied
    per-fold). Bootstrap bad-session 0.4x and ambiguous=0 are baked in here.
    """
    records = []
    for td in sorted(data_root.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            if not ((sd / "candidate_features.npz").exists()
                    and (sd / "labels.mat").exists()):
                continue
            result = tc.load_prospective_session(sd)
            if result is None:
                continue
            X, y_full = result
            X = X.astype(float)
            y = (y_full == 1).astype(int)
            n = len(y)
            is_bs = (sd / "bootstrap_match_stats.json").exists()
            w = np.ones(n, dtype=float)
            if is_bs:
                rec = tc._get_bootstrap_recovery(sd)
                if rec is not None and rec < tc.BAD_SESSION_RECOVERY_THRESHOLD:
                    w *= tc.BAD_SESSION_WEIGHT
                w[tc._get_bootstrap_ambiguous_mask(sd, n)] = 0.0
            records.append({"X": X, "y": y, "is_bootstrap": is_bs,
                            "w_raw": w, "session_dir": sd})
    return records


def fit_eval(X_tr, y_tr, w_tr, X_te, y_te, scaler):
    mask = w_tr > 0
    ep = float(w_tr[(y_tr == 1) & mask].sum())
    en = float(w_tr[(y_tr == 0) & mask].sum())
    spw = en / ep if ep > 0 else 1.0
    clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                        eval_metric="auc", verbosity=0, random_state=42, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(scaler.transform(X_tr), y_tr, sample_weight=w_tr)
    return roc_auc_score(y_te, clf.predict_proba(scaler.transform(X_te))[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bla-weight", type=float, default=1.0,
                    help="Per-candidate weight for appended BLA data in condition B.")
    ap.add_argument("--min-pos", type=int, default=5,
                    help="Min positives for a vCA1 agent session to be a test fold.")
    args = ap.parse_args()

    print("Loading vCA1 ...")
    vca1 = load_area(config_vCA1.DATA_ROOT)
    print("Loading BLA ...")
    bla = load_area(config.DATA_ROOT)

    v_ag = [r for r in vca1 if not r["is_bootstrap"] and int(r["y"].sum()) >= args.min_pos]
    v_bs = [r for r in vca1 if r["is_bootstrap"]]
    print(f"\nvCA1: {len(v_ag)} usable agent (>= {args.min_pos} pos), {len(v_bs)} bootstrap")
    print(f"BLA : {sum(1 for r in bla if not r['is_bootstrap'])} agent, "
          f"{sum(1 for r in bla if r['is_bootstrap'])} bootstrap "
          f"(appended at weight {args.bla_weight} in condition B)")

    if len(v_ag) < 3:
        print("Too few vCA1 agent sessions. Aborting.")
        return

    # vCA1 bootstrap pool (always in train, never in test)
    Xv_bs = np.vstack([r["X"] for r in v_bs])
    yv_bs = np.concatenate([r["y"] for r in v_bs])
    wv_bs = np.concatenate([r["w_raw"] for r in v_bs])

    # ALL BLA data appended in condition B
    X_bla = np.vstack([r["X"] for r in bla])
    y_bla = np.concatenate([r["y"] for r in bla])
    w_bla = np.full(len(y_bla), args.bla_weight, dtype=float)

    # vCA1 agent pool
    Xv_ag = np.vstack([r["X"] for r in v_ag])
    yv_ag = np.concatenate([r["y"] for r in v_ag])
    gv_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(v_ag)])

    # Scaler fit on the union of everything (consistent feature scaling)
    scaler = StandardScaler().fit(np.vstack([Xv_ag, Xv_bs, X_bla]))

    n_splits = min(5, len(v_ag))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs_A, aucs_B = [], []
    for tr, te in cv.split(Xv_ag, yv_ag, gv_ag):
        if len(np.unique(yv_ag[te])) < 2:
            continue
        n_ag_tr = len(tr)
        agw = float(max(np.sqrt(len(yv_bs) / n_ag_tr), MIN_AGENT_WEIGHT))
        w_ag = np.ones(n_ag_tr) * agw

        # Condition A: vCA1 only
        XA = np.vstack([Xv_ag[tr], Xv_bs])
        yA = np.concatenate([yv_ag[tr], yv_bs])
        wA = np.concatenate([w_ag, wv_bs])
        aucs_A.append(fit_eval(XA, yA, wA, Xv_ag[te], yv_ag[te], scaler))

        # Condition B: + all BLA
        XB = np.vstack([XA, X_bla])
        yB = np.concatenate([yA, y_bla])
        wB = np.concatenate([wA, w_bla])
        aucs_B.append(fit_eval(XB, yB, wB, Xv_ag[te], yv_ag[te], scaler))

    mA, mB = float(np.mean(aucs_A)), float(np.mean(aucs_B))
    sA, sB = float(np.std(aucs_A)), float(np.std(aucs_B))
    print(f"\n  {n_splits}-fold grouped CV on held-out vCA1 agent sessions:")
    print(f"  {'Condition':<22}{'AUC':>8}")
    print(f"  {'-'*22}{'-'*8}")
    print(f"  {'A vCA1-only':<22}{mA:>8.3f}  +/- {sA:.3f}")
    print(f"  {'B vCA1 + BLA':<22}{mB:>8.3f}  +/- {sB:.3f}")
    print(f"  {'Delta (B - A)':<22}{mB - mA:>+8.3f}")
    print(f"\n  {'helps' if mB - mA > 0.005 else 'no benefit (within noise)' if abs(mB-mA) <= 0.005 else 'hurts'}"
          f" — combined model {'beats' if mB > mA else 'does not beat'} vCA1-only.")


if __name__ == "__main__":
    main()
