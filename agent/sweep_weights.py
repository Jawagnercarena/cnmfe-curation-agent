"""
sweep_weights.py
Joint re-sweep of the two training weight knobs on held-out agent OOF folds:
  - MIN_AGENT_WEIGHT (the agent up-weight floor)
  - BAD_SESSION_WEIGHT (down-weight for bootstrap sessions with recovery < 40%)

Replicates the March 2026 ad-hoc methodology (see feedback_agent_weight_floor memory)
so results are directly comparable.  Two 1-D sweeps:
  Sweep 1: floor in {1,3,4,5,8} at bad_w = current 0.4
  Sweep 2: bad_w in {0,0.2,0.4,0.6,1.0} at floor = current 4.0

Each config: 5-fold grouped OOF on agent test folds (clean labels), reporting
OOF AUC plus false-AR / garbage caught at a reference threshold.

Area-agnostic (uses config). For BLA:  python sweep_weights.py
For vCA1: inject config_vCA1 first (or add a wrapper) -- not needed yet.

    python sweep_weights.py                 # threshold 0.11
    python sweep_weights.py --threshold 0.15
"""
import argparse
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from config import DATA_ROOT, AREA
import train_classifier as tc

BAD_RECOVERY = tc.BAD_SESSION_RECOVERY_THRESHOLD   # 0.40
MIN_POS = 5


def load_records():
    """Per-session X, y, is_bootstrap, is_bad, ambig mask -- weights applied later."""
    recs = []
    for td in sorted(DATA_ROOT.iterdir()):
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
            y = (y_full == 1).astype(int)
            n = len(y)
            is_bs = (sd / "bootstrap_match_stats.json").exists()
            is_bad = False
            ambig = np.zeros(n, dtype=bool)
            if is_bs:
                rec = tc._get_bootstrap_recovery(sd)
                is_bad = rec is not None and rec < BAD_RECOVERY
                ambig = tc._get_bootstrap_ambiguous_mask(sd, n)
            recs.append({"X": X.astype(float), "y": y, "is_bootstrap": is_bs,
                         "is_bad": is_bad, "ambig": ambig})
    return recs


def oof_eval(recs, floor, bad_w, threshold):
    """5-fold grouped OOF on agent sessions. Returns (auc, false_ar%, garbage%)."""
    ag = [r for r in recs if not r["is_bootstrap"] and int(r["y"].sum()) >= MIN_POS]
    bs = [r for r in recs if r["is_bootstrap"]]

    X_ag = np.vstack([r["X"] for r in ag])
    y_ag = np.concatenate([r["y"] for r in ag])
    g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag)])

    X_bs = np.vstack([r["X"] for r in bs])
    y_bs = np.concatenate([r["y"] for r in bs])
    w_bs = np.concatenate([
        np.where(r["ambig"], 0.0, (bad_w if r["is_bad"] else 1.0)) for r in bs
    ])
    n_bs = len(y_bs)

    scaler = StandardScaler().fit(np.vstack([X_ag, X_bs]))
    Xs_ag, Xs_bs = scaler.transform(X_ag), scaler.transform(X_bs)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(y_ag), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in cv.split(Xs_ag, y_ag, g_ag):
            agw = float(max(np.sqrt(n_bs / len(tr)), floor))
            w_ag = np.ones(len(tr)) * agw
            Xtr = np.vstack([Xs_ag[tr], Xs_bs])
            ytr = np.concatenate([y_ag[tr], y_bs])
            wtr = np.concatenate([w_ag, w_bs])
            m = wtr > 0
            ep = float(wtr[(ytr == 1) & m].sum())
            en = float(wtr[(ytr == 0) & m].sum())
            spw = en / ep if ep > 0 else 1.0
            clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                eval_metric="auc", verbosity=0, random_state=42, n_jobs=-1)
            clf.fit(Xtr, ytr, sample_weight=wtr)
            oof[te] = clf.predict_proba(Xs_ag[te])[:, 1]

    pos, neg = y_ag == 1, y_ag == 0
    auc = roc_auc_score(y_ag, oof)
    far = float((oof[pos] < threshold).sum()) / pos.sum() * 100
    gc = float((oof[neg] < threshold).sum()) / neg.sum() * 100
    return auc, far, gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.11,
                    help="Reference threshold for false-AR / garbage columns.")
    args = ap.parse_args()

    print(f"Loading {AREA} records ...")
    recs = load_records()
    n_ag = sum(1 for r in recs if not r["is_bootstrap"])
    n_bad = sum(1 for r in recs if r["is_bad"])
    print(f"  {n_ag} agent sessions, {sum(1 for r in recs if r['is_bootstrap'])} bootstrap "
          f"({n_bad} bad sessions, recovery < {BAD_RECOVERY:.0%})")
    print(f"  Metrics at reference threshold = {args.threshold}\n")

    print("=" * 64)
    print(f"SWEEP 1: agent_weight floor  (bad_session_weight fixed at 0.4)")
    print("=" * 64)
    print(f"  {'floor':>6}  {'OOF AUC':>8}  {'false-AR':>9}  {'garbage':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}")
    for floor in [1.0, 3.0, 4.0, 5.0, 8.0]:
        auc, far, gc = oof_eval(recs, floor, 0.4, args.threshold)
        star = "  <- current" if floor == 4.0 else ""
        print(f"  {floor:>6.1f}  {auc:>8.3f}  {far:>8.1f}%  {gc:>7.1f}%{star}")

    print("\n" + "=" * 64)
    print(f"SWEEP 2: bad_session_weight  (agent_weight floor fixed at 4.0)")
    print("=" * 64)
    print(f"  {'bad_w':>6}  {'OOF AUC':>8}  {'false-AR':>9}  {'garbage':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}")
    for bad_w in [0.0, 0.2, 0.4, 0.6, 1.0]:
        auc, far, gc = oof_eval(recs, 4.0, bad_w, args.threshold)
        star = "  <- current" if bad_w == 0.4 else ""
        print(f"  {bad_w:>6.1f}  {auc:>8.3f}  {far:>8.1f}%  {gc:>7.1f}%{star}")


if __name__ == "__main__":
    main()
