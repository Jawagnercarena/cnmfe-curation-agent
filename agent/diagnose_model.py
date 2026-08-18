"""
diagnose_model.py
Deep comparison: LR vs XGBoost (agent-only) vs XGBoost (full w/ bootstrap)
Plus: Pareto-optimal threshold analysis using the real deployed model's weighting.

Section 1: Per-session LOO AUC on agent sessions
Section 2: Score distributions on 4 pending sessions at key thresholds
Section 3: Threshold sweep using REAL weighting (bad-session 0.4x, ambiguous mask=0,
           agent_weight) — finds the Pareto-optimal operating threshold
"""
import json
import warnings
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

AGENT_DIR = Path(__file__).parent
from config import DATA_ROOT, MODEL_DIR
MODEL_PATH = MODEL_DIR / "classifier.joblib"

# Reuse the validated prospective loader (handles the labels.mat/features
# size-mismatch by reconstructing auto-rejected candidates as label=0).
import train_classifier as _tc

# Weighting constants and helpers come straight from train_classifier so this
# harness can never drift from the deployed weighting. (A previous forked copy
# of _get_bootstrap_ambiguous_mask read JSON key "ambiguous_candidate_indices",
# which no producer writes — the mask was silently all-False in every run.)
BAD_SESSION_RECOVERY_THRESHOLD = _tc.BAD_SESSION_RECOVERY_THRESHOLD
BAD_SESSION_WEIGHT             = _tc.BAD_SESSION_WEIGHT
MIN_AGENT_WEIGHT               = _tc.MIN_AGENT_WEIGHT

_get_bootstrap_recovery       = _tc._get_bootstrap_recovery
_get_bootstrap_ambiguous_mask = _tc._get_bootstrap_ambiguous_mask


# -----------------------------------------------------------------------
# Data loading (using all 105 sessions, same logic as train_classifier.py)
# -----------------------------------------------------------------------

def load_all_records():
    """Load all sessions with labels; compute and attach real per-sample weights."""
    records = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."): continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir(): continue
            feat_f  = sd / "candidate_features.npz"
            label_f = sd / "labels.mat"
            if not feat_f.exists() or not label_f.exists(): continue
            # Reconstructs full labels for prospective sessions where labels.mat
            # only covers the non-auto-rejected subset (returns None on failure).
            result = _tc.load_prospective_session(sd)
            if result is None:
                continue
            X, y_full = result
            X = X.astype(float)
            y = (y_full == 1).astype(int)
            is_bs = _tc._is_bootstrap_session(sd)
            records.append({
                "name": f"{td.name}/{sd.name}",
                "X": X,
                "y": y,
                "is_bootstrap": is_bs,
                "session_dir": sd,
            })

    # Compute agent_weight exactly as train_classifier.py does
    n_bs = sum(len(r["y"]) for r in records if r["is_bootstrap"])
    n_ag = sum(len(r["y"]) for r in records if not r["is_bootstrap"])
    agent_weight = (float(max(np.sqrt(n_bs / n_ag), MIN_AGENT_WEIGHT))
                    if n_ag > 0 and n_bs > 0 else MIN_AGENT_WEIGHT)

    # Attach per-sample weights
    for r in records:
        n = len(r["y"])
        w = np.ones(n, dtype=float)
        if not r["is_bootstrap"]:
            w *= agent_weight
        else:
            recovery = _get_bootstrap_recovery(r["session_dir"])
            if recovery is not None and recovery < BAD_SESSION_RECOVERY_THRESHOLD:
                w *= BAD_SESSION_WEIGHT
            ambig = _get_bootstrap_ambiguous_mask(r["session_dir"], n)
            w[ambig] = 0.0
        r["w"] = w

    return records, agent_weight


def make_clf(mtype, spw=1.0):
    # Delegate to the trainer's factory so hyperparameters can never diverge.
    # Historical callers pass "xgb"/"lr"; anything non-"lr" was always XGBoost.
    return _tc._make_clf("lr" if mtype == "lr" else "xgboost", spw)


def compute_spw(y, w):
    mask = w > 0
    pos = float(w[(y == 1) & mask].sum())
    neg = float(w[(y == 0) & mask].sum())
    return neg / pos if pos > 0 else 1.0


def _find_pending_sessions(limit: int = 4):
    """Agent-pipeline sessions that were curated but not yet MATLAB-reviewed.

    These have candidate_features.npz (curator ran) and ROIs_candidates.jpg
    (agent pipeline) but no labels.mat yet.  Discovered from DATA_ROOT so the
    tool works for any brain area instead of a hardcoded BLA path list.
    """
    out = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            if ((sd / "candidate_features.npz").exists()
                    and (sd / "ROIs_candidates.jpg").exists()
                    and not (sd / "labels.mat").exists()):
                out.append(sd)
    return out[:limit]


# -----------------------------------------------------------------------
# Section 1: Per-session LOO AUC
# -----------------------------------------------------------------------

def run_loo_analysis(records):
    ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
    bs_recs = [r for r in records if r["is_bootstrap"]]

    X_bs = np.vstack([r["X"] for r in bs_recs]) if bs_recs else None
    y_bs = np.concatenate([r["y"] for r in bs_recs]) if bs_recs else None
    w_bs = np.concatenate([r["w"] for r in bs_recs]) if bs_recs else None

    print("\n" + "="*80)
    print("1. PER-SESSION LOO AUC  (train on N-1 agent sessions, test on 1)")
    print("   A=LR agent-only  B=XGB agent-only  C=XGB+bootstrap (real weights)")
    print("="*80)
    print(f"\n  {'Session':<40}  {'LR':>6}  {'XGB-ag':>7}  {'XGB-bs':>7}  {'B-A':>6}  {'C-A':>6}")
    print(f"  {'-'*40}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")

    aucs = {"lr": [], "xgb_ag": [], "xgb_bs": []}

    for i, te in enumerate(ag_recs):
        tr_ag = [r for j, r in enumerate(ag_recs) if j != i]
        if len(tr_ag) < 3: continue
        X_tr = np.vstack([r["X"] for r in tr_ag])
        y_tr = np.concatenate([r["y"] for r in tr_ag])
        X_te, y_te = te["X"], te["y"]
        if len(np.unique(y_te)) < 2: continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # A: LR agent-only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cA = make_clf("lr"); cA.fit(X_tr_s, y_tr)
            auc_A = roc_auc_score(y_te, cA.predict_proba(X_te_s)[:, 1])

        # B: XGB agent-only, uniform weight
        spw_B = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
        cB = make_clf("xgb", spw_B)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cB.fit(X_tr_s, y_tr)
        auc_B = roc_auc_score(y_te, cB.predict_proba(X_te_s)[:, 1])

        # C: XGB + bootstrap, real weights
        if X_bs is not None:
            # Recompute agent_weight for this fold's training set
            n_ag_tr = len(X_tr)
            n_bs_tr = len(X_bs)
            ag_w_C  = float(max(np.sqrt(n_bs_tr / n_ag_tr), MIN_AGENT_WEIGHT)) if n_ag_tr > 0 else MIN_AGENT_WEIGHT
            w_ag_C  = np.ones(n_ag_tr) * ag_w_C
            X_tr_C  = np.vstack([X_tr, X_bs])
            y_tr_C  = np.concatenate([y_tr, y_bs])
            w_tr_C  = np.concatenate([w_ag_C, w_bs])

            sc_C    = StandardScaler()
            X_tr_Cs = sc_C.fit_transform(X_tr_C)
            X_te_Cs = sc_C.transform(X_te)

            spw_C = compute_spw(y_tr_C, w_tr_C)
            cC = make_clf("xgb", spw_C)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cC.fit(X_tr_Cs, y_tr_C, sample_weight=w_tr_C)
            auc_C = roc_auc_score(y_te, cC.predict_proba(X_te_Cs)[:, 1])
        else:
            auc_C = auc_B

        aucs["lr"].append(auc_A); aucs["xgb_ag"].append(auc_B); aucs["xgb_bs"].append(auc_C)
        sname = te["name"]
        sname = (sname[-40:] if len(sname) > 40 else sname).ljust(40)
        print(f"  {sname}  {auc_A:.3f}  {auc_B:7.3f}  {auc_C:7.3f}  "
              f"{auc_B-auc_A:+6.3f}  {auc_C-auc_A:+6.3f}")

    print(f"\n  {'MEAN':<40}  {np.mean(aucs['lr']):.3f}  "
          f"{np.mean(aucs['xgb_ag']):7.3f}  {np.mean(aucs['xgb_bs']):7.3f}  "
          f"{np.mean(aucs['xgb_ag'])-np.mean(aucs['lr']):+6.3f}  "
          f"{np.mean(aucs['xgb_bs'])-np.mean(aucs['lr']):+6.3f}")
    print(f"  {'STD':<40}  {np.std(aucs['lr']):.3f}  "
          f"{np.std(aucs['xgb_ag']):7.3f}  {np.std(aucs['xgb_bs']):7.3f}")


# -----------------------------------------------------------------------
# Section 2: Score distributions on the 4 pending sessions
# -----------------------------------------------------------------------

def run_pending_analysis(records):
    pending = _find_pending_sessions()
    if not pending:
        print("\n" + "="*80)
        print("2. SCORE DISTRIBUTIONS ON PENDING SESSIONS")
        print("="*80)
        print("\n  (no pending/unreviewed sessions found - skipping)")
        return

    d = joblib.load(str(MODEL_PATH))
    scaler_dep, clf_dep = d["scaler"], d["clf"]

    ag_recs = [r for r in records if not r["is_bootstrap"]]
    X_ag = np.vstack([r["X"] for r in ag_recs])
    y_ag = np.concatenate([r["y"] for r in ag_recs])
    w_ag = np.concatenate([r["w"] for r in ag_recs])

    sc_lr = StandardScaler(); X_lr = sc_lr.fit_transform(X_ag)
    clf_lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    clf_lr.fit(X_lr, y_ag)

    sc_xgb = StandardScaler(); X_xgb = sc_xgb.fit_transform(X_ag)
    spw = compute_spw(y_ag, np.ones(len(y_ag)))
    clf_xgb_ag = make_clf("xgb", spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_xgb_ag.fit(X_xgb, y_ag)

    # `pending` is discovered dynamically at the top of this function via
    # _find_pending_sessions() — no hardcoded session paths to maintain.
    print("\n" + "="*80)
    print("2. SCORE DISTRIBUTIONS ON 4 PENDING SESSIONS (no ground truth)")
    print("   LR@0.10=old operating point  |  XGB-ag=agent-only  |  XGB-full=deployed")
    print("="*80)
    for sd in pending:
        feat_f = sd / "candidate_features.npz"
        if not feat_f.exists(): continue
        npz = np.load(str(feat_f), allow_pickle=True)
        X = npz["feature_matrix"].astype(float)
        N = len(X)
        s_lr   = clf_lr.predict_proba(sc_lr.transform(X))[:, 1]
        s_xag  = clf_xgb_ag.predict_proba(sc_xgb.transform(X))[:, 1]
        s_xful = clf_dep.predict_proba(scaler_dep.transform(X))[:, 1]

        sname = f"{sd.parent.name}/{sd.name[-33:]}"
        print(f"\n  {sname}  (N={N})")
        print(f"  {'Thresh':>7}  {'LR rej':>7}  {'XGB-ag rej':>11}  {'XGB-full rej':>13}")
        for t in [0.05, 0.10, 0.12, 0.15, 0.20, 0.25]:
            print(f"  {t:7.2f}  "
                  f"{int((s_lr<t).sum()):4d} ({int((s_lr<t).sum())/N:.0%})  "
                  f"{int((s_xag<t).sum()):5d} ({int((s_xag<t).sum())/N:.0%})       "
                  f"{int((s_xful<t).sum()):5d} ({int((s_xful<t).sum())/N:.0%})")
        print(f"  Score pct  LR: p5={np.percentile(s_lr,5):.3f} p10={np.percentile(s_lr,10):.3f} "
              f"p50={np.percentile(s_lr,50):.3f} | "
              f"XGB-full: p5={np.percentile(s_xful,5):.3f} p10={np.percentile(s_xful,10):.3f} "
              f"p50={np.percentile(s_xful,50):.3f}")


# -----------------------------------------------------------------------
# Section 3: Pareto-optimal threshold using REAL weighting
# -----------------------------------------------------------------------

def run_threshold_sweep_real_weights(records):
    """
    5-fold grouped OOF CV using the REAL training weighting:
    - agent_weight = sqrt(n_bs_candidates / n_ag_candidates)  [per fold]
    - bad_session_weight = 0.4x for bootstrap sessions with recovery < 40%
    - ambiguous_mask = weight=0 for false-negative contaminated negatives

    After collecting OOF scores on agent test folds (clean labels), sweep thresholds
    and compute:
      - false_ar: % of real neurons wrongly auto-rejected
      - garbage:  % of garbage (label=0) correctly auto-rejected
      - efficiency: marginal garbage caught per 1% increase in false_ar
    """
    ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
    bs_recs = [r for r in records if r["is_bootstrap"]]

    if len(ag_recs) < 3:
        print("  Too few agent sessions. Skipping.")
        return

    X_ag  = np.vstack([r["X"] for r in ag_recs])
    y_ag  = np.concatenate([r["y"] for r in ag_recs])
    g_ag  = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag_recs)])

    X_bs  = np.vstack([r["X"] for r in bs_recs]) if bs_recs else None
    y_bs  = np.concatenate([r["y"] for r in bs_recs]) if bs_recs else None
    w_bs  = np.concatenate([r["w"] for r in bs_recs]) if bs_recs else None  # real weights

    print("\n" + "="*80)
    print("3. PARETO-OPTIMAL THRESHOLD  (5-fold grouped OOF, REAL weights)")
    print("   Real weights: agent_weight per fold, bad-session 0.4x, ambiguous=0")
    print("   Test folds = agent sessions only (clean human labels)")
    print("="*80)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lr   = np.full(len(y_ag), np.nan)
    oof_xgb  = np.full(len(y_ag), np.nan)

    for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
        y_tr = y_ag[tr_idx]
        X_tr = X_ag[tr_idx]
        X_te = X_ag[te_idx]

        # ---- LR: agent-only, uniform weight ----
        sc_lr = StandardScaler()
        X_tr_s = sc_lr.fit_transform(X_tr)
        X_te_s = sc_lr.transform(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_lr = make_clf("lr")
            clf_lr.fit(X_tr_s, y_tr)
            oof_lr[te_idx] = clf_lr.predict_proba(X_te_s)[:, 1]

        # ---- XGB + bootstrap, REAL weights ----
        if X_bs is not None:
            n_ag_tr  = len(tr_idx)
            n_bs_tr  = len(y_bs)
            ag_w_f   = float(max(np.sqrt(n_bs_tr / n_ag_tr), MIN_AGENT_WEIGHT)) if n_ag_tr > 0 else MIN_AGENT_WEIGHT
            w_ag_f   = np.ones(n_ag_tr) * ag_w_f

            X_tr_C   = np.vstack([X_tr, X_bs])
            y_tr_C   = np.concatenate([y_tr, y_bs])
            w_tr_C   = np.concatenate([w_ag_f, w_bs])   # real bootstrap weights

            sc_xgb   = StandardScaler()
            X_tr_Cs  = sc_xgb.fit_transform(X_tr_C)
            X_te_Cs  = sc_xgb.transform(X_te)

            spw_C    = compute_spw(y_tr_C, w_tr_C)
            clf_xgb  = make_clf("xgb", spw_C)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf_xgb.fit(X_tr_Cs, y_tr_C, sample_weight=w_tr_C)
                oof_xgb[te_idx] = clf_xgb.predict_proba(X_te_Cs)[:, 1]
        else:
            # Fallback: XGB agent-only
            sc_xgb = StandardScaler()
            X_tr_s2 = sc_xgb.fit_transform(X_tr)
            X_te_s2 = sc_xgb.transform(X_te)
            spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
            clf_xgb = make_clf("xgb", spw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf_xgb.fit(X_tr_s2, y_tr)
                oof_xgb[te_idx] = clf_xgb.predict_proba(X_te_s2)[:, 1]

    pos_mask = y_ag == 1
    neg_mask = y_ag == 0
    n_pos    = pos_mask.sum()
    n_neg    = neg_mask.sum()

    print(f"\n  OOF pool: {len(y_ag)} candidates, {n_pos} real neurons ({n_pos/len(y_ag):.1%}), "
          f"{n_neg} garbage ({n_neg/len(y_ag):.1%})")

    # Print models' overall AUC on the test folds
    valid = ~np.isnan(oof_lr) & ~np.isnan(oof_xgb)
    auc_lr  = roc_auc_score(y_ag[valid], oof_lr[valid])
    auc_xgb = roc_auc_score(y_ag[valid], oof_xgb[valid])
    print(f"  OOF AUC — LR: {auc_lr:.3f}  |  XGB-full: {auc_xgb:.3f}")

    thresholds = np.round(np.arange(0.01, 0.41, 0.01), 2)

    # Collect per-threshold stats for both models
    def sweep(scores):
        rows = []
        for t in thresholds:
            far = float((scores[pos_mask] < t).sum()) / n_pos * 100
            gc  = float((scores[neg_mask] < t).sum()) / n_neg * 100
            rows.append((t, far, gc))
        return rows

    rows_lr  = sweep(oof_lr)
    rows_xgb = sweep(oof_xgb)

    print(f"\n  {'T':>5}  {'LR false-ar':>11}  {'LR gc':>6}  | "
          f"{'XGB false-ar':>12}  {'XGB gc':>6}  {'Eff*':>6}  Note")
    print(f"  {'—'*5}  {'—'*11}  {'—'*6}  | {'—'*12}  {'—'*6}  {'—'*6}  {'—'*20}")

    prev_far_xgb = 0.0
    prev_gc_xgb  = 0.0

    for (t, far_lr, gc_lr), (_, far_xgb, gc_xgb) in zip(rows_lr, rows_xgb):
        # Efficiency: extra garbage caught per extra 1% false AR increase vs previous step
        d_far = far_xgb - prev_far_xgb
        d_gc  = gc_xgb  - prev_gc_xgb
        eff   = d_gc / d_far if d_far > 0.001 else float("inf")
        eff_s = f"{eff:5.1f}x" if np.isfinite(eff) else "  inf"

        note = ""
        if abs(t - 0.10) < 0.005: note = "<-- LR operating point"
        elif abs(t - 0.03) < 0.005: note = "<-- prev threshold"

        print(f"  {t:5.2f}  {far_lr:9.1f}%  {gc_lr:4.1f}%  | "
              f"{far_xgb:10.1f}%  {gc_xgb:4.1f}%  {eff_s}  {note}")

        prev_far_xgb = far_xgb
        prev_gc_xgb  = gc_xgb

    print(f"\n  * Eff = (extra %garbage caught) / (extra %false-AR) vs previous step.")
    print(f"    inf = free garbage (no extra false-AR cost). Threshold is optimal")
    print(f"    when Eff drops below 1.0 — paying more in false-AR than gained in garbage.")

    # Find the Pareto-optimal recommendation
    # Rule: maximize garbage caught subject to false_ar <= 2x LR's rate
    lr_far_at_010 = rows_lr[9][1]   # index 9 = threshold 0.10
    budget = lr_far_at_010 * 2      # allow up to 2x LR's false-AR as the outer limit

    # Also find the "free garbage" plateau (where Eff = inf)
    plateau_end = 0.10
    for (t, far_xgb, gc_xgb), (t2, far_xgb2, gc_xgb2) in zip(
            [(r[0], r[1], r[2]) for r in rows_xgb],
            [(r[0], r[1], r[2]) for r in rows_xgb[1:]]):
        if far_xgb2 > far_xgb + 0.001:
            plateau_end = t
            break

    # Best threshold at budget constraint
    best_t = best_gc = best_far = None
    for t, far, gc in rows_xgb:
        if far <= budget:
            if best_gc is None or gc > best_gc:
                best_t, best_gc, best_far = t, gc, far

    # Find the "efficiency knee": last threshold where Eff >= 2.0 before a sustained drop
    # (ignoring inf steps which are free)
    knee_t = None
    prev_far2 = 0.0
    prev_gc2  = 0.0
    for t, far, gc in rows_xgb:
        d_far = far - prev_far2
        d_gc  = gc  - prev_gc2
        eff   = d_gc / d_far if d_far > 0.001 else float("inf")
        if 0.001 < d_far and eff < 2.0:
            # First step where you pay >= 1% false-AR per 2% garbage gain
            knee_t = t
            break
        prev_far2 = far
        prev_gc2  = gc

    # Stats at key operating points
    def xgb_at(t):
        idx = round(t / 0.01) - 1
        if 0 <= idx < len(rows_xgb):
            return rows_xgb[idx][1], rows_xgb[idx][2]
        return None, None

    far_lr010, gc_lr010 = rows_lr[9][1], rows_lr[9][2]   # LR at 0.10

    print(f"\n  --- Recommendations ---")
    print(f"  LR@0.10 baseline:      {far_lr010:.1f}% false-AR,  {gc_lr010:.1f}% garbage caught")
    print(f"")
    for t_rec, label in [(0.10, "Conservative (matched false-AR)"),
                          (0.15, "Balanced"),
                          (0.18, "Aggressive (2x LR false-AR)")]:
        far, gc = xgb_at(t_rec)
        if far is not None:
            gain = gc / gc_lr010 if gc_lr010 > 0 else float("inf")
            print(f"  XGB@{t_rec:.2f} {label:<30}: "
                  f"{far:.1f}% false-AR,  {gc:.1f}% garbage  ({gain:.1f}x LR garbage)")
    if knee_t is not None:
        far_k, gc_k = xgb_at(knee_t)
        gain_k = gc_k / gc_lr010 if gc_lr010 > 0 else float("inf")
        print(f"  XGB@{knee_t:.2f} Efficiency knee (Eff drops <2x){'':<14}: "
              f"{far_k:.1f}% false-AR,  {gc_k:.1f}% garbage  ({gain_k:.1f}x LR garbage)")
    rec_t = knee_t if knee_t is not None else plateau_end
    far_rec, gc_rec = xgb_at(rec_t) if rec_t is not None else (None, None)
    print(f"\n  --- Computed (area-agnostic) ---")
    print(f"  Free-garbage plateau ends at: {plateau_end}")
    print(f"  Efficiency knee (Eff drops below 2x): {knee_t}")
    if far_rec is not None:
        print(f"  Suggested operating threshold ~ {rec_t:.2f}: "
              f"{far_rec:.1f}% false-AR, {gc_rec:.1f}% garbage caught.")
    print(f"  Rule of thumb: pick the highest threshold whose Eff stays >= ~2x and "
          f"false-AR is acceptable for the area.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("Loading training data with real weights...")
    records, agent_weight = load_all_records()
    ag = sum(1 for r in records if not r["is_bootstrap"])
    bs = sum(1 for r in records if r["is_bootstrap"])
    print(f"  {len(records)} sessions ({ag} agent, {bs} bootstrap), "
          f"agent_weight={agent_weight:.2f}x")

    run_loo_analysis(records)
    run_pending_analysis(records)
    run_threshold_sweep_real_weights(records)


if __name__ == "__main__":
    main()
