"""
eval_motion.py -- motion-separability go/no-go eval (handoff Part 1, Step 1.2).

This is the tool the motion-integration plan is gated on. Reviewers tag brain-
motion artifacts with (m); CNMFe_final_save writes a `motion_delete` vector into
labels.mat alongside the binary keep/delete `labels`. The trained classifier
still ignores motion_delete -- to it an (m) is just a (d). Before we change that,
we have to answer: is "motion" actually a separable signature in the features,
and is that separation robust across ANIMALS (not just an artifact of which
cohort happened to get tagged)?

It answers two questions, each with two CV schemes:

  Q1  motion-deletes vs KEEPS        -- can we flag them at all? (a floor; motion
                                        is a delete, so this should be easy)
  Q2  motion-deletes vs OTHER deletes -- the hard, useful one: is motion its own
                                        signature, or just "bad neuron"?

  CV schemes per question:
    - grouped-by-session (StratifiedGroupKFold): OPTIMISTIC. Because motion tags
      are concentrated in a few animals, a session split lets the model see the
      same animal in train and test, so a pure animal-detector scores high here.
    - leave-one-animal-out (LOAO): HONEST. Holding out an entire animal is the
      only way to tell motion-signal from cohort-signal. The leave-bla37-out row
      is decisive today (bla37 carries ~75% of BLA's tags).

Scope: uses the ACTIVE config's DATA_ROOT (BLA by default; vCA1 has zero tags).
Only the 13 currently-deployed features are on disk, so this measures whether
TODAY'S features already separate motion. The staged trace features
(pop_coherence / pop_sync_frac) are forward-only and cannot be backfilled to
these already-finalized sessions, so they are not evaluated here.

Read-only: loads candidate_features.npz + labels.mat, writes nothing.

Usage:
    python eval_motion.py
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))

import train_classifier as tc
from config import AREA, DATA_ROOT


def log(m=""):
    print(m, flush=True)


def animal_of(name: str) -> str:
    """Extract the animal id (e.g. bla37, bla12, pnb88) from a session dir name."""
    m = re.search(r"-[Tt][Ss]eries-[0-9]+-([A-Za-z]*\d+)", name)
    return m.group(1) if m else "?"


# ---------------------------------------------------------------------------
# Per-session loading: reconstruct BOTH keep/delete and motion labels onto the
# full candidate set, exactly as train_classifier.load_prospective_session does.
# ---------------------------------------------------------------------------

def load_motion_session(sd: Path):
    """Return (X, y_keep, y_motion, has_motion) or None.

    y_keep   : 1 = kept, 0 = deleted (via the validated prospective loader).
    y_motion : 1 = deleted specifically as a motion artifact, 0 = otherwise.
               Auto-rejected candidates (never shown to the reviewer) are 0.
    has_motion: whether labels.mat actually carries a motion_delete vector.
    """
    r = tc.load_prospective_session(sd)
    if r is None:
        return None
    X, y_keep = r
    N = len(X)

    ldata = sio.loadmat(str(sd / "labels.mat"))
    y_motion = np.zeros(N, dtype=float)
    has_motion = "motion_delete" in ldata
    if has_motion:
        md = ldata["motion_delete"].flatten().astype(float)
        if N == len(md):
            y_motion = md
        else:
            npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
            auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
            review_idx = [i for i in range(N) if i not in auto]
            if len(review_idx) == len(md):
                for k, idx in enumerate(review_idx):
                    y_motion[idx] = md[k]
            else:
                # Size bookkeeping did not line up; treat as no motion info
                # rather than risk mislabelling.
                has_motion = False
                y_motion = np.zeros(N, dtype=float)

    return X, y_keep.astype(int), y_motion.astype(int), has_motion


def collect_agent_sessions():
    """All agent (non-bootstrap) labeled sessions. Motion tags only ever appear
    in agent sessions, and their keep/delete labels are clean (bootstrap labels
    are ~41% noisy), so restricting here keeps the separability test clean."""
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
            if (sd / "bootstrap_match_stats.json").exists():
                continue  # agent sessions only
            r = load_motion_session(sd)
            if r is None:
                continue
            X, yk, ym, has_m = r
            recs.append({
                "X": X, "yk": yk, "ym": ym, "has_motion": has_m,
                "animal": animal_of(sd.name),
                "name": f"{td.name}/{sd.name}",
            })
    return recs


# ---------------------------------------------------------------------------
# Task assembly: motion (positive) vs a chosen negative pool.
# ---------------------------------------------------------------------------

def build_task(recs, which: str):
    """Stack a binary problem. which='Q1' -> negatives are keeps;
    which='Q2' -> negatives are non-motion deletes. Returns X, y, group, animal."""
    Xs, ys, gs, ans = [], [], [], []
    for gi, r in enumerate(recs):
        ym, yk = r["ym"], r["yk"]
        pos = ym == 1
        neg = (yk == 1) if which == "Q1" else ((yk == 0) & (ym == 0))
        sel = pos | neg
        if not sel.any():
            continue
        lab = np.zeros(int(sel.sum()), dtype=int)
        lab[pos[sel]] = 1
        Xs.append(r["X"][sel])
        ys.append(lab)
        gs.append(np.full(int(sel.sum()), gi))
        ans.append(np.array([r["animal"]] * int(sel.sum())))
    X = np.vstack(Xs)
    return X, np.concatenate(ys), np.concatenate(gs), np.concatenate(ans)


def _fit_predict(X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X_tr), y_tr)
        return clf.predict_proba(sc.transform(X_te))[:, 1]


def grouped_oof(X, y, g, n_splits=5):
    """Pooled out-of-fold scores from StratifiedGroupKFold (split by session)."""
    n_pos_groups = len(set(g[y == 1].tolist()))
    n_splits = max(2, min(n_splits, n_pos_groups))
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y, g):
        oof[te] = _fit_predict(X[tr], y[tr], X[te])
    valid = ~np.isnan(oof)
    auc = roc_auc_score(y[valid], oof[valid]) if len(np.unique(y[valid])) > 1 else float("nan")
    ap = average_precision_score(y[valid], oof[valid])
    return auc, ap, n_splits


def leave_one_animal_out(X, y, g, a):
    """For each animal that carries motion positives, train on all OTHER animals
    and test on the held-out animal. This defeats the cohort confound."""
    rows = []
    animals_with_pos = sorted(set(a[y == 1].tolist()))
    for held in animals_with_pos:
        te = a == held
        tr = ~te
        n_pos = int((y[te] == 1).sum())
        n_neg = int((y[te] == 0).sum())
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            rows.append((held, n_pos, n_neg, None, None))
            continue
        p = _fit_predict(X[tr], y[tr], X[te])
        rows.append((held, n_pos, n_neg,
                     roc_auc_score(y[te], p), average_precision_score(y[te], p)))
    return rows


def run_question(recs, which, title):
    X, y, g, a = build_task(recs, which)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    base = n_pos / len(y)
    log("\n" + "=" * 74)
    log(f"{which}: {title}")
    log("=" * 74)
    log(f"  positives (motion): {n_pos}   negatives: {n_neg}   "
        f"(motion base rate {base:.1%})")

    auc, ap, k = grouped_oof(X, y, g)
    log(f"\n  Grouped-by-session CV ({k}-fold, OPTIMISTIC -- animal can leak "
        f"across folds):")
    log(f"    ROC-AUC {auc:.3f}   PR-AUC {ap:.3f}  (chance PR-AUC = {base:.3f})")

    log("\n  Leave-one-animal-out (HONEST -- train excludes the test animal):")
    log(f"    {'held-out':<10} {'test pos':>8} {'test neg':>8}  {'ROC-AUC':>8}  {'PR-AUC':>7}")
    for held, tp, tn, auc_h, ap_h in leave_one_animal_out(X, y, g, a):
        if auc_h is None:
            log(f"    {held:<10} {tp:>8} {tn:>8}  {'  n/a':>8}  {'  n/a':>7}  (one class)")
        else:
            log(f"    {held:<10} {tp:>8} {tn:>8}  {auc_h:>8.3f}  {ap_h:>7.3f}")
    return X, y


def feature_importance_report(recs):
    X, y, _, _ = build_task(recs, "Q2")
    try:
        # Feature names from any session's npz (all share the 13-col schema).
        npz = np.load(str(next(DATA_ROOT.glob("*/*/candidate_features.npz"))),
                      allow_pickle=True)
        names = [str(n) for n in npz["feature_names"]]
    except Exception:
        names = [f"f{i}" for i in range(X.shape[1])]
    sc = StandardScaler().fit(X)
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X), y)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    log("\n" + "=" * 74)
    log("Feature importance for motion vs other deletes (Q2, full fit)")
    log("=" * 74)
    log("  (which of the 13 deployed features, if any, carry the motion signal)")
    for rank, i in enumerate(order, 1):
        mark = "  <-- hand-designed motion feature" if names[i] == "motion_correlation" else ""
        log(f"    {rank:>2}. {names[i]:<22} {imp[i]:.4f}{mark}")


def main():
    log("=" * 74)
    log(f"MOTION-SEPARABILITY EVAL  (area: {AREA})")
    log("=" * 74)

    recs = collect_agent_sessions()
    motion_recs = [r for r in recs if r["ym"].sum() > 0]
    total_tags = int(sum(r["ym"].sum() for r in recs))

    log(f"\nAgent sessions loaded: {len(recs)}")
    log(f"Sessions with motion tags: {len(motion_recs)}   total motion tags: {total_tags}")
    if total_tags == 0:
        log("\nNo motion tags in this area -- nothing to evaluate. "
            "(vCA1 has none; run with the BLA config.)")
        return

    by_animal = {}
    for r in motion_recs:
        by_animal[r["animal"]] = by_animal.get(r["animal"], 0) + int(r["ym"].sum())
    log("\nMotion tags by animal (the confound to beat):")
    for an, n in sorted(by_animal.items(), key=lambda x: -x[1]):
        log(f"    {an:<8} {n:>4}  ({n/total_tags:.0%})")

    run_question(recs, "Q1", "motion-deletes vs KEEPS  (floor: can we flag them at all)")
    run_question(recs, "Q2", "motion-deletes vs OTHER deletes  (is motion its own signature)")
    feature_importance_report(recs)

    log("\n" + "=" * 74)
    log("How to read this")
    log("=" * 74)
    log("  Q2 is the decision. If grouped-by-session AUC is high but the "
        "leave-bla37-out")
    log("  AUC collapses toward 0.5, the 'separation' is a cohort detector, not "
        "motion --")
    log("  hold for more animal spread / the forward-only trace features. If "
        "leave-one-")
    log("  animal-out stays materially above 0.5 (esp. bla37), motion is a real, "
        "portable")
    log("  signature and the deployed features already carry it (path a/b: "
        "feature-only")
    log("  + a motion upweight). PR-AUC vs its chance line shows the practical "
        "precision.")


if __name__ == "__main__":
    main()
