"""
test_spatial_stability.py -- does the backfilled footprint-vs-video spatial-
stability feature (motion_diag.mat, from extract_motion_diag.m) separate motion
artifacts from real cells where the deployed trace/shape features could not?

Restricted to the 8 motion-tagged BLA sessions -- the only ones where the feature
could be backfilled from retained movies + candidate footprints. Within those
sessions it compares three feature sets on motion separability:
    existing-13  the deployed features, subset to the reviewed candidates
    spatial-7    the new motion_diag stability features
    combined-20  both
for Q1 (motion vs keeps) and Q2 (motion vs OTHER deletes), each with grouped-by-
session CV (optimistic) and leave-one-animal-out (the honest, cohort-robust test;
bla37 is the decisive hold-out at 75% of tags).

Read-only. Run after run_motion_diag.m has produced motion_diag.mat for all 8.

    python test_spatial_stability.py
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))
import train_classifier as tc
from config import DATA_ROOT

SESSIONS = [
    ("6odorDualDiffRew", "AVG5x-TSeries-061226-bla37-213um-37z-000"),
    ("Block_Valence",    "AVG5x-TSeries-070226-bla37-262um-37z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-061126-bla37-277um-35z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-060426-bla37-275um-35z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-052026-bla36-669um-29z-000"),
    ("2tones",           "AVG5x-TSeries-101525-bla12-660um-23z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-052826-bla37-216um-37z-000"),
    ("2tones",           "AVG5x-TSeries-101525-bla16-278um-36z-000"),
]


def log(m=""):
    print(m, flush=True)


def animal_of(nm):
    m = re.search(r"-[Tt][Ss]eries-[0-9]+-([A-Za-z]*\d+)", nm)
    return m.group(1) if m else "?"


def load_session(task, nm):
    sd = DATA_ROOT / task / nm
    md = sio.loadmat(str(sd / "motion_diag.mat"))
    feats = md["feats"].astype(float)
    snames = [str(x[0]) for x in md["feature_names"][0]]

    lab = sio.loadmat(str(sd / "labels.mat"))
    y  = lab["labels"].flatten().astype(int)
    ym = lab["motion_delete"].flatten().astype(int)

    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = npz["feature_matrix"].astype(float)
    auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
    ridx = [i for i in range(len(X)) if i not in auto]
    X13 = X[ridx]
    enames = [str(n) for n in npz["feature_names"]]

    assert len(X13) == len(y) == feats.shape[0], \
        f"{nm}: align mismatch X13={len(X13)} y={len(y)} sp={feats.shape[0]}"
    return dict(X13=X13, sp=feats, y=y, ym=ym, animal=animal_of(nm),
                enames=enames, snames=snames, name=f"{task}/{nm}")


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]


def grouped_auc(X, y, g):
    n_pos_groups = len(set(g[y == 1].tolist()))
    k = max(2, min(5, n_pos_groups))
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y, g):
        oof[te] = fit_predict(X[tr], y[tr], X[te])
    v = ~np.isnan(oof)
    return roc_auc_score(y[v], oof[v]) if len(np.unique(y[v])) > 1 else float("nan")


def loao(X, y, a):
    out = {}
    for held in sorted(set(a[y == 1].tolist())):
        te = a == held
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            out[held] = (int((y[te] == 1).sum()), None)
            continue
        p = fit_predict(X[tr], y[tr], X[te])
        out[held] = (int((y[te] == 1).sum()), roc_auc_score(y[te], p))
    return out


def build(recs, which, kind):
    Xs, ys, gs, ans = [], [], [], []
    for gi, r in enumerate(recs):
        pos = r["ym"] == 1
        neg = (r["y"] == 1) if which == "Q1" else ((r["y"] == 0) & (r["ym"] == 0))
        sel = pos | neg
        if not sel.any():
            continue
        if kind == "e13":
            M = r["X13"]
        elif kind == "sp7":
            M = r["sp_imp"]
        else:
            M = np.hstack([r["X13"], r["sp_imp"]])
        lab = np.zeros(int(sel.sum()), dtype=int)
        lab[pos[sel]] = 1
        Xs.append(M[sel]); ys.append(lab)
        gs.append(np.full(int(sel.sum()), gi))
        ans.append(np.array([r["animal"]] * int(sel.sum())))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ans)


def main():
    recs = [load_session(t, n) for t, n in SESSIONS
            if (DATA_ROOT / t / n / "motion_diag.mat").exists()]
    log("=" * 78)
    log(f"SPATIAL-STABILITY MOTION TEST  ({len(recs)}/8 sessions with motion_diag)")
    log("=" * 78)

    # Impute spatial NaNs (unscored candidates) with global column medians.
    allsp = np.vstack([r["sp"] for r in recs])
    med = np.nanmedian(allsp, axis=0)
    for r in recs:
        sp = r["sp"].copy()
        bad = np.where(np.isnan(sp))
        sp[bad] = np.take(med, bad[1])
        r["sp_imp"] = sp

    snames = recs[0]["snames"]
    n_motion = int(sum(r["ym"].sum() for r in recs))
    by_animal = {}
    for r in recs:
        by_animal[r["animal"]] = by_animal.get(r["animal"], 0) + int(r["ym"].sum())
    log(f"motion cells: {n_motion}   by animal: "
        + ", ".join(f"{a}={n}" for a, n in sorted(by_animal.items(), key=lambda x: -x[1])))

    # ---- Per-feature group means (spatial features) ----
    log("\n" + "-" * 78)
    log("Spatial-feature group means  (do motion cells look different?)")
    log("-" * 78)
    sp_all = np.vstack([r["sp_imp"] for r in recs])
    y_all  = np.concatenate([r["y"]  for r in recs])
    ym_all = np.concatenate([r["ym"] for r in recs])
    keep = y_all == 1
    od   = (y_all == 0) & (ym_all == 0)
    mo   = ym_all == 1
    log(f"  {'feature':<16}{'keep':>9}{'other-del':>11}{'MOTION':>9}")
    for j, s in enumerate(snames):
        log(f"  {s:<16}{sp_all[keep,j].mean():>9.3f}"
            f"{sp_all[od,j].mean():>11.3f}{sp_all[mo,j].mean():>9.3f}")

    # ---- Separability comparison ----
    for which, desc in [("Q2", "motion vs OTHER deletes  (the decision)"),
                        ("Q1", "motion vs keeps  (floor)")]:
        log("\n" + "=" * 78)
        log(f"{which}: {desc}")
        log("=" * 78)
        log(f"  {'feature set':<14}{'grouped AUC':>12}   leave-one-animal-out AUC "
            f"(n motion)")
        for kind, label in [("e13", "existing-13"), ("sp7", "spatial-7"),
                            ("c20", "combined-20")]:
            X, y, g, a = build(recs, which, kind)
            gauc = grouped_auc(X, y, g)
            lo = loao(X, y, a)
            lo_s = "  ".join(
                f"{an}:{('%.3f' % v[1]) if v[1] is not None else ' n/a '}(n={v[0]})"
                for an, v in sorted(lo.items(), key=lambda kv: -kv[1][0]))
            log(f"  {label:<14}{gauc:>12.3f}   {lo_s}")

    # ---- Does the model use the spatial features? ----
    X, y, g, a = build(recs, "Q2", "c20")
    names = recs[0]["enames"] + snames
    sc = StandardScaler().fit(X)
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X), y)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    log("\n" + "=" * 78)
    log("Combined-20 feature importance (Q2 full fit) -- top 10")
    log("=" * 78)
    for rank, i in enumerate(order[:10], 1):
        tag = "  [SPATIAL]" if names[i] in snames else ""
        log(f"  {rank:>2}. {names[i]:<18} {imp[i]:.4f}{tag}")

    log("\n" + "=" * 78)
    log("Verdict guide")
    log("=" * 78)
    log("  If combined-20 leave-bla37-out Q2 AUC clearly beats existing-13's, and")
    log("  spatial features rank high in importance, the feature adds real, animal-")
    log("  portable motion signal -> worth the forward-only deploy. If it matches")
    log("  existing-13 (~0.66) or the spatial group means barely differ for motion,")
    log("  these artifacts are not spatially unstable in a way this feature catches.")


if __name__ == "__main__":
    main()
