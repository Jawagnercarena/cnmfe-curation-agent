"""
test_motion_vectors.py -- does LOCAL MOTION measured directly from the raw movie
at each candidate's patch (motion_vec.mat, from extract_motion_vectors.m) separate
motion artifacts from real cells?

This is the one approach that does NOT rely on CNMF-E's extracted footprint/trace:
it estimates per-frame in-plane displacement (Lucas-Kanade) in the patch around
each outline and summarises how much the location physically moves and whether
that motion tracks the cell's activity. Scored identically to the other tests:
existing-13 vs motionvec-6 vs combined-19, Q1/Q2, grouped CV + leave-one-animal-out.

Read-only. Run after run_motion_vec.m has produced motion_vec.mat for all 8.
    python test_motion_vectors.py
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
    md = sio.loadmat(str(sd / "motion_vec.mat"))
    mvfeats = md["feats"].astype(float)
    mvnames = [str(x[0]) for x in md["feature_names"][0]]

    lab = sio.loadmat(str(sd / "labels.mat"))
    y  = lab["labels"].flatten().astype(int)
    ym = lab["motion_delete"].flatten().astype(int)

    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = npz["feature_matrix"].astype(float)
    auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
    ridx = [i for i in range(len(X)) if i not in auto]
    X13 = X[ridx]
    enames = [str(n) for n in npz["feature_names"]]

    assert len(X13) == len(y) == mvfeats.shape[0], \
        f"{nm}: align mismatch X13={len(X13)} y={len(y)} mv={mvfeats.shape[0]}"
    return dict(X13=X13, mv=mvfeats, y=y, ym=ym, animal=animal_of(nm),
                enames=enames, mvnames=mvnames, name=f"{task}/{nm}")


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]


def grouped_auc(X, y, g):
    k = max(2, min(5, len(set(g[y == 1].tolist()))))
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
        else:
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
        elif kind == "mv":
            M = r["mv_imp"]
        else:
            M = np.hstack([r["X13"], r["mv_imp"]])
        lab = np.zeros(int(sel.sum()), dtype=int)
        lab[pos[sel]] = 1
        Xs.append(M[sel]); ys.append(lab)
        gs.append(np.full(int(sel.sum()), gi))
        ans.append(np.array([r["animal"]] * int(sel.sum())))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ans)


def main():
    recs = [load_session(t, n) for t, n in SESSIONS]
    log("=" * 78)
    log(f"LOCAL-MOTION-VECTOR MOTION TEST  ({len(recs)} sessions)")
    log("=" * 78)

    allmv = np.vstack([r["mv"] for r in recs])
    med = np.nanmedian(allmv, axis=0)
    for r in recs:
        m = r["mv"].copy()
        bad = np.where(np.isnan(m))
        m[bad] = np.take(med, bad[1])
        r["mv_imp"] = m

    mvnames = recs[0]["mvnames"]
    n_motion = int(sum(r["ym"].sum() for r in recs))
    by_an = {}
    for r in recs:
        by_an[r["animal"]] = by_an.get(r["animal"], 0) + int(r["ym"].sum())
    log(f"motion cells: {n_motion}   by animal: "
        + ", ".join(f"{a}={n}" for a, n in sorted(by_an.items(), key=lambda x: -x[1])))

    mv_all = np.vstack([r["mv_imp"] for r in recs])
    y_all  = np.concatenate([r["y"]  for r in recs])
    ym_all = np.concatenate([r["ym"] for r in recs])
    keep = y_all == 1
    od   = (y_all == 0) & (ym_all == 0)
    mo   = ym_all == 1
    log("\n" + "-" * 78)
    log("Local-motion group means  (do motion cells sit in higher-motion spots?)")
    log("-" * 78)
    log(f"  {'feature':<18}{'keep':>9}{'other-del':>11}{'MOTION':>9}")
    for j, s in enumerate(mvnames):
        log(f"  {s:<18}{mv_all[keep,j].mean():>9.3f}"
            f"{mv_all[od,j].mean():>11.3f}{mv_all[mo,j].mean():>9.3f}")

    for which, desc in [("Q2", "motion vs OTHER deletes  (the decision)"),
                        ("Q1", "motion vs keeps  (floor)")]:
        log("\n" + "=" * 78)
        log(f"{which}: {desc}")
        log("=" * 78)
        log(f"  {'feature set':<14}{'grouped AUC':>12}   leave-one-animal-out AUC (n motion)")
        for kind, label in [("e13", "existing-13"), ("mv", "motionvec-6"),
                            ("c19", "combined-19")]:
            X, y, g, a = build(recs, which, kind)
            gauc = grouped_auc(X, y, g)
            lo = loao(X, y, a)
            lo_s = "  ".join(
                f"{an}:{('%.3f' % v[1]) if v[1] is not None else ' n/a '}(n={v[0]})"
                for an, v in sorted(lo.items(), key=lambda kv: -kv[1][0]))
            log(f"  {label:<14}{gauc:>12.3f}   {lo_s}")

    X, y, g, a = build(recs, "Q2", "c19")
    names = recs[0]["enames"] + mvnames
    sc = StandardScaler().fit(X)
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X), y)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    log("\n" + "=" * 78)
    log("Combined-19 feature importance (Q2 full fit) -- top 10")
    log("=" * 78)
    for rank, i in enumerate(order[:10], 1):
        tag = "  [MOTION-VEC]" if names[i] in mvnames else ""
        log(f"  {rank:>2}. {names[i]:<18} {imp[i]:.4f}{tag}")


if __name__ == "__main__":
    main()
