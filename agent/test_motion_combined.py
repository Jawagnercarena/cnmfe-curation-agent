"""
test_motion_combined.py -- best-case check for the new motion features. Stacks
the spatial-stability (motion_diag) and local-motion-vector (motion_vec) features
with the deployed 13, and also tests a WITHIN-SESSION-normalized variant (z-score
each new feature per session, to isolate 'this cell's spot moves more than its
session-mates' from 'this whole session is motion-heavy'). Q2/Q1, leave-one-
animal-out. Read-only.
"""
import re, sys, warnings
from pathlib import Path
import numpy as np, scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT_DIR = Path(__file__).parent; sys.path.insert(0, str(AGENT_DIR))
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


def log(m=""): print(m, flush=True)
def animal_of(nm):
    m = re.search(r"-[Tt][Ss]eries-[0-9]+-([A-Za-z]*\d+)", nm)
    return m.group(1) if m else "?"


def zscore_cols(M):
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0) + 1e-9
    return (M - mu) / sd


def load_session(task, nm):
    sd = DATA_ROOT / task / nm
    lab = sio.loadmat(str(sd / "labels.mat"))
    y = lab["labels"].flatten().astype(int); ym = lab["motion_delete"].flatten().astype(int)
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = npz["feature_matrix"].astype(float)
    auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
    ridx = [i for i in range(len(X)) if i not in auto]
    X13 = X[ridx]
    sp = sio.loadmat(str(sd / "motion_diag.mat"))["feats"].astype(float)
    mv = sio.loadmat(str(sd / "motion_vec.mat"))["feats"].astype(float)
    assert len(X13) == len(y) == sp.shape[0] == mv.shape[0], f"{nm}: align"
    new = np.hstack([sp, mv])                       # 13 new features
    return dict(X13=X13, new=new, new_z=zscore_cols(new),
                y=y, ym=ym, animal=animal_of(nm))


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
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42).split(X, y, g):
        oof[te] = fit_predict(X[tr], y[tr], X[te])
    v = ~np.isnan(oof)
    return roc_auc_score(y[v], oof[v])


def loao(X, y, a):
    out = {}
    for held in sorted(set(a[y == 1].tolist())):
        te = a == held; tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            out[held] = (int((y[te] == 1).sum()), None)
        else:
            out[held] = (int((y[te] == 1).sum()),
                         roc_auc_score(y[te], fit_predict(X[tr], y[tr], X[te])))
    return out


def build(recs, which, kind):
    imp_med = None
    if kind != "e13":
        stack = np.vstack([r["new" if kind == "raw" else "new_z"] for r in recs])
        imp_med = np.nanmedian(stack, axis=0)
    Xs, ys, gs, ans = [], [], [], []
    for gi, r in enumerate(recs):
        pos = r["ym"] == 1
        neg = (r["y"] == 1) if which == "Q1" else ((r["y"] == 0) & (r["ym"] == 0))
        sel = pos | neg
        if not sel.any(): continue
        if kind == "e13":
            M = r["X13"]
        else:
            nf = r["new" if kind == "raw" else "new_z"].copy()
            bad = np.where(np.isnan(nf)); nf[bad] = np.take(imp_med, bad[1])
            M = np.hstack([r["X13"], nf])
        lab = np.zeros(int(sel.sum()), dtype=int); lab[pos[sel]] = 1
        Xs.append(M[sel]); ys.append(lab)
        gs.append(np.full(int(sel.sum()), gi))
        ans.append(np.array([r["animal"]] * int(sel.sum())))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ans)


def main():
    recs = [load_session(t, n) for t, n in SESSIONS]
    log("=" * 82)
    log("BEST-CASE MOTION FEATURE STACK  (existing-13 + spatial-7 + motionvec-6)")
    log("=" * 82)
    for which, desc in [("Q2", "motion vs OTHER deletes (the decision)"),
                        ("Q1", "motion vs keeps")]:
        log(f"\n{which}: {desc}")
        log(f"  {'feature set':<28}{'grpAUC':>8}   leave-one-animal-out (n motion)")
        for kind, label in [("e13", "existing-13 (baseline)"),
                            ("raw", "+ all 13 new (raw)"),
                            ("z",   "+ all 13 new (within-session z)")]:
            X, y, g, a = build(recs, which, kind)
            ga = grouped_auc(X, y, g)
            lo = loao(X, y, a)
            s = "  ".join(f"{an}:{('%.3f'%v[1]) if v[1] is not None else 'n/a'}(n={v[0]})"
                         for an, v in sorted(lo.items(), key=lambda kv: -kv[1][0]))
            log(f"  {label:<28}{ga:>8.3f}   {s}")


if __name__ == "__main__":
    main()
