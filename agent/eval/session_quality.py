"""
Quality check on the near-chance early BLA sessions (option 2).

For each agent session, distinguish LABEL NOISE from DISTRIBUTION SHIFT:
  - within-session AUC : 5-fold CV trained ONLY on this session (LR, standardized).
      If the session's own features can separate its real/junk, the labels are
      internally coherent. If NOT, either labels are noise or data is degenerate.
  - cross-session AUC  : train on all OTHER agent sessions, test on this one.
      Low cross while high within = the features mean something different here
      (covariate shift) -> down-weight/normalize, NOT a labeling problem.
      Low both = suspect labels or unusable data.
  - feature health     : NaN/Inf, and max per-feature z-shift vs the corpus mean.
  - keep rate + auto-rejected count for context.
Baseline: a few good modern sessions for comparison.
"""
import sys, warnings
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
import diagnose_model as dm

TARGETS = ["bla2-695", "bla3-667", "bla4-751", "bla5-527", "bla3-665", "bla4-755"]
BASELINE = ["061826-bla37-209", "062626-bla37-200", "101725-bla16-283",
            "052826-bla37-216", "121225-bla12-652"]

records, aw = dm.load_all_records()
ag = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]

# feature names
fn = np.load(ag[0]["session_dir"] / "candidate_features.npz", allow_pickle=True)["feature_names"]
FEAT = [str(x) for x in fn]

X_all = np.vstack([r["X"] for r in ag]); y_all = np.concatenate([r["y"] for r in ag])
mu, sd = X_all.mean(0), X_all.std(0); sd[sd == 0] = 1.0

def within_auc(X, y):
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    k = min(5, n_pos, n_neg)
    if k < 2: return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)

def cross_auc(target_r):
    others = [r for r in ag if r is not target_r]
    Xtr = np.vstack([r["X"] for r in others]); ytr = np.concatenate([r["y"] for r in others])
    sc = StandardScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = dm.make_clf("xgb", float((ytr == 0).sum()) / max((ytr == 1).sum(), 1))
        clf.fit(sc.fit_transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(target_r["X"]))[:, 1]
    return roc_auc_score(target_r["y"], p)

def row(r, tag):
    X, y = r["X"], r["y"]
    npz = np.load(r["session_dir"] / "candidate_features.npz", allow_pickle=True)
    auto = int(np.asarray(npz["auto_rejected"]).sum()) if "auto_rejected" in npz else -1
    n = len(y); npos = int(y.sum())
    nan = int(np.isnan(X).sum() + np.isinf(X).sum())
    zsh = np.abs((X.mean(0) - mu) / sd)          # per-feature shift of THIS session
    jmax = int(np.argmax(zsh))
    wa = within_auc(X, y); ca = cross_auc(r)
    print(f"{tag:5} {r['name'].split('/')[-1][:30]:30} {n:5d}{npos:5d}{npos/n*100:5.1f}%"
          f"{auto:6d}{nan:5d}  {wa:5.2f}  {ca:5.2f}  {zsh.max():5.1f} {FEAT[jmax][:16]:16}")
    return wa, ca

print(f"agent sessions={len(ag)}  features={len(FEAT)}")
print("FEATURES:", ", ".join(FEAT))
print()
print(f"{'':5} {'session':30} {'tot':>5}{'real':>5}{'keep':>6}{'auto':>6}{'nan':>5}  "
      f"{'winAUC':>5}  {'xAUC':>5}  {'maxZ':>5} {'@feature':16}")
print("-" * 104)
print("SUSPECT (near-chance cross-session):")
sus = []
for r in ag:
    if any(t in r["name"] for t in TARGETS):
        sus.append(row(r, "SUS"))
print("\nBASELINE (good modern sessions):")
for r in ag:
    if any(b in r["name"] for b in BASELINE):
        row(r, "BASE")

print("\nInterpretation key:")
print("  winAUC LOW + xAUC LOW  -> labels not separable even within-session = suspect labels / bad data")
print("  winAUC HIGH + xAUC LOW -> covariate shift (features differ here) = down-weight, not a label problem")
