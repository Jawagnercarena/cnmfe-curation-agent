"""C10: the Step 4 training-data confound, tested empirically.

Historical backfill can never supply v2 for auto-rejected candidates. At
Step 4, training data must either:
  (a) exclude auto-rejected rows from agent training (what Step 2 did), or
  (b) include them with v2=0 + flag=0 — creating flag<->label correlation in
      agent history that INVERTS in production (all rows flag=1, real v2).

Arms (mixed corpus, rankv2_35 layout, 8 seeds, grouped CV, eval ALWAYS on the
same reviewed rows):
  a       agent train rows = reviewed only            (= Step 2 protocol)
  b       agent train rows = full set, autorej v2=0+flag=0
  b_nf    arm b without the flag column (autorej v2=0, no marker)

Diagnostics:
  1. flag<->label table among agent train rows in arm b.
  2. Flag-flip test on auto-rejected OOF rows: score(flag=0,v2=0) vs
     score(flag=1,v2=0) under arm-b fold models -> how much junkness is
     anchored on the flag itself?
  3. Auto-reject competence per arm: OOF scores of auto-rejected rows
     (production-like flag=1, v2=0 approximation) — fraction < 0.05 / < 0.12.
  4. Arm-b model's gain importance of the flag column.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

RT = Path(__file__).parent
sys.path.insert(0, str(RT))
warnings.simplefilter("ignore")
from redteam_lib import (SEEDS, MIN_AGENT_WEIGHT, load_pool, make_xgb,
                         spw_of, summarize, fmt)

t0 = time.time()
records = load_pool()

VKEY = "Xv2b" if (len(sys.argv) > 1 and sys.argv[1] == "v2b") else "Xv2"
OUT = "c10_oof_v2b.npz" if VKEY == "Xv2b" else "c10_oof.npz"
print(f"feature set: {VKEY}")

FLAG_COL = 34  # 26 + 8 = index of flag in the 35-col layout

ag_blocks, bs_X, bs_y, bs_w = [], [], [], []
for rec in records:
    X13 = rec["X13"]
    ranks = rankdata(X13, axis=0, method="average") / len(X13)
    X26 = np.hstack([X13, ranks])
    if rec["is_bootstrap"]:
        z = np.zeros((len(X13), 9))
        bs_X.append(np.hstack([X26, z]))
        bs_y.append(rec["y"]); bs_w.append(rec["w"])
        continue
    n = rec["n_cand"]
    V = np.zeros((n, 8))
    V[rec["ridx"]] = rec[VKEY]
    flag = np.zeros((n, 1))
    flag[rec["ridx"]] = 1.0
    Xfull = np.hstack([X26, V, flag])
    is_rev = np.zeros(n, dtype=bool)
    is_rev[rec["ridx"]] = True
    if rec["y"].sum() >= 5:
        ag_blocks.append((rec["name"], Xfull, rec["y"], is_rev))

X_ag = np.vstack([b[1] for b in ag_blocks])
y_ag = np.concatenate([b[2] for b in ag_blocks])
g_ag = np.concatenate([[i] * len(b[2]) for i, b in enumerate(ag_blocks)])
rev = np.concatenate([b[3] for b in ag_blocks])
X_bs, y_bs, w_bs = np.vstack(bs_X), np.concatenate(bs_y), np.concatenate(bs_w)

n_autorej = int((~rev).sum())
print(f"agent rows {len(y_ag)} (reviewed {int(rev.sum())}, autorej {n_autorej}); "
      f"bootstrap {len(y_bs)}")
tab = [((~rev) & (y_ag == 0)).sum(), ((~rev) & (y_ag == 1)).sum(),
       (rev & (y_ag == 0)).sum(), (rev & (y_ag == 1)).sum()]
print(f"flag<->label (agent train rows, arm b): flag0/junk {tab[0]}, "
      f"flag0/real {tab[1]}, flag1/junk {tab[2]}, flag1/real {tab[3]}")


def run_arm(train_mask, drop_flag=False, collect_flagflip=False):
    """OOF over agent rows; train folds restricted to train_mask (+bootstrap);
    eval on reviewed rows of test folds."""
    cols = [c for c in range(X_ag.shape[1]) if not (drop_flag and c == FLAG_COL)]
    Xa, Xb = X_ag[:, cols], X_bs[:, cols]
    oof = np.full((len(SEEDS), len(y_ag)), np.nan)
    flagflip = []
    gains = []
    for si, seed in enumerate(SEEDS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in cv.split(Xa, y_ag, g_ag):
            tr_idx = tr_idx[train_mask[tr_idx]]
            n_ag_tr = len(tr_idx)
            ag_w = max(np.sqrt(len(y_bs) / n_ag_tr), MIN_AGENT_WEIGHT)
            X_tr = np.vstack([Xa[tr_idx], Xb])
            y_tr = np.concatenate([y_ag[tr_idx], y_bs])
            w_tr = np.concatenate([np.full(n_ag_tr, ag_w), w_bs])
            sc = StandardScaler()
            clf = make_xgb(spw_of(y_tr, w_tr))
            clf.fit(sc.fit_transform(X_tr), y_tr, sample_weight=w_tr)
            oof[si, te_idx] = clf.predict_proba(sc.transform(Xa[te_idx]))[:, 1]
            if collect_flagflip and si == 0:
                ar = te_idx[~rev[te_idx]]
                if len(ar):
                    X0 = Xa[ar].copy()
                    X1 = Xa[ar].copy()
                    X1[:, cols.index(FLAG_COL)] = 1.0
                    s0 = clf.predict_proba(sc.transform(X0))[:, 1]
                    s1 = clf.predict_proba(sc.transform(X1))[:, 1]
                    flagflip.append(np.column_stack([s0, s1]))
                imp = clf.feature_importances_
                gains.append(imp[cols.index(FLAG_COL)] / imp.sum())
    return oof, (np.vstack(flagflip) if flagflip else None), gains


b13 = np.load(RT / "my_reviewed_b13_oof.npz", allow_pickle=True)
base_oof, base_y = b13["oof_seeds"], b13["y"]

# arm a: train reviewed-only (Step 2 protocol) — recompute here so the
# auto-rejected-row scoring uses the SAME models arm-b is compared with
oof_a, _, _ = run_arm(rev)
s_a = summarize(oof_a[:, rev], y_ag[rev], base_oof, base_y)
print(fmt("arm a (exclude autorej)", s_a))

oof_b, ff, gains_b = run_arm(np.ones(len(y_ag), dtype=bool),
                             collect_flagflip=True)
s_b = summarize(oof_b[:, rev], y_ag[rev], base_oof, base_y)
print(fmt("arm b (autorej v2=0+f0)", s_b))

oof_bnf, _, _ = run_arm(np.ones(len(y_ag), dtype=bool), drop_flag=True)
s_bnf = summarize(oof_bnf[:, rev], y_ag[rev], base_oof, base_y)
print(fmt("arm b_nf (no flag col)", s_bnf))

print(f"\nflag gain-importance in arm b (per fold, seed 42): "
      f"{np.array2string(np.array(gains_b), precision=3)}")
if ff is not None:
    d = ff[:, 1] - ff[:, 0]
    print(f"flag-flip on {len(ff)} autorej OOF rows (arm b, seed 42): "
          f"score(f=0) median {np.median(ff[:, 0]):.4f} -> "
          f"score(f=1) median {np.median(ff[:, 1]):.4f}; "
          f"shift mean {d.mean():+.4f}, p95 {np.percentile(d, 95):+.4f}, "
          f"max {d.max():+.4f}")
    print(f"  autorej rows scoring >=0.12 with f=0: {(ff[:, 0] >= 0.12).mean()*100:.1f}%  "
          f"with f=1: {(ff[:, 1] >= 0.12).mean()*100:.1f}%")

# auto-reject competence: OOF scores on autorej rows, production-like flag=1
print("\nauto-reject competence (OOF autorej rows, seed-mean):")
for tag, oof in (("arm a", oof_a), ("arm b", oof_b), ("arm b_nf", oof_bnf)):
    m = np.nanmean(oof, axis=0)[~rev]
    print(f"  {tag}: autorej rows n={len(m)}  <0.05: {(m < 0.05).mean()*100:.1f}%  "
          f"<0.12: {(m < 0.12).mean()*100:.1f}%  median {np.median(m):.4f}")
print("  (arm a never trained on these rows; all arms see v2=0 there — "
      "production would supply real v2, so this is an approximation)")

np.savez(RT / OUT, oof_a=oof_a, oof_b=oof_b, oof_bnf=oof_bnf,
         y=y_ag, rev=rev,
         session=np.concatenate([[b[0]] * len(b[2]) for b in ag_blocks]))
print(f"total {time.time() - t0:.0f}s")
