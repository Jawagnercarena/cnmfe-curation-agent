"""
Reviewer-quality / model-health check for the BLA model.

Question: is the recent false-AR rise coming from genuinely hard/dim neurons,
or from reviewers mislabeling (garbage labeled 'real', which both inflates
false-AR and poisons training)?

Method: multi-seed StratifiedGroupKFold OOF with the REAL deployed weighting,
averaged per candidate. Then a PER-SESSION breakdown:
  - false-AR@0.12 rate  (reviewer-called-real that the model rejects)
  - mean OOF prob on the session's real neurons  (calibration)
  - n_confident_wrong   (real-labeled scoring <0.02 -> model is SURE it's garbage)
The degradation signature is a FEW sessions with high false-AR AND many
confident-wrong reals. Hard-but-honest data shows false-AR spread thinly with
reals sitting just under the boundary (0.05-0.12), not deep in garbage.
"""
import sys, warnings, datetime as dt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
import diagnose_model as dm

T = 0.12                       # deployed operating threshold
CONF = 0.02                    # "model is sure it's garbage" cutoff
RECENT_AFTER = dt.datetime(2026, 7, 30)   # last threshold decision / commit
SEEDS = [42, 1, 7, 13, 100]

records, agent_weight = dm.load_all_records()
ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
bs_recs = [r for r in records if r["is_bootstrap"]]

# attach labels.mat mtime
for r in ag_recs:
    r["mtime"] = dt.datetime.fromtimestamp((r["session_dir"] / "labels.mat").stat().st_mtime)

X_ag = np.vstack([r["X"] for r in ag_recs])
y_ag = np.concatenate([r["y"] for r in ag_recs])
g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag_recs)])
X_bs = np.vstack([r["X"] for r in bs_recs])
y_bs = np.concatenate([r["y"] for r in bs_recs])
w_bs = np.concatenate([r["w"] for r in bs_recs])

# multi-seed OOF, averaged per candidate
oof_sum = np.zeros(len(y_ag)); oof_n = np.zeros(len(y_ag)); aucs = []
for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y_ag), np.nan)
    for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
        y_tr, X_tr, X_te = y_ag[tr_idx], X_ag[tr_idx], X_ag[te_idx]
        ag_w = float(max(np.sqrt(len(y_bs) / len(tr_idx)), dm.MIN_AGENT_WEIGHT))
        w_ag = np.ones(len(tr_idx)) * ag_w
        X_trC = np.vstack([X_tr, X_bs]); y_trC = np.concatenate([y_tr, y_bs])
        w_trC = np.concatenate([w_ag, w_bs])
        sc = StandardScaler(); X_trCs = sc.fit_transform(X_trC); X_teCs = sc.transform(X_te)
        clf = dm.make_clf("xgb", dm.compute_spw(y_trC, w_trC))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_trCs, y_trC, sample_weight=w_trC)
            oof[te_idx] = clf.predict_proba(X_teCs)[:, 1]
    aucs.append(roc_auc_score(y_ag, oof))
    oof_sum += oof; oof_n += 1
oof = oof_sum / oof_n

pos = y_ag == 1; neg = y_ag == 0
print(f"agent sessions={len(ag_recs)}  bootstrap={len(bs_recs)}  agent_weight={agent_weight:.2f}")
print(f"OOF pool: {len(y_ag)} cands, {pos.sum()} real, {neg.sum()} garbage")
print(f"OOF AUC (avg-prob) {np.mean(aucs):.3f}  seeds={SEEDS}")
print(f"Pool false-AR@{T}: {(oof[pos] < T).sum()}/{pos.sum()} = {(oof[pos] < T).mean()*100:.2f}%\n")

# ---- per-session table ----
rows = []
gstart = 0
for i, r in enumerate(ag_recs):
    n = len(r["y"]); sl = slice(gstart, gstart + n); gstart += n
    p = oof[sl]; yy = r["y"]
    reals = p[yy == 1]
    n_real = int((yy == 1).sum())
    far_n = int((reals < T).sum())
    conf_wrong = int((reals < CONF).sum())
    rows.append({
        "name": r["name"].split("/")[-1],
        "mtime": r["mtime"], "recent": r["mtime"] > RECENT_AFTER,
        "n_real": n_real, "n_gar": int((yy == 0).sum()),
        "far_n": far_n, "far_pct": far_n / n_real * 100,
        "mean_real_p": float(reals.mean()), "conf_wrong": conf_wrong,
    })

# global reference: where do real neurons sit?
print("Distribution of OOF prob on ALL real neurons (calibration):")
allreal = oof[pos]
for lo, hi in [(0,.02),(.02,.05),(.05,.12),(.12,.30),(.30,1.01)]:
    c = ((allreal >= lo) & (allreal < hi)).sum()
    print(f"   [{lo:.2f},{hi:.2f}) : {c:4d}  {c/len(allreal)*100:5.1f}%")
print()

print(f"{'session':<26}{'date':<11}{'real':>5}{'gar':>5}{'FAR@.12':>9}{'meanP':>7}{'sure-garb':>10}")
print("-" * 73)
for row in sorted(rows, key=lambda z: (-z["far_pct"], -z["conf_wrong"])):
    tag = " *NEW" if row["recent"] else ""
    print(f"{row['name']:<26}{row['mtime']:%Y-%m-%d} {row['n_real']:>5}{row['n_gar']:>5}"
          f"{row['far_n']:>4}/{row['far_pct']:>4.0f}%{row['mean_real_p']:>7.3f}"
          f"{row['conf_wrong']:>8}{tag}")

# ---- recent vs established aggregate ----
def agg(subset):
    tot_real = sum(z["n_real"] for z in subset)
    tot_far  = sum(z["far_n"] for z in subset)
    tot_cw   = sum(z["conf_wrong"] for z in subset)
    return tot_real, tot_far, tot_cw
rec = [z for z in rows if z["recent"]]; est = [z for z in rows if not z["recent"]]
rr, rf, rc = agg(rec); er, ef, ec = agg(est)
print("\n" + "=" * 73)
print("RECENT (since 2026-07-30) vs ESTABLISHED sessions:")
print(f"  recent      {len(rec):>2} sessions: {rf:>3}/{rr:<4} false-AR = {rf/max(rr,1)*100:5.2f}%   sure-garbage reals={rc}")
print(f"  established  {len(est):>2} sessions: {ef:>3}/{er:<4} false-AR = {ef/max(er,1)*100:5.2f}%   sure-garbage reals={ec}")

# ---- counterfactual: false-AR with recent sessions REMOVED from pool ----
print("\nCounterfactual pool false-AR@.12 (this OOF, just excluding recent rows from the tally):")
est_mask = np.concatenate([[not r["mtime"] > RECENT_AFTER] * len(r["y"]) for r in ag_recs])
ep = oof[pos & est_mask]
print(f"  established-only reals: {(ep < T).sum()}/{len(ep)} = {(ep < T).mean()*100:.2f}%")
