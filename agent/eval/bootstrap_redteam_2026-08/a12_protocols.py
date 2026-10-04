"""
a12_protocols.py -- attack #12: the vCA1 bootstrap-contribution sign flip.

Claims: train_classifier --eval (_bootstrap_roi_eval) reported agent-only
0.869 -> +bootstrap 0.878 (+0.009); diagnose_model.run_loo_analysis reported
XGB-ag 0.888 -> XGB-bs 0.877 (-0.011).  The two protocols differ in:
  fold structure   5-fold grouped (seed 42)      vs leave-one-session-out
  aggregation      mean of per-FOLD AUCs         vs mean of per-SESSION AUCs
  scaler           fit once on ALL rows          vs per fold
  agent weight     sqrt/4.0 per fold (both -- the override was not read)
Factorial re-run on the as-of-08-24 pool (reproduce both numbers first) and
on the pinned pool: protocol {5fold, LOO} x aggregation {pooled, per-unit
mean} x weight {sqrt, fixed 5.0, uniform 1.0} x scaler {global, per-fold}
x 8 seeds -> Cond B minus Cond A.  The deploy-relevant question is whether
"fixed 5.0 is not worse than sqrt on FAR@0.05 at matched junk" holds in
every protocol cell.  Read-only.
"""
import itertools
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import rt_lib as rt

INGESTED_0826 = {"6odorDualDiffRew/AVG5x-TSeries-061826-pnb97-679um-24z-000",
                 "6odorDualDiffRew/AVG5x-TSeries-061926-pnb97-610um-24z-000"}
PARKED = rt.DATA_ROOT["vCA1"] / ".excluded" / "3odor" / "AVG5x-TSeries-030426-pnb88-187um-35z-000"
CLAIMED = {"eval_A": 0.869, "eval_B": 0.878, "loo_ag": 0.888, "loo_bs": 0.877}


def run_cell(recs, protocol, weight, scaler_mode, seed):
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y_ag, g, rev, _, _ = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    n_bs = len(y_bs)
    if scaler_mode == "global":
        sc = StandardScaler().fit(np.vstack([r["X"] for r in recs]))
        X_ag_s, X_bs_s = sc.transform(X_ag), sc.transform(X_bs)
    else:
        X_ag_s, X_bs_s = X_ag, X_bs
    per_fold = {"A": [], "B": []}
    oofA = np.full(len(y_ag), np.nan)
    oofB = np.full(len(y_ag), np.nan)
    if protocol == "5fold":
        splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X_ag, y_ag, g))
        xgb_seed = 42
    else:
        splits = [(np.flatnonzero(g != i), np.flatnonzero(g == i)) for i in np.unique(g)]
        xgb_seed = seed
    for tr, te in splits:
        if len(np.unique(y_ag[te])) < 2:
            continue
        # Cond A: agent only, uniform
        sA = rt.fit_predict(X_ag_s[tr], y_ag[tr], np.ones(len(tr)), X_ag_s[te],
                            use_scaler=(scaler_mode == "perfold"), xgb_seed=xgb_seed)
        # Cond B: + bootstrap
        agw = {"sqrt": None, "fixed5": 5.0, "uniform": 1.0}[weight]
        agw = rt.fold_agent_weight(n_bs, len(tr), agw)
        sB = rt.fit_predict(np.vstack([X_ag_s[tr], X_bs_s]), np.concatenate([y_ag[tr], y_bs]),
                            np.concatenate([np.full(len(tr), agw), w_bs]), X_ag_s[te],
                            use_scaler=(scaler_mode == "perfold"), xgb_seed=xgb_seed)
        oofA[te], oofB[te] = sA, sB
        per_fold["A"].append(rt.auc(y_ag[te], sA))
        per_fold["B"].append(rt.auc(y_ag[te], sB))
    ok = ~np.isnan(oofA)
    return {"A_unit_mean": float(np.mean(per_fold["A"])), "B_unit_mean": float(np.mean(per_fold["B"])),
            "A_pooled": rt.auc(y_ag[ok], oofA[ok]), "B_pooled": rt.auc(y_ag[ok], oofB[ok]),
            "n_units": len(per_fold["A"]),
            "far_B_0.05": rt.far_junk(oofB[ok], y_ag[ok], 0.05)[0],
            "junk_B_0.05": rt.far_junk(oofB[ok], y_ag[ok], 0.05)[1]}, oofB[ok], y_ag[ok]


def factorial(recs, label, seeds):
    out = {}
    for protocol, weight, scaler_mode in itertools.product(("5fold", "LOO"), ("sqrt", "fixed5", "uniform"),
                                                           ("global", "perfold")):
        cells = [run_cell(recs, protocol, weight, scaler_mode, s)[0] for s in seeds]
        key = f"{protocol}|{weight}|{scaler_mode}"
        out[key] = {k: rt.summarize([c[k] for c in cells]) for k in cells[0]}
        out[key]["delta_unit_mean"] = rt.paired_delta([c["A_unit_mean"] for c in cells], [c["B_unit_mean"] for c in cells])
        out[key]["delta_pooled"] = rt.paired_delta([c["A_pooled"] for c in cells], [c["B_pooled"] for c in cells])
        rt.log(f"  [{label}] {key:22s}: unit-mean A {out[key]['A_unit_mean']['mean']:.4f} B {out[key]['B_unit_mean']['mean']:.4f} "
               f"({out[key]['delta_unit_mean']['mean']:+.4f}) | pooled A {out[key]['A_pooled']['mean']:.4f} B {out[key]['B_pooled']['mean']:.4f} "
               f"({out[key]['delta_pooled']['mean']:+.4f}) | FAR_B@0.05 {out[key]['far_B_0.05']['mean']:.2f}%")
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    res = {}
    recs_0824 = rt.load_pool("vCA1", width=13, extra_dirs=[PARKED] if PARKED.exists() else None,
                             exclude=INGESTED_0826)
    rt.log("[repro] the two claimed protocols on the as-of-08-24 pool, seed 42")
    ev = run_cell(recs_0824, "5fold", "sqrt", "global", 42)[0]
    loo = run_cell(recs_0824, "LOO", "sqrt", "perfold", 42)[0]
    res["repro"] = {"eval_protocol": ev, "loo_protocol": loo, "claimed": CLAIMED,
                    "eval_delta_repro": ev["B_unit_mean"] - ev["A_unit_mean"],
                    "loo_delta_repro": loo["B_unit_mean"] - loo["A_unit_mean"]}
    rt.log(f"  --eval protocol: A {ev['A_unit_mean']:.3f} B {ev['B_unit_mean']:.3f} (claimed 0.869 -> 0.878); "
           f"LOO protocol: ag {loo['A_unit_mean']:.3f} bs {loo['B_unit_mean']:.3f} (claimed 0.888 -> 0.877)")
    rt.log("[factorial] as-of-08-24 pool, 8 seeds")
    res["factorial_0824"] = factorial(recs_0824, "08-24", rt.SEEDS)
    recs = rt.load_pool("vCA1", width=13)
    rt.log("[factorial] pinned pool, 8 seeds")
    res["factorial_now"] = factorial(recs, "now", rt.SEEDS)
    # attribution: which factor flips the sign of the bootstrap contribution?
    f = res["factorial_0824"]
    signs = {k: np.sign(v["delta_unit_mean"]["mean"]) for k, v in f.items()}
    res["sign_by_cell_0824"] = {k: float(v) for k, v in signs.items()}
    # deploy-relevant: fixed5 vs sqrt on FAR@0.05 per cell (B condition), both pools
    dep = {}
    for pool_key in ("factorial_0824", "factorial_now"):
        for protocol in ("5fold", "LOO"):
            for scaler_mode in ("global", "perfold"):
                k5 = f"{protocol}|fixed5|{scaler_mode}"
                ks = f"{protocol}|sqrt|{scaler_mode}"
                dep[f"{pool_key}|{protocol}|{scaler_mode}"] = {
                    "far_fixed5": res[pool_key][k5]["far_B_0.05"]["mean"],
                    "far_sqrt": res[pool_key][ks]["far_B_0.05"]["mean"],
                    "auc_fixed5": res[pool_key][k5]["B_pooled"]["mean"],
                    "auc_sqrt": res[pool_key][ks]["B_pooled"]["mean"],
                    "fixed5_not_worse_far": res[pool_key][k5]["far_B_0.05"]["mean"] <= res[pool_key][ks]["far_B_0.05"]["mean"] + 0.1}
    res["deploy_relevant"] = dep
    checks = {
        "eval_repro_sign_positive": res["repro"]["eval_delta_repro"] > 0,
        "loo_repro_sign_negative": res["repro"]["loo_delta_repro"] < 0,
        "fixed5_not_worse_far_in_all_cells": all(v["fixed5_not_worse_far"] for v in dep.values()),
    }
    res["checks"] = checks
    res["verdict"] = "PASS" if checks["fixed5_not_worse_far_in_all_cells"] else "FAIL"
    res["criterion"] = ("the sign flip is attributed to named protocol factors (table); the deploy-relevant "
                        "comparison (fixed 5.0 not worse than sqrt on FAR@0.05) holds in every protocol cell")
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a12", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
