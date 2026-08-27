"""
a08_animal_cv.py -- attack #8: animal / FOV leakage.

The harnesses group CV folds by SESSION.  Bootstrap sessions from the same
animal as an agent test session (BLA: bla8 x3, bla16 x2, bla12 x1) sit in the
training fold, and agent sessions of the same animal (bla37 x20, bla36 x19 ...)
straddle folds.  This re-runs the deployed-contract A/B under
  session grouping  (must reproduce the pins), and
  animal grouping   (BLA: StratifiedGroupKFold(5) over the 6 agent animals;
                     vCA1: 2-animal leave-one-animal-out, xgb seed varied)
with, under animal grouping, the bootstrap sessions of the test-fold animals
DROPPED from training.  Reports side by side, 8 seeds, per variant: AUC
full / reviewed, FAR + junk at the deployed T, rule-chosen T, and the
bootstrap-contribution delta (Cond A agent-only vs Cond B +bootstrap).
Variants: BLA b13 + rankv2b_35 (live files); vCA1 b13 (live, the deployed
13-col) + rankv2b_35 arm b0 (the pending deploy).  Writes fixtures/a08_*.npz
for attack #10.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

VCA1_ARMS = rt.EXT["vCA1"] / "_arms"


def vca1_ff(sd):
    if rt.is_bootstrap(sd):
        return VCA1_ARMS / f"{rt.key(rt.rel(sd))}__b0.npz"
    return sd / "candidate_features_v2.npz"


def run_area(area, pool_kw, variants, T, override):
    recs = rt.load_pool(area, **pool_kw)
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g_sess, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    bs_animal = np.concatenate([[r["animal"] or "?"] * len(r["y"]) for r in bs])
    an_ids = {a: i for i, a in enumerate(sorted(set(animals)))}
    g_anim = np.array([an_ids[a] for a in animals])
    n_anim = len(an_ids)
    out = {"n_cv_sessions": len(cv), "n_rows": int(len(y)), "n_real": int(y.sum()),
           "animals": {a: int((animals == a).sum()) for a in an_ids},
           "bootstrap_rows_by_cv_animal": {a: int((bs_animal == a).sum()) for a in an_ids}}
    fixtures = {"y": y, "reviewed": rev, "animals": animals, "groups_session": g_sess, "groups_animal": g_anim,
                "names": names}
    for grouping in ("session", "animal"):
        gout = {}
        for vname, sl in variants.items():
            oof_B = np.full((len(rt.SEEDS), len(y)), np.nan)
            oof_A = np.full((len(rt.SEEDS), len(y)), np.nan)
            for si, seed in enumerate(rt.SEEDS):
                if grouping == "session":
                    oof_B[si] = rt.run_oof(X_ag[:, sl], y, g_sess, X_bs[:, sl], y_bs, w_bs, seed,
                                           agent_weight=override)
                    oof_A[si] = rt.run_oof(X_ag[:, sl], y, g_sess, X_bs[:0, sl], y_bs[:0], w_bs[:0], seed,
                                           agent_weight=1.0)
                else:
                    if n_anim >= 3:
                        def mask_fn(te, _an=animals, _bsan=bs_animal):
                            return ~np.isin(_bsan, np.unique(_an[te]))
                        oof_B[si] = rt.run_oof(X_ag[:, sl], y, g_anim, X_bs[:, sl], y_bs, w_bs, seed,
                                               agent_weight=override, n_splits=min(5, n_anim),
                                               bs_train_mask=mask_fn)
                        oof_A[si] = rt.run_oof(X_ag[:, sl], y, g_anim, X_bs[:0, sl], y_bs[:0], w_bs[:0], seed,
                                               agent_weight=1.0, n_splits=min(5, n_anim))
                    else:   # 2 animals: deterministic LOAO, seed varies the fit
                        for a, ai in an_ids.items():
                            te = g_anim == ai
                            tr = ~te
                            keep_bs = ~np.isin(bs_animal, [a])
                            agw = rt.fold_agent_weight(int(keep_bs.sum()), int(tr.sum()), override)
                            X_tr = np.vstack([X_ag[tr][:, sl], X_bs[keep_bs][:, sl]])
                            y_tr = np.concatenate([y[tr], y_bs[keep_bs]])
                            w_tr = np.concatenate([np.full(int(tr.sum()), agw), w_bs[keep_bs]])
                            oof_B[si, te] = rt.fit_predict(X_tr, y_tr, w_tr, X_ag[te][:, sl], xgb_seed=seed)
                            oof_A[si, te] = rt.fit_predict(X_ag[tr][:, sl], y[tr], np.ones(int(tr.sum())),
                                                           X_ag[te][:, sl], xgb_seed=seed)
            fixtures[f"oof_{grouping}_{vname}"] = oof_B
            fixtures[f"oof_{grouping}_{vname}_agentonly"] = oof_A
            aucs_full = [rt.auc(y, o) for o in oof_B]
            aucs_rev = [rt.auc(y, o, rev) for o in oof_B]
            aucs_A = [rt.auc(y, o) for o in oof_A]
            aucs_A_rev = [rt.auc(y, o, rev) for o in oof_A]
            fj = [rt.far_junk(o, y, T, rev) for o in oof_B]
            tab = rt.threshold_table(oof_B, y, rev)
            cT, gate, det = rt.rule_T(tab)
            per_animal = {}
            for a, ai in an_ids.items():
                m = g_anim == ai
                if len(np.unique(y[m])) < 2:
                    continue
                per_animal[a] = {"auc": rt.summarize([rt.auc(y[m], o[m]) for o in oof_B]),
                                 "far_at_T": rt.summarize([rt.far_junk(o[m], y[m], T)[0] for o in oof_B])}
            gout[vname] = {
                "auc_full": rt.summarize(aucs_full), "auc_reviewed": rt.summarize(aucs_rev),
                "auc_full_agentonly": rt.summarize(aucs_A), "auc_reviewed_agentonly": rt.summarize(aucs_A_rev),
                "bootstrap_contribution_full": rt.paired_delta(aucs_A, aucs_full),
                "bootstrap_contribution_reviewed": rt.paired_delta(aucs_A_rev, aucs_rev),
                "far_at_T": rt.summarize([x[0] for x in fj]), "junk_full_at_T": rt.summarize([x[1] for x in fj]),
                "junk_reviewed_at_T": rt.summarize([x[2] for x in fj]),
                "rule_T": cT, "rule_gate": gate, "rule_detail": det, "threshold_table": tab,
                "per_animal": per_animal, "T": T,
            }
            rt.log(f"  {area} [{grouping:7s}] {vname:11s}: AUC full {np.mean(aucs_full):.4f}+/-{np.std(aucs_full):.4f} "
                   f"rev {np.mean(aucs_rev):.4f} | agent-only {np.mean(aucs_A):.4f} (bs delta {np.mean(aucs_full)-np.mean(aucs_A):+.4f}) "
                   f"| T={T}: FAR {np.mean([x[0] for x in fj]):.2f}% (max {max(x[0] for x in fj):.2f}) junk {np.mean([x[1] for x in fj]):.1f}% "
                   f"| rule T {cT}", )
        out[grouping] = gout
    np.savez(rt.FIXTURES / f"a08_{area}.npz", **fixtures)
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    rt.FIXTURES.mkdir(exist_ok=True)
    res = {"areas": {}}
    rt.log("BLA (live 35-col files)")
    res["areas"]["BLA"] = run_area("BLA", {"width": 35},
                                   {"b13": slice(0, 13), "rankv2b_35": slice(0, 35)}, rt.DEPLOYED_T["BLA"], None)
    rt.log("vCA1 (13-col live = deployed; rankv2b_35 = arm b0 parallel files)")
    res["areas"]["vCA1_13col"] = run_area("vCA1", {"width": 13}, {"b13": slice(0, 13)}, rt.DEPLOYED_T["vCA1"],
                                          rt.AGENT_WEIGHT_OVERRIDE["vCA1"])
    res["areas"]["vCA1_v2_b0"] = run_area("vCA1", {"width": 35, "feature_file_of": vca1_ff},
                                          {"b13": slice(0, 13), "rankv2b_35": slice(0, 35)}, rt.DEPLOYED_T["vCA1"],
                                          rt.AGENT_WEIGHT_OVERRIDE["vCA1"])
    # decision checks
    b = res["areas"]["BLA"]
    v = res["areas"]["vCA1_13col"]
    v2 = res["areas"]["vCA1_v2_b0"]
    checks = {
        "bla_session_reproduces_pin": abs(b["session"]["rankv2b_35"]["auc_full"]["mean"] - 0.9283) < 5e-4,
        "bla_v2b_beats_b13_under_animal": b["animal"]["rankv2b_35"]["auc_reviewed"]["mean"] > b["animal"]["b13"]["auc_reviewed"]["mean"],
        "bla_far_at_0.04_worst_seed_le_1pct_under_animal": b["animal"]["rankv2b_35"]["far_at_T"]["max"] <= 1.0,
        "bla_bootstrap_contribution_sign_stable": np.sign(b["animal"]["rankv2b_35"]["bootstrap_contribution_full"]["mean"]) == np.sign(b["session"]["rankv2b_35"]["bootstrap_contribution_full"]["mean"]) or abs(b["animal"]["rankv2b_35"]["bootstrap_contribution_full"]["mean"]) < 0.005,
        "vca1_13col_far_at_0.05_worst_seed_le_1pct_under_animal": v["animal"]["b13"]["far_at_T"]["max"] <= 1.0,
        "vca1_v2_b0_beats_b13_under_animal": v2["animal"]["rankv2b_35"]["auc_reviewed"]["mean"] > v2["animal"]["b13"]["auc_reviewed"]["mean"],
        "vca1_v2_b0_far_at_0.05_worst_seed_le_1pct_under_animal": v2["animal"]["rankv2b_35"]["far_at_T"]["max"] <= 1.0,
    }
    res["checks"] = {k: bool(v_) for k, v_ in checks.items()}
    res["deflation"] = {
        "BLA_rankv2b_35_auc_full": b["session"]["rankv2b_35"]["auc_full"]["mean"] - b["animal"]["rankv2b_35"]["auc_full"]["mean"],
        "vCA1_b13_auc_full": v["session"]["b13"]["auc_full"]["mean"] - v["animal"]["b13"]["auc_full"]["mean"],
        "vCA1_v2_b0_auc_full": v2["session"]["rankv2b_35"]["auc_full"]["mean"] - v2["animal"]["rankv2b_35"]["auc_full"]["mean"],
    }
    # verdict: the deploy decisions survive if the paired signs hold; FAR ceilings are reported, absolute deflation is a number
    ranking = ["bla_session_reproduces_pin", "bla_v2b_beats_b13_under_animal", "vca1_v2_b0_beats_b13_under_animal"]
    ceilings = ["bla_far_at_0.04_worst_seed_le_1pct_under_animal",
                "vca1_13col_far_at_0.05_worst_seed_le_1pct_under_animal",
                "vca1_v2_b0_far_at_0.05_worst_seed_le_1pct_under_animal"]
    res["ranking_decisions_survive"] = all(res["checks"][k] for k in ranking)
    res["threshold_ceilings_hold_under_animal_grouping"] = all(res["checks"][k] for k in ceilings)
    res["verdict"] = "PASS" if res["ranking_decisions_survive"] and res["threshold_ceilings_hold_under_animal_grouping"] else "FAIL"
    res["criterion"] = ("paired signs (v2 > b13; bootstrap contribution) hold under animal grouping with the "
                        "test animals' bootstrap sessions dropped; FAR ceilings at the deployed T reported per grouping; "
                        "absolute AUC deflation reported, not judged")
    rt.log(f"checks {res['checks']}\ndeflation {res['deflation']}\nVERDICT {res['verdict']}")
    rt.write_result("a08", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
