"""
a09_seeds.py -- attack #9: the 3-seed decisions, re-run at 8 seeds.

(i)  vCA1 agent-weight sweep {1, 2, 3.5, 5, 7.01}: reproduce the 3-seed log
     (c3_vca1_weight_sweep.log: w=5.0 AUC 0.8857, FAR@0.05 1.2%; w=7.01
     0.8843, 2.5%) on the AS-OF-08-24 pool (22 agent = today's 23 minus the
     two pnb97 sessions ingested 08-26 12:16, plus the parked
     .excluded/3odor/030426 read explicitly) with that script's exact
     protocol (no scaler, weight-0 rows dropped, ALL non-test sessions in
     train, seeds 42/43/44, pooled test scores), then 8 canonical seeds on the
     pinned pool.  Decision test: FAR@0.05(7.01) - FAR@0.05(5.0) > 0 on
     >= 7/8 seeds and > 2 SE; plus rule-chosen T and junk at matched FAR.
(ii) duplicate handling masked / label-0 / label-1 at 8 seeds, both areas,
     deployed weights (c3_assumption_checks --mode dups was 3 seeds).
Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

INGESTED_0826 = {"6odorDualDiffRew/AVG5x-TSeries-061826-pnb97-679um-24z-000",
                 "6odorDualDiffRew/AVG5x-TSeries-061926-pnb97-610um-24z-000"}
PARKED = rt.DATA_ROOT["vCA1"] / ".excluded" / "3odor" / "AVG5x-TSeries-030426-pnb88-187um-35z-000"
WEIGHTS = [1.0, 2.0, 3.5, 5.0, 7.01]
T_LIST = [0.05, 0.07]


def sweep_protocol(recs, w_agent, seed, use_scaler=False, thresholds=T_LIST):
    """c3_vca1_weight_sweep.py's loop: every non-test session trains (agent
    sessions with < 5 positives included), weight-0 rows dropped, no scaler."""
    ag_ok = [i for i, r in enumerate(recs) if not r["is_bootstrap"] and int(r["y"].sum()) >= rt.MIN_POS]
    X_all = np.vstack([r["X"] for r in recs])
    y_all = np.concatenate([r["y"] for r in recs])
    groups = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(recs)])
    is_bs = np.concatenate([[r["is_bootstrap"]] * len(r["y"]) for r in recs])
    w_row = np.concatenate([r["w"] for r in recs])
    ag_mask = np.isin(groups, ag_ok)
    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full(len(y_all), np.nan)
    for tr_i, te_i in skf.split(np.zeros(ag_mask.sum()), y_all[ag_mask], groups[ag_mask]):
        te_sessions = np.unique(groups[ag_mask][te_i])
        train = ~np.isin(groups, te_sessions)
        w = w_row.copy()
        w[~is_bs] = w_agent
        m = train & (w > 0)
        s = rt.fit_predict(X_all[m], y_all[m], w[m], X_all[np.isin(groups, te_sessions)],
                           use_scaler=use_scaler, drop_zero_weight=False)
        scores[np.isin(groups, te_sessions)] = s
    ok = ~np.isnan(scores)
    y, s = y_all[ok], scores[ok]
    out = {"auc": rt.auc(y, s)}
    for t in thresholds:
        far, junk, _ = rt.far_junk(s, y, t)
        out[f"far_{t}"] = far
        out[f"junk_{t}"] = junk
    return out, s, y


def part_i():
    res = {}
    # (a) as-of-08-24 pool, the sweep script's protocol, seeds 42/43/44
    recs_0824 = rt.load_pool("vCA1", width=13, extra_dirs=[PARKED] if PARKED.exists() else None,
                             exclude=INGESTED_0826)
    n_ag = sum(1 for r in recs_0824 if not r["is_bootstrap"])
    rt.log(f"  as-of-08-24 pool: {len(recs_0824)} sessions ({n_ag} agent), CV-eligible "
           f"{sum(1 for r in recs_0824 if not r['is_bootstrap'] and r['y'].sum() >= 5)}")
    res["pool_0824"] = {"n_sessions": len(recs_0824), "n_agent": n_ag, "parked_included": PARKED.exists()}
    res["repro_3seed"] = {}
    for w in WEIGHTS:
        runs = [sweep_protocol(recs_0824, w, s)[0] for s in rt.SEEDS3]
        res["repro_3seed"][str(w)] = {k: rt.summarize([r[k] for r in runs]) for k in runs[0]}
        rt.log(f"    [08-24 pool, 3 seeds] w={w:5.2f}: AUC {res['repro_3seed'][str(w)]['auc']['mean']:.4f}"
               f"+/-{res['repro_3seed'][str(w)]['auc']['sd']:.4f} FAR@0.05 {res['repro_3seed'][str(w)]['far_0.05']['mean']:.1f}% "
               f"junk {res['repro_3seed'][str(w)]['junk_0.05']['mean']:.1f}%")
    # (b) pinned pool, 8 seeds, same protocol, plus the harness protocol (scaler) for reference
    recs = rt.load_pool("vCA1", width=13)
    res["pool_now"] = {"n_sessions": len(recs), "n_agent": sum(1 for r in recs if not r["is_bootstrap"])}
    res["eight_seed"] = {}
    per_seed = {}
    harness_oof = {}
    for w in WEIGHTS:
        runs = []
        for s in rt.SEEDS:
            o, sc, y = sweep_protocol(recs, w, s)
            runs.append(o)
        per_seed[w] = runs
        # rule-chosen T and junk at matched FAR need per-seed OOF vectors: recompute via harness protocol
        cv, rest, bs = rt.split_pool(recs)
        X_ag, y_ag, g, rev, _, _ = rt.stack_agent(cv)
        X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
        oof = rt.run_oof_seeds(X_ag, y_ag, g, X_bs, y_bs, w_bs, agent_weight=w)
        harness_oof[w] = oof
        tab = rt.threshold_table(oof, y_ag, rev)
        cT, gate, det = rt.rule_T(tab)
        res["eight_seed"][str(w)] = {
            "sweep_protocol": {k: rt.summarize([r[k] for r in runs]) for k in runs[0]},
            "harness_auc_full": rt.summarize([rt.auc(y_ag, o) for o in oof]),
            "harness_auc_reviewed": rt.summarize([rt.auc(y_ag, o, rev) for o in oof]),
            "harness_far_0.05": rt.summarize([rt.far_junk(o, y_ag, 0.05)[0] for o in oof]),
            "harness_junk_0.05": rt.summarize([rt.far_junk(o, y_ag, 0.05)[1] for o in oof]),
            "rule_T": cT, "rule_detail": det,
        }
        e = res["eight_seed"][str(w)]
        rt.log(f"    [pinned pool, 8 seeds] w={w:5.2f}: sweep-protocol AUC {e['sweep_protocol']['auc']['mean']:.4f} "
               f"FAR@0.05 {e['sweep_protocol']['far_0.05']['mean']:.2f}% | harness AUC {e['harness_auc_full']['mean']:.4f} "
               f"FAR@0.05 {e['harness_far_0.05']['mean']:.2f}% (max {e['harness_far_0.05']['max']:.2f}) junk {e['harness_junk_0.05']['mean']:.1f}% "
               f"| rule T {cT} ({det.get('junk_full', float('nan')):.1f}% junk)")
    # the fair weight comparison: junk caught at MATCHED false-AR, every weight vs the deployed 5.0
    cv, rest, bs = rt.split_pool(recs)
    _, y_ag, _, rev, _, _ = rt.stack_agent(cv)
    res["matched_op_vs_w5"] = {}
    for w in WEIGHTS:
        mop = rt.matched_operating_point(harness_oof[5.0], harness_oof[w], y_ag, 0.05)
        res["matched_op_vs_w5"][str(w)] = {k: v for k, v in mop.items() if k.endswith("_mean")}
        rt.log(f"    matched-op w={w:5.2f} vs 5.0@0.05: FAR at matched junk {mop['far_ref_mean']:.2f}% -> "
               f"{mop['far_new_at_matched_junk_mean']:.2f}%; junk at matched FAR {mop['junk_ref_mean']:.1f}% -> "
               f"{mop['junk_new_at_matched_far_mean']:.1f}%")
    np.savez(rt.FIXTURES / "a09_vca1_weight_oof.npz", y=y_ag, reviewed=rev,
             **{f"oof_w{w:g}": harness_oof[w] for w in WEIGHTS})
    # decision test: 7.01 vs 5.0 on FAR@0.05, per seed (sweep protocol)
    d = np.array([per_seed[7.01][i]["far_0.05"] - per_seed[5.0][i]["far_0.05"] for i in range(len(rt.SEEDS))])
    res["decision_far_7_minus_5"] = rt.paired_delta([per_seed[5.0][i]["far_0.05"] for i in range(8)],
                                                    [per_seed[7.01][i]["far_0.05"] for i in range(8)])
    da = rt.paired_delta([per_seed[5.0][i]["auc"] for i in range(8)], [per_seed[7.01][i]["auc"] for i in range(8)])
    res["decision_auc_7_minus_5"] = da
    res["decision_survives"] = bool(res["decision_far_7_minus_5"]["n_positive"] >= 7
                                    and res["decision_far_7_minus_5"]["mean"] > 2 * res["decision_far_7_minus_5"]["se"]
                                    and abs(da["mean"]) < 0.01)
    rt.log(f"  decision (7.01 vs 5.0): FAR@0.05 delta {res['decision_far_7_minus_5']} | AUC delta {da} -> "
           f"{'SURVIVES' if res['decision_survives'] else 'DOES NOT SURVIVE'}")
    return res


def part_ii():
    res = {}
    for area in ("BLA", "vCA1"):
        recs = rt.load_pool(area, width=rt.EXPECTED_WIDTH[area])
        cv, rest, bs = rt.split_pool(recs)
        X_ag, y_ag, g, rev, _, _ = rt.stack_agent(cv)
        X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
        # duplicate rows: weight 0 AND labelled 0 AND listed as duplicates (not ambiguous)
        dup_mask = np.zeros(len(y_bs), bool)
        off = 0
        for r in bs:
            js = rt.load_json(r["session_dir"])
            for i in js.get("duplicate_candidate_indices", []):
                if 0 <= i < len(r["y"]):
                    dup_mask[off + i] = True
            off += len(r["y"])
        T = rt.DEPLOYED_T[area]
        override = rt.AGENT_WEIGHT_OVERRIDE.get(area)
        res[area] = {"n_dup_rows": int(dup_mask.sum())}
        oofs = {}
        for mode in ("masked", "label0", "label1"):
            yb = y_bs.copy()
            wb = w_bs.copy()
            if mode == "label0":
                wb[dup_mask] = 1.0
            elif mode == "label1":
                wb[dup_mask] = 1.0
                yb[dup_mask] = 1
            oof = rt.run_oof_seeds(X_ag, y_ag, g, X_bs, yb, wb, agent_weight=override)
            oofs[mode] = oof
            tab = rt.threshold_table(oof, y_ag, rev)
            cT, gate, det = rt.rule_T(tab)
            res[area][mode] = {"auc_full": rt.summarize([rt.auc(y_ag, o) for o in oof]),
                               "auc_reviewed": rt.summarize([rt.auc(y_ag, o, rev) for o in oof]),
                               "far_at_T": rt.summarize([rt.far_junk(o, y_ag, T)[0] for o in oof]),
                               "junk_at_T": rt.summarize([rt.far_junk(o, y_ag, T)[1] for o in oof]),
                               "rule_T": cT, "rule_detail": det}
            rt.log(f"  {area} dups={mode:6s}: AUC {res[area][mode]['auc_full']['mean']:.4f} rev {res[area][mode]['auc_reviewed']['mean']:.4f} "
                   f"FAR@{T} {res[area][mode]['far_at_T']['mean']:.2f}% junk {res[area][mode]['junk_at_T']['mean']:.1f}% rule T {cT}")
        a_m = [rt.auc(y_ag, o) for o in oofs["masked"]]
        res[area]["label0_minus_masked"] = rt.paired_delta(a_m, [rt.auc(y_ag, o) for o in oofs["label0"]])
        res[area]["label1_minus_masked"] = rt.paired_delta(a_m, [rt.auc(y_ag, o) for o in oofs["label1"]])
        res[area]["matched_op_label0_vs_masked"] = rt.matched_operating_point(oofs["masked"], oofs["label0"], y_ag, T)
        res[area]["matched_op_label1_vs_masked"] = rt.matched_operating_point(oofs["masked"], oofs["label1"], y_ag, T)
        rt.log(f"  {area}: label0-masked {res[area]['label0_minus_masked']['mean']:+.4f} ({res[area]['label0_minus_masked']['n_positive']}/8); "
               f"label1-masked {res[area]['label1_minus_masked']['mean']:+.4f} ({res[area]['label1_minus_masked']['n_positive']}/8); "
               f"junk at matched FAR: label0 {res[area]['matched_op_label0_vs_masked']['junk_new_at_matched_far_mean']:.1f}% "
               f"label1 {res[area]['matched_op_label1_vs_masked']['junk_new_at_matched_far_mean']:.1f}% vs masked {res[area]['matched_op_label0_vs_masked']['junk_ref_mean']:.1f}%")
    return res


def main():
    t = rt.Timer()
    rt.assert_pinned()
    rt.log("[i] vCA1 weight decision")
    I = part_i()
    rt.log("[ii] duplicate handling at 8 seeds")
    II = part_ii()
    r5 = I["repro_3seed"]["5.0"]
    r7 = I["repro_3seed"]["7.01"]
    checks = {
        "repro_w5_auc_within_0.003": abs(r5["auc"]["mean"] - 0.8857) < 0.003,
        "repro_w7_auc_within_0.003": abs(r7["auc"]["mean"] - 0.8843) < 0.003,
        "repro_far_ordering": r7["far_0.05"]["mean"] > r5["far_0.05"]["mean"],
        "w5_decision_survives_8_seeds": I["decision_survives"],
        "dups_masked_ge_label0_within_noise": all(II[a]["label0_minus_masked"]["mean"] < 2 * II[a]["label0_minus_masked"]["se"] + 0.002 for a in II),
        "dups_label1_worse_or_equal": all(II[a]["label1_minus_masked"]["mean"] <= 0.002 for a in II),
    }
    res = {"part_i": I, "part_ii": II, "checks": checks,
           "verdict": "PASS" if all(checks.values()) else "FAIL",
           "criterion": "3-seed numbers reproduce on the 08-24 pool (AUC within 0.003, FAR ordering); the w=5.0 vs 7.01 "
                        "FAR advantage holds at 8 seeds (>= 7/8 positive, > 2 SE) with AUC within noise; masked >= label0 "
                        "within noise and label1 not better"}
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a09", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
