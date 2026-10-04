"""
a14_bla_v2b.py -- attack #14 / hypothesis D14: BLA bootstrap rows are
feature-degenerate (zero-filled v2b, v2_present=0).

Gate on the real-v2b arms built by prep_bla_v2b_arms.py (fixtures/
bla_v2b_arms.npz): arm a = live files (zero v2b), b0_lso (Cn=None, hiconf from
leave-session-out 13-col scores), b0prod (hiconf from the deployed companion
model, in-sample), b1_lso (real ring_contrast where Cn is same-res).  8 seeds,
rankv2b_35, the deployed sqrt/4.0 recipe, paired against
c3_bla_gate8_oof.npz (arm a must reproduce it).  Reports reviewed/full AUC
deltas (mean / min / n positive), false-AR at matched junk and junk at
matched false-AR per seed, rule-chosen T, the bla21 smoke cells, and the
v2b column distributions bootstrap vs agent (trace-regime shift).  The
pre-registered bar is vCA1's: delta >= max(0.005, 2 SE), >= 7/8 seeds
positive, junk at matched false-AR not worse.  Informational for BLA (no
deploy change here); a hold becomes recommendation #1.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

C3_OOF = rt.AGENT / "eval" / "bootstrap_matching_2026-08" / "c3_bla_gate8_oof.npz"
ARMS = ("a", "b0_lso", "b0prod", "b1_lso")
T = rt.DEPLOYED_T["BLA"]
SMOKE_REL = "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"
SMOKE_IDX = [21, 24]


def main():
    t = rt.Timer()
    rt.assert_pinned()
    armz = np.load(rt.FIXTURES / "bla_v2b_arms.npz", allow_pickle=True)
    fix = np.load(C3_OOF, allow_pickle=True)
    recs = rt.load_pool("BLA", width=35)
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g, rev, animals, names = rt.stack_agent(cv)
    smoke_rows = []
    off = 0
    for r in cv:
        if r["name"] == SMOKE_REL:
            smoke_rows = [off + i for i in SMOKE_IDX]
        off += len(r["y"])
    y_bs = np.concatenate([r["y"] for r in bs])
    w_bs = np.concatenate([r["w"] for r in bs])
    res = {"n_pool": int(len(y)), "n_real": int(y.sum()), "arms": {}}
    oof = {}
    for arm in ARMS:
        if arm == "a":
            X_bs = np.vstack([r["X"] for r in bs])
        else:
            X_bs = np.vstack([armz[f"{rt.key(r['name'])}__{arm}"] for r in bs])
        assert X_bs.shape[1] == 35
        oof[arm] = rt.run_oof_seeds(X_ag, y, g, X_bs, y_bs, w_bs, agent_weight=None)
        res["arms"][arm] = {
            "bootstrap_flag_rows": int(X_bs[:, 34].sum()), "ring_nonzero_rows": int((X_bs[:, 33] != 0).sum()),
            "auc_full": rt.summarize([rt.auc(y, o) for o in oof[arm]]),
            "auc_reviewed": rt.summarize([rt.auc(y, o, rev) for o in oof[arm]]),
            "far_at_T": rt.summarize([rt.far_junk(o, y, T)[0] for o in oof[arm]]),
            "junk_at_T": rt.summarize([rt.far_junk(o, y, T)[1] for o in oof[arm]]),
            "smoke": {str(ci): float(np.mean(oof[arm][:, r_])) for ci, r_ in zip(SMOKE_IDX, smoke_rows)},
        }
        tab = rt.threshold_table(oof[arm], y, rev)
        cT, gate, det = rt.rule_T(tab)
        res["arms"][arm].update({"rule_T": cT, "rule_detail": det, "threshold_table": tab})
        rt.log(f"  arm {arm:7s}: AUC full {res['arms'][arm]['auc_full']['mean']:.4f}+/-{res['arms'][arm]['auc_full']['sd']:.4f} "
               f"reviewed {res['arms'][arm]['auc_reviewed']['mean']:.4f} | FAR@{T} {res['arms'][arm]['far_at_T']['mean']:.2f}% "
               f"junk {res['arms'][arm]['junk_at_T']['mean']:.1f}% | rule T {cT} | smoke {res['arms'][arm]['smoke']}", )
    # arm a must reproduce the fixture
    md = float(np.max(np.abs(oof["a"] - fix["oof_v2b"])))
    res["arm_a_reproduces_fixture"] = {"max_abs_score_diff": md, "pass": bool(md < 1e-6)}
    rt.log(f"  arm a vs c3_bla_gate8_oof v2b: max|diff| {md:.2e}")
    # paired deltas vs arm a
    a_full = [rt.auc(y, o) for o in oof["a"]]
    a_rev = [rt.auc(y, o, rev) for o in oof["a"]]
    for arm in ARMS[1:]:
        d_full = rt.paired_delta(a_full, [rt.auc(y, o) for o in oof[arm]])
        d_rev = rt.paired_delta(a_rev, [rt.auc(y, o, rev) for o in oof[arm]])
        mop = rt.matched_operating_point(oof["a"], oof[arm], y, T)
        bar = max(0.005, 2 * d_rev["se"])
        res["arms"][arm].update({
            "delta_full_vs_a": d_full, "delta_reviewed_vs_a": d_rev, "bar": bar,
            "beats_bar": bool(d_rev["mean"] >= bar and d_rev["n_positive"] >= 7),
            "matched_op_vs_a": {k: v for k, v in mop.items() if k.endswith("_mean")},
            "junk_at_matched_far_not_worse": bool(mop["junk_new_at_matched_far_mean"] >= mop["junk_ref_mean"] - 0.5),
        })
        rt.log(f"  {arm:7s} vs a: reviewed {d_rev['mean']:+.4f} (min {d_rev['min']:+.4f}, {d_rev['n_positive']}/8, bar {bar:.4f}) "
               f"full {d_full['mean']:+.4f} | matched-op: FAR at matched junk {mop['far_ref_mean']:.2f}% -> "
               f"{mop['far_new_at_matched_junk_mean']:.2f}%; junk at matched FAR {mop['junk_ref_mean']:.1f}% -> "
               f"{mop['junk_new_at_matched_far_mean']:.1f}%")
    # trace-regime shift: v2b column distributions, bootstrap (b0_lso) vs agent rows
    Xb = np.vstack([armz[f"{rt.key(r['name'])}__b0_lso"] for r in bs])
    shift = {}
    for j, nm in enumerate(rt.F.V2B_NAMES):
        col = 26 + j
        ab, aa = Xb[:, col], X_ag[:, col]
        pb, pa = Xb[y_bs == 1, col], X_ag[y == 1, col]
        shift[nm] = {"bootstrap_all_p50": float(np.median(ab)), "agent_all_p50": float(np.median(aa)),
                     "bootstrap_pos_p50": float(np.median(pb)), "agent_pos_p50": float(np.median(pa)),
                     "bootstrap_pos_mean": float(pb.mean()), "agent_pos_mean": float(pa.mean())}
    res["v2b_regime_shift"] = shift
    hyp = res["arms"]["b0_lso"]
    checks = {"arm_a_reproduces_fixture": res["arm_a_reproduces_fixture"]["pass"],
              "b0_lso_beats_bar": hyp["beats_bar"], "b0_lso_junk_not_worse": hyp["junk_at_matched_far_not_worse"],
              "b0prod_beats_bar": res["arms"]["b0prod"]["beats_bar"]}
    res["checks"] = checks
    res["hypothesis_D14_supported"] = bool(checks["b0_lso_beats_bar"] and checks["b0_lso_junk_not_worse"])
    res["verdict"] = "PASS" if checks["arm_a_reproduces_fixture"] else "FAIL"
    res["criterion"] = ("arm a reproduces the fixture (harness fidelity); D14 is SUPPORTED if real v2b on bootstrap rows "
                        "beats zero-fill by >= max(0.005, 2 SE) on reviewed AUC with >= 7/8 seeds and junk at matched "
                        "false-AR not worse (vCA1's pre-registered bar)")
    rt.log(f"checks {checks}\nD14 supported: {res['hypothesis_D14_supported']}\nVERDICT {res['verdict']}")
    rt.write_result("a14", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
