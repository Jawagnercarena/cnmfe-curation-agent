"""
a11_bla_g5.py -- attack #11: BLA G5 (per-animal / early-era), never re-run.

Part A (no refits): the pre-fix step5_oof.npz and the post-fix
c3_bla_gate8_oof.npz share rows, names, groups and seeds -> per animal (6)
and per era, per seed, AUC(pre-fix corpus) vs AUC(post-fix corpus) on the
deployed variant (rankv2b_35) and on b13; paired deltas; FAR at 0.04 per
animal.  Eras: (a) recording date < 2026-01-01 from the session name;
(b) the prior red team's labels_mtime < 2026-04-15 (pin manifest).
Part B (refits on the live 35-col files): leave-one-ANIMAL-out and
leave-the-early-era-out; train on the remaining agent sessions + all
bootstrap, and a variant dropping the held-out animal's own bootstrap
sessions; 8 xgb seeds; b13 vs rankv2b_35.  Read-only.
"""
import datetime as dt
import sys

import numpy as np

import rt_lib as rt

STEP5_OOF = rt.AGENT / "eval" / "step4_2026-08" / "step5_oof.npz"
C3_OOF = rt.AGENT / "eval" / "bootstrap_matching_2026-08" / "c3_bla_gate8_oof.npz"
ERA_A = dt.date(2026, 1, 1)
ERA_B_MTIME = dt.datetime(2026, 4, 15).timestamp()
T = rt.DEPLOYED_T["BLA"]


def cell(y, pre, post, rev=None):
    if len(np.unique(y)) < 2:
        return None
    a_pre = [rt.auc(y, o) for o in pre]
    a_post = [rt.auc(y, o) for o in post]
    d = rt.paired_delta(a_pre, a_post)
    far_pre = [rt.far_junk(o, y, T)[0] for o in pre]
    far_post = [rt.far_junk(o, y, T)[0] for o in post]
    return {"n_rows": int(len(y)), "n_pos": int(y.sum()), "auc_pre": rt.summarize(a_pre),
            "auc_post": rt.summarize(a_post), "delta": d,
            "far_pre_at_0.04": rt.summarize(far_pre), "far_post_at_0.04": rt.summarize(far_post)}


def part_a(pin_rows):
    f5 = np.load(STEP5_OOF, allow_pickle=True)
    c3 = np.load(C3_OOF, allow_pickle=True)
    same = (list(f5["names"]) == list(c3["names"]) and np.array_equal(f5["y"], c3["y"])
            and np.array_equal(f5["groups"], c3["groups"]) and np.array_equal(f5["reviewed"], c3["reviewed"])
            and np.array_equal(f5["smoke_rows"], c3["smoke_rows"]))
    assert same, "fixtures do not share rows -- pairing invalid"
    names = list(c3["names"])
    g, y = c3["groups"], c3["y"]
    animals = np.array([rt.animal_of("BLA", n.split("/")[-1]) for n in names])
    dates = [rt.date_of(n.split("/")[-1]) for n in names]
    mt = {r["rel"]: r.get("labels_mtime") for r in pin_rows}
    era_a = np.array([d is not None and d < ERA_A for d in dates])
    era_b = np.array([(mt.get(n) or 0) < ERA_B_MTIME for n in names])
    out = {"fixtures_share_rows": True, "n_sessions": len(names),
           "sessions_era_a_early": int(era_a.sum()), "sessions_era_b_early": int(era_b.sum()),
           "by_animal": {}, "by_era": {}}
    for v in ("oof_v2b", "oof_b13"):
        out["by_animal"][v] = {}
        for a in sorted(set(animals)):
            m = np.isin(g, np.flatnonzero(animals == a))
            c = cell(y[m], f5[v][:, m], c3[v][:, m])
            if c:
                c["n_sessions"] = int((animals == a).sum())
                out["by_animal"][v][a] = c
                rt.log(f"  [{v}] animal {a:6s} {c['n_sessions']:2d} sess {c['n_pos']:4d} real: "
                       f"AUC {c['auc_pre']['mean']:.4f} -> {c['auc_post']['mean']:.4f} "
                       f"({c['delta']['mean']:+.4f}, min {c['delta']['min']:+.4f}, {c['delta']['n_positive']}/8) "
                       f"FAR@0.04 {c['far_pre_at_0.04']['mean']:.2f} -> {c['far_post_at_0.04']['mean']:.2f}")
        out["by_era"][v] = {}
        for label, mask_s in (("a_early_2025", era_a), ("a_late_2026", ~era_a),
                              ("b_early_mtime", era_b), ("b_late_mtime", ~era_b)):
            m = np.isin(g, np.flatnonzero(mask_s))
            c = cell(y[m], f5[v][:, m], c3[v][:, m])
            if c:
                c["n_sessions"] = int(mask_s.sum())
                out["by_era"][v][label] = c
                rt.log(f"  [{v}] era {label:14s} {c['n_sessions']:2d} sess {c['n_pos']:4d} real: "
                       f"AUC {c['auc_pre']['mean']:.4f} -> {c['auc_post']['mean']:.4f} "
                       f"({c['delta']['mean']:+.4f}, min {c['delta']['min']:+.4f}, {c['delta']['n_positive']}/8)")
    return out


def part_b():
    recs = rt.load_pool("BLA", width=35)
    ag = [r for r in recs if not r["is_bootstrap"]]
    bs = [r for r in recs if r["is_bootstrap"]]
    out = {}
    holdouts = {}
    for a in sorted({r["animal"] for r in ag}):
        holdouts[f"animal_{a}"] = (lambda r, _a=a: r["animal"] == _a)
    holdouts["era_a_early_2025"] = lambda r: r["date"] is not None and r["date"] < ERA_A
    for name, is_held in holdouts.items():
        test = [r for r in ag if is_held(r) and int(r["y"].sum()) >= rt.MIN_POS]
        train_ag = [r for r in ag if not is_held(r)]
        if not test or len(train_ag) < 3:
            continue
        y_te = np.concatenate([r["y"] for r in test])
        rev_te = np.concatenate([r["reviewed"] for r in test])
        X_te = np.vstack([r["X"] for r in test])
        X_tr_ag = np.vstack([r["X"] for r in train_ag])
        y_tr_ag = np.concatenate([r["y"] for r in train_ag])
        held_animals = {r["animal"] for r in test}
        out[name] = {"n_test_sessions": len(test), "n_test_rows": int(len(y_te)), "n_test_pos": int(y_te.sum())}
        for bs_variant in ("all_bootstrap", "drop_heldout_animal_bootstrap"):
            use_bs = bs if bs_variant == "all_bootstrap" else [r for r in bs if r["animal"] not in held_animals]
            X_bs = np.vstack([r["X"] for r in use_bs]) if use_bs else np.zeros((0, 35))
            y_bs = np.concatenate([r["y"] for r in use_bs]) if use_bs else np.zeros(0, int)
            w_bs = np.concatenate([r["w"] for r in use_bs]) if use_bs else np.zeros(0)
            agw = rt.fold_agent_weight(len(y_bs), len(y_tr_ag), None)
            res_v = {}
            per_seed = {}
            for vname, sl in (("b13", slice(0, 13)), ("rankv2b_35", slice(0, 35))):
                X_tr = np.vstack([X_tr_ag[:, sl], X_bs[:, sl]])
                y_tr = np.concatenate([y_tr_ag, y_bs])
                w_tr = np.concatenate([np.full(len(y_tr_ag), agw), w_bs])
                aucs, aucs_rev, fars = [], [], []
                for seed in rt.SEEDS:
                    s = rt.fit_predict(X_tr, y_tr, w_tr, X_te[:, sl], xgb_seed=seed)
                    aucs.append(rt.auc(y_te, s))
                    aucs_rev.append(rt.auc(y_te, s, rev_te))
                    fars.append(rt.far_junk(s, y_te, T)[0])
                per_seed[vname] = aucs
                res_v[vname] = {"auc": rt.summarize(aucs), "auc_reviewed": rt.summarize(aucs_rev),
                                "far_at_0.04": rt.summarize(fars)}
            res_v["delta_v2b_minus_b13"] = rt.paired_delta(per_seed["b13"], per_seed["rankv2b_35"])
            res_v["n_bootstrap_sessions"] = len(use_bs)
            res_v["agent_weight"] = agw
            out[name][bs_variant] = res_v
            rt.log(f"  LOAO {name:22s} [{bs_variant:30s}] test {len(test)} sess / {int(y_te.sum())} real: "
                   f"b13 {res_v['b13']['auc']['mean']:.4f} v2b {res_v['rankv2b_35']['auc']['mean']:.4f} "
                   f"(reviewed {res_v['rankv2b_35']['auc_reviewed']['mean']:.4f}) FAR@0.04 {res_v['rankv2b_35']['far_at_0.04']['mean']:.2f}%")
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    pin_rows = rt.load_pin()["areas"]["BLA"]["sessions"]
    rt.log("[A] paired per-animal / per-era from the two fixtures (no refits)")
    A = part_a(pin_rows)
    rt.log("[B] leave-one-animal-out / leave-early-era-out refits on the live files")
    B = part_b()
    neg_cells = []
    for v in ("oof_v2b",):
        for a, c in A["by_animal"][v].items():
            if c["delta"]["mean"] < -2 * c["delta"]["se"] and c["delta"]["mean"] < -0.002:
                neg_cells.append(("animal", a, c["delta"]["mean"]))
        for e, c in A["by_era"][v].items():
            if c["delta"]["mean"] < -2 * c["delta"]["se"] and c["delta"]["mean"] < -0.002:
                neg_cells.append(("era", e, c["delta"]["mean"]))
    checks = {"fixtures_paired": A["fixtures_share_rows"],
              "no_animal_or_era_worse_beyond_noise": len(neg_cells) == 0,
              "early_era_improved": A["by_era"]["oof_v2b"]["a_early_2025"]["delta"]["mean"] > 0}
    res = {"part_a": A, "part_b": B, "negative_cells": neg_cells, "checks": checks,
           "verdict": "PASS" if all(checks.values()) else "FAIL",
           "criterion": "no animal / era where the fixed corpus is worse than the pre-fix corpus beyond seed noise "
                        "(paired, same rows/seeds); early era reported explicitly; LOAO refits reported"}
    rt.log(f"negative cells {neg_cells}\nchecks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a11", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
