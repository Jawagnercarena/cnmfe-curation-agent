"""
a18_global_animal.py -- attack #18: the global (cross-area) model, animal-grouped, 8 seeds.

Re-implements c5_global_model.py's Cond A / Cond B (target's own bootstrap +
deployed weighting; Cond B appends every other area's rows at source_w 1.0
and 0.3; shared first 13 columns) under
  session grouping  (reproduce the 3-seed log: vCA1 -0.0038 / -0.0023,
                     BLA +0.0003 / -0.0011, DG_AL +0.0145 / +0.0110), and
  animal grouping   (BLA 6-animal grouped folds; vCA1 2-animal LOAO; DG_AL
                     2-animal LOAO plus FOV grouping: A/B planes of one day and
                     the non-averaged duplicate of an AVG4x recording are one FOV).
8 seeds (CV seed for k-fold; xgb seed for LOAO).  Reports Cond B - A per cell
with min / all-positive, and junk at matched false-AR.  Writes
fixtures/a18_oof.npz for attack #19.  Read-only.
"""
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import rt_lib as rt

SOURCE_W = [1.0, 0.3]
CLAIMED = {"vCA1": {"1.0": -0.0023, "0.3": -0.0038}, "BLA": {"1.0": -0.0011, "0.3": 0.0003},
           "DG_AL": {"1.0": 0.0110, "0.3": 0.0145}}


def fov_of(area, name, animal, date):
    import re
    d = [t for t in rt._tokens_after_tseries(name) if re.fullmatch(r"\d+um", t)]
    return f"{date}|{animal}|{'-'.join(d)}" if area == "DG_AL" else f"{animal}|{d[0] if d else '?'}"


def load_area13(area):
    recs = rt.load_pool(area, width=rt.EXPECTED_WIDTH[area])
    for r in recs:
        r["X"] = r["X"][:, :13]
        r["fov"] = fov_of(area, r["session_dir"].name, r["animal"], r["date"])
    return recs


def evaluate(target, pools, grouping, seeds):
    recs = pools[target]
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g_sess, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs, width=X_ag.shape[1])
    others = [r for a, rs in pools.items() if a != target for r in rs]
    X_ot = np.vstack([r["X"] for r in others])
    y_ot = np.concatenate([r["y"] for r in others])
    w_ot = np.concatenate([r["w"] for r in others])
    T = rt.DEPLOYED_T[target] if target != "DG_AL" else 0.05      # c5 used 0.05 for DG
    override = rt.AGENT_WEIGHT_OVERRIDE.get(target)
    if grouping == "session":
        groups = g_sess
    elif grouping == "animal":
        ids = {a: i for i, a in enumerate(sorted(set(animals)))}
        groups = np.array([ids[a] for a in animals])
    else:   # fov
        fovs = np.concatenate([[r["fov"]] * len(r["y"]) for r in cv])
        ids = {f: i for i, f in enumerate(sorted(set(fovs)))}
        groups = np.array([ids[f] for f in fovs])
    n_groups = len(np.unique(groups))
    out = {"n_cv_sessions": len(cv), "n_rows": int(len(y)), "n_real": int(y.sum()), "n_groups": int(n_groups), "T": T}
    oof_store = {}
    for sw in SOURCE_W:
        A_auc, B_auc, A_far, B_far, A_junk, B_junk, jm = [], [], [], [], [], [], []
        oofA_seeds, oofB_seeds = [], []
        for seed in seeds:
            oofA = np.full(len(y), np.nan)
            oofB = np.full(len(y), np.nan)
            if n_groups >= 3:
                splits = list(StratifiedGroupKFold(n_splits=min(5, n_groups), shuffle=True, random_state=seed).split(X_ag, y, groups))
                xseed = 42
            else:
                splits = [(np.flatnonzero(groups != k), np.flatnonzero(groups == k)) for k in np.unique(groups)]
                xseed = seed
            for tr, te in splits:
                if len(np.unique(y[te])) < 2:
                    continue
                if grouping == "session":
                    keep = np.ones(len(y_bs), bool)
                else:
                    te_animals = set(animals[te])
                    bs_an = np.concatenate([[r["animal"]] * len(r["y"]) for r in bs]) if bs else np.zeros(0)
                    keep = ~np.isin(bs_an, list(te_animals)) if len(bs_an) else np.ones(0, bool)
                agw = rt.fold_agent_weight(int(keep.sum()), len(tr), override)
                base_X = [X_ag[tr], X_bs[keep]]
                base_y = [y[tr], y_bs[keep]]
                base_w = [np.full(len(tr), agw), w_bs[keep]]
                for cond in ("A", "B"):
                    Xs, ys, ws = list(base_X), list(base_y), list(base_w)
                    if cond == "B":
                        Xs.append(X_ot); ys.append(y_ot); ws.append(w_ot * sw)
                    Xt, yt, wt = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
                    s = rt.fit_predict(Xt, yt, wt, X_ag[te], xgb_seed=xseed, drop_zero_weight=True)
                    (oofA if cond == "A" else oofB)[te] = s
            ok = ~np.isnan(oofA)
            A_auc.append(rt.auc(y[ok], oofA[ok])); B_auc.append(rt.auc(y[ok], oofB[ok]))
            A_far.append(rt.far_junk(oofA[ok], y[ok], T)[0]); B_far.append(rt.far_junk(oofB[ok], y[ok], T)[0])
            A_junk.append(rt.far_junk(oofA[ok], y[ok], T)[1]); B_junk.append(rt.far_junk(oofB[ok], y[ok], T)[1])
            oofA_seeds.append(oofA); oofB_seeds.append(oofB)
        oofA_seeds, oofB_seeds = np.array(oofA_seeds), np.array(oofB_seeds)
        ok = ~np.isnan(oofA_seeds[0])
        mop = rt.matched_operating_point(oofA_seeds[:, ok], oofB_seeds[:, ok], y[ok], T)
        out[str(sw)] = {"A_auc": rt.summarize(A_auc), "B_auc": rt.summarize(B_auc), "delta": rt.paired_delta(A_auc, B_auc),
                        "A_far": rt.summarize(A_far), "B_far": rt.summarize(B_far), "A_junk": rt.summarize(A_junk),
                        "B_junk": rt.summarize(B_junk), "junk_B_at_matched_far": mop["junk_new_at_matched_far_mean"],
                        "junk_A_ref": mop["junk_ref_mean"], "far_B_at_matched_junk": mop["far_new_at_matched_junk_mean"]}
        oof_store[f"oofA_{sw}"] = oofA_seeds
        oof_store[f"oofB_{sw}"] = oofB_seeds
        d = out[str(sw)]
        rt.log(f"  [{target:5s} {grouping:7s} w={sw}] A {d['A_auc']['mean']:.4f} B {d['B_auc']['mean']:.4f} delta {d['delta']['mean']:+.4f} "
               f"(min {d['delta']['min']:+.4f}, {d['delta']['n_positive']}/8) | T={T}: junk A {d['A_junk']['mean']:.1f}% B {d['B_junk']['mean']:.1f}% "
               f"| junk at matched FAR {d['junk_A_ref']:.1f}% -> {d['junk_B_at_matched_far']:.1f}%")
    oof_store.update({"y": y, "reviewed": rev, "groups": groups, "animals": animals})
    return out, oof_store


def main():
    t = rt.Timer()
    rt.assert_pinned()
    pools = {a: load_area13(a) for a in rt.AREAS}
    for a, rs in pools.items():
        rt.log(f"{a}: {len(rs)} sessions ({sum(1 for r in rs if not r['is_bootstrap'])} agent), {sum(len(r['y']) for r in rs)} rows")
    res = {"targets": {}}
    fixtures = {}
    for target in ("vCA1", "BLA", "DG_AL"):
        res["targets"][target] = {}
        groupings = ("session", "animal") + (("fov",) if target == "DG_AL" else ())
        for grouping in groupings:
            out, store = evaluate(target, pools, grouping, rt.SEEDS)
            res["targets"][target][grouping] = out
            for k, v in store.items():
                fixtures[f"{target}_{grouping}_{k}"] = v
    np.savez(rt.FIXTURES / "a18_oof.npz", **fixtures)
    # reproduction of the 3-seed log (session grouping; seeds differ, so within noise)
    repro = {}
    for target in ("vCA1", "BLA", "DG_AL"):
        for sw in ("1.0", "0.3"):
            d = res["targets"][target]["session"][sw]["delta"]["mean"]
            repro[f"{target}_w{sw}"] = {"own": d, "claimed": CLAIMED[target][sw], "abs_diff": abs(d - CLAIMED[target][sw])}
    res["repro_session_grouping"] = repro
    dg = res["targets"]["DG_AL"]
    checks = {
        # BLA / vCA1: 75 and 16 test sessions -> the 3-seed log is reproducible to a few thousandths;
        # DG_AL: 9 sessions / 2 animals -> its 3-seed number carries ~0.01 of seed noise, so only the sign
        # and the order of magnitude are checkable.
        "bla_vca1_session_grouping_within_0.006": all(v["abs_diff"] < 0.006 for k, v in repro.items() if not k.startswith("DG_AL")),
        "dg_session_grouping_same_sign_within_0.012": all(np.sign(v["own"]) == np.sign(v["claimed"]) and v["abs_diff"] < 0.012
                                                          for k, v in repro.items() if k.startswith("DG_AL")),
        "dg_gain_survives_animal_loao_w0.3": dg["animal"]["0.3"]["delta"]["mean"] > 0 and dg["animal"]["0.3"]["delta"]["n_positive"] >= 7,
        "dg_gain_survives_fov_grouping_w0.3": dg["fov"]["0.3"]["delta"]["mean"] > 0 and dg["fov"]["0.3"]["delta"]["n_positive"] >= 7,
        "bla_null_under_animal": all(abs(res["targets"]["BLA"]["animal"][sw]["delta"]["mean"]) < 0.01 for sw in ("1.0", "0.3")),
        "vca1_null_under_session": all(abs(res["targets"]["vCA1"]["session"][sw]["delta"]["mean"]) < 0.01 for sw in ("1.0", "0.3")),
    }
    res["checks"] = checks
    res["finding_vca1_pooling_under_loao"] = {sw: res["targets"]["vCA1"]["animal"][sw]["delta"] for sw in ("1.0", "0.3")}
    res["verdict"] = "PASS" if (checks["bla_vca1_session_grouping_within_0.006"] and checks["dg_session_grouping_same_sign_within_0.012"]) else "FAIL"
    res["dg_pooled_prior_holds"] = bool(checks["dg_gain_survives_animal_loao_w0.3"] and checks["dg_gain_survives_fov_grouping_w0.3"])
    res["criterion"] = ("own re-implementation reproduces the logged session-grouped deltas within 0.006; the DG_AL gain "
                        "is judged real only if it survives leave-one-animal-out AND FOV grouping at 8 seeds (>= 7/8)")
    rt.log(f"checks {checks}\nDG pooled prior holds: {res['dg_pooled_prior_holds']}\nVERDICT {res['verdict']}")
    rt.write_result("a18", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
