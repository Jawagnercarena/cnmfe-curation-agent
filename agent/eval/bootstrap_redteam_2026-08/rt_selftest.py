"""
rt_selftest.py -- prove the independent evaluator before any attack runs.

  1. Synthetic orientation red/green: an asymmetric footprint compared with
     itself scores ~0 under the mixed metric (C-order rows vs F-order rows) and
     exactly 1.0 under a consistent one; square and 48x80 frames.
  2. BLA pin reproduction: rt_lib's pool + harness reproduce
     agent/eval/bootstrap_matching_2026-08/c3_bla_gate8_oof.npz (b13 and
     rankv2b_35, 8 seeds; sqrt/4.0 recipe) -- same rows, names, groups,
     reviewed mask; max|score diff| < 1e-6 and per-seed AUC to 1e-4.
  3. vCA1 pin reproduction: parallel v2 files + arm b0 at fixed weight 5.0
     reproduce agent/eval/vca1_v2_2026-08/gate_b0_w5_oof.npz (b13 and v35) and
     the b13 pin D:\\...\\vCA1\\.feature_expansion\\_pinned\\baseline_oof.npz.
  4. Masked bootstrap row sums equal the deployed joblibs' n_excluded_ambiguous
     (BLA 4,494; vCA1 6,662); pool counts 170 / 134 / 9.

Writes results/selftest.json.  Read-only on the corpus.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

import rt_lib as rt

BLA_FIX = rt.AGENT / "eval" / "bootstrap_matching_2026-08" / "c3_bla_gate8_oof.npz"
VCA1_FIX = rt.AGENT / "eval" / "vca1_v2_2026-08" / "gate_b0_w5_oof.npz"
VCA1_PIN = rt.EXT["vCA1"] / "_pinned" / "baseline_oof.npz"
VCA1_ARMS = rt.EXT["vCA1"] / "_arms"


def synthetic_orientation():
    out = {}
    for (d1, d2) in ((64, 64), (48, 80)):
        img = np.zeros((d1, d2))
        ys, xs = np.mgrid[0:d1, 0:d2]
        # asymmetric blob: elongated, off-diagonal so its transpose is a different image
        img = np.exp(-(((ys - 12) / 3.0) ** 2 + ((xs - 40 if d2 > 48 else xs - 30) / 9.0) ** 2))
        rows_C = img.flatten(order="C")[None, :]
        rows_F = img.flatten(order="F")[None, :]
        mixed = float(rt.cosine_rows(rows_C, rows_F)[0, 0])
        consistent = float(rt.cosine_rows(rows_C, rows_C)[0, 0])
        # round-trip: F-order columns back to the image
        A = img.flatten(order="F")[:, None]
        rt_img = rt.fcols_to_images(A, d1, d2)[0]
        wrong = rt.ccols_to_images(A, d1, d2)[0]
        out[f"{d1}x{d2}"] = {
            "mixed_metric_self_cosine": mixed, "consistent_self_cosine": consistent,
            "fcols_roundtrip_maxdiff": float(np.max(np.abs(rt_img - img))),
            "ccols_roundtrip_maxdiff": float(np.max(np.abs(wrong - img))),
            "pass": bool(mixed < 0.2 and abs(consistent - 1.0) < 1e-12
                         and np.max(np.abs(rt_img - img)) == 0.0
                         and np.max(np.abs(wrong - img)) > 0.5),
        }
        print(f"  synthetic {d1}x{d2}: mixed {mixed:.4f}, consistent {consistent:.6f}, "
              f"F round-trip {out[f'{d1}x{d2}']['fcols_roundtrip_maxdiff']:.1e}, "
              f"C round-trip {out[f'{d1}x{d2}']['ccols_roundtrip_maxdiff']:.2f} -> "
              f"{'PASS' if out[f'{d1}x{d2}']['pass'] else 'FAIL'}")
    return out


def compare_fixture(fix, names, y, g, rev, oofs: dict, label):
    res = {"rows_match": bool(len(fix["y"]) == len(y)) and bool((fix["y"] == y).all()),
           "groups_match": bool((fix["groups"] == g).all()) if len(fix["groups"]) == len(g) else False,
           "reviewed_match": bool((fix["reviewed"] == rev).all()) if len(fix["reviewed"]) == len(rev) else False,
           "names_match": bool(list(fix["names"]) == list(names))}
    for fk, ok in oofs.items():
        ref = fix[fk]
        md = float(np.max(np.abs(ref - ok)))
        auc_ref = [rt.auc(y, o) for o in ref]
        auc_own = [rt.auc(y, o) for o in ok]
        dauc = float(np.max(np.abs(np.array(auc_ref) - np.array(auc_own))))
        res[fk] = {"max_abs_score_diff": md, "max_abs_auc_diff": dauc,
                   "auc_ref_mean": float(np.mean(auc_ref)), "auc_own_mean": float(np.mean(auc_own)),
                   "pass": bool(md < 1e-6 and dauc < 1e-4)}
        print(f"  {label} {fk}: max|score diff| {md:.2e}, max|AUC diff| {dauc:.2e}, "
              f"AUC ref {np.mean(auc_ref):.4f} own {np.mean(auc_own):.4f} -> "
              f"{'PASS' if res[fk]['pass'] else 'FAIL'}")
    print(f"  {label} rows/groups/reviewed/names match: {res['rows_match']}/"
          f"{res['groups_match']}/{res['reviewed_match']}/{res['names_match']}")
    return res


def bla_pin():
    fix = np.load(BLA_FIX, allow_pickle=True)
    recs = rt.load_pool("BLA", width=35)
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    masked = int((w_bs == 0).sum())
    print(f"  BLA pool: {len(recs)} sessions ({len(cv)+len(rest)} agent, {len(bs)} bootstrap), "
          f"CV {len(cv)} sessions / {len(y)} rows / {int(y.sum())} real; masked {masked}")
    oofs = {}
    for fk, sl in (("oof_b13", slice(0, 13)), ("oof_v2b", slice(0, 35))):
        oofs[fk] = rt.run_oof_seeds(X_ag[:, sl], y, g, X_bs[:, sl], y_bs, w_bs,
                                    agent_weight=None)
        print(f"    {fk} done", flush=True)
    res = compare_fixture(fix, names, y, g, rev, oofs, "BLA")
    res.update({"n_sessions": len(recs), "n_agent": len(cv) + len(rest), "n_bootstrap": len(bs),
                "n_cv": len(cv), "n_rows": int(len(y)), "n_real": int(y.sum()), "masked": masked})
    np.savez(rt.FIXTURES / "selftest_bla_oof.npz", y=y, groups=g, reviewed=rev, names=names,
             animals=animals, **oofs)
    return res


def vca1_pin():
    fix = np.load(VCA1_FIX, allow_pickle=True)
    pin = np.load(VCA1_PIN, allow_pickle=True)

    def ff(sd):
        if rt.is_bootstrap(sd):
            return VCA1_ARMS / f"{rt.key(rt.rel(sd))}__b0.npz"
        return sd / "candidate_features_v2.npz"
    recs = rt.load_pool("vCA1", width=35, feature_file_of=ff)
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    masked = int((w_bs == 0).sum())
    print(f"  vCA1 pool: {len(recs)} sessions ({len(cv)+len(rest)} agent, {len(bs)} bootstrap), "
          f"CV {len(cv)} sessions / {len(y)} rows / {int(y.sum())} real; masked {masked}; "
          f"bootstrap flag rows {int(X_bs[:, 34].sum())}, ring nonzero {int((X_bs[:, 33] != 0).sum())}")
    oofs = {}
    for fk, sl in (("oof_b13", slice(0, 13)), ("oof_v35", slice(0, 35))):
        oofs[fk] = rt.run_oof_seeds(X_ag[:, sl], y, g, X_bs[:, sl], y_bs, w_bs,
                                    agent_weight=rt.AGENT_WEIGHT_OVERRIDE["vCA1"])
        print(f"    {fk} done", flush=True)
    res = compare_fixture(fix, names, y, g, rev, oofs, "vCA1")
    md = float(np.max(np.abs(pin["oof_seeds"] - oofs["oof_b13"]))) \
        if pin["oof_seeds"].shape == oofs["oof_b13"].shape else float("nan")
    res["b13_vs_pinned_baseline"] = {"max_abs_score_diff": md, "pass": bool(md < 1e-6)}
    print(f"  vCA1 b13 vs _pinned/baseline_oof.npz: max|diff| {md:.2e}")
    res.update({"n_sessions": len(recs), "n_agent": len(cv) + len(rest), "n_bootstrap": len(bs),
                "n_cv": len(cv), "n_rows": int(len(y)), "n_real": int(y.sum()), "masked": masked})
    np.savez(rt.FIXTURES / "selftest_vca1_oof.npz", y=y, groups=g, reviewed=rev, names=names,
             animals=animals, **oofs)
    return res


def joblib_meta(area):
    d = joblib.load(rt.MODEL_DIR[area] / "classifier.joblib")
    return {k: d.get(k) for k in ("model_type", "reject_threshold", "agent_weight", "n_sessions",
                                  "n_features", "n_excluded_ambiguous", "feature_version")}


def main():
    t = rt.Timer()
    rt.FIXTURES.mkdir(exist_ok=True)
    pin_hash = rt.assert_pinned()
    print(f"pin {pin_hash[:12]} unchanged")
    res = {"pin_hash": pin_hash}
    print("\n[1] synthetic orientation")
    res["synthetic"] = synthetic_orientation()
    print("\n[2] BLA pin reproduction")
    res["bla"] = bla_pin()
    print("\n[3] vCA1 pin reproduction")
    res["vca1"] = vca1_pin()
    print("\n[4] joblib metadata vs pools")
    res["joblib"] = {a: joblib_meta(a) for a in ("BLA", "vCA1", "DG_AL")}
    dg = rt.load_pool("DG_AL")
    res["dg_al"] = {"n_sessions": len(dg), "n_agent": sum(1 for r in dg if not r["is_bootstrap"])}
    for a in ("BLA", "vCA1"):
        print(f"  {a} joblib: {res['joblib'][a]}")
    print(f"  DG_AL: {res['dg_al']}")
    checks = {
        "synthetic": all(v["pass"] for v in res["synthetic"].values()),
        "bla_b13": res["bla"]["oof_b13"]["pass"], "bla_v2b": res["bla"]["oof_v2b"]["pass"],
        "bla_rows": res["bla"]["rows_match"] and res["bla"]["names_match"] and res["bla"]["reviewed_match"],
        "vca1_b13": res["vca1"]["oof_b13"]["pass"], "vca1_v35": res["vca1"]["oof_v35"]["pass"],
        "vca1_rows": res["vca1"]["rows_match"] and res["vca1"]["names_match"] and res["vca1"]["reviewed_match"],
        "vca1_pin_b13": res["vca1"]["b13_vs_pinned_baseline"]["pass"],
        "masked_bla": res["bla"]["masked"] == res["joblib"]["BLA"]["n_excluded_ambiguous"] == 4494,
        "masked_vca1": res["vca1"]["masked"] == res["joblib"]["vCA1"]["n_excluded_ambiguous"] == 6662,
        "counts": (res["bla"]["n_sessions"], res["vca1"]["n_sessions"], res["dg_al"]["n_sessions"]) == (170, 134, 9),
        "joblib_bla_sessions": res["joblib"]["BLA"]["n_sessions"] == res["bla"]["n_sessions"],
        "joblib_vca1_sessions_parked": res["joblib"]["vCA1"]["n_sessions"] == res["vca1"]["n_sessions"] - 1,
    }
    res["checks"] = checks
    res["all_pass"] = all(checks.values())
    print("\nCHECKS:")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    print("SELFTEST:", "ALL PASS" if res["all_pass"] else "FAIL", f"({t.s()} s)")
    rt.write_result("selftest", res, __file__, t)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
