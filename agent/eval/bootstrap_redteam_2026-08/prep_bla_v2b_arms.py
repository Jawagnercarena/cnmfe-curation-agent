"""
prep_bla_v2b_arms.py -> fixtures/bla_v2b_arms.npz  (attack #14 data prep)

Real v2b for the 91 BLA bootstrap sessions from the persisted candidate traces
and footprints (bootstrap_candidates.npz: C_raw + C-order footprints), the
mirror of the vCA1 arm-b0 backfill.  Arm matrices are written ONLY here,
never into session dirs.

Arms (all keep the live first 26 columns bit-identical; flag column = 1):
  b0_lso  : Cn=None (ring_contrast 0), hiconf = leave-session-out 13-col
            scores >= 0.5 (fixtures/boot_oos_BLA.npz, out-of-sample)
  b0prod  : Cn=None, hiconf = the deployed joblib's companion first-pass
            model on the live 13 columns (in-sample for these rows -- the
            production analogue, deliberately the opposite extreme)
  b1_lso  : as b0_lso but with the session Cn.mat where its resolution
            equals (d1, d2) -- real ring_contrast on those sessions
Read-only on the corpus.
"""
import sys

import joblib
import numpy as np

import rt_lib as rt

ARMS = ("b0_lso", "b0prod", "b1_lso")


def main():
    t = rt.Timer()
    rt.assert_pinned()
    oos = np.load(rt.FIXTURES / "boot_oos_BLA.npz", allow_pickle=True)
    oos_sess, oos_row, oos13 = oos["session"], oos["row"], oos["score_13"]
    jl = joblib.load(rt.MODEL_DIR["BLA"] / "classifier.joblib")
    fp_sc, fp_clf = jl["first_pass_scaler"], jl["first_pass_clf"]
    assert fp_sc.n_features_in_ == 13
    out, meta = {}, {"sessions": [], "n_cn_same_res": 0, "hiconf_agreement": []}
    for sd in rt.labeled_sessions("BLA"):
        if not rt.is_bootstrap(sd):
            continue
        r = rt.load_record(sd, "BLA", width=35)
        X13 = r["X"][:, :13]
        bc = rt.load_bootstrap_candidates(sd)
        assert bc["C_raw"].shape[0] == len(X13) == bc["A_rows_C"].shape[0], r["name"]
        imgs = rt.bootstrap_images(bc)                       # (N, d1, d2) order='C'
        m = oos_sess == r["name"]
        assert m.sum() == len(X13) and (oos_row[m] == np.arange(len(X13))).all(), r["name"]
        hi_lso = oos13[m] >= rt.F.HICONF_SCORE
        s_prod = fp_clf.predict_proba(fp_sc.transform(X13))[:, 1]
        hi_prod = s_prod >= rt.F.HICONF_SCORE
        Cn = rt.load_cn(sd / "Cn.mat")
        cn_ok = Cn is not None and Cn.shape == (bc["d1"], bc["d2"])
        meta["n_cn_same_res"] += int(cn_ok)
        v2b = {
            "b0_lso": rt.F.compute_v2b_features(bc["C_raw"], imgs, None, hi_lso),
            "b0prod": rt.F.compute_v2b_features(bc["C_raw"], imgs, None, hi_prod),
        }
        v2b["b1_lso"] = rt.F.compute_v2b_features(bc["C_raw"], imgs, Cn if cn_ok else None, hi_lso)
        for arm in ARMS:
            X35 = rt.F.assemble_v2_matrix(X13, v2b[arm], 1.0)
            assert X35.shape == r["X"].shape
            assert np.array_equal(X35[:, :26], r["X"][:, :26]), f"{r['name']}: first 26 cols moved"
            out[f"{rt.key(r['name'])}__{arm}"] = X35
        agree = float(np.mean(hi_lso == hi_prod))
        meta["hiconf_agreement"].append(agree)
        meta["sessions"].append({"rel": r["name"], "n": int(len(X13)), "cn_same_res": bool(cn_ok),
                                 "hiconf_lso": int(hi_lso.sum()), "hiconf_prod": int(hi_prod.sum()),
                                 "hiconf_agreement": agree,
                                 "ring_nonzero_b1": int((v2b["b1_lso"][:, 7] != 0).sum())})
        rt.log(f"  {r['name']}: N {len(X13)} hiconf lso {int(hi_lso.sum())} / prod {int(hi_prod.sum())} "
               f"(agree {agree:.3f}) Cn same-res {cn_ok}")
    np.savez(rt.FIXTURES / "bla_v2b_arms.npz", **out)
    meta["hiconf_agreement_mean"] = float(np.mean(meta["hiconf_agreement"]))
    meta["hiconf_agreement_min"] = float(np.min(meta["hiconf_agreement"]))
    rt.log(f"done: {len(meta['sessions'])} sessions, Cn same-res {meta['n_cn_same_res']}, "
           f"hiconf agreement mean {meta['hiconf_agreement_mean']:.3f} min {meta['hiconf_agreement_min']:.3f}")
    rt.write_result("bla_v2b_arms_meta", meta, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
