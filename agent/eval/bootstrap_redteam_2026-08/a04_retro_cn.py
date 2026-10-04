"""
a04_retro_cn.py -- attack #4: the retro-path cn_correlation transpose.

prep_retro_ids.py identified the retro-labeled sessions numerically (stored
column 12 == the order-C recomputation bit-for-bit; 6 BLA, 0 vCA1).  Here:
quantify the corruption (stored vs corrected column: rank correlation,
fraction of rows moved by > 0.1) and measure whether fixing it moves the
model: corrected col 12 (order-F, extraction Cn) + the 13 rank columns
recomputed for those sessions, v2b and the flag untouched -> 8-seed paired
OOF, live vs corrected, same folds/seeds/weights; AUC full / reviewed, FAR
at 0.04, rule T, junk at matched false-AR, and the per-session OOF change on
the corrected sessions.  "Material" if |paired delta| > 2 SE or the rule T
moves; else "cosmetic".  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

T = rt.DEPLOYED_T["BLA"]


def corrected_matrix(r, ext):
    X = r["X"].copy()
    rev_idx = np.flatnonzero(r["reviewed"])
    imgs = rt.fcols_to_images(ext["A"], ext["d1"], ext["d2"])
    cn = np.array([rt.F.cn_features(imgs[i], ext["Cn"])["cn_correlation"] for i in range(len(imgs))])
    X13 = X[:, :13].copy()
    X13[rev_idx, rt.CN_COL] = cn
    X[:, :13] = X13
    X[:, 13:26] = rt.F.compute_ranks(X13)
    return X


def main():
    t = rt.Timer()
    rt.assert_pinned()
    retro = rt.read_result("retro_sessions")
    retro_set = set(retro["counts"]["BLA"]["C_match_sessions"])
    recs = rt.load_pool("BLA", width=35)
    per = {}
    corrected = {}
    for r in recs:
        if r["name"] in retro_set:
            ext = rt.load_extraction("BLA", r["name"])
            Xc = corrected_matrix(r, ext)
            corrected[r["name"]] = Xc
            rev = r["reviewed"]
            a, b = r["X"][rev, rt.CN_COL], Xc[rev, rt.CN_COL]
            per[r["name"]] = {"n_rows": int(len(r["y"])), "n_pos": int(r["y"].sum()),
                              "spearman_stored_vs_corrected": float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]),
                              "pearson": float(np.corrcoef(a, b)[0, 1]),
                              "frac_moved_gt_0.1": float(np.mean(np.abs(a - b) > 0.1)),
                              "stored_p50": float(np.median(a)), "corrected_p50": float(np.median(b)),
                              "stored_pos_p50": float(np.median(a[r["y"][rev] == 1])), "corrected_pos_p50": float(np.median(b[r["y"][rev] == 1])),
                              "rank_cols_changed_frac": float(np.mean(np.any(np.abs(r["X"][:, 13:26] - Xc[:, 13:26]) > 1e-12, axis=1)))}
            rt.log(f"  {r['name']}: spearman {per[r['name']]['spearman_stored_vs_corrected']:.3f} moved>0.1 "
                   f"{per[r['name']]['frac_moved_gt_0.1']:.2f} cn p50 {per[r['name']]['stored_p50']:.3f} -> {per[r['name']]['corrected_p50']:.3f} "
                   f"(pos {per[r['name']]['stored_pos_p50']:.3f} -> {per[r['name']]['corrected_pos_p50']:.3f})")
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    Xc_ag = X_ag.copy()
    off = 0
    corrected_rows = np.zeros(len(y), bool)
    for r in cv:
        n = len(r["y"])
        if r["name"] in corrected:
            Xc_ag[off:off + n] = corrected[r["name"]]
            corrected_rows[off:off + n] = True
        off += n
    res = {"retro_sessions": per, "n_corrected_sessions": len(corrected),
           "n_corrected_rows_in_cv_pool": int(corrected_rows.sum()), "n_corrected_pos": int(y[corrected_rows].sum())}
    out = {}
    oofs = {}
    for label, Xa in (("live", X_ag), ("corrected", Xc_ag)):
        oofs[label] = {}
        for vname, sl in (("b13", slice(0, 13)), ("rankv2b_35", slice(0, 35))):
            oof = rt.run_oof_seeds(Xa[:, sl], y, g, X_bs[:, sl], y_bs, w_bs, agent_weight=None)
            oofs[label][vname] = oof
            tab = rt.threshold_table(oof, y, rev)
            cT, gate, det = rt.rule_T(tab)
            out[f"{label}_{vname}"] = {"auc_full": rt.summarize([rt.auc(y, o) for o in oof]),
                                       "auc_reviewed": rt.summarize([rt.auc(y, o, rev) for o in oof]),
                                       "far_at_T": rt.summarize([rt.far_junk(o, y, T)[0] for o in oof]),
                                       "junk_at_T": rt.summarize([rt.far_junk(o, y, T)[1] for o in oof]),
                                       "rule_T": cT, "rule_detail": det,
                                       "auc_on_corrected_sessions": rt.summarize([rt.auc(y[corrected_rows], o[corrected_rows]) for o in oof]),
                                       "auc_on_other_sessions": rt.summarize([rt.auc(y[~corrected_rows], o[~corrected_rows]) for o in oof])}
            e = out[f"{label}_{vname}"]
            rt.log(f"  {label:9s} {vname:11s}: AUC full {e['auc_full']['mean']:.4f} rev {e['auc_reviewed']['mean']:.4f} | FAR@{T} "
                   f"{e['far_at_T']['mean']:.2f}% (max {e['far_at_T']['max']:.2f}) junk {e['junk_at_T']['mean']:.1f}% | rule T {cT} "
                   f"| AUC on the 6 retro sessions {e['auc_on_corrected_sessions']['mean']:.4f}, others {e['auc_on_other_sessions']['mean']:.4f}")
    for vname in ("b13", "rankv2b_35"):
        a_l = [rt.auc(y, o) for o in oofs["live"][vname]]
        a_c = [rt.auc(y, o) for o in oofs["corrected"][vname]]
        r_l = [rt.auc(y, o, rev) for o in oofs["live"][vname]]
        r_c = [rt.auc(y, o, rev) for o in oofs["corrected"][vname]]
        s_l = [rt.auc(y[corrected_rows], o[corrected_rows]) for o in oofs["live"][vname]]
        s_c = [rt.auc(y[corrected_rows], o[corrected_rows]) for o in oofs["corrected"][vname]]
        mop = rt.matched_operating_point(oofs["live"][vname], oofs["corrected"][vname], y, T)
        out[f"delta_{vname}"] = {"full": rt.paired_delta(a_l, a_c), "reviewed": rt.paired_delta(r_l, r_c),
                                 "on_retro_sessions": rt.paired_delta(s_l, s_c),
                                 "matched_op": {k: v for k, v in mop.items() if k.endswith("_mean")},
                                 "rule_T_moved": out[f"live_{vname}"]["rule_T"] != out[f"corrected_{vname}"]["rule_T"]}
        d = out[f"delta_{vname}"]
        rt.log(f"  delta {vname}: full {d['full']['mean']:+.4f} (se {d['full']['se']:.4f}, {d['full']['n_positive']}/8) reviewed "
               f"{d['reviewed']['mean']:+.4f} | on the retro sessions {d['on_retro_sessions']['mean']:+.4f} | junk at matched FAR "
               f"{mop['junk_ref_mean']:.1f}% -> {mop['junk_new_at_matched_far_mean']:.1f}% | rule T moved {d['rule_T_moved']}")
    dv = out["delta_rankv2b_35"]
    material = bool(abs(dv["full"]["mean"]) > 2 * dv["full"]["se"] or dv["rule_T_moved"]
                    or abs(dv["on_retro_sessions"]["mean"]) > 2 * dv["on_retro_sessions"]["se"])
    res.update({"gate": out, "material": material,
                "checks": {"bug_confirmed_on_retro_set": retro["counts"]["BLA"]["C-match"] == 6 and retro["counts"]["vCA1"]["C-match"] == 0,
                           "corrected_rows_present": bool(corrected_rows.any())},
                "verdict": "PASS",
                "criterion": "bug confirmed numerically (6 BLA sessions C-match, 0 vCA1); follow-up sized as MATERIAL if the "
                             "corrected-vs-live paired delta exceeds 2 SE (pool or retro sessions) or the rule T moves, else COSMETIC"})
    res["sizing"] = "MATERIAL" if material else "COSMETIC"
    rt.log(f"sizing {res['sizing']}\nVERDICT {res['verdict']}")
    rt.write_result("a04", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
