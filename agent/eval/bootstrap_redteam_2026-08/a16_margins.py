"""
a16_margins.py -- hypothesis D16: bootstrap positives are easy.

Out-of-sample score distributions of bootstrap positives (leave-session-out,
fixtures/boot_oos_{area}.npz; BLA under the deployed 35-col contract and the
13-col companion, vCA1 13-col) against agent positives' OOF scores (the
reproduced pins, fixtures/selftest_*_oof.npz): quantiles, the hard band
[0.02, 0.10), fraction below the deployed T, fraction >= 0.5; score vs pair
similarity; per session.  Supported if the bootstrap hard-band fraction is
less than half the agent positives'.  Read-only, no fits.
"""
import sys

import numpy as np

import rt_lib as rt


def dist(s, T):
    s = np.asarray(s, float)
    return {"n": int(len(s)), "p05": float(np.percentile(s, 5)), "p25": float(np.percentile(s, 25)),
            "p50": float(np.median(s)), "p75": float(np.percentile(s, 75)),
            "frac_hard_band_0.02_0.10": float(np.mean((s >= 0.02) & (s < 0.10))),
            "frac_below_T": float(np.mean(s < T)), "frac_below_0.02": float(np.mean(s < 0.02)),
            "frac_ge_0.5": float(np.mean(s >= 0.5)), "frac_ge_0.9": float(np.mean(s >= 0.9))}


def main():
    t = rt.Timer()
    rt.assert_pinned()
    res = {"areas": {}}
    for area, fix_name, keys in (("BLA", "selftest_bla_oof.npz", [("score_35", "oof_v2b"), ("score_13", "oof_b13")]),
                                 ("vCA1", "selftest_vca1_oof.npz", [("score_13", "oof_b13")])):
        oos = np.load(rt.FIXTURES / f"boot_oos_{area}.npz", allow_pickle=True)
        fix = np.load(rt.FIXTURES / fix_name, allow_pickle=True)
        T = rt.DEPLOYED_T[area]
        yb, wb, sess = oos["y"], oos["w"], oos["session"]
        y_ag = fix["y"]
        # pair similarity per bootstrap positive row
        sim_of = np.full(len(yb), np.nan)
        for s in np.unique(sess):
            js = rt.load_json(rt.DATA_ROOT[area] / s)
            m = sess == s
            rows = oos["row"][m]
            simmap = {int(c): float(v) for c, v in zip(js["candidate_indices"][:js["n_matched"]], js["pair_similarities"][:js["n_matched"]])}
            sim_of[np.flatnonzero(m)] = [simmap.get(int(r), np.nan) for r in rows]
        out = {}
        for oos_key, fix_key in keys:
            sb = oos[oos_key]
            sa = fix[fix_key].mean(axis=0)              # seed-mean OOF per agent row
            pos_b = sb[yb == 1]
            pos_a = sa[y_ag == 1]
            neg_b = sb[(yb == 0) & (wb > 0)]
            neg_a = sa[y_ag == 0]
            d = {"bootstrap_pos": dist(pos_b, T), "agent_pos": dist(pos_a, T),
                 "bootstrap_neg_unmasked": dist(neg_b, T), "agent_neg": dist(neg_a, T),
                 "auc_bootstrap_rows_unmasked": rt.auc(yb[wb > 0], sb[wb > 0]),
                 "auc_agent_rows": rt.auc(y_ag, sa)}
            # score vs similarity among bootstrap positives
            sp = sim_of[yb == 1]
            ok = np.isfinite(sp)
            bins = [(0.45, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.01)]
            d["by_similarity"] = {f"{lo:.2f}-{hi:.2f}": {"n": int(((sp >= lo) & (sp < hi) & ok).sum()),
                                                        "score_p50": float(np.median(pos_b[(sp >= lo) & (sp < hi) & ok])) if ((sp >= lo) & (sp < hi) & ok).any() else None,
                                                        "frac_hard": float(np.mean((pos_b[(sp >= lo) & (sp < hi) & ok] >= 0.02) & (pos_b[(sp >= lo) & (sp < hi) & ok] < 0.10))) if ((sp >= lo) & (sp < hi) & ok).any() else None}
                                  for lo, hi in bins}
            d["spearman_score_vs_sim"] = float(np.corrcoef(np.argsort(np.argsort(pos_b[ok])), np.argsort(np.argsort(sp[ok])))[0, 1])
            # per-session hard-band fraction for bootstrap positives
            per = {}
            for s in np.unique(sess):
                m = (sess == s) & (yb == 1)
                if m.sum() >= 5:
                    per[s] = float(np.mean((sb[m] >= 0.02) & (sb[m] < 0.10)))
            d["per_session_hard_frac"] = {"p50": float(np.median(list(per.values()))), "p90": float(np.percentile(list(per.values()), 90)),
                                          "n_sessions": len(per)}
            d["supported"] = bool(d["bootstrap_pos"]["frac_hard_band_0.02_0.10"] < 0.5 * d["agent_pos"]["frac_hard_band_0.02_0.10"])
            out[oos_key] = d
            rt.log(f"  {area} {oos_key}: bootstrap pos hard-band {d['bootstrap_pos']['frac_hard_band_0.02_0.10']:.3f} "
                   f"(<T {d['bootstrap_pos']['frac_below_T']:.3f}, p50 {d['bootstrap_pos']['p50']:.3f}) vs agent pos hard-band "
                   f"{d['agent_pos']['frac_hard_band_0.02_0.10']:.3f} (<T {d['agent_pos']['frac_below_T']:.3f}, p50 {d['agent_pos']['p50']:.3f}) "
                   f"| spearman score~sim {d['spearman_score_vs_sim']:.2f} | supported {d['supported']}")
        res["areas"][area] = out
    sup = {a: all(v["supported"] for v in res["areas"][a].values()) for a in res["areas"]}
    res["hypothesis_D16_supported"] = sup
    res["verdict"] = "PASS"
    res["criterion"] = ("D16 supported for an area if the bootstrap positives' hard-band fraction [0.02,0.10) is < half "
                        "the agent positives'; the attack itself is descriptive (PASS = computed)")
    rt.log(f"D16 supported {sup}")
    rt.write_result("a16", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
