"""
a03_labels_from_mat.py -- attack #3: are agent-session labels unaffected?

Recompute labels.mat for all 102 labeled agent sessions from footprints only:
review side = the .feature_expansion extraction A (bit-identical to
review_neuron.mat, MATLAB F-order columns), final side = the session's
spatial_footprints.mat (written from the same neuron.A the labels were
matched against, CNMFe_final_save.m:725-737).  Both rules are applied and
reported: the greedy per-final-neuron rule at 0.60 (CNMFe_final_save.m:
628-655, prospective sessions) and Hungarian at 0.60 (train_classifier.py:
212-221, the retro path).  Control: transpose the review side (order-C) and
recompute -- agreement must collapse.  The MATLAB load-only step lives in
matlab/m03_loadonly_check.m and closes the last link (full(neuron.A) ==
spatial_footprints, full(rn.neuron.A) == extraction A).  Read-only.
"""
import json
import sys

import numpy as np

import rt_lib as rt


def labels_from(rev_rows, fin_rows, rule):
    sim = rt.cosine_rows(rev_rows, fin_rows)           # N_review x N_final
    if rule == "greedy":
        return rt.greedy_per_final(sim, rt.FINAL_SAVE_THR), sim
    lab = np.zeros(sim.shape[0], int)
    for j, k, s in rt.hungarian_pairs(sim, rt.FINAL_SAVE_THR):
        lab[j] = 1
    return lab, sim


def main():
    t = rt.Timer()
    rt.assert_pinned()
    retro = rt.read_result("retro_sessions") if (rt.RESULTS / "retro_sessions.json").exists() else None
    retro_set = set()
    if retro:
        for a in retro["counts"]:
            retro_set |= set(retro["counts"][a]["C_match_sessions"])
    rows = []
    for area in ("BLA", "vCA1"):
        for sd in rt.labeled_sessions(area):
            if rt.is_bootstrap(sd):
                continue
            r = rt.load_record(sd, area)
            ext = rt.load_extraction(area, r["name"])
            if ext is None:
                rows.append({"rel": r["name"], "status": "no_extraction"})
                continue
            rev_idx = np.flatnonzero(r["reviewed"])
            y_rev = r["y"][rev_idx]
            fin = rt.load_curated_stack(sd)                     # final kept set
            d1, d2 = ext["d1"], ext["d2"]
            if fin.shape[1:] != (d1, d2):
                rows.append({"rel": r["name"], "status": "dims_mismatch", "fin": list(fin.shape)})
                continue
            A = ext["A"]
            rev_F = A.T                                          # MATLAB pixel order, as-is
            fin_F = rt.stack_to_rows(fin, "F")                   # final images in MATLAB order
            # consistent comparison (both MATLAB order)
            res = {}
            for rule in ("greedy", "hungarian"):
                lab, sim = labels_from(rev_F, fin_F, rule)
                res[rule] = {"agree": float(np.mean(lab == y_rev)), "n_mismatch": int((lab != y_rev).sum()),
                             "n_pos_recomputed": int(lab.sum())}
            # control: review side transposed (order-C reshape of F-order columns, then C-flatten)
            rev_T = rt.stack_to_rows(rt.ccols_to_images(A, d1, d2), "F")
            lab_c, _ = labels_from(rev_T, fin_F, "greedy")
            res["transposed_control"] = {"agree": float(np.mean(lab_c == y_rev)),
                                         "n_pos_recomputed": int(lab_c.sum())}
            best = max(("greedy", "hungarian"), key=lambda k: res[k]["agree"])
            rows.append({"rel": r["name"], "status": "ok", "n_reviewed": int(len(rev_idx)),
                         "n_pos_labels": int(y_rev.sum()), "n_final": int(fin.shape[0]),
                         "in_retro_set": r["name"] in retro_set, **res, "best_rule": best,
                         "exact": bool(res[best]["n_mismatch"] == 0)})
            rt.log(f"  {area} {r['name']}: greedy agree {res['greedy']['agree']:.4f} "
                   f"({res['greedy']['n_mismatch']} off) hungarian {res['hungarian']['agree']:.4f} "
                   f"({res['hungarian']['n_mismatch']} off) | transposed {res['transposed_control']['agree']:.3f} "
                   f"| labels {int(y_rev.sum())} final {fin.shape[0]}")
    ok = [x for x in rows if x["status"] == "ok"]
    exact = [x for x in ok if x["exact"]]
    mism = [x for x in ok if not x["exact"]]
    summary = {
        "n_sessions": len(rows), "n_ok": len(ok), "n_exact": len(exact),
        "n_exact_greedy": sum(1 for x in ok if x["greedy"]["n_mismatch"] == 0),
        "n_exact_hungarian": sum(1 for x in ok if x["hungarian"]["n_mismatch"] == 0),
        "mean_agree_consistent": float(np.mean([x[x["best_rule"]]["agree"] for x in ok])),
        "mean_agree_transposed": float(np.mean([x["transposed_control"]["agree"] for x in ok])),
        "mean_pos_recomputed_transposed": float(np.mean([x["transposed_control"]["n_pos_recomputed"] for x in ok])),
        "mean_pos_labels": float(np.mean([x["n_pos_labels"] for x in ok])),
        "mismatched_sessions": [{"rel": x["rel"], "greedy_off": x["greedy"]["n_mismatch"],
                                 "hungarian_off": x["hungarian"]["n_mismatch"], "n_reviewed": x["n_reviewed"]}
                                for x in mism],
    }
    checks = {"ge_100_of_102_exact": len(exact) >= 100 and len(ok) == 102,
              "transposed_collapses": summary["mean_agree_transposed"] < summary["mean_agree_consistent"] - 0.2
              or summary["mean_pos_recomputed_transposed"] < 0.5 * summary["mean_pos_labels"]}
    res = {"sessions": rows, "summary": summary, "checks": checks,
           "verdict": "PASS" if all(checks.values()) else "FAIL",
           "criterion": ">= 100/102 sessions reproduce labels.mat exactly from footprints under consistent "
                        "(MATLAB) pixel order, and the transposed control collapses; MATLAB load-only "
                        "check in matlab/m03_loadonly_check.m closes the neuron.mat / review_neuron.mat link",
           "matlab_check": "see results/a03_matlab.json (written by the MATLAB step's wrapper)"}
    rt.log(f"summary {json.dumps({k: v for k, v in summary.items() if k != 'mismatched_sessions'})}")
    rt.log(f"mismatched: {summary['mismatched_sessions']}")
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a03", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
