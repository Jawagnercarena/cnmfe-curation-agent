"""
a01_orientation.py -- attack #1: re-derive the orientation claim without bmlib.

Step 1  MATLAB linearization: for one bootstrap session per area (plus one
        v7.3 stack), stack[n].flatten('F') must equal A.txt[:, n] to text
        precision and the C-order flatten must not.
Step 2  Reproduce the stored March numbers on the 4 real _bootstrap_validate
        sandboxes: MIXED-order cosine (sandbox candidates flattened C-order vs
        the session's final curated footprints flattened F-order) -> Hungarian
        -> best-first pair sims must equal validation_match_stats.json
        (bla21: 37/50 above 0.45, top-5 0.944/0.924/0.881/0.865/0.802).
        Consistent order -> recovery, median sim, centroid distance to the
        TRUE position vs to the MIRROR position under both metrics.
Step 3  What cannot be reproduced: the pre-fix corpus JSONs are gone, so the
        "median best pair 0.852 / never 0.99" signature is INCONCLUSIVE-by-loss
        (stated, not scored).
Step 4  All 202 live bootstrap sessions: recomputed FIXED-metric cosine
        (candidate C-order rows vs curated C-order rows) == stored sim_matrix;
        Hungarian pairs == JSON pairs; labels.mat positives ==
        candidate_indices[:n_matched]; N agrees across npz / JSON / npz-cand /
        labels.  Read-only.
"""
import json
import sys

import numpy as np
from scipy import sparse

import rt_lib as rt

SANDBOXES = [
    ("BLA", "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"),
    ("BLA", "2tones/AVG5x-TSeries-100125-bla12-639um-23z-000"),
    ("BLA", "4odorDO/AVG5x-TSeries-02092026-bla12-681um-22z-000"),
    ("BLA", "Valence/AVG5x-TSeries-121225-bla12-652um-23z-000"),
    ("BLA", "2tones/AVG5x-TSeries-100125-bla8-735um-23z-000"),      # empty shell
]
CLAIMED_BLA21 = {"matched_45": 37, "top5": [0.944, 0.924, 0.881, 0.865, 0.802]}


def step1():
    out = {}
    picks = []
    for area in ("BLA", "vCA1"):
        for sd in rt.labeled_sessions(area):
            if rt.is_bootstrap(sd) and (sd / "A.txt").exists():
                picks.append((area, sd))
                break
    # one v7.3 stack (vCA1 has 33)
    for sd in rt.labeled_sessions("vCA1"):
        if rt.is_bootstrap(sd):
            try:
                import scipy.io as sio
                sio.loadmat(str(sd / "spatial_footprints.mat"))
            except NotImplementedError:
                picks.append(("vCA1-v7.3", sd))
                break
    for tag, sd in picks:
        stack = rt.load_curated_stack(sd)
        A = rt.load_A_txt(sd)
        n = min(stack.shape[0], A.shape[1])
        scale = float(np.max(np.abs(A))) or 1.0
        dF = max(float(np.max(np.abs(stack[k].flatten(order="F") - A[:, k]))) for k in range(n)) / scale
        dC = max(float(np.max(np.abs(stack[k].flatten(order="C") - A[:, k]))) for k in range(n)) / scale
        out[tag] = {"session": rt.rel(sd), "n": int(n), "stack_shape": list(stack.shape),
                    "rel_diff_F": dF, "rel_diff_C": dC, "pass": bool(dF < 1e-3 and dC > 100 * max(dF, 1e-12))}
        rt.log(f"  step1 {tag} {rt.rel(sd)}: N {n} rel diff F {dF:.2e} vs C {dC:.2e} -> "
               f"{'PASS' if out[tag]['pass'] else 'FAIL'}")
    return out


def step2():
    out = {}
    for area, relp in SANDBOXES:
        sd = rt.DATA_ROOT[area] / relp
        sb = sd / "_bootstrap_validate"
        rec = {"sandbox_exists": sb.exists(),
               "sandbox_files": sorted(p.name for p in sb.iterdir()) if sb.exists() else []}
        js_path = sd / "validation_match_stats.json"
        tj_path = sd / "validation_match_stats_temporal.json"
        rec["json_present"] = js_path.exists()
        if not (sb / "spatial_footprints.mat").exists():
            rec["status"] = "no_candidates_cached"
            out[relp] = rec
            rt.log(f"  step2 {relp}: no cached candidates -> unreproducible")
            continue
        cand = rt.load_stack(sb / "spatial_footprints.mat")           # (N_cand, d1, d2)
        cur = rt.load_curated_stack(sd)                                # (N_final, d1, d2)
        stored = json.loads(js_path.read_text()) if js_path.exists() else None
        tj = json.loads(tj_path.read_text()) if tj_path.exists() else None
        stored_sims = None
        if stored is not None:
            stored_sims = sorted(stored["pair_similarities"], reverse=True)
        elif tj is not None:
            stored_sims = sorted(tj["strategies"]["spatial"]["pair_scores"], reverse=True)
        rows_C = rt.stack_to_rows(cand, "C")
        cur_C = rt.stack_to_rows(cur, "C")
        cur_F = rt.stack_to_rows(cur, "F")
        mixed = rt.cosine_rows(rows_C, cur_F)
        fixed = rt.cosine_rows(rows_C, cur_C)
        pm = rt.hungarian_pairs(mixed, -1.0)                            # all pairs, best-first
        pf = rt.hungarian_pairs(fixed, -1.0)
        sims_m = [p[2] for p in pm]
        sims_f = [p[2] for p in pf]
        cc = rt.centroids(cand)
        kc = rt.centroids(cur)
        def dists(pairs):
            true_d, mirror_d = [], []
            for j, k, s in pairs:
                if s > rt.MATCH_THR:
                    true_d.append(float(np.hypot(*(cc[j] - kc[k]))))
                    mirror_d.append(float(np.hypot(cc[j][0] - kc[k][1], cc[j][1] - kc[k][0])))
            return true_d, mirror_d
        tm, mm = dists(pm)
        tf, mf = dists(pf)
        rec.update({
            "n_candidates": int(cand.shape[0]), "n_final": int(cur.shape[0]),
            "stored_sims_top5": stored_sims[:5] if stored_sims else None,
            "stored_matched_45": int(sum(s > rt.MATCH_THR for s in stored_sims)) if stored_sims else None,
            "mixed_sims_top5": [round(s, 4) for s in sims_m[:5]],
            "mixed_matched_45": int(sum(s > rt.MATCH_THR for s in sims_m)),
            "mixed_max_abs_diff_vs_stored": float(np.max(np.abs(np.array(sims_m) - np.array(stored_sims)))) if stored_sims and len(stored_sims) == len(sims_m) else None,
            "mixed_median_true_dist_px": float(np.median(tm)) if tm else None,
            "mixed_median_mirror_dist_px": float(np.median(mm)) if mm else None,
            "fixed_matched_45": int(sum(s > rt.MATCH_THR for s in sims_f)),
            "fixed_median_sim": float(np.median(sims_f)),
            "fixed_min_sim": float(min(sims_f)),
            "fixed_median_true_dist_px": float(np.median(tf)) if tf else None,
            "fixed_median_mirror_dist_px": float(np.median(mf)) if mf else None,
        })
        rec["reproduces_stored"] = bool(rec["mixed_max_abs_diff_vs_stored"] is not None
                                        and rec["mixed_max_abs_diff_vs_stored"] < 2e-3
                                        and rec["mixed_matched_45"] == rec["stored_matched_45"])
        rec["fixed_recovers_all"] = bool(rec["fixed_matched_45"] == rec["n_final"])
        rec["status"] = "ok"
        out[relp] = rec
        rt.log(f"  step2 {relp}: cand {cand.shape[0]} final {cur.shape[0]} | mixed top5 "
               f"{rec['mixed_sims_top5']} matched {rec['mixed_matched_45']}/{cur.shape[0]} "
               f"(stored {rec['stored_matched_45']}, max|diff| {rec['mixed_max_abs_diff_vs_stored']}) "
               f"true {rec['mixed_median_true_dist_px']} px vs mirror {rec['mixed_median_mirror_dist_px']} px"
               f" | fixed matched {rec['fixed_matched_45']}/{cur.shape[0]} median sim "
               f"{rec['fixed_median_sim']:.3f} true {rec['fixed_median_true_dist_px']} px")
    return out


def step4():
    out = {"per_session": [], "counts": {}}
    for area in ("vCA1", "BLA"):
        n_ok = 0
        for sd in rt.labeled_sessions(area):
            if not rt.is_bootstrap(sd):
                continue
            js = rt.load_json(sd)
            bc = rt.load_bootstrap_candidates(sd)
            r = rt.load_record(sd, area)
            cur = rt.load_curated_stack(sd)
            cur_C = rt.stack_to_rows(cur, "C")
            csr = bc["A_rows_C"]
            # cosine: sparse rows vs dense curated rows, both C-order
            num = np.asarray(csr.dot(cur_C.T))
            n_c = np.sqrt(np.asarray(csr.multiply(csr).sum(axis=1)).ravel()) + 1e-12
            n_k = np.linalg.norm(cur_C, axis=1) + 1e-12
            sim = num / n_c[:, None] / n_k[None, :]
            d_sim = float(np.max(np.abs(sim - bc["sim_matrix"]))) if sim.size else 0.0
            pairs = rt.hungarian_pairs(sim, -1.0)
            n_m = js["n_matched"]
            own_pos = sorted(int(j) for j, k, s in pairs if s > rt.MATCH_THR)
            json_pos = sorted(int(j) for j in js["candidate_indices"][:n_m])
            lab_pos = sorted(int(j) for j in np.flatnonzero(r["y"] == 1))
            own_map = {k: j for j, k, s in pairs if s > rt.MATCH_THR}
            json_map = {int(k): int(j) for j, k in zip(js["candidate_indices"][:n_m], js["curated_indices"][:n_m])}
            rec = {"rel": r["name"], "n_npz": int(r["n_cand"]), "n_json": int(js["n_candidates"]),
                   "n_cand_npz": int(csr.shape[0]), "n_labels": int(len(rt.load_labels(sd))),
                   "n_curated": int(js["n_curated"]), "n_matched": int(n_m),
                   "max_abs_sim_diff": d_sim, "pairs_equal_json": own_map == json_map,
                   "positives_equal_json": own_pos == json_pos, "labels_equal_json": lab_pos == json_pos,
                   "n_equal": int(r["n_cand"]) == int(js["n_candidates"]) == int(csr.shape[0]) == int(len(rt.load_labels(sd)))}
            rec["pass"] = bool(rec["max_abs_sim_diff"] < 1e-5 and rec["pairs_equal_json"]
                               and rec["labels_equal_json"] and rec["n_equal"])
            n_ok += rec["pass"]
            if not rec["pass"]:
                rt.log(f"  step4 FAIL {r['name']}: {rec}")
            out["per_session"].append(rec)
        tot = sum(1 for p in out["per_session"] if p["rel"].startswith(tuple(f"{t.name}/" for t in rt.DATA_ROOT[area].iterdir() if t.is_dir() and not t.name.startswith("."))))
        out["counts"][area] = {"pass": n_ok}
        rt.log(f"  step4 {area}: {n_ok} sessions pass")
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    res = {}
    rt.log("[step 1] MATLAB linearization vs A.txt")
    res["step1"] = step1()
    rt.log("[step 2] March numbers on the sandboxes")
    res["step2"] = step2()
    res["step3"] = {"claim": "corpus signature: median best pair 0.852, no session reaches 0.99 (pre-fix)",
                    "verdict": "INCONCLUSIVE-by-loss",
                    "reason": "pre-fix bootstrap_match_stats.json and candidate caches were overwritten by "
                              "the corpus re-run and never versioned; only per-session counts survive "
                              "(a3_damage.json n_matched_old, b2_gate_b.json old_matched)."}
    rt.log("[step 4] live labels consistent with the fixed metric (all 202)")
    res["step4"] = step4()
    n4 = sum(1 for p in res["step4"]["per_session"] if p["pass"])
    res["n_step4_pass"] = n4
    res["n_step4"] = len(res["step4"]["per_session"])
    s2 = [v for v in res["step2"].values() if v.get("status") == "ok"]
    bla21 = res["step2"].get(SANDBOXES[0][1], {})
    res["claim_bla21"] = {"claimed": CLAIMED_BLA21,
                          "own_mixed_top5": bla21.get("mixed_sims_top5"),
                          "own_mixed_matched_45": bla21.get("mixed_matched_45"),
                          "own_fixed_matched_45": bla21.get("fixed_matched_45")}
    checks = {
        "step1_all": all(v["pass"] for v in res["step1"].values()),
        "step2_reproduces_stored": all(v["reproduces_stored"] for v in s2) and len(s2) == 4,
        "step2_fixed_recovers_all": all(v["fixed_recovers_all"] for v in s2),
        "step2_mixed_mirror_closer": all((v["mixed_median_mirror_dist_px"] or 1e9) < (v["mixed_median_true_dist_px"] or 0) for v in s2),
        "step2_fixed_true_closer": all((v["fixed_median_true_dist_px"] or 1e9) < 3.0 for v in s2),
        "step4_all": n4 == res["n_step4"] == 202,
    }
    res["checks"] = checks
    res["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    res["criterion"] = ("step1 F-order matches A.txt and C-order does not; step2 mixed metric reproduces "
                        "the stored March pair sims (4/4, atol 2e-3) and the consistent metric recovers every "
                        "final neuron with true-position matches < 3 px; step4 202/202 live sessions consistent")
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a01", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
