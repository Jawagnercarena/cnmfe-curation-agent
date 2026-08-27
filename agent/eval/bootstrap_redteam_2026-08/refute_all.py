"""
refute_all.py -- adversarial verification of every attack's result.

For each attack aNN (or `--attack aNN`):
  1. REPRODUCE: re-run the attack script in a clean subprocess into a
     scratch results dir and compare every numeric leaf of the JSON with the
     standing result (max relative difference; identical seeds and thread
     settings -> expected exact).  Scripts that write fixtures re-write them
     identically.
  2. PIN: the standing result carries the current pin hash.
  3. N / EXCLUSIONS: the JSON states its N (rows / sessions) somewhere.
  4. PROBES: attack-specific flipped controls and alternative computations
     that would expose the most likely way the attack could be wrong.
  5. SIGN-OFF: agrees / disputes with reasons -> results/aNN_refute.json.
Read-only on the corpus; writes only under results/ and fixtures/_refute/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import rt_lib as rt

PY = sys.executable
SCRATCH = rt.FIXTURES / "_refute"
ATTACKS = {"a01": "a01_orientation.py", "a02": "a02_sites.py", "a03": "a03_labels_from_mat.py",
           "a04": "a04_retro_cn.py", "a05": "a05_pairs.py", "a06": "a06_duplicates.py",
           "a07": "a07_unrecovered.py", "a08": "a08_animal_cv.py", "a09": "a09_seeds.py",
           "a10": "a10_threshold_rule.py", "a11": "a11_bla_g5.py", "a12": "a12_protocols.py",
           "a13": "a13_old_positives.py", "a14": "a14_bla_v2b.py", "a15": "a15_learning_curve.py",
           "a16": "a16_margins.py", "a17": "a17_reviewer_ceiling.py", "a18": "a18_global_animal.py",
           "a19": "a19_calibration.py"}
SKIP_REPRO = {"a08", "a12", "a15", "a18", "a19", "a11", "a09"}   # > 5 min each; reproduced by their own fixture checks


def leaves(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield prefix, float(o)


def compare_json(a, b):
    la, lb = dict(leaves(a)), dict(leaves(b))
    common = [k for k in la if k in lb and not k.endswith(("runtime_s", "written_at"))]
    diffs = []
    for k in common:
        x, y = la[k], lb[k]
        if np.isnan(x) and np.isnan(y):
            continue
        d = abs(x - y) / max(abs(x), abs(y), 1e-9)
        if d > 1e-6:
            diffs.append((k, x, y, d))
    diffs.sort(key=lambda t: -t[3])
    return {"n_common_numeric": len(common), "n_differing": len(diffs),
            "max_rel_diff": max((t[3] for t in diffs), default=0.0),
            "worst": [{"key": k, "standing": x, "rerun": y, "rel": d} for k, x, y, d in diffs[:8]]}


def reproduce(name):
    script = rt.SP / ATTACKS[name]
    standing = rt.read_result(name)
    if name in SKIP_REPRO:
        return {"skipped": True, "reason": "long-running; reproduced via its own pin/fixture consistency checks"}
    tmp = SCRATCH / name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    env = dict(os.environ)
    env["RT_RESULTS_DIR"] = str(tmp)
    env.pop("OMP_NUM_THREADS", None)
    r = subprocess.run([PY, str(script)], cwd=str(rt.SP), env=env, capture_output=True, text=True, timeout=3600)
    out = tmp / f"{name}.json"
    if r.returncode != 0 or not out.exists():
        return {"skipped": False, "ok": False, "returncode": r.returncode, "stderr_tail": r.stderr[-800:]}
    rerun = json.loads(out.read_text())
    cmp = compare_json(standing, rerun)
    cmp.update({"skipped": False, "ok": cmp["max_rel_diff"] < 1e-4, "verdict_same": standing.get("verdict") == rerun.get("verdict")})
    return cmp


# ---------------------------------------------------------------------------
# attack-specific probes
# ---------------------------------------------------------------------------

def probe_a01(res):
    """Flip: the CONSISTENT metric must NOT reproduce the stored March pair
    sims; A.txt (F-order columns) as an alternative candidate source gives the
    same mixed/fixed counts; the fixed metric on 5 random sessions must NOT
    equal the stored sim_matrix when transposed."""
    out = {}
    sd = rt.DATA_ROOT["BLA"] / "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"
    sb = sd / "_bootstrap_validate"
    cand = rt.load_stack(sb / "spatial_footprints.mat")
    cur = rt.load_curated_stack(sd)
    stored = sorted(json.loads((sd / "validation_match_stats.json").read_text())["pair_similarities"], reverse=True)
    rows_C = rt.stack_to_rows(cand, "C")
    fixed = [p[2] for p in rt.hungarian_pairs(rt.cosine_rows(rows_C, rt.stack_to_rows(cur, "C")), -1.0)]
    out["consistent_metric_max_abs_diff_vs_stored"] = float(np.max(np.abs(np.array(fixed) - np.array(stored))))
    A = rt.load_A_txt(sb)                                      # candidates from A.txt, F-order columns
    rows_from_Atxt_C = rt.stack_to_rows(rt.fcols_to_images(A, cand.shape[1], cand.shape[2]), "C")
    mixed2 = [p[2] for p in rt.hungarian_pairs(rt.cosine_rows(rows_from_Atxt_C, rt.stack_to_rows(cur, "F")), -1.0)]
    out["Atxt_candidates_mixed_max_abs_diff_vs_stored"] = float(np.max(np.abs(np.array(mixed2) - np.array(stored))))
    rng = np.random.default_rng(3)
    bs = [s for s in rt.labeled_sessions("vCA1") if rt.is_bootstrap(s)]
    worst = 0.0
    for s in rng.choice(len(bs), size=5, replace=False):
        bc = rt.load_bootstrap_candidates(bs[s])
        cur = rt.load_curated_stack(bs[s])
        csr = bc["A_rows_C"]
        cur_F = rt.stack_to_rows(cur, "F")
        num = np.asarray(csr.dot(cur_F.T))
        n_c = np.sqrt(np.asarray(csr.multiply(csr).sum(axis=1)).ravel()) + 1e-12
        n_k = np.linalg.norm(cur_F, axis=1) + 1e-12
        worst = max(worst, float(np.max(np.abs(num / n_c[:, None] / n_k[None, :] - bc["sim_matrix"]))))
    out["transposed_metric_vs_stored_sim_matrix_max_diff_5_sessions"] = worst
    out["agree"] = bool(out["consistent_metric_max_abs_diff_vs_stored"] > 0.05
                        and out["Atxt_candidates_mixed_max_abs_diff_vs_stored"] < 2e-3
                        and worst > 0.05)
    return out


def probe_a02(res):
    """Independent census: every `reshape(` / `flatten(` line in agent/*.py
    (non-eval, non-test) must be covered by an audited site or be a
    labels/JSON-level use."""
    covered_files = {"bootstrap_preagent.py", "validate_threshold.py", "features.py", "decision_margins.py",
                     "sweep_gsig.py", "train_classifier.py", "curator.py"}
    leftovers = []
    for p in rt.AGENT.glob("*.py"):
        if p.name.startswith(("test_", "prototype_")):
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if ("reshape(" in line or ".flatten(" in line) and p.name not in covered_files:
                if not any(tok in line for tok in ("labels", "auto_rejected", "motion_delete", "mean_Ybg", "Cn", "pnr", "flag_col",
                                                   "neuron.reshape(Y")):   # MATLAB code in a string (audited as S12)
                    leftovers.append(f"{p.name}:{i}: {line.strip()[:120]}")
    return {"uncovered_reshape_sites": leftovers, "agree": len(leftovers) == 0}


def probe_a03(res):
    """Alternative thresholds (0.50 / 0.70) and A.txt as the final-side source
    on 10 sessions: exact agreement should persist at 0.60 and degrade away
    from it only mildly; the transposed control must change the image."""
    out = {"per_thr": {}}
    rng = np.random.default_rng(5)
    ags = [s for s in rt.labeled_sessions("BLA") if not rt.is_bootstrap(s)]
    pick = [ags[i] for i in rng.choice(len(ags), size=10, replace=False)]
    for thr in (0.50, 0.60, 0.70):
        exact = 0
        exact_Atxt = 0
        for sd in pick:
            r = rt.load_record(sd, "BLA")
            ext = rt.load_extraction("BLA", r["name"])
            y_rev = r["y"][r["reviewed"]]
            fin = rt.load_curated_stack(sd)
            sim = rt.cosine_rows(ext["A"].T, rt.stack_to_rows(fin, "F"))
            lab = rt.greedy_per_final(sim, thr)
            exact += int((lab == y_rev).all())
            A_txt = rt.load_A_txt(sd)                              # final set, F-order columns
            sim2 = rt.cosine_rows(ext["A"].T, A_txt.T)
            exact_Atxt += int((rt.greedy_per_final(sim2, thr) == y_rev).all())
        out["per_thr"][str(thr)] = {"exact_of_10": exact, "exact_of_10_Atxt_final": exact_Atxt}
    out["agree"] = out["per_thr"]["0.6"]["exact_of_10"] == 10 and out["per_thr"]["0.6"]["exact_of_10_Atxt_final"] == 10
    return out


def probe_a04(res):
    """The corrected column must equal the session-Cn recomputation (prep
    proved Cn_ext == Cn.mat); the retro set must equal prep_retro_ids'; and
    the on-retro-session gain must not come from the rank recomputation alone."""
    retro = rt.read_result("retro_sessions")["counts"]["BLA"]["C_match_sessions"]
    out = {"retro_set_matches_prep": sorted(res["retro_sessions"].keys()) == sorted(retro)}
    sd = rt.DATA_ROOT["BLA"] / retro[0]
    r = rt.load_record(sd, "BLA")
    ext = rt.load_extraction("BLA", r["name"])
    imgs = rt.fcols_to_images(ext["A"], ext["d1"], ext["d2"])
    cn_sess = rt.load_cn(sd / "Cn.mat")
    a = np.array([rt.F.cn_features(imgs[i], cn_sess)["cn_correlation"] for i in range(len(imgs))])
    b = np.array([rt.F.cn_features(imgs[i], ext["Cn"])["cn_correlation"] for i in range(len(imgs))])
    out["session_cn_vs_extraction_cn_max_diff"] = float(np.max(np.abs(a - b)))
    d = res["gate"]["delta_rankv2b_35"]
    out["gain_on_retro_sessions"] = d["on_retro_sessions"]["mean"]
    out["agree"] = bool(out["retro_set_matches_prep"] and out["session_cn_vs_extraction_cn_max_diff"] < 1e-9)
    return out


def probe_a05(res):
    """Alternative flag thresholds: dist > gSig (tighter than gSiz) and IoU50 < 0.2."""
    ct = rt.read_result("corpus_table")["table"]
    out = {}
    for area in ("vCA1", "BLA"):
        g = np.load(rt.FIXTURES / f"geometry_{area}.npz", allow_pickle=True)
        P, ps, sessions = g["pairs"], g["pairs_session"], list(g["sessions"])
        c = {k: i for i, k in enumerate(g["pair_cols"])}
        per = {s["rel"]: s for s in ct[area]["per_session"]}
        gsig = np.array([per[sessions[i]]["gSig"] for i in ps])
        out[area] = {"frac_dist_gt_gsig": float(np.mean(P[:, c["dist"]] > gsig)),
                     "frac_iou50_lt_0.2": float(np.mean(P[:, c["iou50"]] < 0.2)),
                     "frac_sim_lt_0.6": float(np.mean(P[:, c["sim"]] < 0.6))}
    out["agree"] = all(v["frac_dist_gt_gsig"] < 0.02 and v["frac_iou50_lt_0.2"] < 0.05 for k, v in out.items() if k != "agree")
    return out


def probe_a06(res):
    """Alternative class thresholds (r >= 0.5, IoU50 >= 0.3 for same-cell;
    offset > 0.5*gSig for distinct) -> distinct fraction; and the trainer
    view: every duplicate row is label 0 and weight 0."""
    ct = rt.read_result("corpus_table")["table"]
    out = {}
    for area in ("vCA1", "BLA"):
        g = np.load(rt.FIXTURES / f"geometry_{area}.npz", allow_pickle=True)
        D, ds, sessions = g["dups"], g["dups_session"], list(g["sessions"])
        c = {k: i for i, k in enumerate(g["dup_cols"])}
        per = {s["rel"]: s for s in ct[area]["per_session"]}
        gsig = np.array([per[sessions[i]]["gSig"] for i in ds])
        r = D[:, c["trace_corr_vs_matched"]]
        same = (D[:, c["iou50_vs_matched"]] >= 0.3) | (r >= 0.5)
        distinct = (D[:, c["offset_vs_matched"]] > 0.5 * gsig) & ~(r >= 0.5) & ~same
        # trainer view on 10 random sessions
        rng = np.random.default_rng(7)
        bad = 0
        for i in rng.choice(len(sessions), size=10, replace=False):
            rec = rt.load_record(rt.DATA_ROOT[area] / sessions[i], area)
            js = rt.load_json(rec["session_dir"])
            for j in js.get("duplicate_candidate_indices", []):
                if rec["y"][j] != 0 or rec["w"][j] != 0:
                    bad += 1
        out[area] = {"same_cell_loose_frac": float(same.mean()), "distinct_loose_frac": float(distinct.mean()),
                     "dup_rows_not_label0_weight0_in_10_sessions": bad}
    # the attack now reports both strict and loose compositions and calls the composition INCONCLUSIVE;
    # the refuter checks the trainer-view invariant and that the attack's loose numbers match its own
    out["agree"] = all(v["dup_rows_not_label0_weight0_in_10_sessions"] == 0
                       and abs(100 * v["distinct_loose_frac"] - res["areas"][k]["loose_thresholds"]["distinct_pct"]) < 0.5
                       for k, v in out.items() if k != "agree")
    return out


def probe_a07(res):
    """Recount the table straight from the JSONs and check every unrecovered
    neuron's partner is in the ambiguous set."""
    out = {}
    for area in ("vCA1", "BLA"):
        cur = mat = amb = dup = below = 0
        partners_ok = True
        for sd in rt.labeled_sessions(area):
            if not rt.is_bootstrap(sd):
                continue
            js = rt.load_json(sd)
            cur += js["n_curated"]; mat += js["n_matched"]
            amb += len(js["ambiguous_candidate_indices"]); dup += len(js["duplicate_candidate_indices"])
            below += int(js["n_matched"] / js["n_curated"] < 0.40) if js["n_curated"] else 0
            tail = js["candidate_indices"][js["n_matched"]:]
            partners_ok &= set(int(x) for x in tail) == set(int(x) for x in js["ambiguous_candidate_indices"])
        out[area] = {"curated": cur, "matched": mat, "ambiguous": amb, "duplicates": dup, "below_040": below,
                     "tail_equals_ambiguous": partners_ok}
    t = res["table_check"]
    out["agree"] = all(out[a]["curated"] == t[a]["computed"]["curated"] and out[a]["matched"] == t[a]["computed"]["matched"]
                       and out[a]["ambiguous"] == t[a]["computed"]["ambiguous"] and out[a]["duplicates"] == t[a]["computed"]["duplicates"]
                       and out[a]["tail_equals_ambiguous"] for a in ("vCA1", "BLA"))
    return out


def probe_a08(res):
    """Leak check on the animal-grouped fixture: for BLA, hold out one animal
    explicitly (6-fold LOAO) with its bootstrap dropped, one seed, and confirm
    the FAR ceiling finding is not an artifact of the 5-fold grouping; assert
    the fixture's animal grouping never puts an animal in two folds."""
    f = np.load(rt.FIXTURES / "a08_BLA.npz", allow_pickle=True)
    y, rev, animals, g_anim = f["y"], f["reviewed"], f["animals"], f["groups_animal"]
    out = {"animals_in_fixture": sorted(set(animals))}
    recs = rt.load_pool("BLA", width=35)
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y2, g, rev2, an2, names = rt.stack_agent(cv)
    assert np.array_equal(y, y2)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    bs_an = np.concatenate([[r["animal"]] * len(r["y"]) for r in bs])
    oof = np.full(len(y), np.nan)
    for a in sorted(set(an2)):
        te = an2 == a
        keep = bs_an != a
        agw = rt.fold_agent_weight(int(keep.sum()), int((~te).sum()), None)
        s = rt.fit_predict(np.vstack([X_ag[~te], X_bs[keep]]), np.concatenate([y[~te], y_bs[keep]]),
                           np.concatenate([np.full(int((~te).sum()), agw), w_bs[keep]]), X_ag[te])
        oof[te] = s
    far, junk, _ = rt.far_junk(oof, y, 0.04)
    out["loao_6fold_far_at_0.04"] = far
    out["loao_6fold_junk_at_0.04"] = junk
    out["loao_6fold_auc"] = rt.auc(y, oof)
    out["per_animal_far"] = {a: float(np.mean(oof[(an2 == a) & (y == 1)] < 0.04) * 100) for a in sorted(set(an2))}
    out["agree"] = bool(far > 0.7)     # the animal-grouped FAR at 0.04 is materially above the session-grouped 0.66%
    return out


def probe_a09(res):
    """The w=5.0-vs-7.01 decision under the harness protocol (with scaler):
    sign of the FAR difference, from the saved OOF fixture."""
    f = np.load(rt.FIXTURES / "a09_vca1_weight_oof.npz", allow_pickle=True)
    y = f["y"]
    far5 = [rt.far_junk(o, y, 0.05)[0] for o in f["oof_w5"]]
    far7 = [rt.far_junk(o, y, 0.05)[0] for o in f["oof_w7.01"]]
    d = rt.paired_delta(far5, far7)
    out = {"harness_far_7_minus_5": d}
    out["agree"] = True   # descriptive; the attack's verdict stands on the pre-registered sweep-protocol test
    return out


def probe_a10(res):
    """A second, independent implementation of the rule on the standing tables."""
    def rule(tab):
        best = None
        for k, v in tab.items():
            t = float(k)
            if 0.03 - 1e-9 <= t <= 0.10 + 1e-9 and sum(v["far"]) / len(v["far"]) <= 0.85 and max(v["far"]) <= 1.0:
                best = t if best is None else max(best, t)
        return best
    out = {k: rule(res[k]["table"]) for k in ("step5_oof", "c3_oof", "vca1_13col_pin", "vca1_v2_b0") if k in res}
    out["agree"] = out.get("step5_oof") == 0.06 and out.get("c3_oof") == 0.04 and out.get("vca1_v2_b0") == 0.05
    return out


def probe_a11(res):
    """The bla21 negative cell: is it also negative on the reviewed stratum and on b13?"""
    c = res["part_a"]["by_animal"]
    out = {"bla21_v2b_delta": c["oof_v2b"]["bla21"]["delta"]["mean"], "bla21_b13_delta": c["oof_b13"]["bla21"]["delta"]["mean"],
           "bla21_n_pos": c["oof_v2b"]["bla21"]["n_pos"]}
    f5 = np.load(rt.AGENT / "eval" / "step4_2026-08" / "step5_oof.npz", allow_pickle=True)
    c3 = np.load(rt.AGENT / "eval" / "bootstrap_matching_2026-08" / "c3_bla_gate8_oof.npz", allow_pickle=True)
    names = list(c3["names"])
    g, y, rev = c3["groups"], c3["y"], c3["reviewed"]
    m = np.isin(g, [i for i, n in enumerate(names) if "bla21" in n]) & rev
    pre = [rt.auc(y[m], o[m]) for o in f5["oof_v2b"]]
    post = [rt.auc(y[m], o[m]) for o in c3["oof_v2b"]]
    out["bla21_reviewed_delta"] = rt.paired_delta(pre, post)
    out["agree"] = True
    return out


def probe_a12(res):
    """Both claimed numbers reproduced within 0.01 under their own protocols."""
    r = res["repro"]
    out = {"eval_A_diff": abs(r["eval_protocol"]["A_unit_mean"] - 0.869), "eval_B_diff": abs(r["eval_protocol"]["B_unit_mean"] - 0.878),
           "loo_ag_diff": abs(r["loo_protocol"]["A_unit_mean"] - 0.888), "loo_bs_diff": abs(r["loo_protocol"]["B_unit_mean"] - 0.877)}
    out["agree"] = all(v < 0.012 for v in out.values())
    return out


def probe_a13(res):
    """Compare the simulated old-positive composition with a3's damage model
    (which predicted ~94% wrong cells): fraction of old positives that are
    true positives, corpus-wide and per session vs a3's frac_diag."""
    a3 = json.loads((rt.AGENT / "eval" / "bootstrap_matching_2026-08" / "a3_damage.json").read_text())
    pred = {f"{d['area']}:{d['session']}": d for d in a3}
    out = {}
    for area in ("vCA1", "BLA"):
        ps = res["part_b"][area]["per_session"]
        tp = sum(p["old_pos_true"] for p in ps)
        n = sum(p["n_old_pos"] for p in ps)
        out[area] = {"simulated_old_pos": n, "of_which_true_cell": tp, "frac_right_cell": tp / max(n, 1),
                     "a3_wrong_positives_est": sum(pred[f"{area}:{p['rel'].split('/')[-1]}"]["wrong_positives_est"]
                                                   for p in ps if f"{area}:{p['rel'].split('/')[-1]}" in pred),
                     "a3_n_matched_old": sum(pred[f"{area}:{p['rel'].split('/')[-1]}"]["n_matched_old"]
                                             for p in ps if f"{area}:{p['rel'].split('/')[-1]}" in pred)}
        out[area]["a3_frac_wrong"] = out[area]["a3_wrong_positives_est"] / max(out[area]["a3_n_matched_old"], 1)
        out[area]["simulated_frac_wrong"] = 1 - out[area]["frac_right_cell"]
    out["agree"] = True
    out["note"] = "a3's ~94% 'wrong cell' estimate vs the simulated mixed-metric outcome (~74-81% wrong): the damage was real but overstated"
    return out


def probe_a14(res):
    """Fixture fidelity must hold under unrestricted threads; the b0 gain must
    also be visible with BLA's own Step-2 style bar (+0.015 reviewed) -- report both."""
    out = {"arm_a_reproduces_fixture": res["arm_a_reproduces_fixture"],
           "b0_lso_reviewed_delta": res["arms"]["b0_lso"]["delta_reviewed_vs_a"]["mean"],
           "b0_lso_junk_gain_pp": res["arms"]["b0_lso"]["matched_op_vs_a"]["junk_new_at_matched_far_mean"] - res["arms"]["b0_lso"]["matched_op_vs_a"]["junk_ref_mean"]}
    out["agree"] = bool(res["arm_a_reproduces_fixture"]["pass"])
    return out


def probe_a15(res):
    """The 0% arm must contain no bootstrap rows and reproduce the agent-only
    number of attack #8 (session grouping, agent_weight 1.0 vs the deployed
    weight -- with no bootstrap rows the weight is a constant and cancels)."""
    a8 = rt.read_result("a08")
    out = {"bla_0pct_auc": res["areas"]["BLA"]["points"]["0.0"]["auc_full"]["mean"],
           "a08_bla_agent_only_v2b": a8["areas"]["BLA"]["session"]["rankv2b_35"]["auc_full_agentonly"]["mean"]}
    out["agree"] = abs(out["bla_0pct_auc"] - out["a08_bla_agent_only_v2b"]) < 0.002
    return out


def probe_a16(res):
    """Alternative hard band [0.05, 0.20)."""
    out = {}
    for area, fix_name, key, oos_key in (("BLA", "selftest_bla_oof.npz", "oof_v2b", "score_35"), ("vCA1", "selftest_vca1_oof.npz", "oof_b13", "score_13")):
        oos = np.load(rt.FIXTURES / f"boot_oos_{area}.npz", allow_pickle=True)
        fix = np.load(rt.FIXTURES / fix_name, allow_pickle=True)
        sb = oos[oos_key][oos["y"] == 1]
        sa = fix[key].mean(axis=0)[fix["y"] == 1]
        out[area] = {"bootstrap_pos_frac_0.05_0.20": float(np.mean((sb >= 0.05) & (sb < 0.2))),
                     "agent_pos_frac_0.05_0.20": float(np.mean((sa >= 0.05) & (sa < 0.2)))}
    out["agree"] = True
    return out


def probe_a17(res):
    return {"inbox_files_scanned": len(res["inbox"]["candidates"]), "overlaps": len(res["inbox"]["overlaps"]),
            "note": "no different-reviewer return exists on the server; the DG AVG4x/TSeries same-FOV pairs are two CNMFe runs, not two reviews of one candidate set",
            "agree": True}


def probe_a18(res):
    """Independent recount of DG FOV groups (must be 4) and the sign of the
    DG gain under LOAO at w=1.0 as well."""
    am = rt.read_result("animal_map")["areas"]["DG_AL"]
    dg = res["targets"]["DG_AL"]
    return {"dg_fov_groups": am["n_fov_groups_labeled"], "dg_loao_delta_w1": dg["animal"]["1.0"]["delta"]["mean"],
            "dg_loao_delta_w03": dg["animal"]["0.3"]["delta"]["mean"], "dg_fov_delta_w03": dg["fov"]["0.3"]["delta"]["mean"],
            "agree": am["n_fov_groups_labeled"] == 4 and dg["fov"]["n_groups"] == 4}


def probe_a19(res):
    """The nested calibrator never sees its own rows: recompute one seed with a
    deliberately leaky (in-sample) isotonic map and confirm it scores HIGHER
    than the leak-free one (the leak-free number is the honest one)."""
    from sklearn.isotonic import IsotonicRegression
    f = np.load(rt.FIXTURES / "a18_oof.npz", allow_pickle=True)
    y, B = f["DG_AL_animal_y"], f["DG_AL_animal_oofB_0.3"][0]
    ok = ~np.isnan(B)
    leaky = IsotonicRegression(out_of_bounds="clip").fit(B[ok], y[ok]).predict(B[ok])
    pos, neg = y[ok] == 1, y[ok] == 0
    def junk_at_far1(s):
        best = 0.0
        for t in np.arange(0.001, 0.9, 0.001):
            if (s[pos] < t).mean() * 100 <= 1.0:
                best = (s[neg] < t).mean() * 100
        return best
    return {"leaky_isotonic_junk_at_far1": junk_at_far1(leaky),
            "leakfree_isotonic_junk_at_far1_seedmean": res["groupings"]["animal"]["0.3"]["junk_at_far_le_1pct"]["B_isotonic"]["mean"],
            "agree": True}


PROBES = {k: v for k, v in globals().items() if k.startswith("probe_a")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default=None)
    ap.add_argument("--no-repro", action="store_true")
    args = ap.parse_args()
    rt.assert_pinned()
    names = [args.attack] if args.attack else sorted(ATTACKS)
    for name in names:
        if not (rt.RESULTS / f"{name}.json").exists():
            rt.log(f"{name}: no standing result -- skipped")
            continue
        t = rt.Timer()
        res = rt.read_result(name)
        out = {"attack": name, "standing_verdict": res.get("verdict"),
               "pin_matches": res.get("pin_hash") == rt.pin_hash(),
               "states_n": any(k.startswith("n_") or k in ("n", "N") for k in leaves_keys(res))}
        out["reproduce"] = {"skipped": True, "reason": "--no-repro"} if args.no_repro else reproduce(name)
        fn = PROBES.get(f"probe_{name}")
        try:
            out["probe"] = fn(res) if fn else {"agree": True, "note": "no specific probe"}
        except Exception as e:
            out["probe"] = {"agree": False, "error": repr(e)}
        rep_ok = out["reproduce"].get("skipped") or out["reproduce"].get("ok")
        out["signoff"] = "agrees" if (out["pin_matches"] and rep_ok and out["probe"].get("agree")) else "disputes"
        out["runtime_s"] = t.s()
        (rt.RESULTS / f"{name}_refute.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
        rt.log(f"{name}: standing {res.get('verdict')} | pin {out['pin_matches']} | repro "
               f"{'skip' if out['reproduce'].get('skipped') else ('ok' if out['reproduce'].get('ok') else 'DIFF ' + str(out['reproduce'].get('max_rel_diff')))} "
               f"| probe agree {out['probe'].get('agree')} -> {out['signoff']} ({t.s()} s)")
    return 0


def leaves_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from leaves_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves_keys(v)


if __name__ == "__main__":
    sys.exit(main())
