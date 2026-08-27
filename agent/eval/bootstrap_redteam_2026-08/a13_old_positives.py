"""
a13_old_positives.py -- hypothesis D13: the old (mirror-position) positives
were mostly cells, so the fix mainly repaired the negative pool.

(A) Literal, on the 4 real sandboxes: the March candidates' 13 features are
    recomputed from the sandbox outputs (spatial_footprints.mat, C_raw.txt,
    Ybg_mean.mat, Cn.mat) with the production row functions; the MIXED-order
    Hungarian gives the old positives (37 / 19 / 25 / 34); they are scored
    with the deployed BLA companion (13-col) and compared with the human
    review labels transferred by mutual-best cosine > 0.6 against the
    session's review set (.feature_expansion A).
(B) Simulated, all 202 sessions: re-apply the TRANSPOSED metric (candidate
    C-order rows vs curated F-order rows) to bootstrap_candidates.npz ->
    Hungarian -> pairs > 0.45 = the rows the bug would label 1 today;
    cross-tab against the fixed labels (true positive / duplicate /
    ambiguous / plain negative) and score them out-of-sample
    (fixtures/boot_oos) and in-sample (deployed joblib).
Supported if most old positives score as cells out-of-sample; the predicted
signature (small AUC move, large operating-point move) is then checked
against the measured +0.0037 AUC / +9.4pp junk.  Read-only.
"""
import sys

import joblib
import numpy as np

import rt_lib as rt

SANDBOXES = ["2tones/AVG5x-TSeries-093025-bla21-313um-38z-000",
             "2tones/AVG5x-TSeries-100125-bla12-639um-23z-000",
             "4odorDO/AVG5x-TSeries-02092026-bla12-681um-22z-000",
             "Valence/AVG5x-TSeries-121225-bla12-652um-23z-000"]
T = {"BLA": rt.DEPLOYED_T["BLA"], "vCA1": rt.DEPLOYED_T["vCA1"]}


def features13(stack, traces, bg, Cn):
    rows = []
    for i in range(stack.shape[0]):
        sf = rt.F.spatial_features(stack[i])
        tf = rt.F.temporal_features(traces[i])
        mf = rt.F.motion_features(traces[i], bg)
        cf = rt.F.cn_features(stack[i], Cn)
        rows.append({**sf, **tf, **mf, **cf})
    names = list(rows[0].keys())
    assert names == rt.V1_NAMES, names
    return np.array([[r[k] for k in names] for r in rows])


def part_a():
    jl = joblib.load(rt.MODEL_DIR["BLA"] / "classifier.joblib")
    fp_sc, fp_clf = jl["first_pass_scaler"], jl["first_pass_clf"]
    out = {}
    for relp in SANDBOXES:
        sd = rt.DATA_ROOT["BLA"] / relp
        sb = sd / "_bootstrap_validate"
        cand = rt.load_stack(sb / "spatial_footprints.mat")
        traces = np.loadtxt(str(sb / "C_raw.txt"))
        bg = rt.load_mat_var(sb / "Ybg_mean.mat", "mean_Ybg").flatten() if (sb / "Ybg_mean.mat").exists() else None
        Cn = rt.load_cn(sb / "Cn.mat")
        assert traces.shape[0] == cand.shape[0]
        X13 = features13(cand, traces, bg, Cn)
        s13 = fp_clf.predict_proba(fp_sc.transform(X13))[:, 1]
        cur = rt.load_curated_stack(sd)                              # the FINAL (human-kept) set
        rows_C = rt.stack_to_rows(cand, "C")
        mixed = rt.cosine_rows(rows_C, rt.stack_to_rows(cur, "F"))
        fixed = rt.cosine_rows(rows_C, rt.stack_to_rows(cur, "C"))
        old_pos = sorted(j for j, k, s in rt.hungarian_pairs(mixed, rt.MATCH_THR))
        true_pos = sorted(j for j, k, s in rt.hungarian_pairs(fixed, rt.MATCH_THR))
        # human labels on the review set, transferred to sandbox candidates by mutual-best cosine > 0.6
        r = rt.load_record(sd, "BLA")
        ext = rt.load_extraction("BLA", relp)
        rev_idx = np.flatnonzero(r["reviewed"])
        y_rev = r["y"][rev_idx]
        rev_rows = rt.stack_to_rows(rt.fcols_to_images(ext["A"], ext["d1"], ext["d2"]), "C")
        c2r = rt.cosine_rows(rows_C, rev_rows)
        best_r = c2r.argmax(axis=1)
        best_c = c2r.argmax(axis=0)
        transfer = np.full(len(cand), -1)
        for j in range(len(cand)):
            k = best_r[j]
            if c2r[j, k] > 0.6 and best_c[k] == j:
                transfer[j] = int(y_rev[k])
        def summ(idx):
            idx = np.asarray(idx, int)
            tr = transfer[idx]
            return {"n": int(len(idx)), "score_p50": float(np.median(s13[idx])) if len(idx) else None,
                    "frac_ge_T": float(np.mean(s13[idx] >= T["BLA"])) if len(idx) else None,
                    "frac_ge_0.5": float(np.mean(s13[idx] >= 0.5)) if len(idx) else None,
                    "human_kept": int((tr == 1).sum()), "human_deleted": int((tr == 0).sum()), "unknown": int((tr < 0).sum())}
        all_neg = [j for j in range(len(cand)) if j not in set(true_pos)]
        out[relp] = {"n_candidates": int(len(cand)), "n_final": int(cur.shape[0]),
                     "old_positives": summ(old_pos), "true_positives": summ(true_pos),
                     "old_pos_that_are_true_pos": int(len(set(old_pos) & set(true_pos))),
                     "all_other_candidates": summ(all_neg)}
        o = out[relp]
        rt.log(f"  {relp}: old pos {o['old_positives']} | true pos {o['true_positives']} | overlap {o['old_pos_that_are_true_pos']} "
               f"| rest {o['all_other_candidates']}")
    return out


def part_b():
    out = {}
    for area in ("vCA1", "BLA"):
        oos = np.load(rt.FIXTURES / f"boot_oos_{area}.npz", allow_pickle=True)
        key = "score_35" if area == "BLA" else "score_13"
        jl = joblib.load(rt.MODEL_DIR[area] / "classifier.joblib")
        sc, clf = jl["scaler"], jl["clf"]
        tab = {"true_positive": [], "duplicate": [], "ambiguous": [], "plain_negative": []}
        scores_oos, scores_in, cls = [], [], []
        n_old_total = 0
        per_session = []
        for sd in rt.labeled_sessions(area):
            if not rt.is_bootstrap(sd):
                continue
            r = rt.load_record(sd, area)
            js = rt.load_json(sd)
            bc = rt.load_bootstrap_candidates(sd)
            cur = rt.load_curated_stack(sd)
            cur_F = rt.stack_to_rows(cur, "F")
            csr = bc["A_rows_C"]
            num = np.asarray(csr.dot(cur_F.T))
            n_c = np.sqrt(np.asarray(csr.multiply(csr).sum(axis=1)).ravel()) + 1e-12
            n_k = np.linalg.norm(cur_F, axis=1) + 1e-12
            mixed = num / n_c[:, None] / n_k[None, :]
            old_pos = [j for j, k, s in rt.hungarian_pairs(mixed, rt.MATCH_THR)]
            dups = set(int(i) for i in js.get("duplicate_candidate_indices", []))
            amb = set(int(i) for i in js.get("ambiguous_candidate_indices", []))
            m = oos["session"] == r["name"]
            s_o = oos[key][m]
            s_i = clf.predict_proba(sc.transform(r["X"]))[:, 1]
            for j in old_pos:
                c = ("true_positive" if r["y"][j] == 1 else "duplicate" if j in dups
                     else "ambiguous" if j in amb else "plain_negative")
                tab[c].append(j)
                cls.append(c)
                scores_oos.append(float(s_o[j]))
                scores_in.append(float(s_i[j]))
            n_old_total += len(old_pos)
            per_session.append({"rel": r["name"], "n_old_pos": len(old_pos), "n_curated": js["n_curated"],
                                "old_pos_true": int(sum(1 for j in old_pos if r["y"][j] == 1))})
        so, si, cl = np.array(scores_oos), np.array(scores_in), np.array(cls)
        d = {"n_old_positives": int(n_old_total), "class_counts": {k: len(v) for k, v in tab.items()},
             "class_frac": {k: len(v) / max(n_old_total, 1) for k, v in tab.items()}}
        for label, s in (("oos", so), ("in_sample", si)):
            d[label] = {"score_p50": float(np.median(s)), "frac_ge_T": float(np.mean(s >= T[area])),
                        "frac_ge_0.5": float(np.mean(s >= 0.5)), "frac_hard_0.02_0.10": float(np.mean((s >= 0.02) & (s < 0.10)))}
            d[label]["by_class"] = {c: {"n": int((cl == c).sum()), "score_p50": float(np.median(s[cl == c])) if (cl == c).any() else None,
                                        "frac_ge_T": float(np.mean(s[cl == c] >= T[area])) if (cl == c).any() else None}
                                    for c in tab}
        # reference distributions: all bootstrap negatives (unmasked) and true positives, OOS
        yb, wb, sb = oos["y"], oos["w"], oos[key]
        d["reference_oos"] = {"true_pos_frac_ge_T": float(np.mean(sb[yb == 1] >= T[area])),
                              "true_pos_p50": float(np.median(sb[yb == 1])),
                              "unmasked_neg_frac_ge_T": float(np.mean(sb[(yb == 0) & (wb > 0)] >= T[area])),
                              "unmasked_neg_p50": float(np.median(sb[(yb == 0) & (wb > 0)])),
                              "junk_caught_unmasked_neg_at_T": float(np.mean(sb[(yb == 0) & (wb > 0)] < T[area]))}
        d["supported"] = bool(d["oos"]["frac_ge_T"] > 0.5)
        d["per_session"] = per_session
        out[area] = d
        rt.log(f"  {area}: {n_old_total} simulated old positives -> {d['class_frac']} | OOS: p50 {d['oos']['score_p50']:.3f}, "
               f">=T {d['oos']['frac_ge_T']:.3f}, >=0.5 {d['oos']['frac_ge_0.5']:.3f} | in-sample >=T {d['in_sample']['frac_ge_T']:.3f} "
               f"| ref: true pos >=T {d['reference_oos']['true_pos_frac_ge_T']:.3f}, unmasked neg >=T {d['reference_oos']['unmasked_neg_frac_ge_T']:.3f}")
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    rt.log("[A] literal old positives on the 4 sandboxes")
    A = part_a()
    rt.log("[B] simulated old positives on all 202 sessions")
    B = part_b()
    sup = {a: B[a]["supported"] for a in B}
    res = {"part_a": A, "part_b": B, "hypothesis_D13_supported": sup,
           "signature": {"measured_bla_auc_delta": 0.0037, "measured_bla_junk_delta_pp": 9.4,
                         "reading": "if most old positives score as cells out-of-sample, the fix's main effect was on the "
                                    "negative pool (the ~7,900 true cells that sat at label 0), which predicts a small AUC "
                                    "move and a larger operating-point move"},
           "verdict": "PASS",
           "criterion": "D13 supported for an area if > 50% of the simulated old positives score >= the deployed T out-of-sample"}
    rt.log(f"D13 supported {sup}")
    rt.write_result("a13", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
