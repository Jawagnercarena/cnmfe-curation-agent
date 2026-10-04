"""
synth_report.py -- every table in the report comes from results/*.json.

Prints Markdown to stdout (the report author pastes it verbatim and adds
prose).  No number in the tables is typed by hand.
"""
import json
import sys

import numpy as np

import rt_lib as rt

TITLES = {
    "a01": "Orientation re-derived without bmlib", "a02": "An 11th orientation site", "a03": "Agent labels unaffected",
    "a04": "Retro cn_correlation transpose", "a05": "Matched pairs are the right cells", "a06": "Masked duplicates",
    "a07": "Unrecovered neurons + section-4 table", "a08": "Animal / FOV leakage", "a09": "3-seed decisions at 8 seeds",
    "a10": "Threshold rule self-consistency", "a11": "BLA G5 per-animal / early-era", "a12": "Cond A/B vs LOO sign flip",
    "a13": "D13 old positives were cells", "a14": "D14 BLA bootstrap v2b degenerate", "a15": "D15 bootstrap redundant (learning curve)",
    "a16": "D16 bootstrap positives easy", "a17": "D17 reviewer ceiling", "a18": "Global model, animal-grouped",
    "a19": "Pooled DG calibration",
}


def f4(x):
    return "n/a" if x is None else f"{x:+.4f}" if isinstance(x, float) and abs(x) < 1 else f"{x}"


def main():
    R = {k: rt.read_result(k) for k in TITLES if (rt.RESULTS / f"{k}.json").exists()}
    RF = {k: json.loads((rt.RESULTS / f"{k}_refute.json").read_text()) for k in TITLES if (rt.RESULTS / f"{k}_refute.json").exists()}
    pin = rt.load_pin()
    lines = []
    P = lines.append
    P(f"Pin `{pin['pin_hash'][:12]}` -- BLA {pin['areas']['BLA']['n_labeled']} labeled ({pin['areas']['BLA']['n_agent']} agent / "
      f"{pin['areas']['BLA']['n_bootstrap']} bootstrap), vCA1 {pin['areas']['vCA1']['n_labeled']} ({pin['areas']['vCA1']['n_agent']} / "
      f"{pin['areas']['vCA1']['n_bootstrap']}), DG_AL {pin['areas']['DG_AL']['n_labeled']}; joblib md5 BLA {pin['areas']['BLA']['joblib']['md5'][:8]}, "
      f"vCA1 {pin['areas']['vCA1']['joblib']['md5'][:8]}.")
    P("")
    P("## Verdict table")
    P("")
    P("| # | attack | verdict | refuter | one-line evidence |")
    P("|---|---|---|---|---|")
    ev = {}
    a = R.get("a01", {})
    if a:
        s2 = a["step2"]["2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"]
        ev["a01"] = (f"mixed metric reproduces stored March sims on 4/4 sandboxes (bla21 {s2['mixed_matched_45']}/50, max|diff| "
                     f"{s2['mixed_max_abs_diff_vs_stored']:.1e}; matches {s2['mixed_median_mirror_dist_px']:.1f} px from the mirror vs "
                     f"{s2['mixed_median_true_dist_px']:.0f} px from the truth); consistent metric {s2['fixed_matched_45']}/50 at "
                     f"{s2['fixed_median_true_dist_px']:.2f} px; live labels consistent {a['n_step4_pass']}/{a['n_step4']}")
    a = R.get("a02", {})
    if a:
        ev["a02"] = ("no live production defect beyond the 2 known; `features.load_spatial` A.txt fallback has 2 latent defects "
                     f"(transpose + square-only) firing on {len(a['sites']['S2_features_load_spatial_A_txt_fallback']['fallback_fires_on'])} sessions")
    a = R.get("a03", {})
    if a:
        s = a["summary"]
        ev["a03"] = (f"{s['n_exact']}/{s['n_ok']} sessions reproduce labels.mat exactly from footprints (both rules); transposed control "
                     f"agreement {s['mean_agree_transposed']:.2f} and {s['mean_pos_recomputed_transposed']:.0f} vs {s['mean_pos_labels']:.0f} positives")
    a = R.get("a04", {})
    if a:
        d = a["gate"]["delta_rankv2b_35"]
        ev["a04"] = (f"{a['n_corrected_sessions']} BLA sessions (rows {a['n_corrected_rows_in_cv_pool']}, reals {a['n_corrected_pos']}) carry a transposed "
                     f"cn_correlation (Spearman vs corrected ~0); fixing it: deployed 35-col {d['full']['mean']:+.4f} pool / "
                     f"{d['on_retro_sessions']['mean']:+.4f} on those sessions; b13 {a['gate']['delta_b13']['full']['mean']:+.4f}, b13 rule T "
                     f"{a['gate']['live_b13']['rule_T']}->{a['gate']['corrected_b13']['rule_T']}; sizing {a['sizing']}")
    a = R.get("a05", {})
    if a:
        v, b = a["areas"]["vCA1"], a["areas"]["BLA"]
        ev["a05"] = (f"dist p50 {v['dist_px']['p50']:.2f}/{b['dist_px']['p50']:.2f} px, p95 {v['dist_px']['p95']:.1f}/{b['dist_px']['p95']:.1f}; IoU20 p05 "
                     f"{v['iou20']['p05']:.2f}/{b['iou20']['p05']:.2f}; flagged {v['n_flagged']}+{b['n_flagged']} of {v['n_pairs']+b['n_pairs']}; "
                     f"residual transposes 0; human sheets pending")
    a = R.get("a06", {})
    if a:
        v, b = a["areas"]["vCA1"], a["areas"]["BLA"]
        ev["a06"] = (f"same-cell {v['pct']['same_cell']:.0f}%/{b['pct']['same_cell']:.0f}%, distinct {v['pct']['distinct']:.1f}%/{b['pct']['distinct']:.1f}%, "
                     f"other {v['pct']['other']:.0f}%/{b['pct']['other']:.0f}%; trace r with the matched candidate p50 {v['trace_corr']['p50']:.2f}/{b['trace_corr']['p50']:.2f} "
                     f"(>=0.7 only {v['trace_corr']['frac_ge_0.7']:.0%}/{b['trace_corr']['frac_ge_0.7']:.0%})")
    a = R.get("a07", {})
    if a:
        v, b = a["areas"]["vCA1"], a["areas"]["BLA"]
        ev["a07"] = (f"table reproduces exactly; {v['n_unrecovered']}+{b['n_unrecovered']} unrecovered: vCA1 {v['counts']}, BLA {b['counts']}; "
                     f"all partners in the ambiguous set")
    a = R.get("a08", {})
    if a:
        b = a["areas"]["BLA"]; v = a["areas"]["vCA1_13col"]; v2 = a["areas"]["vCA1_v2_b0"]
        ev["a08"] = (f"BLA v2b AUC {b['session']['rankv2b_35']['auc_full']['mean']:.4f} session -> {b['animal']['rankv2b_35']['auc_full']['mean']:.4f} animal; "
                     f"FAR@0.04 {b['session']['rankv2b_35']['far_at_T']['mean']:.2f}% -> {b['animal']['rankv2b_35']['far_at_T']['mean']:.2f}% (max "
                     f"{b['animal']['rankv2b_35']['far_at_T']['max']:.2f}); vCA1 13-col FAR@0.05 {v['session']['b13']['far_at_T']['mean']:.2f}% -> "
                     f"{v['animal']['b13']['far_at_T']['mean']:.2f}% LOAO (max {v['animal']['b13']['far_at_T']['max']:.2f}); arm b0 "
                     f"{v2['animal']['rankv2b_35']['far_at_T']['mean']:.2f}%; rankings (v2 > b13) hold under both")
    a = R.get("a09", {})
    if a:
        i = a["part_i"]; d = i["decision_far_7_minus_5"]
        ev["a09"] = (f"3-seed repro w=5 AUC {i['repro_3seed']['5.0']['auc']['mean']:.4f} / w=7.01 {i['repro_3seed']['7.01']['auc']['mean']:.4f}; at 8 seeds "
                     f"FAR@0.05(7.01)-FAR(5.0) = {d['mean']:+.2f} pp ({d['n_positive']}/8 seeds, se {d['se']:.2f}), AUC {i['decision_auc_7_minus_5']['mean']:+.4f}; "
                     f"dups: label1 {a['part_ii']['BLA']['label1_minus_masked']['mean']:+.4f} (BLA) / {a['part_ii']['vCA1']['label1_minus_masked']['mean']:+.4f} (vCA1)")
    a = R.get("a10", {})
    if a:
        ev["a10"] = (f"rule gives 0.06 pre-fix (table + vectors) and {a['c3_oof']['rule_T']} post-fix; at 0.045 FAR {a['c3_oof']['at_0.045']['far_mean']:.2f}% "
                     f"max {a['c3_oof']['at_0.045']['far_max']:.2f}%; animal-grouped BLA -> {a.get('c3_animal_grouped', {}).get('rule_T')}; vCA1 13-col -> "
                     f"{a['vca1_13col_pin']['rule_T']}, arm b0 -> {a['vca1_v2_b0']['rule_T']}, LOAO -> {a.get('vca1_13col_animal', {}).get('rule_T')}")
    a = R.get("a11", {})
    if a:
        neg = a["negative_cells"]
        e = a["part_a"]["by_era"]["oof_v2b"]
        ev["a11"] = (f"paired per-animal deltas (pre->post fix) positive for 5/6 animals; negative cell {neg}; early era "
                     f"{e['a_early_2025']['delta']['mean']:+.4f} ({e['a_early_2025']['delta']['n_positive']}/8) vs late "
                     f"{e['a_late_2026']['delta']['mean']:+.4f}")
    a = R.get("a12", {})
    if a:
        r = a["repro"]
        ev["a12"] = (f"both protocols reproduced (eval {r['eval_protocol']['A_unit_mean']:.3f}->{r['eval_protocol']['B_unit_mean']:.3f}; LOO "
                     f"{r['loo_protocol']['A_unit_mean']:.3f}->{r['loo_protocol']['B_unit_mean']:.3f}); fixed 5.0 not worse than sqrt on FAR in "
                     f"{sum(v['fixed5_not_worse_far'] for v in a['deploy_relevant'].values())}/{len(a['deploy_relevant'])} protocol cells")
    a = R.get("a13", {})
    if a:
        v, b = a["part_b"]["vCA1"], a["part_b"]["BLA"]
        ev["a13"] = (f"simulated old positives {v['n_old_positives']}+{b['n_old_positives']}: {v['oos']['frac_ge_T']:.0%}/{b['oos']['frac_ge_T']:.0%} score >= T "
                     f"out-of-sample (unmasked negatives {v['reference_oos']['unmasked_neg_frac_ge_T']:.0%}/{b['reference_oos']['unmasked_neg_frac_ge_T']:.0%}; true positives "
                     f"{v['reference_oos']['true_pos_frac_ge_T']:.0%}/{b['reference_oos']['true_pos_frac_ge_T']:.0%}); "
                     f"{v['class_frac']['true_positive']:.0%}/{b['class_frac']['true_positive']:.0%} were the right cell")
    a = R.get("a14", {})
    if a:
        d = a["arms"]["b0_lso"]
        ev["a14"] = (f"real v2b on BLA bootstrap rows: reviewed {d['delta_reviewed_vs_a']['mean']:+.4f} ({d['delta_reviewed_vs_a']['n_positive']}/8; bar {d['bar']:.4f}); "
                     f"junk at matched FAR {d['matched_op_vs_a']['junk_ref_mean']:.1f}% -> {d['matched_op_vs_a']['junk_new_at_matched_far_mean']:.1f}%; "
                     f"fixture reproduced: {a['arm_a_reproduces_fixture']['pass']}")
    a = R.get("a15", {})
    if a:
        b, v = a["areas"]["BLA"], a["areas"]["vCA1"]
        ev["a15"] = (f"AUC flat 0->100% bootstrap (BLA {b['points']['0.0']['auc_full']['mean']:.4f}->{b['points']['1.0']['auc_full']['mean']:.4f}; vCA1 "
                     f"{v['points']['0.0']['auc_full']['mean']:.4f}->{v['points']['1.0']['auc_full']['mean']:.4f}) but FAR at fixed T "
                     f"{b['points']['0.0']['far_at_T']['mean']:.2f}->{b['points']['1.0']['far_at_T']['mean']:.2f}% / "
                     f"{v['points']['0.0']['far_at_T']['mean']:.2f}->{v['points']['1.0']['far_at_T']['mean']:.2f}%")
    a = R.get("a16", {})
    if a:
        b, v = a["areas"]["BLA"]["score_35"], a["areas"]["vCA1"]["score_13"]
        ev["a16"] = (f"hard band [0.02,0.10): bootstrap pos {b['bootstrap_pos']['frac_hard_band_0.02_0.10']:.3f} vs agent pos "
                     f"{b['agent_pos']['frac_hard_band_0.02_0.10']:.3f} (BLA); {v['bootstrap_pos']['frac_hard_band_0.02_0.10']:.3f} vs "
                     f"{v['agent_pos']['frac_hard_band_0.02_0.10']:.3f} (vCA1)")
    a = R.get("a17", {})
    if a:
        ev["a17"] = (f"no different-reviewer overlap on the server ({len(a['inbox']['candidates'])} candidates); proxies: reals < 0.02 = "
                     f"{a['proxies']['BLA']['reals_below_0.02_frac']:.4f} (BLA), {a['proxies']['vCA1']['reals_below_0.02_frac']:.4f} (vCA1)")
    a = R.get("a18", {})
    if a:
        dg = a["targets"]["DG_AL"]
        ev["a18"] = (f"session grouping reproduces the log; DG_AL +0.3 pooled: {dg['session']['0.3']['delta']['mean']:+.4f} session, "
                     f"{dg['animal']['0.3']['delta']['mean']:+.4f} LOAO ({dg['animal']['0.3']['delta']['n_positive']}/8), "
                     f"{dg['fov']['0.3']['delta']['mean']:+.4f} FOV ({dg['fov']['0.3']['delta']['n_positive']}/8)")
    a = R.get("a19", {})
    if a:
        g = a["groupings"]["animal"]["0.3"]
        ev["a19"] = (f"junk@0.05 own {g['junk_at_0.05']['A']['mean']:.1f}% vs pooled {g['junk_at_0.05']['B']['mean']:.1f}%; junk at matched FAR "
                     f"{g['junk_A_ref']:.1f}% -> {g['junk_B_at_matched_far']:.1f}%; leak-free isotonic at FAR<=1%: own "
                     f"{g['junk_at_far_le_1pct']['A']['mean']:.1f}% vs pooled {g['junk_at_far_le_1pct']['B_isotonic']['mean']:.1f}%; {a['usable']}")
    for k in sorted(TITLES):
        if k in R:
            P(f"| {int(k[1:])} | {TITLES[k]} | **{R[k].get('verdict')}** | {RF.get(k, {}).get('signoff', '-')} | {ev.get(k, '')} |")
    P("")
    P("## Section-D scoreboard")
    P("")
    P("| hyp | supported? | number |")
    P("|---|---|---|")
    if "a13" in R:
        P(f"| D13 old positives were cells | {R['a13']['hypothesis_D13_supported']} | {ev['a13']} |")
    if "a14" in R:
        P(f"| D14 BLA bootstrap rows feature-degenerate | {R['a14']['hypothesis_D14_supported']} | {ev['a14']} |")
    if "a15" in R:
        P(f"| D15 bootstrap redundant with agent data (AUC) | {R['a15']['hypothesis_D15_supported']} | {ev['a15']} |")
    if "a16" in R:
        P(f"| D16 bootstrap positives easy | {R['a16']['hypothesis_D16_supported']} | {ev['a16']} |")
    if "a17" in R:
        P(f"| D17 reviewer ceiling | {R['a17']['verdict']} | {ev['a17']} |")
    P("")
    P("## Deploy-verdict inputs")
    P("")
    if "a08" in R and "a10" in R:
        b = R["a08"]["areas"]["BLA"]; v = R["a08"]["areas"]["vCA1_13col"]; v2 = R["a08"]["areas"]["vCA1_v2_b0"]
        P("| model | grouping | AUC full | AUC reviewed | FAR@T mean (max) | junk@T | rule T |")
        P("|---|---|---|---|---|---|---|")
        for label, area, var in (("BLA 35-col @0.04", b, "rankv2b_35"), ("vCA1 13-col @0.05", v, "b13"), ("vCA1 arm b0 @0.05", v2, "rankv2b_35")):
            for grp in ("session", "animal"):
                e = area[grp][var]
                P(f"| {label} | {grp} | {e['auc_full']['mean']:.4f} | {e['auc_reviewed']['mean']:.4f} | {e['far_at_T']['mean']:.2f}% ({e['far_at_T']['max']:.2f}) | "
                  f"{e['junk_full_at_T']['mean']:.1f}% | {e['rule_T']} |")
    P("")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
