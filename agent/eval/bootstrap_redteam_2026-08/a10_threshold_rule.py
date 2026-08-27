"""
a10_threshold_rule.py -- attack #10: the Step-5 threshold rule.

Own implementation of the rule (largest T in [0.03, 0.10] with mean false-AR
<= 0.85% and worst-seed <= 1.0%; gate junk_full >= 30%) applied to:
  * the pinned pre-fix baseline: step5_results.json's own table AND the raw
    step5_oof.npz vectors -> must give 0.06 (self-consistency);
  * the post-fix c3_bla_gate8_oof.npz vectors -> 0.04, on a 0.005 grid too
    (0.035 / 0.045 / 0.055), which settles the undocumented "0.045 fails
    the worst-seed rule" sentence with a computed number;
  * the reviewed stratum (junk_reviewed in the gate instead of junk_full);
  * the animal-grouped OOF from attack #8 (fixtures/a08_BLA.npz);
  * vCA1: the 13-col pin (_pinned/baseline_oof.npz) -> 0.03, the arm-b0 gate
    fixture (gate_b0_w5_oof.npz) -> 0.05, and their animal-grouped versions.
Read-only; no refits.
"""
import json
import sys

import numpy as np

import rt_lib as rt

STEP5_JSON = rt.AGENT / "eval" / "step4_2026-08" / "step5_results.json"
STEP5_OOF = rt.AGENT / "eval" / "step4_2026-08" / "step5_oof.npz"
C3_OOF = rt.AGENT / "eval" / "bootstrap_matching_2026-08" / "c3_bla_gate8_oof.npz"
VCA1_PIN = rt.EXT["vCA1"] / "_pinned" / "baseline_oof.npz"
VCA1_B0 = rt.AGENT / "eval" / "vca1_v2_2026-08" / "gate_b0_w5_oof.npz"
FINE = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10]


def apply(oof, y, rev, label, thresholds=FINE):
    tab = rt.threshold_table(oof, y, rev, thresholds)
    cT, gate, det = rt.rule_T(tab)
    # reviewed-stratum variant of the gate
    cT_rev = None
    for k in sorted(tab, key=float):
        t = float(k)
        if 0.03 - 1e-9 <= t <= 0.10 + 1e-9 and np.mean(tab[k]["far"]) <= 0.85 and max(tab[k]["far"]) <= 1.0:
            cT_rev = t
    gate_rev = bool(cT_rev is not None and np.nanmean(tab[f"{cT_rev:.3f}"]["junk_reviewed"]) >= 30.0)
    row045 = tab.get("0.045")
    out = {"rule_T": cT, "gate": gate, "detail": det, "rule_T_reviewed_gate": cT_rev, "gate_reviewed": gate_rev,
           "table": {k: {kk: [round(float(x), 4) for x in vv] for kk, vv in v.items()} for k, v in tab.items()},
           "at_0.045": {"far_mean": float(np.mean(row045["far"])), "far_max": float(max(row045["far"])),
                        "junk_full": float(np.mean(row045["junk_full"]))} if row045 else None}
    rt.log(f"  {label}: rule T = {cT} (gate {gate}; reviewed-gate T {cT_rev} {gate_rev}); "
           + (f"at 0.045: FAR {out['at_0.045']['far_mean']:.2f}% max {out['at_0.045']['far_max']:.2f}%" if row045 else ""))
    for k in sorted(tab, key=float):
        v = tab[k]
        rt.log(f"     T {float(k):.3f}: FAR {np.mean(v['far']):.2f} (max {max(v['far']):.2f})  junk {np.mean(v['junk_full']):.1f}  rev {np.nanmean(v['junk_reviewed']):.1f}")
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    res = {}
    # 1. self-consistency on the pinned table itself (their numbers, our rule)
    s5 = json.loads(STEP5_JSON.read_text())
    tab5 = {k: v for k, v in s5["threshold_table"].items()}
    cT5, g5, d5 = rt.rule_T(tab5)
    res["step5_table_rule_T"] = {"rule_T": cT5, "gate": g5, "claimed": s5.get("chosen_T"), "match": cT5 == s5.get("chosen_T")}
    rt.log(f"step5_results.json table -> rule T {cT5} (claimed {s5.get('chosen_T')})")
    # 2. raw vectors
    f5 = np.load(STEP5_OOF, allow_pickle=True)
    res["step5_oof"] = apply(f5["oof_v2b"], f5["y"], f5["reviewed"], "pinned pre-fix (step5_oof v2b)")
    c3 = np.load(C3_OOF, allow_pickle=True)
    res["c3_oof"] = apply(c3["oof_v2b"], c3["y"], c3["reviewed"], "post-fix (c3_bla_gate8 v2b)")
    res["c3_oof_b13"] = apply(c3["oof_b13"], c3["y"], c3["reviewed"], "post-fix b13 (for reference)")
    # 3. animal-grouped (from attack #8) if present
    a8 = rt.FIXTURES / "a08_BLA.npz"
    if a8.exists():
        f8 = np.load(a8, allow_pickle=True)
        res["c3_animal_grouped"] = apply(f8["oof_animal_rankv2b_35"], f8["y"], f8["reviewed"], "post-fix, ANIMAL-grouped v2b")
        res["c3_session_grouped_a08"] = apply(f8["oof_session_rankv2b_35"], f8["y"], f8["reviewed"], "post-fix, session-grouped (a08 rerun)")
    # 4. vCA1
    vp = np.load(VCA1_PIN, allow_pickle=True)
    res["vca1_13col_pin"] = apply(vp["oof_seeds"], vp["y"], vp["reviewed"], "vCA1 13-col pin (deployed model)")
    vb = np.load(VCA1_B0, allow_pickle=True)
    res["vca1_v2_b0"] = apply(vb["oof_v35"], vb["y"], vb["reviewed"], "vCA1 arm b0 (pending deploy)")
    a8v = rt.FIXTURES / "a08_vCA1.npz"
    if a8v.exists():
        f8v = np.load(a8v, allow_pickle=True)
        keyb = "oof_animal_rankv2b_35" if "oof_animal_rankv2b_35" in f8v.files else None
        res["vca1_13col_animal"] = apply(f8v["oof_animal_b13"], f8v["y"], f8v["reviewed"], "vCA1 13-col, ANIMAL-grouped (LOAO)")
        if keyb:
            res["vca1_v2_b0_animal"] = apply(f8v[keyb], f8v["y"], f8v["reviewed"], "vCA1 arm b0, ANIMAL-grouped (LOAO)")
    checks = {
        "pinned_table_gives_0.06": cT5 == 0.06,
        "pinned_vectors_give_0.06": res["step5_oof"]["rule_T"] == 0.06,
        "postfix_vectors_give_0.04": res["c3_oof"]["rule_T"] == 0.04,
        "0.045_fails_worst_seed": res["c3_oof"]["at_0.045"]["far_max"] > 1.0,
        "vca1_13col_gives_0.03": res["vca1_13col_pin"]["rule_T"] == 0.03,
        "vca1_b0_gives_0.05": res["vca1_v2_b0"]["rule_T"] == 0.05,
    }
    if "c3_animal_grouped" in res:
        checks["0.04_holds_under_animal_grouping"] = res["c3_animal_grouped"]["rule_T"] == 0.04
    res["checks"] = checks
    core = ["pinned_table_gives_0.06", "pinned_vectors_give_0.06", "postfix_vectors_give_0.04"]
    res["verdict"] = "PASS" if all(checks[k] for k in core) else "FAIL"
    res["criterion"] = ("own rule reproduces 0.06 on the pinned baseline (table and vectors) and 0.04 post-fix; "
                        "0.045 and the animal-grouped / reviewed-stratum outcomes are reported as numbers")
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a10", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
