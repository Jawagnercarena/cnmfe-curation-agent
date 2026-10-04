"""
Step 3d: apply the pre-registered decision rules to the gate outputs.

The rules and their constants were fixed BEFORE any gate ran (see the approved
plan, section 4), so this script only reads gate_*.json and reports.  Nothing
here is tuned after seeing results.

  seed noise   a paired per-seed delta d "beats" if
                   mean(d) >= max(0.005, 2 * sd(d)/sqrt(n))  AND  >= 7/8 seeds > 0
  arm          b0 vs a, then b1 vs b0; ties go to arm (a)      [user decision D2]
  v2 at all    on the WINNING arm: reviewed paired delta >= +0.010,
               all 8 seeds > 0, and false-AR at matched junk no worse than b13
  weight       switch off 5.0 only if W beats it on reviewed AUC and FAR@0.05
               does not worsen
  threshold    Step 4 rule verbatim: largest T in [0.03, 0.10] with mean
               false-AR <= 0.85% AND worst-seed <= 1.0%; then junk_full >= 30%.
               No qualifying T -> STOP (deploy_ok False), report the table.

  python decide_vca1.py --selftest        prove the rule reproduces Step 4's 0.06
  python decide_vca1.py --stage arm
  python decide_vca1.py --stage weight --arm <winner>
  python decide_vca1.py --stage threshold --arm <winner> --agent-weight <W>
  python decide_vca1.py --stage all
"""
import argparse
import glob
import json
import sys

import numpy as np

import vca1_common as vc

MIN_EFFECT = 0.005          # floor on "beats beyond seed noise"
SHIP_DELTA = 0.010          # reviewed-AUC bar for adopting the 35-col contract
MIN_SEEDS_POSITIVE = 7      # of 8
FAR_TOLERANCE = 0.2         # pp; matched-junk false-AR may not worsen by more


def load_gate(arm, w):
    f = vc.SP / f"gate_{arm}_w{w:g}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def beats(d):
    """Paired per-seed delta list -> (verdict, detail). The seed-noise rule."""
    d = np.asarray(d, dtype=float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    bar = max(MIN_EFFECT, 2 * se)
    n_pos = int((d > 0).sum())
    verdict = bool(d.mean() >= bar and n_pos >= MIN_SEEDS_POSITIVE)
    return verdict, {"mean": float(d.mean()), "min": float(d.min()),
                     "se": se, "bar": bar, "n_positive": n_pos,
                     "n_seeds": len(d), "beats": verdict}


def chosen_T(table):
    """Step 4 rule verbatim (threshold_sweep_v2.py:188-196)."""
    chosen = None
    for t in sorted(float(k) for k in table):
        if t > 0.10:
            continue
        far = table[f"{t:.2f}"]["far"]
        if np.mean(far) <= 0.85 and max(far) <= 1.0:
            chosen = t
    return chosen


def selftest():
    """The rule must reproduce Step 4's published choice of 0.06 for BLA."""
    f = vc.AGENT / "eval" / "step4_2026-08" / "step5_results.json"
    s4 = json.loads(f.read_text())
    got = chosen_T(s4["threshold_table"])
    exp = s4["chosen_T"]
    ok = got == exp
    print(f"selftest: Step 4 table -> chosen T {got} (published {exp}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def stage_arm():
    print("=== STAGE: bootstrap arm ===")
    gates = {a: load_gate(a, vc.AGENT_WEIGHT) for a in (vc.ARM_A, vc.ARM_B0, vc.ARM_B1)}
    missing = [a for a, g in gates.items() if g is None]
    if missing:
        print(f"  missing gate runs for arm(s) {missing} -- run gate_vca1.py first")
        return None, {}

    detail = {}
    for a, g in gates.items():
        rev = g["auc_reviewed"]
        print(f"  arm {a:<3} reviewed AUC b13 {np.mean(rev['b13']):.4f} -> "
              f"v35 {np.mean(rev['rankv2b_35']):.4f}  "
              f"(paired {np.mean(g['delta_reviewed']):+.4f}); "
              f"chosen T {g['chosen_T']}; "
              f"bootstrap flagged rows {g['bootstrap_flagged_rows']}")

    # b0 vs a, on the SAME test rows -> pair the per-seed v35 AUCs
    def paired(a1, a2):
        return [x - y for x, y in zip(gates[a2]["auc_reviewed"]["rankv2b_35"],
                                      gates[a1]["auc_reviewed"]["rankv2b_35"])]

    d_b0_a = paired(vc.ARM_A, vc.ARM_B0)
    v_b0, det_b0 = beats(d_b0_a)
    detail["b0_vs_a"] = det_b0
    print(f"\n  b0 vs a : mean {det_b0['mean']:+.4f}, bar {det_b0['bar']:.4f}, "
          f"{det_b0['n_positive']}/{det_b0['n_seeds']} seeds positive -> "
          f"{'b0 beats a' if v_b0 else 'no (tie -> arm a)'}")

    winner = vc.ARM_B0 if v_b0 else vc.ARM_A

    d_b1_b0 = paired(vc.ARM_B0, vc.ARM_B1)
    v_b1, det_b1 = beats(d_b1_b0)
    detail["b1_vs_b0"] = det_b1
    print(f"  b1 vs b0: mean {det_b1['mean']:+.4f}, bar {det_b1['bar']:.4f}, "
          f"{det_b1['n_positive']}/{det_b1['n_seeds']} seeds positive -> "
          f"{'b1 beats b0' if v_b1 else 'no'}")

    d_b1_a = paired(vc.ARM_A, vc.ARM_B1)
    v_b1a, det_b1a = beats(d_b1_a)
    detail["b1_vs_a"] = det_b1a
    if v_b1 and (winner == vc.ARM_B0 or v_b1a):
        winner = vc.ARM_B1
    print(f"  b1 vs a : mean {det_b1a['mean']:+.4f} -> {'beats' if v_b1a else 'no'}")

    print(f"\n  ARM WINNER: {winner}")
    if winner == vc.ARM_B1:
        print("  -> follow-on candidate: regenerating candidate-resolution Cn for the 40 "
              "sessions without one (26 half-res imresize, 14 MATLAB pre-pass) could add "
              "ring_contrast there. NOT this project.")
    if winner != vc.ARM_A:
        print("  -> PRODUCTION FOLLOW-ON REQUIRED before any deploy: "
              "bootstrap_preagent.py:402-404 zero-fills unconditionally, so a future "
              "vCA1 bootstrap run would write arm-(a) rows into an arm-(b) corpus; and "
              "the leave-session-out hiconf used here has no production analogue.")
    return winner, detail


def stage_weight(arm):
    print(f"\n=== STAGE: agent weight (arm {arm}) ===")
    found = {}
    for f in sorted(glob.glob(str(vc.SP / f"gate_{arm}_w*.json"))):
        g = json.loads(open(f).read())
        found[g["agent_weight"]] = g
    if not found:
        print("  no gate runs found")
        return None, {}
    base = found.get(vc.AGENT_WEIGHT)
    for w in sorted(found):
        g = found[w]
        far05 = np.mean(g["threshold_table"]["0.05"]["far"])
        print(f"  w={w:<5g} reviewed v35 AUC {np.mean(g['auc_reviewed']['rankv2b_35']):.4f}"
              f"  FAR@0.05 {far05:5.2f}%  junk@0.05 "
              f"{np.mean(g['threshold_table']['0.05']['junk_full']):.1f}%"
              f"  chosen T {g['chosen_T']}")
    if base is None:
        print(f"  no run at the deployed weight {vc.AGENT_WEIGHT} -- keeping it by default")
        return vc.AGENT_WEIGHT, {}
    best, detail = vc.AGENT_WEIGHT, {}
    for w, g in sorted(found.items()):
        if w == vc.AGENT_WEIGHT:
            continue
        d = [x - y for x, y in zip(g["auc_reviewed"]["rankv2b_35"],
                                   base["auc_reviewed"]["rankv2b_35"])]
        v, det = beats(d)
        far_ok = (np.mean(g["threshold_table"]["0.05"]["far"])
                  <= np.mean(base["threshold_table"]["0.05"]["far"]) + FAR_TOLERANCE)
        detail[str(w)] = {**det, "far_not_worse": bool(far_ok)}
        if v and far_ok:
            best = w
    print(f"  WEIGHT: {best:g}" + ("" if best != vc.AGENT_WEIGHT else "  (deployed value kept)"))
    return best, detail


def stage_threshold(arm, w):
    print(f"\n=== STAGE: threshold (arm {arm}, weight {w:g}) ===")
    g = load_gate(arm, w)
    if g is None:
        print("  gate run missing")
        return None, {}
    t = chosen_T(g["threshold_table"])
    if t is None:
        print("  NO T in [0.03, 0.10] satisfies mean false-AR <= 0.85% and "
              "worst-seed <= 1.0%.")
        print("  -> STOP: report the table, no deploy. The user decides whether to "
              "accept a T outside the rule or wait for more reviewer returns.")
        return None, {"stop": True, "table": g["threshold_table"]}
    row = g["threshold_table"][f"{t:.2f}"]
    junk_ok = np.mean(row["junk_full"]) >= 30.0
    print(f"  chosen T = {t:.2f}: false-AR {np.mean(row['far']):.2f}% "
          f"(worst seed {max(row['far']):.2f}%), junk full {np.mean(row['junk_full']):.1f}%, "
          f"junk reviewed {np.mean(row['junk_reviewed']):.1f}%")
    print(f"  junk >= 30% gate: {'PASS' if junk_ok else 'FAIL'}")
    return t, {"stop": False, "junk_gate_pass": bool(junk_ok),
               "far_mean": float(np.mean(row["far"])), "far_max": float(max(row["far"])),
               "junk_full": float(np.mean(row["junk_full"])),
               "junk_reviewed": float(np.mean(row["junk_reviewed"]))}


def stage_ship(arm, w):
    """The 'is v2 worth it at all' criterion, evaluated on the WINNING arm."""
    print(f"\n=== STAGE: adopt the 35-column contract? (arm {arm}, weight {w:g}) ===")
    g = load_gate(arm, w)
    d = g["delta_reviewed"]
    all_pos = all(x > 0 for x in d)
    big = np.mean(d) >= SHIP_DELTA
    mj = g["matched_junk"]
    far_ok = (mj["far_v35"] is not None
              and mj["far_v35"] <= mj["far_b13"] + FAR_TOLERANCE)
    ship = bool(all_pos and big and far_ok)
    print(f"  reviewed paired delta {np.mean(d):+.4f} (bar +{SHIP_DELTA:.3f}) -> {big}")
    print(f"  all {len(d)} seeds positive -> {all_pos} (min {min(d):+.4f})")
    v35_txt = "n/a" if mj["far_v35"] is None else f"{mj['far_v35']:.2f}%"
    print(f"  false-AR at matched junk: b13 {mj['far_b13']:.2f}% -> v35 {v35_txt}"
          f" -> not worse: {far_ok}")
    print(f"  ADOPT v2: {'YES' if ship else 'NO'}")
    return ship, {"delta_mean": float(np.mean(d)), "delta_min": float(min(d)),
                  "all_seeds_positive": all_pos, "meets_bar": bool(big),
                  "far_not_worse": bool(far_ok),
                  "far_b13": mj["far_b13"], "far_v35": mj["far_v35"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=["arm", "weight", "threshold", "all"], default="all")
    ap.add_argument("--arm")
    ap.add_argument("--agent-weight", type=float)
    args = ap.parse_args()
    vc.configure()
    if args.selftest:
        return selftest()

    out = {"rules": {"min_effect": MIN_EFFECT, "ship_delta": SHIP_DELTA,
                     "min_seeds_positive": MIN_SEEDS_POSITIVE,
                     "far_tolerance_pp": FAR_TOLERANCE,
                     "threshold_rule": "largest T in [0.03,0.10] with mean far<=0.85 and max far<=1.0; junk_full>=30"}}

    arm = args.arm
    if args.stage in ("arm", "all"):
        arm, out["arm_detail"] = stage_arm()
        out["arm"] = arm
    if arm is None:
        print("\nno arm decided -- stopping")
        return 1

    w = args.agent_weight
    if args.stage in ("weight", "all"):
        w, out["weight_detail"] = stage_weight(arm)
        out["agent_weight"] = w
    w = w if w is not None else vc.AGENT_WEIGHT

    ship, out["ship_detail"] = stage_ship(arm, w)
    out["adopt_v2"] = ship

    if args.stage in ("threshold", "all"):
        t, out["threshold_detail"] = stage_threshold(arm, w)
        out["chosen_T"] = t

    out["production_followon_required"] = bool(arm != vc.ARM_A)
    out["deploy_ok"] = bool(ship and out.get("chosen_T") is not None
                            and out.get("threshold_detail", {}).get("junk_gate_pass"))

    print("\n" + "=" * 62)
    print(f"ARM {out.get('arm')}   WEIGHT {out.get('agent_weight')}   "
          f"T {out.get('chosen_T')}   ADOPT {out.get('adopt_v2')}")
    print(f"deploy_ok = {out['deploy_ok']}"
          + ("   (production follow-on required first)" if out["production_followon_required"] else ""))
    print("=" * 62)
    (vc.SP / "gate_decision.json").write_text(json.dumps(out, indent=1))
    print("wrote gate_decision.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
