"""
pareto_sweep.py -- the false-auto-reject / junk-caught trade-off curve for the
two DEPLOYED classifiers, on a fine threshold grid.

Both curves come from out-of-fold scores produced by the red team's independent
evaluator (agent/eval/bootstrap_redteam_2026-08/rt_lib.py), 8 seeds
[42, 1, 7, 13, 100, 2024, 31337, 9], StratifiedGroupKFold(5) with bootstrap
sessions always in train.  Nothing here refits a deployed model or touches a
session dir -- it is read-only.

Sources per area:
  BLA  (35-col rankv2b_35, agent weight sqrt/4.0, deployed T=0.04)
       recomputed from the CURRENT live candidate_features.npz, because the
       2026-08-26 cn_correlation refresh changed 6 sessions AFTER the red team's
       fixtures were pinned.  Self-checked against cnfix_gate.json "corrected".
  vCA1 (35-col arm b0, agent weight 5.0, deployed T=0.04)
       read from the gate fixture gate_b0_w5_oof.npz, which rt_selftest.py
       reproduced with max score difference 0.0.
  Animal-grouped ("new animal") curves: vCA1 from the red team's a08 fixture;
  BLA recomputed here on the corrected corpus.

Metrics at each T (mean over seeds, plus worst seed for false-AR):
  false_ar        % of REAL cells scored below T (cells silently lost)
  junk_caught     % of JUNK candidates scored below T
  junk_reviewed   % of junk among human-reviewed rows only
  auto_rejected   % of ALL candidates removed  (= the review-burden saving)
  reals_lost_per_session / candidates_removed_per_session (absolute)

Usage:  python pareto_sweep.py            (writes pareto.json + prints tables)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RT = HERE.parents[0] / "bootstrap_redteam_2026-08"
sys.path.insert(0, str(RT))
import rt_lib as rt  # noqa: E402

CNFIX_GATE = HERE.parents[0] / "cnfix_2026-08" / "cnfix_gate.json"
VCA1_GATE_OOF = HERE.parents[0] / "vca1_v2_2026-08" / "gate_b0_w5_oof.npz"
A08_VCA1 = RT / "fixtures" / "a08_vCA1.npz"
OUT_JSON = HERE / "pareto.json"
OUT_NPZ = HERE / "bla_deployed_oof.npz"

GRID = [round(x, 4) for x in np.arange(0.005, 0.2001, 0.005)]
DEPLOYED_T = {"BLA": 0.04, "vCA1": 0.04}


def curve(oof_seeds, y, reviewed, n_sessions, grid=GRID):
    """Per-threshold summary over seeds.  oof_seeds: (n_seeds, n_rows)."""
    pos, neg = y == 1, y == 0
    nr = neg & reviewed
    n_all = len(y)
    rows = []
    for t in grid:
        far = np.array([(o[pos] < t).sum() / pos.sum() * 100 for o in oof_seeds])
        jf = np.array([(o[neg] < t).sum() / neg.sum() * 100 for o in oof_seeds])
        jr = np.array([(o[nr] < t).sum() / nr.sum() * 100 for o in oof_seeds])
        ar = np.array([(o < t).sum() / n_all * 100 for o in oof_seeds])
        lost = np.array([(o[pos] < t).sum() for o in oof_seeds])
        rem = np.array([(o < t).sum() for o in oof_seeds])
        rows.append({
            "T": t,
            "false_ar_mean": float(far.mean()), "false_ar_worst": float(far.max()),
            "false_ar_sd": float(far.std(ddof=1)),
            "junk_caught_mean": float(jf.mean()),
            "junk_reviewed_mean": float(jr.mean()),
            "auto_rejected_mean": float(ar.mean()),
            "reals_lost_mean": float(lost.mean()),
            "reals_lost_per_session": float(lost.mean() / n_sessions),
            "candidates_removed_per_session": float(rem.mean() / n_sessions),
        })
    return rows


def summarize(area, oof_seeds, y, reviewed, names, extra=None):
    n_sessions = len(names)
    aucs_full = [rt.auc(y, o) for o in oof_seeds]
    aucs_rev = [rt.auc(y, o, reviewed) for o in oof_seeds]
    d = {
        "area": area,
        "n_sessions": int(n_sessions),
        "n_candidates": int(len(y)),
        "n_real": int(y.sum()),
        "n_junk": int((y == 0).sum()),
        "n_reviewed_junk": int(((y == 0) & reviewed).sum()),
        "n_seeds": int(len(oof_seeds)),
        "auc_full_mean": float(np.mean(aucs_full)),
        "auc_full_sd": float(np.std(aucs_full, ddof=1)),
        "auc_reviewed_mean": float(np.mean(aucs_rev)),
        "deployed_T": DEPLOYED_T[area],
        "curve": curve(oof_seeds, y, reviewed, n_sessions),
    }
    if extra:
        d.update(extra)
    return d


def bla_deployed():
    """Recompute the deployed BLA model's OOF on the CURRENT (corrected) corpus."""
    rt.log("BLA: loading the live 35-col pool ...")
    recs = rt.load_pool("BLA", width=35)
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs)
    rt.log(f"BLA: {len(recs)} labeled sessions -> {len(cv)} CV / {len(bs)} bootstrap; "
           f"{len(y)} rows, {int(y.sum())} real, {int(w_bs.size - (w_bs > 0).sum())} masked bootstrap rows")
    # animal groups
    uniq = {a: i for i, a in enumerate(sorted(set(animals)))}
    g_animal = np.array([uniq[a] for a in animals])
    rt.log("BLA: 8-seed session-grouped OOF ...")
    oof_sess = rt.run_oof_seeds(X_ag, y, g, X_bs, y_bs, w_bs, agent_weight=None)
    rt.log("BLA: 8-seed animal-grouped OOF ...")
    oof_anim = rt.run_oof_seeds(X_ag, y, g_animal, X_bs, y_bs, w_bs, agent_weight=None)
    np.savez_compressed(OUT_NPZ, y=y, reviewed=rev, animals=animals, groups=g,
                        groups_animal=g_animal, names=names,
                        oof_session=oof_sess, oof_animal=oof_anim)
    return oof_sess, oof_anim, y, rev, names, animals


def selfcheck_bla(oof_sess, y, rev):
    """The recomputed curve must match cnfix_gate.json's 'corrected' variant."""
    ref = json.loads(CNFIX_GATE.read_text())["corrected"]
    got = {
        "auc_full": float(np.mean([rt.auc(y, o) for o in oof_sess])),
        "auc_reviewed": float(np.mean([rt.auc(y, o, rev) for o in oof_sess])),
        "far_at_0.04": float(np.mean([rt.far_junk(o, y, 0.04)[0] for o in oof_sess])),
        "junk_at_0.04": float(np.mean([rt.far_junk(o, y, 0.04)[1] for o in oof_sess])),
        "far_at_0.03": float(np.mean([rt.far_junk(o, y, 0.03)[0] for o in oof_sess])),
        "junk_at_0.03": float(np.mean([rt.far_junk(o, y, 0.03)[1] for o in oof_sess])),
    }
    out = {}
    for k, v in got.items():
        r = ref[k]["mean"]
        out[k] = {"gate": r, "recomputed": v, "abs_diff": abs(r - v)}
    out["pass"] = all(v["abs_diff"] < 5e-3 for v in out.values() if isinstance(v, dict))
    return out


def vca1_deployed():
    d = np.load(VCA1_GATE_OOF, allow_pickle=True)
    a08 = np.load(A08_VCA1, allow_pickle=True)
    assert np.array_equal(d["y"], a08["y"]), "vCA1 fixtures disagree on rows"
    return (d["oof_v35"], a08["oof_animal_rankv2b_35"], d["y"], d["reviewed"],
            d["names"], d["animals"], d["oof_b13"])


def print_table(name, res, marks):
    print(f"\n=== {name} ===")
    print(f"  {res['n_sessions']} CV sessions | {res['n_candidates']} candidates | "
          f"{res['n_real']} real / {res['n_junk']} junk ({res['n_reviewed_junk']} reviewed junk) | "
          f"AUC {res['auc_full_mean']:.4f}")
    print(f"  {'T':>6} {'falseAR%':>9} {'worst':>7} {'junk%':>7} {'junkRev%':>9} "
          f"{'autoRej%':>9} {'lost/sess':>10} {'cut/sess':>9}")
    for r in res["curve"]:
        if r["T"] not in marks:
            continue
        star = " <-- DEPLOYED" if abs(r["T"] - res["deployed_T"]) < 1e-9 else ""
        print(f"  {r['T']:>6.3f} {r['false_ar_mean']:>9.2f} {r['false_ar_worst']:>7.2f} "
              f"{r['junk_caught_mean']:>7.1f} {r['junk_reviewed_mean']:>9.1f} "
              f"{r['auto_rejected_mean']:>9.1f} {r['reals_lost_per_session']:>10.2f} "
              f"{r['candidates_removed_per_session']:>9.1f}{star}")


def main():
    out = {"grid": GRID, "seeds": rt.SEEDS,
           "note": "8-seed grouped-CV out-of-fold scores; read-only; see module docstring"}

    oof_s, oof_a, y, rev, names, animals = bla_deployed()
    chk = selfcheck_bla(oof_s, y, rev)
    out["bla_selfcheck_vs_cnfix_gate"] = chk
    rt.log(f"BLA self-check vs cnfix_gate.json corrected: "
           f"{'PASS' if chk['pass'] else 'FAIL'} "
           f"(max abs diff {max(v['abs_diff'] for v in chk.values() if isinstance(v, dict)):.2e})")
    out["BLA"] = summarize("BLA", oof_s, y, rev, names,
                           {"grouping": "session", "corpus": "live/corrected 2026-08-26",
                            "contract": "rankv2b_35", "agent_weight": "sqrt/4.0 per fold"})
    out["BLA_animal_grouped"] = summarize("BLA", oof_a, y, rev, names,
                                          {"grouping": "animal (5 folds over 6 animals)"})
    per_animal = {}
    for a in sorted(set(animals)):
        m = animals == a
        per_animal[a] = {"n_rows": int(m.sum()), "n_real": int(y[m].sum()),
                         "auc_session_grouped": float(np.mean([rt.auc(y[m], o[m]) for o in oof_s])),
                         "auc_animal_grouped": float(np.mean([rt.auc(y[m], o[m]) for o in oof_a]))}
    out["BLA_per_animal"] = per_animal

    v_s, v_a, vy, vrev, vnames, vanimals, v_b13 = vca1_deployed()
    out["vCA1"] = summarize("vCA1", v_s, vy, vrev, vnames,
                            {"grouping": "session", "corpus": "pinned 2026-08-26 (arm b0)",
                             "contract": "rankv2b_35 arm b0", "agent_weight": "5.0 fixed"})
    out["vCA1_animal_grouped"] = summarize("vCA1", v_a, vy, vrev, vnames,
                                           {"grouping": "animal (2-animal LOAO)"})
    out["vCA1_previous_13col"] = summarize("vCA1", v_b13, vy, vrev, vnames,
                                           {"grouping": "session", "contract": "13-col (pre-2026-08-26)"})
    out["BLA_previous_13col"] = None

    OUT_JSON.write_text(json.dumps(out, indent=1))
    marks = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]
    print_table("BLA  deployed (35-col rankv2b_35, T=0.04) -- session-grouped", out["BLA"], marks)
    print_table("BLA  same model, ANIMAL-grouped (a new mouse)", out["BLA_animal_grouped"], marks)
    print_table("vCA1 deployed (35-col arm b0, T=0.04) -- session-grouped", out["vCA1"], marks)
    print_table("vCA1 same model, ANIMAL-grouped (a new mouse)", out["vCA1_animal_grouped"], marks)
    print_table("vCA1 PREVIOUS 13-col model -- session-grouped", out["vCA1_previous_13col"], marks)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
