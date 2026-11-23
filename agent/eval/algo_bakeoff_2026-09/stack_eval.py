"""
stack_eval.py -- Stage 2 comparison: the raw-data CNN against the deployed
XGBoost, on identical rows, folds and seeds.

Three comparisons per area and grouping (README.md / plan, pre-registered):
  (a) CNN standalone vs control on the rows that HAVE raw data (auto-rejected
      candidates have none; the same subset is used for both models)
  (b) blend = probability mean of control and CNN where the CNN scored, the
      control alone elsewhere -- defined on every row, so it is compared with
      the control on the full pool exactly like a tabular contender
  (c) rank blend: the same with within-seed ranks (scale-free variant)

Gate (plan, Stage 2 step 4), on BLA under BOTH groupings, for the blend:
  reviewed-pool AUC delta >= +0.005 with 8/8 seeds, junk at the control's
  matched false-AR >= +3 pp, and rule_T finds a T under animal grouping with
  worst-seed false-AR <= 1%.
Kill criterion: CNN standalone reviewed AUC on the raw-data rows below the
control's on the same rows by more than 0.02.

Reads results/<AREA>_<grouping>_{xgb_deployed,cnn}_t8.npz; writes
results/stage2_<AREA>_t8.{md,json}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")


def load(area, grouping, name):
    p = RESULTS / f"{area}_{grouping}_{name}_t8.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def block(y, rev, ref, new, T, label):
    """Paired metrics of `new` vs `ref` (both (seeds, rows) on the same rows)."""
    auc_f = [rt.auc(y, o) for o in new]
    auc_r = [rt.auc(y, o, rev) for o in new]
    ref_f = [rt.auc(y, o) for o in ref]
    ref_r = [rt.auc(y, o, rev) for o in ref]
    d_f, d_r = rt.paired_delta(ref_f, auc_f), rt.paired_delta(ref_r, auc_r)
    tab = rt.threshold_table(new, y, rev)
    cT, gate, det = rt.rule_T(tab)
    mop = rt.matched_operating_point(ref, new, y, T)
    return {
        "label": label, "n_rows": int(len(y)), "n_real": int(y.sum()),
        "auc_full": auc_f, "auc_reviewed": auc_r, "ref_auc_full": ref_f, "ref_auc_reviewed": ref_r,
        "d_auc_full": d_f, "d_auc_reviewed": d_r,
        "rule_T": cT, "rule_gate": gate, "rule_detail": det,
        "far_ref_at_T": mop["far_ref_mean"], "junk_ref_at_T": mop["junk_ref_mean"],
        "far_new_at_matched_junk": mop["far_new_at_matched_junk_mean"],
        "junk_new_at_matched_far": mop["junk_new_at_matched_far_mean"],
        "d_junk_pp": mop["junk_new_at_matched_far_mean"] - mop["junk_ref_mean"],
    }


def fmt(b):
    d = b["d_auc_reviewed"]
    return (f"| {b['label']} | {b['n_rows']} / {b['n_real']} | {np.mean(b['ref_auc_full']):.4f} / {np.mean(b['ref_auc_reviewed']):.4f} "
            f"| {np.mean(b['auc_full']):.4f} / {np.mean(b['auc_reviewed']):.4f} | {d['mean']:+.4f} ({d['min']:+.4f}; {d['n_positive']}/{d['n']}) "
            f"| {b['rule_T']} ({b['rule_detail'].get('far_max', float('nan')):.2f} worst) | {b['far_new_at_matched_junk']:.2f} vs {b['far_ref_at_T']:.2f} "
            f"| {b['junk_new_at_matched_far']:.1f} vs {b['junk_ref_at_T']:.1f} ({b['d_junk_pp']:+.1f}) |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="BLA")
    ap.add_argument("--grouping", default="session,animal")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    area = a.area
    T = B.T_DEPLOY[area]
    out = {"area": area, "T": T, "groupings": {}}
    lines = [f"# Stage 2: raw-data CNN vs deployed XGBoost -- {area} ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
             "Columns: rows / reals used; control AUC full / reviewed; new AUC full / reviewed; paired reviewed-AUC delta "
             "(min; seeds positive); rule_T (worst-seed FAR at it); FAR of new at the control's matched junk vs the control's FAR at T; "
             "junk of new at the control's matched FAR vs the control's junk at T (delta pp).", ""]
    for grouping in [g.strip() for g in a.grouping.split(",") if g.strip()]:
        ctrl, cnn = load(area, grouping, "xgb_deployed"), load(area, grouping, "cnn")
        if ctrl is None or cnn is None:
            lines.append(f"## {grouping}: missing {'control' if ctrl is None else 'cnn'} result")
            continue
        assert np.array_equal(ctrl["y"], cnn["y"]), "rows differ"
        y, rev = ctrl["y"].astype(int), ctrl["reviewed"].astype(bool)
        seeds = [int(s) for s in cnn["seeds"] if int(s) in [int(t) for t in ctrl["seeds"]]]
        ci = [[int(t) for t in ctrl["seeds"]].index(s) for s in seeds]
        ni = [[int(t) for t in cnn["seeds"]].index(s) for s in seeds]
        ref, new = ctrl["oof"][ci], cnn["oof"][ni]
        has = cnn["has_raw"].astype(bool) & ~np.isnan(new).any(axis=0)
        # (a) standalone on the raw-data subset
        a_blk = block(y[has], rev[has], ref[:, has], new[:, has], T, "(a) CNN alone, raw-data rows")
        # (b) probability blend on all rows
        blend = np.where(np.isnan(new), ref, 0.5 * (ref + np.nan_to_num(new)))
        b_blk = block(y, rev, ref, blend, T, "(b) blend mean(control, CNN), all rows")
        # (c) rank blend on all rows (scale-free); rank normalised to [0,1] so T is a quantile
        rk = np.zeros_like(ref)
        for s in range(len(seeds)):
            r1 = rankdata(ref[s]) / len(y)
            r2 = rankdata(np.where(np.isnan(new[s]), ref[s], new[s])) / len(y)
            rk[s] = 0.5 * (r1 + r2)
        c_blk = block(y, rev, ref, rk, T, "(c) rank blend (T is a quantile here)")
        # per-animal (reviewed AUC) for (a)
        animals = np.asarray(ctrl["animals"]).astype(str) if "animals" in ctrl else None
        per_animal = {}
        if animals is not None:
            for an in sorted(set(animals)):
                m = has & (animals == an)
                if y[m].sum() >= 5 and (y[m] == 0).sum() >= 5:
                    per_animal[an] = {"n": int(m.sum()), "n_real": int(y[m].sum()),
                                      "control": float(np.mean([rt.auc(y[m], o[m]) for o in ref])),
                                      "cnn": float(np.mean([rt.auc(y[m], o[m]) for o in new])),
                                      "blend": float(np.mean([rt.auc(y[m], o[m]) for o in blend]))}
        kill = (np.mean(a_blk["auc_reviewed"]) < np.mean(a_blk["ref_auc_reviewed"]) - 0.02)
        gate_auc = b_blk["d_auc_reviewed"]["mean"] >= 0.005 and b_blk["d_auc_reviewed"]["all_positive"]
        gate_junk = b_blk["d_junk_pp"] >= 3.0
        gate_rule = (grouping != "animal") or (b_blk["rule_T"] is not None and b_blk["rule_detail"].get("far_max", 9) <= 1.0)
        out["groupings"][grouping] = {"seeds": seeds, "n_rows_with_raw": int(has.sum()), "a": a_blk, "b": b_blk, "c": c_blk,
                                      "per_animal": per_animal, "kill": bool(kill),
                                      "gate": {"auc": bool(gate_auc), "junk": bool(gate_junk), "rule": bool(gate_rule),
                                               "pass": bool(gate_auc and gate_junk and gate_rule)}}
        lines += [f"## {grouping}-grouped ({len(seeds)} seeds; {int(has.sum())} of {len(y)} rows have raw data)", "",
                  "| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |",
                  "|---|---|---|---|---|---|---|---|",
                  fmt(a_blk), fmt(b_blk), fmt(c_blk), "",
                  f"kill criterion (CNN alone < control - 0.02 on reviewed AUC): {'TRIGGERED' if kill else 'no'}; "
                  f"gate for the blend: AUC {'pass' if gate_auc else 'fail'}, junk {'pass' if gate_junk else 'fail'}, "
                  f"rule {'pass' if gate_rule else 'fail'} -> **{'PASS' if (gate_auc and gate_junk and gate_rule) else 'FAIL'}**", ""]
        if per_animal:
            lines += ["| animal | rows | reals | control AUC | CNN AUC | blend AUC |", "|---|---|---|---|---|---|"]
            lines += [f"| {an} | {v['n']} | {v['n_real']} | {v['control']:.4f} | {v['cnn']:.4f} | {v['blend']:.4f} |"
                      for an, v in per_animal.items()]
            lines.append("")
    md = RESULTS / f"stage2_{area}_t8.md"
    md.write_text("\n".join(lines))
    (RESULTS / f"stage2_{area}_t8.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    print("\n".join(lines))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
