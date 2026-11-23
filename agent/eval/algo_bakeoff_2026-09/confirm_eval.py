"""
confirm_eval.py -- Round 2 confirmation (README "Round 2"): a configuration's
8-seed OOF (results/r2/<variant>_<AREA>_<grouping>.npz) against the deployed
control on all four cells with the pre-declared combiner (fixed 50/50
probability blend; rows without a CNN score keep the control score), judged by
the UNCHANGED Stage 2 gate:

  BLA, both groupings   blend reviewed AUC >= +0.005 with all seeds positive,
                        junk at matched false-AR >= +3 pp, and under animal
                        grouping rule_T exists with worst-seed false-AR <= 1%
  vCA1, both groupings  "not worse": blend reviewed-AUC delta >= 0 and junk at
                        matched false-AR >= -1 pp (definition fixed 2026-09-07
                        before the confirmation run; README amendment)

Also reported: the round-1 CNN's blend on the same seeds (reference), CNN alone
for both, per-animal reviewed AUC.

Usage: python confirm_eval.py --variant valfix+ens3
Writes results/r2/confirm_<variant>.md / .json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import stack_eval as SE  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
R2 = RESULTS / "r2"
CELLS = [("BLA", "session"), ("BLA", "animal"), ("vCA1", "session"), ("vCA1", "animal")]


def blend_of(ref, cnn):
    return np.where(np.isnan(cnn), ref, 0.5 * (ref + np.nan_to_num(cnn)))


def cell(area, grouping, variant):
    vp = R2 / f"{variant}_{area}_{grouping}.npz"
    if not vp.exists():
        return None
    ctrl = np.load(RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
    v = np.load(vp, allow_pickle=True)
    r1p = RESULTS / f"{area}_{grouping}_cnn_t8.npz"           # round-1 CNN: absent in refresh result dirs
    r1 = np.load(r1p, allow_pickle=True) if r1p.exists() else {"y": v["y"], "seeds": v["seeds"], "oof": v["oof"]}
    has_r1 = r1p.exists()
    assert np.array_equal(v["y"], ctrl["y"]) and np.array_equal(v["y"], r1["y"]), "rows differ"
    y, rev = v["y"].astype(int), v["reviewed"].astype(bool)
    seeds = [int(s) for s in v["seeds"]]
    ic = [[int(t) for t in ctrl["seeds"]].index(s) for s in seeds]
    ir = [[int(t) for t in r1["seeds"]].index(s) for s in seeds]
    ref, o1, ov = ctrl["oof"][ic], r1["oof"][ir].astype(float), v["oof"].astype(float)
    T = B.T_DEPLOY[area]
    has = ~np.isnan(ov).any(axis=0) & ~np.isnan(o1).any(axis=0)
    blocks = {
        "cnn_alone": SE.block(y[has], rev[has], ref[:, has], ov[:, has], T, "(a) CNN alone, raw-data rows"),
        "cnn_alone_r1": SE.block(y[has], rev[has], ref[:, has], o1[:, has], T, "(a') round-1 CNN alone, same rows"),
        "blend": SE.block(y, rev, ref, blend_of(ref, ov), T, "(b) blend 50/50, all rows"),
        "blend_r1": SE.block(y, rev, ref, blend_of(ref, o1), T, "(b') round-1 blend 50/50, same seeds"),
    }
    animals = np.asarray(ctrl["animals"]).astype(str)
    per_animal = {}
    bv, b1 = blend_of(ref, ov), blend_of(ref, o1)
    for an in sorted(set(animals)):
        m = animals == an
        if y[m].sum() >= 5 and (y[m] == 0).sum() >= 5:
            per_animal[an] = {"n": int(m.sum()), "n_real": int(y[m].sum()),
                              "control": float(np.mean([rt.auc(y[m], o[m], rev[m]) for o in ref])),
                              "cnn": float(np.mean([rt.auc(y[m & has], o[m & has], rev[m & has]) for o in ov])),
                              "blend_r1": float(np.mean([rt.auc(y[m], o[m], rev[m]) for o in b1])),
                              "blend": float(np.mean([rt.auc(y[m], o[m], rev[m]) for o in bv]))}
    b = blocks["blend"]
    d = b["d_auc_reviewed"]
    if area == "BLA":
        g_auc = d["mean"] >= 0.005 and d["all_positive"]
        g_junk = b["d_junk_pp"] >= 3.0
        g_rule = (grouping != "animal") or (b["rule_T"] is not None and b["rule_detail"].get("far_max", 9) <= 1.0)
    else:
        g_auc = d["mean"] >= 0.0
        g_junk = b["d_junk_pp"] >= -1.0
        g_rule = True
    kill = np.mean(blocks["cnn_alone"]["auc_reviewed"]) < np.mean(blocks["cnn_alone"]["ref_auc_reviewed"]) - 0.02
    if not has_r1:
        for k in ("cnn_alone_r1", "blend_r1"):
            blocks[k]["label"] += " [no round-1 run in this results dir: same as the variant]"
    return {"area": area, "grouping": grouping, "seeds": seeds, "n_rows_with_raw": int(has.sum()), "blocks": blocks,
            "per_animal": per_animal, "kill": bool(kill),
            "gate": {"auc": bool(g_auc), "junk": bool(g_junk), "rule": bool(g_rule), "pass": bool(g_auc and g_junk and g_rule)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cells", default=",".join(f"{a}:{g}" for a, g in CELLS))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    cells = [tuple(c.split(":")) for c in a.cells.split(",") if c.strip()]
    out = {"variant": a.variant, "written_at": dt.datetime.now().isoformat(timespec="seconds"), "cells": {}}
    lines = [f"# Round 2 confirmation: `{a.variant}` vs the deployed control ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
             "Combiner: fixed 50/50 probability blend (pre-declared). Columns as in stage2_*.md.", ""]
    verdict = {}
    for area, grouping in cells:
        c = cell(area, grouping, a.variant)
        if c is None:
            lines.append(f"## {area} {grouping}: not run")
            continue
        out["cells"][f"{area}/{grouping}"] = c
        verdict[(area, grouping)] = c["gate"]["pass"]
        bl = c["blocks"]
        lines += [f"## {area} {grouping}-grouped ({len(c['seeds'])} seeds; {c['n_rows_with_raw']} rows with raw data)", "",
                  "| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |",
                  "|---|---|---|---|---|---|---|---|",
                  SE.fmt(bl["cnn_alone"]), SE.fmt(bl["cnn_alone_r1"]), SE.fmt(bl["blend"]), SE.fmt(bl["blend_r1"]), "",
                  f"kill (CNN alone < control - 0.02): {'TRIGGERED' if c['kill'] else 'no'}; gate: AUC {'pass' if c['gate']['auc'] else 'fail'}, "
                  f"junk {'pass' if c['gate']['junk'] else 'fail'}, rule {'pass' if c['gate']['rule'] else 'fail'} -> "
                  f"**{'PASS' if c['gate']['pass'] else 'FAIL'}**", ""]
        if c["per_animal"]:
            lines += ["| animal | rows | reals | control | CNN | round-1 blend | blend |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {an} | {v['n']} | {v['n_real']} | {v['control']:.4f} | {v['cnn']:.4f} | {v['blend_r1']:.4f} | {v['blend']:.4f} |"
                      for an, v in c["per_animal"].items()]
            lines.append("")
    all_ok = all(verdict.get(c, False) for c in CELLS)
    out["verdict"] = {"per_cell": {f"{a}/{g}": v for (a, g), v in verdict.items()}, "pass": bool(all_ok)}
    lines += ["## Verdict", "", "| cell | gate |", "|---|---|"]
    lines += [f"| {a} {g} | {'PASS' if verdict.get((a, g), False) else 'FAIL / not run'} |" for a, g in CELLS]
    lines += ["", f"**Pre-registered gate (BLA both groupings pass, vCA1 not worse under both): {'MET' if all_ok else 'NOT MET'}**", ""]
    R2.mkdir(exist_ok=True)
    (R2 / f"confirm_{a.variant}.md").write_text("\n".join(lines), encoding="utf-8")
    (R2 / f"confirm_{a.variant}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
