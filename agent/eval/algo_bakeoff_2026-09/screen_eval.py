"""
screen_eval.py -- Round 2 screening table (README "Round 2").

Each variant's OOF (results/r2/<variant>_<AREA>_<grouping>.npz) is paired
against the round-1 CNN (results/<AREA>_<grouping>_cnn_t8.npz, same seeds) and
the deployed control on the screening cells.  Per cell:
  * CNN alone: reviewed AUC round 1 -> variant on the rows both scored, paired
    per seed (mean delta, min, seeds positive)        <- the screening metric
  * blend: 50/50 probability blend with the control, reviewed-AUC delta vs the
    control (round-1 blend -> variant blend) and junk at matched false-AR
  * stack: if results/r2/stack_<variant>_<AREA>_<grouping>.json exists, the
    stacked model's delta vs the pinned control
  * training facts from the variant json: median best epoch, epoch-cap hits,
    inner-validation rows
Selection rule (pre-registered): carry a variant forward if CNN-alone delta
>= 0 on both cells and >= +0.003 on at least one.

Usage: python screen_eval.py [--variants valfix,fixed12,ens3,nbtrace,patch48] [--cells BLA:session,vCA1:animal]
Writes results/r2/screen.md and screen.json.
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


def blend_of(ref, cnn):
    return np.where(np.isnan(cnn), ref, 0.5 * (ref + np.nan_to_num(cnn)))


def cell(area, grouping, variant):
    vp = R2 / f"{variant}_{area}_{grouping}.npz"
    if not vp.exists():
        return None
    ctrl = np.load(RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
    r1 = np.load(RESULTS / f"{area}_{grouping}_cnn_t8.npz", allow_pickle=True)
    v = np.load(vp, allow_pickle=True)
    assert np.array_equal(v["y"], r1["y"]) and np.array_equal(v["y"], ctrl["y"]), "rows differ"
    seeds = [int(s) for s in v["seeds"]]
    ic = [[int(t) for t in ctrl["seeds"]].index(s) for s in seeds]
    ir = [[int(t) for t in r1["seeds"]].index(s) for s in seeds]
    y, rev = v["y"].astype(int), v["reviewed"].astype(bool)
    ref, o1, ov = ctrl["oof"][ic], r1["oof"][ir].astype(float), v["oof"].astype(float)
    has = ~np.isnan(o1).any(axis=0) & ~np.isnan(ov).any(axis=0)
    T = B.T_DEPLOY[area]
    a1 = [rt.auc(y[has], o[has], rev[has]) for o in o1]
    av = [rt.auc(y[has], o[has], rev[has]) for o in ov]
    d_alone = rt.paired_delta(a1, av)
    b1 = SE.block(y, rev, ref, blend_of(ref, o1), T, "blend r1")
    bv = SE.block(y, rev, ref, blend_of(ref, ov), T, "blend variant")
    st = None
    sp = R2 / f"stack_{variant}_{area}_{grouping}.json"
    if sp.exists():
        sj = json.loads(sp.read_text())
        blk = sj["blocks"]["stack_vs_pinned"]
        st = {"d_auc": blk["d_auc_reviewed"]["mean"], "min": blk["d_auc_reviewed"]["min"],
              "n_pos": blk["d_auc_reviewed"]["n_positive"], "n": blk["d_auc_reviewed"]["n"],
              "d_junk_pp": blk["d_junk_pp"], "far_at_matched_junk": blk["far_new_at_matched_junk"]}
    j = json.loads(vp.with_suffix(".json").read_text())
    F = j["folds"]
    be = np.array([np.mean(f["best_epoch"]) for f in F])
    er = np.array([np.mean(f["epochs_run"]) for f in F])
    cap = int(j["cfg"]["fixed_epochs"]) or int(j["cfg"]["epochs"])
    nval = np.array([f["n_val_agent"] + f["n_val_bs"] for f in F])
    return {
        "area": area, "grouping": grouping, "variant": variant, "seeds": seeds, "n_rows_both": int(has.sum()),
        "alone_r1": float(np.mean(a1)), "alone_v": float(np.mean(av)), "d_alone": d_alone,
        "blend_r1_d": b1["d_auc_reviewed"], "blend_v_d": bv["d_auc_reviewed"],
        "blend_r1_junk_pp": b1["d_junk_pp"], "blend_v_junk_pp": bv["d_junk_pp"],
        "blend_v_far_at_matched_junk": bv["far_new_at_matched_junk"], "blend_ref_far": bv["far_ref_at_T"],
        "stack": st, "median_best_epoch": float(np.median(be)), "cap_hits": int((er >= cap).sum()), "n_folds": len(F),
        "n_val_min": int(nval.min()), "n_val_max": int(nval.max()), "runtime_min": j["runtime_s"] / 60,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="valfix,fixed12,ens3,nbtrace,patch48")
    ap.add_argument("--cells", default="BLA:session,vCA1:animal")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    cells = [tuple(c.split(":")) for c in a.cells.split(",") if c.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    rows, out = [], {"written_at": dt.datetime.now().isoformat(timespec="seconds"), "cells": {}}
    lines = [f"# Round 2 screening ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
             "CNN alone = reviewed AUC on the rows both CNNs scored, paired per seed; blend = 50/50 with the control, "
             "delta vs the control (round-1 blend -> variant blend) and junk at the control's matched false-AR; "
             "stack = 35 + CNN + rank XGBoost vs the pinned control when computed.", "",
             "| variant | cell | seeds | CNN alone r1 -> v | d alone (min; +) | blend d AUC r1 -> v | blend junk pp r1 -> v | stack d AUC (junk pp) | best ep | cap hits | inner val rows | min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    carried = {}
    for variant in variants:
        ok_both, big_one, n_cells = True, False, 0
        for area, grouping in cells:
            c = cell(area, grouping, variant)
            if c is None:
                lines.append(f"| {variant} | {area} {grouping} | (not run) | | | | | | | | | |")
                ok_both = False
                continue
            n_cells += 1
            out["cells"][f"{variant}/{area}/{grouping}"] = c
            d = c["d_alone"]
            ok_both &= d["mean"] >= 0
            big_one |= d["mean"] >= 0.003
            st = c["stack"]
            st_s = f"{st['d_auc']:+.4f} ({st['d_junk_pp']:+.1f})" if st else "-"
            lines.append(f"| {variant} | {area} {grouping} | {len(c['seeds'])} | {c['alone_r1']:.4f} -> {c['alone_v']:.4f} "
                         f"| {d['mean']:+.4f} ({d['min']:+.4f}; {d['n_positive']}/{d['n']}) "
                         f"| {c['blend_r1_d']['mean']:+.4f} -> {c['blend_v_d']['mean']:+.4f} ({c['blend_v_d']['n_positive']}/{c['blend_v_d']['n']}) "
                         f"| {c['blend_r1_junk_pp']:+.1f} -> {c['blend_v_junk_pp']:+.1f} | {st_s} | {c['median_best_epoch']:.0f} "
                         f"| {c['cap_hits']}/{c['n_folds']} | {c['n_val_min']}-{c['n_val_max']} | {c['runtime_min']:.0f} |")
        carried[variant] = bool(n_cells == len(cells) and ok_both and big_one)
    lines += ["", "Selection rule: CNN-alone delta >= 0 on both cells and >= +0.003 on at least one.", ""]
    lines += [f"- `{v}`: {'CARRIED FORWARD' if ok else 'not carried'}" for v, ok in carried.items()]
    out["carried"] = carried
    R2.mkdir(exist_ok=True)
    (R2 / "screen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (R2 / "screen.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
