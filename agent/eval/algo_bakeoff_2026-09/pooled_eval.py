"""
pooled_eval.py -- pooled BLA+vCA1 CNN (results/r2/pooled<W>_<variant>_<cell>.npz)
vs the per-area CNN (results/r2/<variant>_<cell>.npz) at the same seeds, and
both blended 50/50 with the deployed control.  README "Pooled BLA + vCA1 CNN".

Usage: python pooled_eval.py --variant fixed12+nbtrace --other-weight 1.0
Writes results/r2/pooled<W>_<variant>.md / .json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fixed12+nbtrace")
    ap.add_argument("--other-weight", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    tag = f"pooled{a.other_weight:g}_{a.variant}"
    out = {"tag": tag, "written_at": dt.datetime.now().isoformat(timespec="seconds"), "cells": {}}
    lines = [f"# Pooled BLA+vCA1 CNN vs per-area CNN — `{a.variant}`, other area x{a.other_weight:g} ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
             "CNN alone: reviewed AUC on rows both scored, pooled minus per-area, paired per seed. Blend: 50/50 with the "
             "deployed control, delta vs control (per-area -> pooled) and junk at matched false-AR / FAR at matched junk.", "",
             "| cell | seeds | CNN alone per-area -> pooled | d alone (min; +) | blend d AUC per-area -> pooled | junk pp per-area -> pooled | FAR @matched junk per-area -> pooled | per animal (per-area -> pooled, CNN alone) |",
             "|---|---|---|---|---|---|---|---|"]
    for area, grouping in CELLS:
        pp = R2 / f"{tag}_{area}_{grouping}.npz"
        vp = R2 / f"{a.variant}_{area}_{grouping}.npz"
        if not pp.exists() or not vp.exists():
            lines.append(f"| {area} {grouping} | (not run) | | | | | | |")
            continue
        ctrl = np.load(RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
        p, v = np.load(pp, allow_pickle=True), np.load(vp, allow_pickle=True)
        assert np.array_equal(p["y"], v["y"]) and np.array_equal(p["y"], ctrl["y"])
        y, rev = p["y"].astype(int), p["reviewed"].astype(bool)
        seeds = [int(s) for s in p["seeds"]]
        iv = [[int(t) for t in v["seeds"]].index(s) for s in seeds]
        ic = [[int(t) for t in ctrl["seeds"]].index(s) for s in seeds]
        ref, ov, op = ctrl["oof"][ic], v["oof"][iv].astype(float), p["oof"].astype(float)
        has = ~np.isnan(ov).any(axis=0) & ~np.isnan(op).any(axis=0)
        T = B.T_DEPLOY[area]
        av = [rt.auc(y[has], o[has], rev[has]) for o in ov]
        apl = [rt.auc(y[has], o[has], rev[has]) for o in op]
        d = rt.paired_delta(av, apl)
        bv = SE.block(y, rev, ref, blend_of(ref, ov), T, "blend per-area")
        bp = SE.block(y, rev, ref, blend_of(ref, op), T, "blend pooled")
        animals = np.asarray(ctrl["animals"]).astype(str)
        pa = []
        for an in sorted(set(animals)):
            m = has & (animals == an)
            if y[m].sum() >= 5 and (y[m] == 0).sum() >= 5:
                pa.append(f"{an} {np.mean([rt.auc(y[m], o[m], rev[m]) for o in ov]):.3f}->{np.mean([rt.auc(y[m], o[m], rev[m]) for o in op]):.3f}")
        out["cells"][f"{area}/{grouping}"] = {"seeds": seeds, "alone_per_area": float(np.mean(av)), "alone_pooled": float(np.mean(apl)),
                                              "d_alone": d, "blend_per_area": bv, "blend_pooled": bp}
        lines.append(f"| {area} {grouping} | {len(seeds)} | {np.mean(av):.4f} -> {np.mean(apl):.4f} | {d['mean']:+.4f} ({d['min']:+.4f}; {d['n_positive']}/{d['n']}) "
                     f"| {bv['d_auc_reviewed']['mean']:+.4f} -> {bp['d_auc_reviewed']['mean']:+.4f} ({bp['d_auc_reviewed']['n_positive']}/{bp['d_auc_reviewed']['n']}) "
                     f"| {bv['d_junk_pp']:+.1f} -> {bp['d_junk_pp']:+.1f} | {bv['far_new_at_matched_junk']:.2f} -> {bp['far_new_at_matched_junk']:.2f} | {'; '.join(pa)} |")
    (R2 / f"{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (R2 / f"{tag}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
