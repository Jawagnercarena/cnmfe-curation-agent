"""clean_seeds_check.py -- the confirmation gate restricted to the 5 seeds never used in
screening (13, 100, 2024, 31337, 9): blend 50/50 vs control, reviewed-AUC delta and junk at
matched false-AR, per cell.  Usage: python clean_seeds_check.py --variant fixed12+nbtrace"""
import os
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B
import stack_eval as SE
rt = B.rt
R = HERE / os.environ.get("BAKEOFF_RESULTS", "results"); R2 = R / "r2"
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
clean = [13, 100, 2024, 31337, 9]
print(f"| cell | seeds | CNN alone d (min; +) | blend d reviewed AUC (min; +) | junk @matched FAR (pp) | FAR @matched junk |")
print("|---|---|---|---|---|---|")
for area, grouping in [("BLA", "session"), ("BLA", "animal"), ("vCA1", "session"), ("vCA1", "animal")]:
    ctrl = np.load(R / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
    v = np.load(R2 / f"{a.variant}_{area}_{grouping}.npz", allow_pickle=True)
    y, rev = v["y"].astype(int), v["reviewed"].astype(bool)
    vs = [int(s) for s in v["seeds"]]; cs = [int(s) for s in ctrl["seeds"]]
    iv = [vs.index(s) for s in clean]; ic = [cs.index(s) for s in clean]
    ref, ov = ctrl["oof"][ic], v["oof"][iv].astype(float)
    has = ~np.isnan(ov).any(axis=0)
    T = B.T_DEPLOY[area]
    alone = SE.block(y[has], rev[has], ref[:, has], ov[:, has], T, "alone")
    blend = SE.block(y, rev, ref, np.where(np.isnan(ov), ref, 0.5 * (ref + np.nan_to_num(ov))), T, "blend")
    da, db = alone["d_auc_reviewed"], blend["d_auc_reviewed"]
    print(f"| {area} {grouping} | {clean} | {da['mean']:+.4f} ({da['min']:+.4f}; {da['n_positive']}/{da['n']}) "
          f"| {db['mean']:+.4f} ({db['min']:+.4f}; {db['n_positive']}/{db['n']}) | {blend['d_junk_pp']:+.1f} "
          f"| {blend['far_new_at_matched_junk']:.2f} vs {blend['far_ref_at_T']:.2f} |")
