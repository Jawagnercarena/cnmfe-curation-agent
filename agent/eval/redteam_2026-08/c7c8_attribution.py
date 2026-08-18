"""C7: which feature group carries the win? Group ablations at 8 seeds on the
reviewed mixed pool, all vs my b13 OOF. C8: v2 vs v2b full variants with
paired per-seed deltas.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

RT = Path(__file__).parent
sys.path.insert(0, str(RT))
warnings.simplefilter("ignore")
from redteam_lib import load_pool, run_oof, summarize, fmt

t0 = time.time()
records = load_pool()
b = np.load(RT / "my_reviewed_b13_oof.npz", allow_pickle=True)
base_oof, base_y = b["oof_seeds"], b["y"]
rv = np.load(RT / "my_rankv2_35_oof.npz", allow_pickle=True)

runs = [
    ("v2_22 (13+v2+flag)", "v2_22", None),
    ("events-only (rank+5ev)", "rankv2_35", [0, 1, 2, 3, 4]),
    ("neighbors-only (rank+2nb)", "rankv2_35", [5, 6]),
    ("ring-only (rank+ring)", "rankv2_35", [7]),
    ("events+ring (no nb)", "rankv2_35", [0, 1, 2, 3, 4, 7]),
    ("rankv2b_35 (v2b full)", "rankv2b_35", None),
    ("v2b_22", "v2b_22", None),
]
res = {"rankv2_35 (full)": summarize(rv["oof_seeds"], rv["y"], base_oof, base_y)}
aucs_by = {"rankv2_35 (full)": np.array(res["rankv2_35 (full)"]["aucs"])}
print(fmt("rankv2_35 (full)", res["rankv2_35 (full)"]))
for tag, variant, cols in runs:
    oof, y, _, _ = run_oof(records, variant, v2_cols=cols)
    s = summarize(oof, y, base_oof, base_y)
    res[tag] = s
    aucs_by[tag] = np.array(s["aucs"])
    print(fmt(tag, s), f" dAUC vs b13 {s['auc_mean'] - 0.89207:+.4f}")

print("\n=== C8 paired per-seed: rankv2_35 - rankv2b_35 ===")
d = aucs_by["rankv2_35 (full)"] - aucs_by["rankv2b_35 (v2b full)"]
print(f"mean {d.mean():+.4f}  sd {d.std():.4f}  min {d.min():+.4f}  "
      f"max {d.max():+.4f}  v2 wins {int((d > 0).sum())}/8 seeds")
print(f"total {time.time() - t0:.0f}s")
