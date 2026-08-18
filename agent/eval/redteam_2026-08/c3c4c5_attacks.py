"""C3/C4/C5 attacks, all on the reviewed-pool mixed arm vs my own b13 OOF.
C3: drop nb_corr_max (col 5); drop both neighbor features (cols 5,6).
C4: delete the v2_present flag column.
C5: remove the 16 autopsy false-AR cells from the EVAL pool (they stay in
    training folds — the attack is on eval-set circularity) and re-measure
    the delta for both b13 and rankv2_35.
"""
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

RT = Path(__file__).parent
sys.path.insert(0, str(RT))
warnings.simplefilter("ignore")
from redteam_lib import PIN, load_pool, run_oof, summarize, fmt

t0 = time.time()
records = load_pool()

b = np.load(RT / "my_reviewed_b13_oof.npz", allow_pickle=True)
base_oof, base_y = b["oof_seeds"], b["y"]
rv = np.load(RT / "my_rankv2_35_oof.npz", allow_pickle=True)
full_s = summarize(rv["oof_seeds"], rv["y"], base_oof, base_y)
print(fmt("rankv2_35 (all 8+flag)", full_s))

# --- C3: neighbor-feature ablations ---
for tag, cols in (("no nb_corr_max", [0, 1, 2, 3, 4, 6, 7]),
                  ("no neighbors at all", [0, 1, 2, 3, 4, 7])):
    oof, y, _, _ = run_oof(records, "rankv2_35", v2_cols=cols)
    s = summarize(oof, y, base_oof, base_y)
    print(fmt(f"C3 {tag}", s),
          f" dAUC vs full {s['auc_mean'] - full_s['auc_mean']:+.4f}")

# --- C4: flag deleted ---
oof, y, _, _ = run_oof(records, "rankv2_35_noflag")
s4 = summarize(oof, y, base_oof, base_y)
print(fmt("C4 no v2_present flag", s4),
      f" dAUC vs full {s4['auc_mean'] - full_s['auc_mean']:+.4f}")

# --- C5: drop the 16 autopsy cells from the eval pool ---
drop = set()
with open(PIN / "autopsy_false_ar.csv", newline="") as f:
    for row in csv.DictReader(f):
        if row["kind"] == "false_AR":
            drop.add((row["session"], int(row["cand_idx0"])))
print(f"\nC5: dropping {len(drop)} autopsy cells from eval pool")
key = [(str(rv["session"][i]), int(rv["idx_in_session"][i]))
       for i in range(len(rv["y"]))]
keep = np.array([k not in drop for k in key])
print(f"  matched {int((~keep).sum())} of {len(drop)} in eval pool")
b_keep = np.array([(str(b["session"][i]), int(b["idx_in_session"][i])) not in drop
                   for i in range(len(base_y))])
s_b13 = summarize(base_oof[:, b_keep], base_y[b_keep],
                  base_oof[:, b_keep], base_y[b_keep])
s_rv = summarize(rv["oof_seeds"][:, keep], rv["y"][keep],
                 base_oof[:, b_keep], base_y[b_keep])
print(fmt("C5 b13 minus 16", s_b13))
print(fmt("C5 rankv2_35 minus 16", s_rv),
      f" delta {s_rv['auc_mean'] - s_b13['auc_mean']:+.4f} (with-16 delta "
      f"{full_s['auc_mean'] - summarize(base_oof, base_y, base_oof, base_y)['auc_mean']:+.4f})")
print(f"total {time.time() - t0:.0f}s")
