"""C2: independent recomputation of the headline numbers.
1. Full-pool b13 -> must reproduce the 0.9099 re-pin AND correlate per-row
   with the pinned baseline_oof.npz (validates my loader/CV end to end).
2. Reviewed-pool b13 vs rankv2_35 (mixed arm) -> claimed 0.8921 / 0.9137.
Saves OOF matrices for downstream attacks.
"""
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

# --- 1. full-pool b13: reproduce the 0.9099 baseline re-pin ---
oof_full, y_full, names_full, fidx_full = run_oof(
    records, "b13", reviewed_only=False)
s = summarize(oof_full, y_full, oof_full)
print(fmt("FULL-POOL b13", s), f"  n={len(y_full)} real={int(y_full.sum())}")
print(f"  claimed (baseline_repin): AUC 0.9099 +/- 0.0017, n=12677, real=2394")

# per-row cross-check vs pinned baseline_oof.npz
pin = np.load(PIN / "baseline_oof.npz", allow_pickle=True)
key_mine = {(str(names_full[i]), int(fidx_full[i])): i for i in range(len(y_full))}
key_pin = {(str(pin["session"][i]), int(pin["idx_in_session"][i])): i
           for i in range(len(pin["y"]))}
common = set(key_mine) & set(key_pin)
print(f"  row alignment: mine={len(key_mine)} pinned={len(key_pin)} "
      f"common={len(common)}")
mi = np.array([key_mine[k] for k in sorted(common)])
pi = np.array([key_pin[k] for k in sorted(common)])
assert (y_full[mi] == pin["y"][pi]).all(), "label mismatch on common rows!"
mine_mean = oof_full.mean(axis=0)[mi]
pin_mean = pin["oof_seeds"].mean(axis=0)[pi]
r = np.corrcoef(mine_mean, pin_mean)[0, 1]
mad = np.mean(np.abs(mine_mean - pin_mean))
print(f"  seed-mean OOF: pearson r={r:.5f}, mean|diff|={mad:.5f}, "
      f"max|diff|={np.max(np.abs(mine_mean - pin_mean)):.5f}")

# --- 2. reviewed-pool headline, mixed arm ---
res = {}
for variant in ("b13", "rank26", "rankv2_35"):
    oof, y, names, fidx = run_oof(records, variant, reviewed_only=True)
    if variant == "b13":
        base_oof, base_y = oof, y
        np.savez(RT / "my_reviewed_b13_oof.npz", oof_seeds=oof, y=y,
                 session=names, idx_in_session=fidx)
    if variant == "rankv2_35":
        np.savez(RT / "my_rankv2_35_oof.npz", oof_seeds=oof, y=y,
                 session=names, idx_in_session=fidx)
    s = summarize(oof, y, base_oof, base_y)
    res[variant] = s
    print(fmt(f"REVIEWED {variant}", s), f"  n={len(y)} real={int(y.sum())}")

d = res["rankv2_35"]["auc_mean"] - res["b13"]["auc_mean"]
pair = np.array(res["rankv2_35"]["aucs"]) - np.array(res["b13"]["aucs"])
print(f"\nDELTA rankv2_35 - b13: {d:+.4f}  (claimed +0.0216)")
print(f"  per-seed paired deltas: {np.array2string(pair, precision=4)}")
print(f"  min paired delta {pair.min():+.4f}; all positive: {(pair > 0).all()}")
print(f"claimed: b13 0.8921+/-0.0027 far@mj 0.86%; "
      f"rankv2_35 0.9137+/-0.0022 far@mj 0.11%")
print(f"total {time.time() - t0:.0f}s")
