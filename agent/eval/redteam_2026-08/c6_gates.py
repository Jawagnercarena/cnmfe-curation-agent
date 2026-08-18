"""C6: gate robustness. The step2 LOAO/early-era gates were single fits at
XGB random_state=42. Re-run b13 / rankv2_35 / rankv2b_35 at 5 xgb seeds and
report per-cell spread. The question: does "improves or holds on all six
animals" survive seed noise, especially bla21 (n=2 sessions) and bla16?
"""
import datetime as dt
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RT = Path(__file__).parent
sys.path.insert(0, str(RT))
warnings.simplefilter("ignore")
from redteam_lib import PIN, load_pool, holdout_eval

ANIMAL_RE = re.compile(r"-(bla\d+)-")
EARLY_CUTOFF = dt.datetime(2026, 4, 15).timestamp()
XGB_SEEDS = [42, 7, 2024, 1, 31337]

t0 = time.time()
records = load_pool()
manifest = {m["rel"]: m for m in json.loads((PIN / "pool_manifest.json").read_text())}
ag_names = [r["name"] for r in records if not r["is_bootstrap"]]
animals = sorted({ANIMAL_RE.search(n).group(1) for n in ag_names})
early = {n for n in ag_names if manifest[n]["labels_mtime"] < EARLY_CUTOFF}
print(f"animals: {animals}; early-era sessions: {len(early)}")
for a in animals:
    ns = [n for n in ag_names if ANIMAL_RE.search(n).group(1) == a]
    print(f"  {a}: {len(ns)} sessions")

results = {}
for variant in ("b13", "rankv2_35", "rankv2b_35"):
    results[variant] = {}
    for cell in animals + ["early_era"]:
        test = early if cell == "early_era" else \
            {n for n in ag_names if ANIMAL_RE.search(n).group(1) == cell}
        aucs = []
        for xs in XGB_SEEDS:
            s, y = holdout_eval(records, variant, test, xgb_seed=xs)
            aucs.append(roc_auc_score(y, s))
        aucs = np.array(aucs)
        results[variant][cell] = aucs
        print(f"{variant:<12} {cell:<10} mean {aucs.mean():.4f}  sd {aucs.std():.4f}  "
              f"min {aucs.min():.4f}  max {aucs.max():.4f}")

print("\n=== paired per-seed deltas (rankv2_35 - b13) ===")
worst_cell = None
for cell in animals + ["early_era"]:
    d = results["rankv2_35"][cell] - results["b13"][cell]
    print(f"{cell:<10} mean {d.mean():+.4f}  min {d.min():+.4f}  "
          f"all>=0: {(d >= 0).all()}")
print("\n=== paired per-seed deltas (rankv2b_35 - b13) ===")
for cell in animals + ["early_era"]:
    d = results["rankv2b_35"][cell] - results["b13"][cell]
    print(f"{cell:<10} mean {d.mean():+.4f}  min {d.min():+.4f}  "
          f"all>=0: {(d >= 0).all()}")

np.savez(RT / "c6_gate_aucs.npz",
         **{f"{v}__{c}": results[v][c] for v in results for c in results[v]})
print(f"total {time.time() - t0:.0f}s")
