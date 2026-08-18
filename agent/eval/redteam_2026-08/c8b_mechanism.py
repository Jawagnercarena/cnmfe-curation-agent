"""C8 supplement: (1) single-feature AUCs of the 8 v2/v2b features on the
reviewed pool (leak smell test — nothing should be implausibly separable);
(2) the marquee mechanism check: bla21 regression cells' stereotypy
percentiles under v2 vs v2b (step2 doc claims 96th/85th pct under v2b,
drifty bla16 N64 low)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RT = Path(__file__).parent
sys.path.insert(0, str(RT))
from redteam_lib import PIN, V2_NAMES, load_pool

records = load_pool(verbose=False)
ag = [r for r in records if not r["is_bootstrap"] and r["y"][r["ridx"]].sum() >= 5]
y = np.concatenate([r["y"][r["ridx"]] for r in ag])
for tag, key in (("v2", "Xv2"), ("v2b", "Xv2b")):
    X = np.vstack([r[key] for r in ag])
    print(f"\nsingle-feature AUC ({tag}), reviewed pool n={len(y)}:")
    for j, nm in enumerate(V2_NAMES):
        auc = roc_auc_score(y, X[:, j])
        print(f"  {nm:<18} {auc:.3f}")

print("\nmechanism check: ev_template_corr percentile within session")
targets = [("2tones/AVG5x-TSeries-093025-bla21-313um-38z-000", 21, "bla21 N22 (regression)"),
           ("2tones/AVG5x-TSeries-093025-bla21-313um-38z-000", 24, "bla21 N25 (regression)"),
           ("2tones/AVG5x-TSeries-100125-bla16-345um-36z-000", 63, "bla16 N64 (drifty marginal)")]
for rel, fullidx, tag in targets:
    rec = next(r for r in records if r["name"] == rel)
    pos = np.where(rec["ridx"] == fullidx)[0][0]
    for vkey, vtag in (("Xv2", "v2"), ("Xv2b", "v2b")):
        col = rec[vkey][:, 2]  # ev_template_corr
        pct = (col < col[pos]).mean() * 100
        n_ev_rate = rec[vkey][pos, 0]
        print(f"  {tag:<28} {vtag:<4} stereotypy pct {pct:5.1f}  "
              f"(value {col[pos]:.3f}, ev_rate {n_ev_rate:.2f}/1k frames)")
