"""C9: real operating point of the new representation (no matched-op
abstraction). Threshold sweeps in diagnose_model-#3 style from the C10 OOF
matrices:
  - reviewed-pool sweep (exact: rows have real v2, flag=1)
  - full-agent-pool sweep (production analog; autorej rows approximated with
    v2=0 — production would supply real v2, so junk-caught there is a floor
    estimate for easy junk under arm a / as-trained representation for arm b)
Reports the false-AR<=target crossings and the [0.05,0.12) band.
"""
import sys
from pathlib import Path

import numpy as np

RT = Path(__file__).parent
sys.path.insert(0, str(RT))

NPZ = sys.argv[1] if len(sys.argv) > 1 else "c10_oof.npz"
d = np.load(RT / NPZ, allow_pickle=True)
print(f"sweeping {NPZ}")
y, rev = d["y"], d["rev"]
THRESHOLDS = np.round(np.arange(0.01, 0.31, 0.01), 2)


def sweep(oof_seeds, y, tag):
    pos, neg = y == 1, y == 0
    print(f"\n--- {tag}  (n={len(y)}, real={int(pos.sum())}, junk={int(neg.sum())}) ---")
    print("  thr   false-AR% (sd)     junk-caught% (sd)")
    rows = []
    for t in THRESHOLDS:
        fars = [(oof_seeds[s][pos] < t).mean() * 100 for s in range(len(oof_seeds))]
        gcs = [(oof_seeds[s][neg] < t).mean() * 100 for s in range(len(oof_seeds))]
        rows.append((t, np.mean(fars), np.std(fars), np.mean(gcs), np.std(gcs)))
    for t, fm, fs, gm, gs in rows:
        mark = ""
        if t == 0.12:
            mark = "   <- current deployed T"
        print(f"  {t:.2f}  {fm:6.2f} ({fs:.2f})      {gm:5.1f} ({gs:.1f}){mark}")
    for target in (0.5, 0.85, 1.0):
        best = None
        for t, fm, fs, gm, gs in rows:
            if fm <= target:
                best = (t, fm, gm)
        if best:
            print(f"  max thr with false-AR <= {target}%: t={best[0]:.2f} "
                  f"(false-AR {best[1]:.2f}%, junk {best[2]:.1f}%)")
    band = [int(((oof_seeds[s][pos] >= 0.05) & (oof_seeds[s][pos] < 0.12)).sum())
            for s in range(len(oof_seeds))]
    print(f"  reals in [0.05,0.12): {np.mean(band):.1f} (seed mean)")
    return rows


for arm in ("oof_a", "oof_b"):
    oof = d[arm]
    sweep(oof[:, rev], y[rev], f"{arm} reviewed pool")
    sweep(oof, y, f"{arm} FULL agent pool (autorej approximated v2=0)")

# reference: current deployed representation, full pool (from C2's run)
# reproduce numbers for context
b = np.load(RT / "my_reviewed_b13_oof.npz", allow_pickle=True)
sweep(b["oof_seeds"], b["y"], "b13 reviewed pool (reference)")
