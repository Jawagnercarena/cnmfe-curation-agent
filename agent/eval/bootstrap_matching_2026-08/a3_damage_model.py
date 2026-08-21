"""
a3_damage_model.py — per-session label-damage estimate for all 202 bootstrap
sessions, from curated footprint geometry (read-only).

Under the pixel-order bug, a stored label can only be accidentally correct when
the curated neuron sits near the image diagonal (its mirror position ~= its
true position). For each session we compute, from spatial_footprints.mat:

  frac_diag   fraction of curated neurons whose centroid mirror-offset
              sqrt(2)*|row-col| is under DIAG_TOL pixels
  damage      estimated wrong training rows = mislabeled positives
              (n_matched outside the diagonal band) + lost true positives
              (n_curated, all of whose true candidates are labeled 0)

Sanity check: old "recovery" should correlate with frac_diag across sessions
(diagonal luck was the only way to match correctly).

Outputs a3_damage.csv + a3_damage.json (this dir). Prints the priority order
for Phase B/C re-runs.

Run:  C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe a3_damage_model.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bmlib

DIAG_TOL = 10.0   # px: mirror within ~2 gSig of true position


def session_row(json_path: Path) -> dict | None:
    sd = json_path.parent
    with open(json_path) as f:
        stats = json.load(f)
    sf = sd / "spatial_footprints.mat"
    if not sf.exists():
        return None
    stack = bmlib.load_stack(sf)
    cen = bmlib.centroids(stack)
    mirror_off = np.sqrt(2.0) * np.abs(cen[:, 0] - cen[:, 1])
    n_cur = stack.shape[0]
    frac_diag = float((mirror_off < DIAG_TOL).mean()) if n_cur else 0.0
    n_matched = int(stats["n_matched"])
    n_cand = int(stats["n_candidates"])
    # wrong positives: matched pairs outside the diagonal band (mirror cells)
    pair_cur = np.asarray(stats["curated_indices"][:n_matched], dtype=int)
    ok = pair_cur[pair_cur < n_cur]
    wrong_pos = int((mirror_off[ok] >= DIAG_TOL).sum()) if len(ok) else 0
    lost_pos = n_cur - (n_matched - wrong_pos)
    return {
        "area": sd.parts[2] if len(sd.parts) > 2 else "?",
        "task": sd.parent.name,
        "session": sd.name,
        "n_curated": n_cur,
        "n_candidates": n_cand,
        "n_matched_old": n_matched,
        "recovery_old": round(n_matched / n_cur, 3) if n_cur else 0.0,
        "frac_diag": round(frac_diag, 3),
        "wrong_positives_est": wrong_pos,
        "lost_positives_est": lost_pos,
        "damage_est": wrong_pos + lost_pos,
    }


if __name__ == "__main__":
    rows = []
    for area in bmlib.AREAS:
        for jp in sorted((bmlib.DATA_ROOT / area).rglob("bootstrap_match_stats.json")):
            if any(p.startswith(".") for p in jp.parts):
                continue
            r = session_row(jp)
            if r:
                r["area"] = area
                rows.append(r)
    print(f"sessions analyzed: {len(rows)}")

    rec = np.array([r["recovery_old"] for r in rows])
    fd = np.array([r["frac_diag"] for r in rows])
    from scipy.stats import spearmanr
    rho, p = spearmanr(fd, rec)
    print(f"sanity: spearman(frac_diag, old recovery) = {rho:.3f} (p={p:.2g})")

    tot_wrong = sum(r["wrong_positives_est"] for r in rows)
    tot_lost = sum(r["lost_positives_est"] for r in rows)
    tot_cur = sum(r["n_curated"] for r in rows)
    tot_match = sum(r["n_matched_old"] for r in rows)
    print(f"corpus: curated {tot_cur} | old matched {tot_match} | "
          f"est wrong positives {tot_wrong} | est lost positives {tot_lost}")

    rows.sort(key=lambda r: -r["damage_est"])
    print("\ntop 20 by estimated damage:")
    for r in rows[:20]:
        print(f"  {r['area']:5s} {r['task']:8s} {r['session'][:44]:44s} "
              f"cur={r['n_curated']:3d} rec_old={r['recovery_old']:.2f} "
              f"diag={r['frac_diag']:.2f} damage={r['damage_est']}")

    out = Path(__file__).parent
    with open(out / "a3_damage.json", "w") as f:
        json.dump(rows, f, indent=1)
    with open(out / "a3_damage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote a3_damage.json / a3_damage.csv ({len(rows)} rows)")
