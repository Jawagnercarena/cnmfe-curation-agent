"""
a1_orientation_audit.py — prove the pixel-order bug and audit every consumer.

Sections
  [1] Synthetic red/green unit test: an asymmetric footprint, MATLAB-style
      F-order column vs numpy C-order flatten. Production formula must FAIL
      (sim << 1); the orientation-consistent formula must give sim == 1.
      Includes a rectangular (d1 != d2) case for the general fix formula.
  [2] Empirical proof on bla21-313um: mismatched matching reproduces the stored
      production/validation scores; consistent matching recovers 50/50; the
      matched "cells" under the mismatch sit at the transposed MIRROR position.
  [3] Latent-fallback scan: sessions where features.load_spatial would fall
      back to A.txt (missing spatial_footprints.mat) and hit the same bug.

READ-ONLY on D:. Exit code 0 iff all assertions pass.
Run:  C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe a1_orientation_audit.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bmlib

FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# [1] Synthetic unit test
# ---------------------------------------------------------------------------

def synthetic_test():
    print("\n[1] Synthetic red/green unit test")
    rng = np.random.default_rng(0)

    for d1, d2, label in ((64, 64, "square 64x64"), (48, 80, "rect 48x80")):
        # Asymmetric off-diagonal blob: an L-shape near (10, d2-15)
        img = np.zeros((d1, d2), dtype=np.float64)
        img[8:14, d2 - 18:d2 - 8] = rng.uniform(0.5, 1.0, (6, 10))
        img[14:24, d2 - 12:d2 - 8] = rng.uniform(0.5, 1.0, (10, 4))

        # MATLAB side: neuron.A column = F-order linearization
        a_col_F = img.flatten(order="F")            # (pixels,)
        A_final = a_col_F[:, None]                  # (pixels, 1) like retro_final.mat

        # Candidate side: spatial_footprints.mat stack, same single footprint
        fp3d = img[None, :, :]

        # --- production formula (bootstrap_preagent._load_candidates L245) ---
        A_review_prod = fp3d.reshape(1, d1 * d2).T          # C-order pixels
        sim_prod = float(bmlib.cosine_matrix(A_review_prod.T, A_final.T)[0, 0])

        # --- orientation-consistent formula (the fix) ---
        A_review_fix = bmlib.stack_to_F(fp3d).T             # F-order pixels
        sim_fix = float(bmlib.cosine_matrix(A_review_fix.T, A_final.T)[0, 0])

        # --- equivalent fix on the other side: convert A_final to a stack ---
        stack_back = bmlib.Fcols_to_stack(A_final, d1, d2)
        roundtrip = float(np.abs(stack_back[0] - img).max())

        check(f"{label}: production formula is broken", sim_prod < 0.5,
              f"sim={sim_prod:.4f} (identical footprint!)")
        check(f"{label}: consistent formula is exact", abs(sim_fix - 1.0) < 1e-6,
              f"sim={sim_fix:.6f}")
        check(f"{label}: Fcols_to_stack round-trips", roundtrip == 0.0,
              f"max|diff|={roundtrip:g}")

        # Regression test against the REAL production fix, if deployed:
        # bootstrap_preagent._reorder_Fcols_to_C must make the production-style
        # comparison exact.
        try:
            agent_dir = Path(__file__).resolve().parents[2]
            if str(agent_dir) not in sys.path:
                sys.path.insert(0, str(agent_dir))
            from bootstrap_preagent import _reorder_Fcols_to_C
        except ImportError:
            print(f"  [SKIP] {label}: bootstrap_preagent has no "
                  "_reorder_Fcols_to_C yet (fix not deployed)")
        else:
            A_fixed = _reorder_Fcols_to_C(A_final, d1, d2)
            sim_deploy = float(bmlib.cosine_matrix(A_review_prod.T, A_fixed.T)[0, 0])
            check(f"{label}: deployed _reorder_Fcols_to_C fixes the match",
                  abs(sim_deploy - 1.0) < 1e-6, f"sim={sim_deploy:.6f}")


# ---------------------------------------------------------------------------
# [2] Empirical proof on bla21-313um
# ---------------------------------------------------------------------------

def empirical_test():
    print("\n[2] Empirical proof — bla21-313um sandbox")
    base = bmlib.SANDBOX_SESSIONS[0]
    cur = bmlib.load_curated_stack(base)          # (50, 512, 512)
    cand = bmlib.load_sandbox_candidates(base)    # (397, 512, 512)

    # Production-style: candidates C-order vs curated F-order (= neuron.A)
    corr_bug = bmlib.cosine_matrix(bmlib.stack_to_C(cand), bmlib.stack_to_F(cur))
    ri_b, ci_b, sims_b = bmlib.hungarian_pairs(corr_bug)
    n_bug = int((sims_b > 0.45).sum())

    # Consistent: both sides F-order
    corr_fix = bmlib.cosine_matrix(bmlib.stack_to_F(cand), bmlib.stack_to_F(cur))
    ri_f, ci_f, sims_f = bmlib.hungarian_pairs(corr_fix)
    n_fix = int((sims_f > 0.45).sum())

    print(f"  mismatched: {n_bug}/{cur.shape[0]} matched at 0.45, "
          f"top5 {np.round(sims_b[:5], 3).tolist()}")
    print(f"  consistent: {n_fix}/{cur.shape[0]} matched at 0.45, "
          f"min sim {sims_f.min():.3f}")

    # Stored production-era scores (validation_match_stats_temporal.json)
    with open(base / "validation_match_stats_temporal.json") as f:
        stored = np.sort(np.array(
            json.load(f)["strategies"]["spatial"]["pair_scores"]))[::-1]

    check("mismatched replica reproduces stored pipeline scores",
          bool(np.allclose(np.sort(sims_b)[::-1], stored, atol=2e-3)),
          f"max|diff|={np.abs(np.sort(sims_b)[::-1] - stored).max():.4f}")
    check("mismatched matching loses neurons", n_bug == 37, f"{n_bug}/50")
    check("consistent matching recovers everything", n_fix == 50, f"{n_fix}/50")
    check("consistent sims are near-perfect", float(np.median(sims_f)) > 0.93,
          f"median={np.median(sims_f):.3f}")

    # Mirror-position geometry of the mismatched "matches"
    cc = bmlib.centroids(cand)
    kc = bmlib.centroids(cur)
    m = sims_b > 0.45
    d_true = np.hypot(*(cc[ri_b[m]] - kc[ci_b[m]]).T)
    d_mirror = np.hypot(*(cc[ri_b[m]] - kc[ci_b[m]][:, ::-1]).T)
    check("mismatched matches sit at the MIRROR position",
          float(np.median(d_mirror)) < 15 < float(np.median(d_true)),
          f"median mirror {np.median(d_mirror):.1f}px vs true {np.median(d_true):.1f}px")

    # Consistent matches sit at the TRUE position
    m2 = sims_f > 0.45
    d_true_f = np.hypot(*(cc[ri_f[m2]] - kc[ci_f[m2]]).T)
    check("consistent matches sit at the TRUE position",
          float(np.median(d_true_f)) < 3,
          f"median true-dist {np.median(d_true_f):.1f}px")


# ---------------------------------------------------------------------------
# [3] Latent A.txt fallback scan (features.load_spatial)
# ---------------------------------------------------------------------------

def fallback_scan():
    print("\n[3] features.load_spatial A.txt-fallback exposure scan")
    hits = []
    for area in bmlib.AREAS + ("DG_AL",):
        root = bmlib.DATA_ROOT / area
        if not root.exists():
            continue
        for a_txt in root.rglob("A.txt"):
            sd = a_txt.parent
            if "_bootstrap" in sd.name or any(p.startswith(".") for p in sd.parts):
                continue
            if not (sd / "spatial_footprints.mat").exists():
                hits.append(sd)
    print(f"  sessions with A.txt but NO spatial_footprints.mat: {len(hits)}")
    for sd in hits[:20]:
        print(f"    {sd}")
    print("  (each of these would get TRANSPOSED footprints from the fallback "
          "at features.py:32-38 — latent bug, fix alongside Phase B)")


if __name__ == "__main__":
    synthetic_test()
    empirical_test()
    fallback_scan()
    print()
    if FAILURES:
        print(f"AUDIT FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("AUDIT OK — all checks passed.")
