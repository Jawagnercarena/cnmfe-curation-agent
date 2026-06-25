"""
validate_threshold.py
Runs the bootstrap pipeline on human-labeled agent sessions to get ground-truth
similarity scores for verified real neurons.  Compares three matching strategies:
  spatial   — cosine similarity of spatial footprints (current approach)
  temporal  — Pearson correlation of calcium traces
  combined  — geometric mean of spatial and temporal scores

Results are saved to validation_match_stats_temporal.json (does NOT touch
validation_match_stats.json or labels.mat).

Usage:
    python validate_threshold.py                 # run all 5 sessions
    python validate_threshold.py --keep-bootstrap  # keep _bootstrap_validate/ for re-runs
    python validate_threshold.py --rematch-only    # skip MATLAB; redo matching only
    python validate_threshold.py --dry-run
"""

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment

AGENT_DIR = Path(__file__).parent
from config import DATA_ROOT

sys.path.insert(0, str(AGENT_DIR))
import params as params_module
import run_cnmfe

VALIDATION_SESSIONS = [
    ("Valence",   "AVG5x-TSeries-121225-bla12-652um-23z-000"),
    ("4odorDO",   "AVG5x-TSeries-02092026-bla12-681um-22z-000"),
    ("2tones",    "AVG5x-TSeries-093025-bla21-313um-38z-000"),
    ("2tones",    "AVG5x-TSeries-100125-bla12-639um-23z-000"),
    ("2tones",    "AVG5x-TSeries-100125-bla8-735um-23z-000"),
]

THRESHOLDS_TO_REPORT = [0.35, 0.40, 0.45, 0.50]
STRATEGIES = ["spatial", "temporal", "combined"]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# MATLAB / IO helpers
# ---------------------------------------------------------------------------

def _extract_final_footprints(session_dir: Path):
    """
    MATLAB: extract A_final + S_final (deconvolved spikes) + gSig/gSiz from neuron.mat.
    Returns (A_final, S_final, gSig, gSiz) or None.
    S_final shape: (N_kept, T_orig)
    """
    from local_config import REPO_ROOT
    repo = str(REPO_ROOT).replace("\\", "/")
    sd   = str(session_dir).replace("\\", "/")
    script = (
        f"cd('{repo}');"
        f" run('cnmfe_setup.m');"
        f" load('{sd}/neuron.mat');"
        f" A_final = full(neuron.A);"
        f" S_final = neuron.S;"
        f" gSig_orig = neuron.options.gSig;"
        f" gSiz_orig = neuron.options.gSiz;"
        # A_final alone can be >100MB — save spatial separately as v7,
        # and write S_final as text to avoid the v7.3/HDF5 size limit.
        f" save('{sd}/retro_final.mat', 'A_final', 'gSig_orig', 'gSiz_orig', '-v7');"
        f" writematrix(S_final, '{sd}/S_final.txt', 'Delimiter', ' ');"
        f" fprintf('Extracted %d final neurons (gSig=%d gSiz=%d T=%d)\\n',"
        f"   size(A_final,2), gSig_orig, gSiz_orig, size(S_final,2));"
    )
    ok = run_cnmfe._run_matlab(script, log, timeout_hours=0.5)
    if not ok:
        return None
    try:
        data      = sio.loadmat(str(session_dir / "retro_final.mat"))
        A_final   = data["A_final"]
        gSig_orig = int(np.asarray(data["gSig_orig"]).flat[0])
        gSiz_orig = int(np.asarray(data["gSiz_orig"]).flat[0])
        S_final   = np.loadtxt(str(session_dir / "S_final.txt"))  # (N_kept, T_orig)
        if S_final.ndim == 1:
            S_final = S_final[np.newaxis, :]
        log(f"  [VALIDATE] S_final shape: {S_final.shape}")
        return A_final, S_final, gSig_orig, gSiz_orig
    except Exception as e:
        log(f"  [VALIDATE] Could not load retro_final: {e}")
        return None
    finally:
        (session_dir / "retro_final.mat").unlink(missing_ok=True)
        (session_dir / "S_final.txt").unlink(missing_ok=True)


def _run_headless_into_bootstrap(tif_path: Path, bootstrap_dir: Path, p: dict) -> bool:
    from local_config import REPO_ROOT
    repo = str(REPO_ROOT).replace("\\", "/")
    nam  = str(tif_path).replace("\\", "/")
    bdir = str(bootstrap_dir).replace("\\", "/") + "/"
    script = (
        f"cd('{repo}');"
        f" run('cnmfe_setup.m');"
        f" nam = '{nam}';"
        f" gSig = {p['gSig']}; gSiz = {p['gSiz']};"
        f" min_corr = {p['min_corr']}; min_pnr = {p['min_pnr']}; bd = {p['bd']};"
        f" dir_nm_override = '{bdir}';"
        f" run('CNMFe_Biane_headless.m');"
    )
    return run_cnmfe._run_matlab(script, log, timeout_hours=4.0)


def _load_candidates(bootstrap_dir: Path):
    """Load (A_review, S_review, H, W) from bootstrap output dir."""
    try:
        sf_file = bootstrap_dir / "spatial_footprints.mat"
        if not sf_file.exists():
            log("  [VALIDATE] spatial_footprints.mat not found.")
            return None
        data = sio.loadmat(str(sf_file))
        fp3d = data["spatial_footprints"]   # (N, H, W)
        N, H, W = fp3d.shape
        A_review = fp3d.reshape(N, H * W).T  # (pixels, N)

        s_file = bootstrap_dir / "S.txt"
        if not s_file.exists():
            log("  [VALIDATE] S.txt not found.")
            return None
        S_review = np.loadtxt(str(s_file))
        if S_review.ndim == 1:
            S_review = S_review[np.newaxis, :]
        log(f"  [VALIDATE] S_review (bootstrap spikes) shape: {S_review.shape}")
        return A_review, S_review, H, W
    except Exception as e:
        log(f"  [VALIDATE] Error loading candidates: {e}")
        log(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------

def _spatial_matrix(A_review, A_final):
    """Cosine similarity: (N_review, N_kept)"""
    nr = np.sqrt((A_review ** 2).sum(axis=0)) + 1e-12
    nf = np.sqrt((A_final  ** 2).sum(axis=0)) + 1e-12
    return (A_review / nr).T @ (A_final / nf)


def _align_traces(C_review, C_final):
    """
    Align frame counts between bootstrap (possibly 2x downsampled) and
    original traces.  Downsamples the longer array to match the shorter.
    Returns (C_review_aligned, C_final_aligned).
    """
    T_review = C_review.shape[1]
    T_final  = C_final.shape[1]

    if T_review == T_final:
        return C_review, C_final

    ratio = T_final / T_review
    if 1.8 < ratio < 2.2:
        # Bootstrap is ~2x downsampled — downsample original to match
        log(f"  [VALIDATE] Frame mismatch: original T={T_final}, bootstrap T={T_review}. "
            f"Downsampling original by 2x.")
        C_final_ds = C_final[:, ::2]
        # Trim to exact same length in case of off-by-one
        T = min(C_review.shape[1], C_final_ds.shape[1])
        return C_review[:, :T], C_final_ds[:, :T]
    else:
        log(f"  [VALIDATE] WARNING: unexpected frame ratio {ratio:.2f}. "
            f"Truncating to shorter length.")
        T = min(T_review, T_final)
        return C_review[:, :T], C_final[:, :T]


def _temporal_matrix(C_review, C_final):
    """
    Pearson correlation matrix: (N_review, N_kept).
    Both inputs already frame-aligned.
    """
    def _normalise(C):
        mu  = C.mean(axis=1, keepdims=True)
        C_c = C - mu
        std = np.sqrt((C_c ** 2).sum(axis=1, keepdims=True)) + 1e-12
        return C_c / std

    A = _normalise(C_review)   # (N_review, T)  unit-normalised
    B = _normalise(C_final)    # (N_kept,   T)  unit-normalised
    # dot product of unit vectors = cosine similarity = Pearson r  (range [-1, 1])
    return A @ B.T             # (N_review, N_kept)


def _hungarian_recover(score_mat, N_kept, label):
    """Run Hungarian matching on score_mat; return (pair_scores, recovery_by_thresh)."""
    row_ind, col_ind = linear_sum_assignment(-score_mat)
    pairs = sorted(
        [(score_mat[r, c], r, c) for r, c in zip(row_ind, col_ind)],
        reverse=True,
    )
    score_strs = [f"{s:.3f}" for s, _, _ in pairs]
    log(f"  [{label}] scores (best first): {score_strs}")
    recovery = {}
    for t in THRESHOLDS_TO_REPORT:
        n = sum(1 for s, _, _ in pairs if s > t)
        pct = n / N_kept * 100 if N_kept else 0
        log(f"    [{label}] thresh={t:.2f}: {n}/{N_kept} ({pct:.0f}%)")
        recovery[str(t)] = n
    return pairs, recovery


# ---------------------------------------------------------------------------
# Per-session validation
# ---------------------------------------------------------------------------

def validate_session(task: str, session_dir: Path, tif_path: Path,
                     keep_bootstrap: bool = False,
                     rematch_only: bool = False) -> dict | None:
    bootstrap_dir = session_dir / "_bootstrap_validate"

    try:
        log(f"\n{'=' * 65}")
        log(f"  VALIDATE: {task}/{session_dir.name}")
        log(f"{'=' * 65}")

        already_have_bootstrap = (bootstrap_dir / "spatial_footprints.mat").exists()

        if rematch_only and not already_have_bootstrap:
            log("  --rematch-only requested but _bootstrap_validate/ missing. Skipping.")
            return None

        # Step 1: extract A_final + C_raw_final
        log("  Step 1: Extracting curated neuron footprints + traces...")
        result1 = _extract_final_footprints(session_dir)
        if result1 is None:
            log("  FAILED at step 1.")
            return None
        A_final, S_final, gSig_orig, gSiz_orig = result1
        N_kept = A_final.shape[1]
        log(f"  Step 1 done: {N_kept} curated neurons (gSig={gSig_orig}, gSiz={gSiz_orig})")

        # Step 2: estimate params
        if not rematch_only:
            log("  Step 2: Estimating CNMFe parameters...")
            p = params_module.estimate_all_params(
                session_dir.name, session_dir, log, bootstrap_mode=True)
            p["gSig"] = gSig_orig
            p["gSiz"] = gSiz_orig
            log(f"  Step 2 done: {p}")

            # Step 3: headless CNMFe (skip if bootstrap dir already populated)
            if already_have_bootstrap:
                log("  Step 3: Skipping CNMFe — _bootstrap_validate/ already exists.")
            else:
                bootstrap_dir.mkdir(exist_ok=True)
                log("  Step 3: Running headless CNMFe (30-120 min)...")
                ok = _run_headless_into_bootstrap(tif_path, bootstrap_dir, p)
                if not ok:
                    log("  FAILED at step 3.")
                    return None

        # Step 4: load candidates
        log("  Step 4: Loading bootstrap candidates...")
        result4 = _load_candidates(bootstrap_dir)
        if result4 is None:
            log("  FAILED at step 4.")
            return None
        A_review, S_review, d1, d2 = result4
        N_review = A_review.shape[1]
        log(f"  [VALIDATE] A_final:  {A_final.shape}  (pixels x N_kept)")
        log(f"  [VALIDATE] A_review: {A_review.shape}  (pixels x N_candidates)")

        # Step 5: align spike train frame counts (bootstrap is 2x temporally downsampled)
        S_review_al, S_final_al = _align_traces(S_review, S_final)

        # Step 6: compute all three score matrices
        log("  Step 6: Running all three matching strategies...")
        S_spatial  = _spatial_matrix(A_review, A_final)             # (N_review, N_kept)
        S_temporal = _temporal_matrix(S_review_al, S_final_al)      # (N_review, N_kept)

        # Clip temporal to [0,1] before combining (negative correlation = no match)
        S_temporal_clipped = np.clip(S_temporal, 0, 1)
        S_combined = np.sqrt(S_spatial * S_temporal_clipped)      # geometric mean

        results = {}
        for label, mat in [("spatial",  S_spatial),
                           ("temporal", S_temporal_clipped),
                           ("combined", S_combined)]:
            log(f"\n  --- {label.upper()} matching ---")
            pairs, recovery = _hungarian_recover(mat, N_kept, label)
            results[label] = {
                "pair_scores":           [round(float(s), 4) for s, _, _ in pairs],
                "candidate_indices":     [int(r) for s, r, c in pairs],
                "curated_indices":       [int(c) for s, r, c in pairs],
                "recovery_by_threshold": recovery,
            }

        # --- Gate analysis ---
        # For every temporal match above 0.45, what is its spatial score?
        # This tells us where to set the spatial gate.
        t_pairs = results["temporal"]
        TEMPORAL_THRESH = 0.45
        temporal_matched = [
            (ts, r, c)
            for ts, r, c in zip(t_pairs["pair_scores"],
                                 t_pairs["candidate_indices"],
                                 t_pairs["curated_indices"])
            if ts > TEMPORAL_THRESH
        ]
        spatial_scores_of_temporal_matches = sorted(
            [round(float(S_spatial[r, c]), 4) for ts, r, c in temporal_matched]
        )
        log(f"\n  --- GATE ANALYSIS (temporal thresh={TEMPORAL_THRESH}) ---")
        log(f"  Spatial scores of {len(temporal_matched)} temporal matches "
            f"(sorted): {spatial_scores_of_temporal_matches}")

        # Bucket into ranges so the distribution is readable
        buckets = [(0.00, 0.10), (0.10, 0.20), (0.20, 0.30),
                   (0.30, 0.45), (0.45, 1.01)]
        bucket_counts = []
        for lo, hi in buckets:
            n = sum(1 for s in spatial_scores_of_temporal_matches if lo <= s < hi)
            bucket_counts.append(n)
            log(f"    spatial [{lo:.2f}, {hi:.2f}): {n} neurons")

        # Sweep gate values: how many temporal matches survive each gate?
        GATE_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
        log(f"  Gate sweep (temporal > {TEMPORAL_THRESH} AND spatial > gate):")
        gate_recovery = {}
        for gate in GATE_VALUES:
            n = sum(1 for s in spatial_scores_of_temporal_matches if s > gate)
            pct = n / N_kept * 100 if N_kept else 0
            log(f"    gate={gate:.2f}: {n}/{N_kept} ({pct:.0f}%)")
            gate_recovery[str(gate)] = n

        results["gate_analysis"] = {
            "spatial_scores_of_temporal_matches": spatial_scores_of_temporal_matches,
            "spatial_bucket_counts":              bucket_counts,
            "gate_sweep":                         gate_recovery,
        }

        # Summary comparison
        header = f"  {'thresh':<8}" + "".join(f"  {s:>12}" for s in STRATEGIES)
        log("\n  === STRATEGY COMPARISON ===")
        log(header)
        for t in THRESHOLDS_TO_REPORT:
            row = f"  {t:<8.2f}" + "".join(
                f"  {results[s]['recovery_by_threshold'][str(t)]:>4}"
                f"({results[s]['recovery_by_threshold'][str(t)] / N_kept * 100:>2.0f}%)"
                for s in STRATEGIES
            )
            log(row)

        stats = {
            "session":     f"{task}/{session_dir.name}",
            "n_candidates": N_review,
            "n_curated":    N_kept,
            "strategies":   results,
        }
        out_path = session_dir / "validation_match_stats_temporal.json"
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        log(f"  Saved: {out_path}")

        return stats

    except Exception as e:
        log(f"  [VALIDATE] Unexpected error: {e}")
        log(traceback.format_exc())
        return None

    finally:
        if bootstrap_dir.exists() and not keep_bootstrap:
            shutil.rmtree(str(bootstrap_dir), ignore_errors=True)
            log("  _bootstrap_validate/ cleaned up.")
        elif bootstrap_dir.exists():
            log("  _bootstrap_validate/ kept (--keep-bootstrap).")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_stats: list[dict]):
    print()
    print("=" * 80)
    print("STRATEGY COMPARISON SUMMARY  (recovery at thresh=0.45)")
    print("=" * 80)
    print(f"{'Session':<38}  {'Curated':>7}", end="")
    for s in STRATEGIES:
        print(f"  {s:>12}", end="")
    print()
    print("-" * 80)

    totals_curated = 0
    totals = {s: 0 for s in STRATEGIES}

    for d in all_stats:
        short     = d["session"][-38:]
        n_curated = d["n_curated"]
        totals_curated += n_curated
        print(f"{short:<38}  {n_curated:>7}", end="")
        for s in STRATEGIES:
            n   = d["strategies"][s]["recovery_by_threshold"]["0.45"]
            pct = n / n_curated * 100 if n_curated else 0
            totals[s] += n
            print(f"  {n:>4}({pct:>2.0f}%)", end="")
        print()

    print("-" * 80)
    print(f"{'TOTAL':<38}  {totals_curated:>7}", end="")
    for s in STRATEGIES:
        n   = totals[s]
        pct = n / totals_curated * 100 if totals_curated else 0
        print(f"  {n:>4}({pct:>2.0f}%)", end="")
    print()

    # Delta vs spatial baseline
    print()
    print("Gain over spatial-only (at thresh=0.45):")
    spatial_total = totals["spatial"]
    for s in STRATEGIES:
        delta = totals[s] - spatial_total
        sign  = "+" if delta >= 0 else ""
        pct   = delta / totals_curated * 100 if totals_curated else 0
        print(f"  {s:>10}: {sign}{delta} neurons ({sign}{pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sessions without running.")
    parser.add_argument("--keep-bootstrap", action="store_true",
                        help="Keep _bootstrap_validate/ after matching (saves re-running MATLAB).")
    parser.add_argument("--rematch-only", action="store_true",
                        help="Skip MATLAB+CNMFe; only redo matching on existing _bootstrap_validate/.")
    args = parser.parse_args()

    sessions_to_run = []
    for task, session_name in VALIDATION_SESSIONS:
        session_dir = DATA_ROOT / task / session_name
        if not session_dir.exists():
            log(f"WARNING: session not found: {session_dir}")
            continue
        tifs = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
        if not tifs:
            log(f"WARNING: no .tif found in {session_dir} — skipping.")
            continue
        sessions_to_run.append((task, session_dir, tifs[0]))

    log(f"Validation sessions: {len(sessions_to_run)}")
    for task, sd, tif in sessions_to_run:
        has_temporal = (sd / "validation_match_stats_temporal.json").exists()
        has_bootstrap = (sd / "_bootstrap_validate" / "spatial_footprints.mat").exists()
        log(f"  {task}/{sd.name}"
            f"{'  [temporal done]' if has_temporal else ''}"
            f"{'  [bootstrap cached]' if has_bootstrap else ''}")

    if args.dry_run:
        return

    all_stats = []
    for task, session_dir, tif_path in sessions_to_run:
        existing = session_dir / "validation_match_stats_temporal.json"
        if existing.exists() and not args.rematch_only:
            log(f"\nSkipping {task}/{session_dir.name} — temporal JSON exists.")
            with open(existing) as f:
                all_stats.append(json.load(f))
            continue

        stats = validate_session(task, session_dir, tif_path,
                                 keep_bootstrap=args.keep_bootstrap,
                                 rematch_only=args.rematch_only)
        if stats:
            all_stats.append(stats)

    if all_stats:
        print_summary(all_stats)
    else:
        log("No sessions completed successfully.")


if __name__ == "__main__":
    main()
