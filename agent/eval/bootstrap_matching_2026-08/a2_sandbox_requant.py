"""
a2_sandbox_requant.py — requantify bootstrap matching on the 4 sandbox sessions
with CORRECT pixel ordering.

Per session:
  1. Correct-order Hungarian matching curated -> re-run candidates:
     recovery at 0.45 + threshold sweep, true-pair similarity distribution vs
     the junk background (max sim over curated for unmatched candidates).
  2. Assignment analysis: Hungarian vs per-curated best candidate (argmax) and
     greedy many-to-one sharing (evidence of merging).
  3. Duplicate analysis: candidates above threshold to an already-matched
     curated neuron (these would be full-weight NEGATIVES under current rules).
  4. Ground-truth transfer: map re-run candidates onto the original reviewed
     candidates (.feature_expansion A, human labels.mat) by mutual-best
     consistent-order cosine > 0.6. Verify matched candidates are human-KEPT,
     and measure what the OLD mismatched labels would have marked positive.

READ-ONLY on D:. Writes a2_results.json into this eval dir only.
Run:  C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe a2_sandbox_requant.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent))
import bmlib

THRESH_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
GT_TRANSFER_THRESH = 0.6
FEXP_DIR = bmlib.DATA_ROOT / "BLA" / ".feature_expansion"


def load_gt(session_dir: Path):
    """Original reviewed candidates: (A_stack (N,d1,d2) float32, labels (N,))."""
    name = f"{session_dir.parent.name}__{session_dir.name}.mat"
    fexp = FEXP_DIR / name
    if not fexp.exists():
        return None, None, f"no extraction {name}"
    d = sio.loadmat(str(fexp))
    d1 = int(d["d1"].flat[0]); d2 = int(d["d2"].flat[0])
    A = d["A"]
    if hasattr(A, "toarray"):                          # stored sparse
        A = A.toarray()
    A = np.asarray(A, dtype=np.float32)                # (pixels, N) F-order cols
    stack = bmlib.Fcols_to_stack(A, d1, d2)
    labels = sio.loadmat(str(session_dir / "labels.mat"))["labels"].ravel()
    if len(labels) != stack.shape[0]:
        return None, None, (f"label/extraction size mismatch "
                            f"{len(labels)} vs {stack.shape[0]}")
    return stack, labels.astype(int), None


def analyze(session_dir: Path) -> dict:
    name = session_dir.name
    print(f"\n=== {session_dir.parent.name}/{name} ===")
    cur = bmlib.load_curated_stack(session_dir)
    cand = bmlib.load_sandbox_candidates(session_dir)
    n_cur, n_cand = cur.shape[0], cand.shape[0]
    print(f"  curated {n_cur} | re-run candidates {n_cand}")

    cand_F = bmlib.stack_to_F(cand)
    cur_F = bmlib.stack_to_F(cur)
    corr = bmlib.cosine_matrix(cand_F, cur_F)          # (n_cand, n_cur)

    out = {"session": name, "n_curated": n_cur, "n_candidates": n_cand}

    # --- 1. Hungarian recovery + sweep -------------------------------------
    ri, ci, sims = bmlib.hungarian_pairs(corr)
    out["recovery_by_threshold"] = {
        str(t): int((sims > t).sum()) for t in THRESH_SWEEP}
    out["pair_sims"] = np.round(sims, 4).tolist()
    print(f"  recovery: " + "  ".join(
        f"{t}:{out['recovery_by_threshold'][str(t)]}" for t in THRESH_SWEEP))

    assigned = dict(zip(ci.tolist(), ri.tolist()))     # curated -> candidate
    matched_cand = {int(ri[k]) for k in range(len(ri)) if sims[k] > 0.45}

    # Junk background: candidates not matched to anything — their best sim to
    # any curated neuron. This is the "wrong pair" score distribution the
    # threshold must reject.
    unmatched_mask = np.ones(n_cand, bool)
    unmatched_mask[list(matched_cand)] = False
    junk_best = corr[unmatched_mask].max(axis=1)
    out["junk_best_sim"] = {
        "p50": float(np.median(junk_best)), "p95": float(np.percentile(junk_best, 95)),
        "p99": float(np.percentile(junk_best, 99)), "max": float(junk_best.max())}
    print(f"  true pairs: min {sims.min():.3f} median {np.median(sims):.3f} | "
          f"junk best-sim: p95 {out['junk_best_sim']['p95']:.3f} "
          f"p99 {out['junk_best_sim']['p99']:.3f} max {out['junk_best_sim']['max']:.3f}")

    # --- 2. Assignment analysis --------------------------------------------
    best_cand = corr.argmax(axis=0)                    # per curated
    best_sim = corr.max(axis=0)
    hung_is_best = sum(1 for c in range(n_cur) if assigned.get(c) == best_cand[c])
    shared = n_cur - len(set(best_cand.tolist()))      # curated sharing a best cand
    out["hungarian_equals_argmax"] = int(hung_is_best)
    out["greedy_shared_candidates"] = int(shared)
    print(f"  Hungarian==argmax for {hung_is_best}/{n_cur} curated | "
          f"greedy sharing (merge evidence): {shared}")

    # --- 3. Duplicates -------------------------------------------------------
    dup = {}
    for t in (0.45, 0.55, 0.60):
        d = 0
        for j in range(n_cand):
            if j in matched_cand:
                continue
            if corr[j].max() > t:
                d += 1
        dup[str(t)] = d
    out["duplicate_negatives"] = dup
    print(f"  duplicate-negatives (unassigned cand > thr to a curated): {dup}")

    # --- 4. Ground-truth transfer -------------------------------------------
    gt_stack, gt_labels, err = load_gt(session_dir)
    if err:
        print(f"  [GT] skipped: {err}")
        out["gt"] = {"error": err}
        return out

    gt_F = bmlib.stack_to_F(gt_stack)
    xg = bmlib.cosine_matrix(cand_F, gt_F)             # (n_cand, n_orig)
    # mutual best above GT_TRANSFER_THRESH
    fwd = xg.argmax(axis=1)                            # cand -> orig
    rev = xg.argmax(axis=0)                            # orig -> cand
    cand_gt = np.full(n_cand, -1)                      # -1 unknown, else 0/1
    for j in range(n_cand):
        o = fwd[j]
        if xg[j, o] > GT_TRANSFER_THRESH and rev[o] == j:
            cand_gt[j] = gt_labels[o]
    n_known = int((cand_gt >= 0).sum())
    out["gt_transfer"] = {"n_known": n_known, "n_unknown": int(n_cand - n_known)}
    print(f"  [GT] transferred labels for {n_known}/{n_cand} re-run candidates")

    # matched candidates should be human-KEPT
    mk = [int(cand_gt[j]) for j in matched_cand]
    out["gt_matched"] = {"kept": mk.count(1), "deleted": mk.count(0),
                         "unknown": mk.count(-1)}
    print(f"  [GT] fixed-matching positives: {mk.count(1)} kept / "
          f"{mk.count(0)} deleted / {mk.count(-1)} unknown")

    # duplicates at 0.45: what are they, per GT?
    dup_idx = [j for j in range(n_cand)
               if j not in matched_cand and corr[j].max() > 0.45]
    dk = [int(cand_gt[j]) for j in dup_idx]
    out["gt_duplicates"] = {"kept": dk.count(1), "deleted": dk.count(0),
                           "unknown": dk.count(-1)}
    print(f"  [GT] duplicate-negatives: {dk.count(1)} kept / {dk.count(0)} deleted "
          f"/ {dk.count(-1)} unknown")

    # what the OLD mismatched matching would have labeled positive
    corr_bug = bmlib.cosine_matrix(bmlib.stack_to_C(cand), cur_F)
    rb, cb, sb = bmlib.hungarian_pairs(corr_bug)
    old_pos = [int(rb[k]) for k in range(len(rb)) if sb[k] > 0.45]
    ok = [int(cand_gt[j]) for j in old_pos]
    out["gt_old_positives"] = {"n": len(old_pos), "kept": ok.count(1),
                              "deleted": ok.count(0), "unknown": ok.count(-1)}
    print(f"  [GT] OLD mismatched positives (n={len(old_pos)}): {ok.count(1)} kept / "
          f"{ok.count(0)} deleted / {ok.count(-1)} unknown")
    return out


if __name__ == "__main__":
    results = []
    for sd in bmlib.SANDBOX_SESSIONS:
        results.append(analyze(sd))
    out_path = Path(__file__).parent / "a2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {out_path}")
