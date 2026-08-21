"""
b2_rerun_report.py — aggregate the Phase B pilot into the Gate B numbers.

Reads the shadow JSONs under D:\\Julian_CNMFe\\.bootstrap_diag\\, the live
session JSONs (old recovery), a3_damage.json (predictions), and
b1_pilot_log.json (runtimes, untouched flags). Also transfers human ground
truth onto the fresh bla21 end-to-end run.

Writes b2_gate_b.json and prints the report body. READ-ONLY outside this dir.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
import bmlib
from a2_sandbox_requant import load_gt, GT_TRANSFER_THRESH

DIAG = bmlib.DATA_ROOT / ".bootstrap_diag"
HERE = Path(__file__).parent


def old_stats(session_dir: Path) -> dict | None:
    """None for agent sessions (bla21 end-to-end) which have no old bootstrap JSON."""
    jp = session_dir / "bootstrap_match_stats.json"
    if not jp.exists():
        return None
    with open(jp) as f:
        return json.load(f)


def load_candidates_npz(shadow: Path):
    z = np.load(shadow / "bootstrap_candidates.npz")
    A = sparse.csr_matrix((z["A_data"], z["A_indices"], z["A_indptr"]),
                          shape=tuple(z["A_shape"]))
    return A, int(z["d1"][0]), int(z["d2"][0])


def gt_check_bla21(shadow: Path) -> dict:
    """Transfer human labels onto the FRESH bla21 run's matched candidates."""
    session_dir = (bmlib.DATA_ROOT / "BLA" / "2tones"
                   / "AVG5x-TSeries-093025-bla21-313um-38z-000")
    gt_stack, gt_labels, err = load_gt(session_dir)
    if err:
        return {"error": err}
    A, d1, d2 = load_candidates_npz(shadow)          # (N, pixels) C-order
    cand_rows = np.asarray(A.todense(), dtype=np.float32)
    gt_rows = bmlib.stack_to_C(gt_stack)             # C-order to match
    xg = bmlib.cosine_matrix(cand_rows, gt_rows)
    fwd = xg.argmax(axis=1)
    rev = xg.argmax(axis=0)
    with open(shadow / "bootstrap_match_stats.json") as f:
        s = json.load(f)
    matched = set(s["candidate_indices"][:s["n_matched"]])
    verdicts = {"kept": 0, "deleted": 0, "unknown": 0}
    for j in matched:
        o = fwd[j]
        if xg[j, o] > GT_TRANSFER_THRESH and rev[o] == j:
            verdicts["kept" if gt_labels[o] == 1 else "deleted"] += 1
        else:
            verdicts["unknown"] += 1
    return verdicts


if __name__ == "__main__":
    with open(HERE / "b1_pilot_log.json") as f:
        pilot_log = json.load(f)
    runlog = {(e["session"], e["runtag"]): e for e in pilot_log}

    rows = []
    for jp in sorted(DIAG.rglob("bootstrap_match_stats.json")):
        shadow = jp.parent
        runtag = shadow.name
        session = shadow.parent.name
        task = shadow.parent.parent.name
        area = shadow.parent.parent.parent.name
        with open(jp) as f:
            s = json.load(f)
        real_dir = bmlib.DATA_ROOT / area / task / session
        old = old_stats(real_dir)
        n_matched_sims = s["pair_similarities"][:s["n_matched"]]
        entry = runlog.get((f"{area}/{task}/{session}", runtag), {})
        rows.append({
            "area": area, "task": task, "session": session, "runtag": runtag,
            "n_curated": s["n_curated"],
            "old_matched": old["n_matched"] if old else None,
            "old_recovery": (round(old["n_matched"] / old["n_curated"], 3)
                             if old else None),
            "new_matched": s["n_matched"],
            "new_recovery": round(s["n_matched"] / s["n_curated"], 3),
            "n_candidates": s["n_candidates"],
            "matched_sim_min": min(n_matched_sims) if n_matched_sims else None,
            "matched_sim_med": (round(float(np.median(n_matched_sims)), 3)
                                if n_matched_sims else None),
            "n_ambiguous": len(s.get("ambiguous_candidate_indices", [])),
            "n_duplicates": len(s.get("duplicate_candidate_indices", [])),
            "recovery_by_threshold": s.get("recovery_by_threshold", {}),
            "min_corr": s.get("cnmfe_params", {}).get("min_corr"),
            "runtime_min": entry.get("runtime_min"),
            "untouched": entry.get("session_dir_untouched"),
        })

    # ---- print table ----
    print(f"{'session':46s} {'tag':10s} {'old':>9s} {'new':>9s} "
          f"{'simmin':>6s} {'dup':>4s} {'amb':>4s} {'min':>6s}")
    for r in sorted(rows, key=lambda r: (r["runtag"], r["area"], r["session"])):
        old_str = (f"{r['old_matched']:>3d}/{r['n_curated']:<3d}"
                   if r["old_matched"] is not None else f"  —/{r['n_curated']:<3d}")
        print(f"{r['session'][:46]:46s} {r['runtag']:10s} "
              f"{old_str}   "
              f"{r['new_matched']:>3d}/{r['n_curated']:<3d}   "
              f"{r['matched_sim_min'] if r['matched_sim_min'] else 0:6.3f} "
              f"{r['n_duplicates']:>4d} {r['n_ambiguous']:>4d} "
              f"{str(r['runtime_min']):>6s}")

    fixed = [r for r in rows if r["runtag"] == "fixed"
             and r["old_matched"] is not None]
    tot_cur = sum(r["n_curated"] for r in fixed)
    tot_old = sum(r["old_matched"] for r in fixed)
    tot_new = sum(r["new_matched"] for r in fixed)
    print(f"\nFIXED runs: {len(fixed)} sessions | curated {tot_cur} | "
          f"old matched {tot_old} ({tot_old / tot_cur:.1%}) | "
          f"new matched {tot_new} ({tot_new / tot_cur:.1%})")

    # threshold stability across fixed runs
    for t in ("0.45", "0.55", "0.60", "0.65"):
        n = sum(r["recovery_by_threshold"].get(t, 0) for r in fixed)
        print(f"  total recovery at thr {t}: {n}/{tot_cur} ({n / tot_cur:.1%})")

    # untouched verification
    bad = [r for r in rows if r["untouched"] is False]
    print(f"\nsession-dir untouched: "
          f"{'ALL VERIFIED' if not bad else f'{len(bad)} VIOLATIONS: ' + str(bad)}")

    # GT check on the fresh bla21 run
    bla21_shadow = (DIAG / "BLA" / "2tones"
                    / "AVG5x-TSeries-093025-bla21-313um-38z-000" / "fixed")
    if (bla21_shadow / "bootstrap_candidates.npz").exists():
        v = gt_check_bla21(bla21_shadow)
        print(f"\nbla21 fresh-run positives vs human labels: {v}")
    else:
        v = {"error": "npz missing"}

    with open(HERE / "b2_gate_b.json", "w") as f:
        json.dump({"rows": rows, "bla21_gt": v}, f, indent=1)
    print(f"\nWrote {HERE / 'b2_gate_b.json'}")
