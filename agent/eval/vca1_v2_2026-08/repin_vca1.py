"""
Step 3a: pin the vCA1 pool, gate provenance, and inventory what the backfill needs.

Port of ../step4_2026-08/repin_manifest.py for vCA1, with two differences:
  * there is no prior vCA1 pin, so the first run CREATES one instead of
    diffing against it (later runs diff and report drift, which is how
    deploy-day detects that a reviewer return moved the pool);
  * it additionally records the bootstrap Cn inventory, because the vCA1
    bootstrap-v2b arms (b0/b1) need to know which sessions can supply a
    same-resolution correlation image for ring_contrast.

Outputs (all in this directory):
  vca1_pool_manifest.json     labeled pool, manifest_util schema
  vca1_extract_sessions.txt   labeled agent rels needing MATLAB extraction
  vca1_pending.json           pending sessions + candidate-file presence
  vca1_cn_status.json         per bootstrap session: Cn readability/resolution

Read-only on session dirs.  Exit 1 on any provenance failure.

  python repin_vca1.py                  pin + inventory
  python repin_vca1.py --check-extract  verify the extraction mats afterwards
"""
import argparse
import json
import sys

import numpy as np
import scipy.io as sio

import vca1_common as vc

sys.path.insert(0, str(vc.AGENT / "eval" / "step2_2026-08"))
import manifest_util                                    # noqa: E402
import train_classifier as tc                           # noqa: E402

MANIFEST = vc.SP / "vca1_pool_manifest.json"
EXTRACT  = vc.SP / "vca1_extract_sessions.txt"
PENDING  = vc.SP / "vca1_pending.json"
CN_STAT  = vc.SP / "vca1_cn_status.json"


def pin():
    live = manifest_util.build_state()          # resolves config.DATA_ROOT == vCA1
    n_ag = sum(1 for s in live if not s["is_bootstrap"])
    print(f"pool: {len(live)} labeled sessions ({n_ag} agent, {len(live) - n_ag} bootstrap)")

    assert not any(vc.PARKED in s["rel"] for s in live), \
        f"{vc.PARKED} is back in the pool -- it was parked in Step 0b"

    # ---- drift vs a previous pin (deploy-day check; first run creates it) ----
    if MANIFEST.exists():
        pinned = {s["rel"]: s for s in json.loads(MANIFEST.read_text())}
        livem  = {s["rel"]: s for s in live}
        new     = sorted(set(livem) - set(pinned))
        gone    = sorted(set(pinned) - set(livem))
        changed = sorted(r for r in set(pinned) & set(livem)
                         if pinned[r]["labels_mtime"] != livem[r]["labels_mtime"]
                         or pinned[r]["n_candidates"] != livem[r]["n_candidates"])
        print(f"\nDrift vs the existing pin ({len(pinned)} sessions): "
              f"new {len(new)}  gone {len(gone)}  changed {len(changed)}")
        for r in new:     print(f"    NEW     {r}")
        for r in gone:    print(f"    GONE    {r}")
        for r in changed: print(f"    CHANGED {r}")
        if new or gone or changed:
            print("  -> every pinned number downstream of this manifest is stale; "
                  "re-run the chain for the affected sessions.")
    else:
        print("\nNo previous pin -- creating one (first run).")

    # ---- provenance gate on labeled agent sessions ----
    need_extract, bad, already = [], [], []
    for s in live:
        if s["is_bootstrap"]:
            continue
        r = s["rel"]
        sd = vc.DATA_ROOT / r
        if (vc.EXT / (vc.key(r) + ".mat")).exists():
            already.append(r)
            continue
        rn, lab = sd / "review_neuron.mat", sd / "labels.mat"
        if not rn.exists():
            bad.append((r, "no review_neuron.mat")); continue
        if rn.stat().st_mtime >= lab.stat().st_mtime:
            bad.append((r, "review_neuron.mat NOT older than labels.mat")); continue
        npz = np.load(sd / vc.V1, allow_pickle=True)
        n_rev = int(npz["n_candidates"][0]) - len(npz["auto_rejected"])
        n_lab = len(sio.loadmat(str(lab))["labels"].flatten())
        if n_rev != n_lab:
            bad.append((r, f"review set {n_rev} != labels {n_lab}")); continue
        need_extract.append(r)

    print(f"\nLabeled agent sessions needing v2b extraction: {len(need_extract)}"
          f"  (already extracted: {len(already)})")
    for r in need_extract:
        print(f"    EXTRACT {r}")
    if bad:
        print(f"\nPROVENANCE FAILURES ({len(bad)}) -- STOP, do not extract these:")
        for r, why in bad:
            print(f"    {r}: {why}")

    # ---- pending inventory (the backfill's third row policy) ----
    _, _, pend, skipped = vc.classify_sessions()
    pend_recs = []
    for sd in pend:
        npz = np.load(sd / vc.V1, allow_pickle=True)
        have = {f: (sd / f).exists() for f in
                ("C_raw.txt", "spatial_footprints.mat", "Cn.mat", "A.txt")}
        pend_recs.append({"rel": vc.rel(sd),
                          "n_candidates": int(npz["n_candidates"][0]),
                          "n_auto_rejected": len(npz["auto_rejected"]),
                          "candidate_files": have})
    n_incomplete = sum(1 for p in pend_recs if not all(p["candidate_files"].values()))
    print(f"\nPending sessions: {len(pend_recs)} "
          f"({sum(p['n_candidates'] for p in pend_recs)} rows); "
          f"{n_incomplete} missing a candidate file")
    for p in pend_recs:
        if not all(p["candidate_files"].values()):
            miss = [k for k, v in p["candidate_files"].items() if not v]
            print(f"    INCOMPLETE {p['rel']}: missing {miss}")
    if skipped:
        print(f"  ({len(skipped)} session(s) have an npz but neither labels nor "
              f"ROIs_candidates.jpg -- ignored)")

    # ---- bootstrap Cn inventory (drives the b0 vs b1 arm split) ----
    _, bs, _, _ = vc.classify_sessions()
    cn = []
    for sd in bs:
        z = np.load(sd / "bootstrap_candidates.npz", allow_pickle=True)
        d1, d2 = int(z["d1"][0]), int(z["d2"][0])
        f = sd / "Cn.mat"
        if not f.exists():
            status, shape, ver = "missing", None, None
        else:
            try:
                shape = sio.loadmat(str(f))["Cn"].shape; ver = "v5"
            except NotImplementedError:
                img = vc.load_cn_any(f)
                shape, ver = (None if img is None else img.shape), "v7.3"
            except Exception:
                shape, ver = None, "unreadable"
            status = ("same_res" if shape == (d1, d2)
                      else "wrong_res" if shape else "unreadable")
        cn.append({"rel": vc.rel(sd), "cand_dims": [d1, d2],
                   "cn_shape": list(shape) if shape else None,
                   "mat_version": ver, "status": status,
                   "ring_available": status == "same_res"})
    from collections import Counter
    tally = Counter(c["status"] for c in cn)
    vers  = Counter(c["mat_version"] for c in cn if c["status"] == "same_res")
    n_ring = sum(1 for c in cn if c["ring_available"])
    print(f"\nBootstrap Cn inventory ({len(cn)} sessions): {dict(tally)}")
    print(f"  ring_contrast computable (same resolution): {n_ring}"
          f"   [by mat version: {dict(vers)}]")
    print(f"  -> arm b1 gets real ring on {n_ring}, zero on {len(cn) - n_ring}; "
          f"arm b0 gets zero on all {len(cn)}")

    MANIFEST.write_text(json.dumps(live, indent=1))
    EXTRACT.write_text("\n".join(need_extract) + ("\n" if need_extract else ""))
    PENDING.write_text(json.dumps(pend_recs, indent=1))
    CN_STAT.write_text(json.dumps(cn, indent=1))
    print(f"\nwrote {MANIFEST.name}, {EXTRACT.name} ({len(need_extract)} rels), "
          f"{PENDING.name}, {CN_STAT.name}")
    return 1 if bad else 0


def check_extract():
    """After the MATLAB extraction: N-parity of every extraction mat."""
    rels = [r for r in EXTRACT.read_text().splitlines() if r.strip()]
    if not rels:
        print("extraction list is empty -- run the pin first."); return 1
    print(f"checking {len(rels)} extraction mats against their npz/labels\n")
    n_ok = n_fail = 0
    for r in rels:
        sd = vc.DATA_ROOT / r
        mat = vc.EXT / (vc.key(r) + ".mat")
        if not mat.exists():
            print(f"  FAIL  {r}: no extraction mat"); n_fail += 1; continue
        m = sio.loadmat(str(mat))
        npz = np.load(sd / vc.V1, allow_pickle=True)
        n_cand = int(npz["n_candidates"][0])
        n_rev  = n_cand - len(npz["auto_rejected"])
        n_lab  = len(sio.loadmat(str(sd / "labels.mat"))["labels"].flatten())
        C = m["C_raw"]
        d1, d2 = int(m["d1"][0][0]), int(m["d2"][0][0])
        A_rows, A_cols = m["A"].shape
        probs = []
        if C.shape[0] != n_rev:            probs.append(f"C_raw rows {C.shape[0]} != review set {n_rev}")
        if n_rev != n_lab:                 probs.append(f"review set {n_rev} != labels {n_lab}")
        if A_cols != C.shape[0]:           probs.append(f"A cols {A_cols} != C_raw rows {C.shape[0]}")
        if A_rows != d1 * d2:              probs.append(f"A rows {A_rows} != d1*d2 {d1*d2}")
        if not np.isfinite(C).all():       probs.append("non-finite C_raw")
        if probs:
            print(f"  FAIL  {r}: {'; '.join(probs)}"); n_fail += 1
        else:
            print(f"  ok    {r}  N={C.shape[0]} T={C.shape[1]} dims={d1}x{d2}"); n_ok += 1
    print(f"\nSUMMARY: {n_ok} pass, {n_fail} fail of {len(rels)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-extract", action="store_true",
                    help="verify the extraction mats instead of re-pinning")
    args = ap.parse_args()
    vc.configure()
    sys.exit(check_extract() if args.check_extract else pin())
