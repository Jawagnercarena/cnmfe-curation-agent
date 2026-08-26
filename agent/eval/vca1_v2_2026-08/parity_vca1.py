"""
Step 3b.4: the vCA1 parity gate -- three phases.

Phase 1  v2b parity.  The SHIPPING features.compute_v2b_features, run on the
         .feature_expansion extractions, must reproduce ref_v2b_vca1.py's
         independently-vendored values (allclose rtol 1e-6, atol 0) on all 23
         labeled agent sessions.  This proves the deployed code computes what we
         are about to evaluate.  (../step4_2026-08/parity_check.py:66-130 does
         the same for BLA against its pinned Step 2 values; vCA1 has no such
         pin, so ref_v2b_vca1.py is the target.)

Phase 2  Loader orientation.  features.load_spatial's footprints must equal
         A.txt columns reshaped order='F' and must NOT equal order='C'.  Proves
         the production path and the extraction path feed compute_v2b_features
         the same geometry.  Delegates to parity_check.phase2_loader_orientation,
         which configure() has re-pointed at the vCA1 root.

Phase 3  Bootstrap orientation + label faithfulness (NEW; vCA1-specific).  The
         bootstrap arms read footprints from bootstrap_candidates.npz, which
         stores them C-order -- the OPPOSITE convention to the extractions.
           (a) faithfulness: recomputing features.spatial_features /
               temporal_features from the npz reproduces the stored
               candidate_features.npz columns, proving the persisted traces and
               footprints ARE the ones the labels were matched against.
               Tolerances are loose because the npz stores float32
               (bootstrap_preagent.py:467,473) while the features came from
               float64 arrays.
           (b) orientation: every vCA1 bootstrap frame is 512x512, so an F/C
               control on SPATIAL features would be VACUOUS -- for a square
               image order='F' of a C-order vector is exactly the transpose, and
               area / circularity / max_weight are transpose-invariant.  The
               control therefore runs on a Cn-dependent quantity, which is not
               transpose-symmetric.

Exit 0 = all phases pass.  Read-only.
"""
import argparse
import json
import sys

import numpy as np
import scipy.io as sio

import vca1_common as vc

RTOL = 1e-6
ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def passed(msg):
    print(f"  ok    {msg}")


def phase1():
    import features
    import parity_check as pc
    print(f"Phase 1: shipping features.compute_v2b_features vs the vendored reference (rtol={RTOL})")
    ref = np.load(vc.PIN / "vca1_v2b_reference.npz", allow_pickle=True)
    hic = np.load(vc.PIN / "hiconf_scores.npz", allow_pickle=True)
    names = [str(x) for x in ref["feature_names"]]
    if names != features.V2B_NAMES:
        fail(f"feature-name order drift: {names} != {features.V2B_NAMES}")
        return
    keys = sorted(k[:-3] for k in ref.files if k.endswith("__X"))
    worst = (0.0, "-", "-")
    n_pass = 0
    for k in keys:
        rel = k.replace("__", "/", 1)
        m = sio.loadmat(str(vc.EXT / (k + ".mat")))
        C = m["C_raw"].astype(float)
        fps = pc.footprints_from_sparse(m["A"], int(m["d1"][0][0]), int(m["d2"][0][0]))
        idx = ref[k + "__idx"]
        hi = hic[rel + "__scores"][idx] >= features.HICONF_SCORE
        got = features.compute_v2b_features(C, fps, m["Cn"], hi)
        exp = ref[k + "__X"]
        if got.shape != exp.shape:
            fail(f"{rel}: shape {got.shape} != {exp.shape}")
            continue
        if np.allclose(got, exp, rtol=RTOL, atol=0):
            n_pass += 1
        else:
            fail(f"{rel}: values differ from the reference")
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-12)
        if np.nanmax(d) > worst[0]:
            worst = (float(np.nanmax(d)), rel, names[int(np.nanargmax(d)) % len(names)])
    (passed if n_pass == len(keys) else fail)(
        f"{n_pass}/{len(keys)} sessions reproduce the reference")
    print(f"        worst relative diff {worst[0]:.2e} ({worst[1]}, {worst[2]})")


def phase2():
    import parity_check as pc
    print("")
    try:
        (passed if pc.phase2_loader_orientation() else fail)("loader orientation (order-F)")
    except Exception as e:
        fail(f"phase 2 raised {type(e).__name__}: {e}")


def phase3():
    import features
    print("\nPhase 3: bootstrap candidate npz -- faithfulness and orientation")
    _, bs, _, _ = vc.classify_sessions()
    cn_stat = {c["rel"]: c for c in json.loads((vc.SP / "vca1_cn_status.json").read_text())}
    same = [r for r in bs if cn_stat[vc.rel(r)]["status"] == "same_res"]
    v73 = [r for r in same if cn_stat[vc.rel(r)]["mat_version"] == "v7.3"]
    wrong = [r for r in bs if cn_stat[vc.rel(r)]["status"] != "same_res"]
    picks = [same[0]] + (v73[:1] if v73 else []) + (wrong[:1] if wrong else [])

    sp_keys = ["area", "circularity", "eccentricity", "compactness",
               "max_weight", "weight_spread"]
    tp_keys = ["peak_snr", "transient_freq", "events_per_min",
               "baseline_stability", "skewness"]

    for sd in picks:
        rel = vc.rel(sd)
        st = cn_stat[rel]
        z = np.load(sd / "bootstrap_candidates.npz", allow_pickle=True)
        fps = vc.bootstrap_footprints(z)                  # order='C'
        C = z["C_raw"].astype(float)
        npz = np.load(sd / vc.V1, allow_pickle=True)
        stored = npz["feature_matrix"]
        fn = [str(x) for x in npz["feature_names"]]
        n = min(12, len(fps))

        got = np.array([[features.spatial_features(fps[i])[k] for k in sp_keys]
                        for i in range(n)])
        exp = stored[:n][:, [fn.index(k) for k in sp_keys]]
        d_sp = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-12)
        (passed if np.nanmax(d_sp) < 1e-5 else fail)(
            f"{rel[:42]:<42} spatial reproduce (max rel {np.nanmax(d_sp):.1e}, float32 npz)")

        gott = np.array([[features.temporal_features(C[i])[k] for k in tp_keys]
                         for i in range(n)])
        expt = stored[:n][:, [fn.index(k) for k in tp_keys]]
        d_tp = np.abs(gott - expt) / np.maximum(np.abs(expt), 1e-12)
        (passed if np.nanmax(d_tp) < 1e-4 else fail)(
            f"{rel[:42]:<42} temporal reproduce (max rel {np.nanmax(d_tp):.1e}) "
            f"-> persisted traces ARE the labelled run's")

        if st["status"] == "same_res":
            # cn_correlation is NOT transpose-symmetric, unlike the spatial cols.
            # The stored value came from the re-run's own Cn (deleted with
            # _bootstrap/), so this is a relative comparison, not an equality:
            # the correct orientation must sit closer to the stored column.
            Cn = vc.load_cn_any(sd / "Cn.mat")
            cc_c = np.array([features.cn_features(fps[i], Cn)["cn_correlation"] for i in range(n)])
            cc_f = np.array([features.cn_features(fps[i].T, Cn)["cn_correlation"] for i in range(n)])
            exp_cc = stored[:n][:, fn.index("cn_correlation")]
            err_c = float(np.nanmean(np.abs(cc_c - exp_cc)))
            err_f = float(np.nanmean(np.abs(cc_f - exp_cc)))
            (passed if err_c < err_f else fail)(
                f"{rel[:42]:<42} order-C beats order-F on cn_correlation "
                f"({err_c:.4f} vs {err_f:.4f}, {st['mat_version']} Cn)")
        else:
            print(f"  --    {rel[:42]:<42} Cn {st['status']} {st['cn_shape']} vs "
                  f"{st['cand_dims']} -> ring_contrast 0 in both arms")


def pending_all():
    import features
    print("\nPending pre-check: production loaders on EVERY pending session")
    _, _, pend, _ = vc.classify_sessions()
    bad = 0
    for sd in pend:
        try:
            n = int(np.load(sd / vc.V1, allow_pickle=True)["n_candidates"][0])
            tr = features.load_traces(sd)
            fp = features.load_spatial(sd)
            cn = features.load_cn(sd)
            if tr.shape[0] != n or fp.shape[0] != n:
                fail(f"{vc.rel(sd)}: traces {tr.shape[0]} / footprints {fp.shape[0]} != {n}")
                bad += 1
            elif cn is None:
                fail(f"{vc.rel(sd)}: load_cn returned None")
                bad += 1
        except Exception as e:
            fail(f"{vc.rel(sd)}: loader raised {type(e).__name__}: {e}")
            bad += 1
    (passed if bad == 0 else fail)(
        f"all {len(pend)} pending sessions load with the production loaders")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending-all", action="store_true",
                    help="also call the production loaders on every pending session")
    args = ap.parse_args()
    vc.configure()
    phase1()
    phase2()
    phase3()
    if args.pending_all:
        pending_all()
    print(f"\nPARITY: {'ALL PASS' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
