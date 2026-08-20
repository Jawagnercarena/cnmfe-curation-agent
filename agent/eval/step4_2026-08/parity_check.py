"""
Step 4 backfill-parity invariant (deploy gate; see the Step 4 plan and
docs/FEATURE_EXPANSION_STEP4_BRIEF.md).

Phase 1 — v2b parity: run the SHIPPING features.compute_v2b_features against
every `.feature_expansion` extraction and compare to the pinned evaluation
values (allclose, rtol 1e-6, per session per column).  This proves the
deployed code computes exactly what Step 2 evaluated and the red-team
confirmed.  The high-confidence neighbor mask is rebuilt exactly the way the
reference did it: pinned grouped-OOF mean scores where the session is in the
OOF pool, current deployed-model scores otherwise (4 sessions; the deployed
joblib predates the pin and has not been retrained since — verified by mtime
before running).

Phase 2 — loader-orientation spot check: on up to 2 PENDING sessions (their
candidate A.txt / spatial_footprints.mat are still the candidate set;
CNMFe_final_save.m overwrites both at review completion), confirm that
features.load_spatial footprints are identical to the pixel-vector columns of
A.txt reshaped column-major (order="F") — the same convention as the
extraction mats.  This proves the production path (load_spatial) and the
backfill path (sparse A -> order-F images) feed compute_v2b_features the same
geometry.

Read-only on session dirs.  Exit code 0 = all invariants hold.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
import features  # noqa: E402  (shipping module under test)

EXT = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion")
PIN = EXT / "_pinned"
DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")
RTOL = 1e-6

HICONF = features.HICONF_SCORE  # 0.5, same constant the reference used


def load_oof_by_session():
    oof = np.load(PIN / "baseline_oof.npz", allow_pickle=True)
    oof_mean = oof["oof_seeds"].mean(axis=0)
    by_sess = {}
    for i in range(len(oof_mean)):
        by_sess.setdefault(str(oof["session"][i]), {})[
            int(oof["idx_in_session"][i])] = float(oof_mean[i])
    return by_sess


def footprints_from_sparse(A, d1, d2):
    """(pixels x N) sparse/dense -> (N, d1, d2), column-major pixel order —
    the extraction-mat convention (matches MATLAB reshape)."""
    A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    n = A.shape[1]
    fps = np.empty((n, d1, d2))
    for k in range(n):
        fps[k] = A[:, k].reshape((d1, d2), order="F")
    return fps


def phase1_v2b_parity():
    pinned = np.load(PIN / "step2_v2b_features.npz", allow_pickle=True)
    names = [str(x) for x in pinned["feature_names"]]
    assert names == features.V2B_NAMES, f"name order drift: {names}"

    model = joblib.load(AGENT / "model" / "BLA" / "classifier.joblib")
    oof_by_sess = load_oof_by_session()

    keys = sorted(k[:-3] for k in pinned.files if k.endswith("__X"))
    print(f"Phase 1: v2b parity on {len(keys)} pinned sessions (rtol={RTOL})")

    n_pass = n_fail = 0
    worst = (0.0, "-", "-")
    for key in keys:
        rel = key.replace("__", "/", 1)
        mat_f = EXT / (key + ".mat")
        m = sio.loadmat(str(mat_f))
        C = m["C_raw"].astype(float)
        A = m["A"]
        Cn = m["Cn"]
        d1, d2 = int(m["d1"][0][0]), int(m["d2"][0][0])

        sd = DATA_ROOT / rel
        npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
        auto_rej = set(int(i) for i in npz["auto_rejected"])
        n_cand = int(npz["n_candidates"][0])
        review_idx = np.array([i for i in range(n_cand) if i not in auto_rej])
        assert (review_idx == pinned[key + "__idx"]).all(), \
            f"{rel}: review_idx != pinned idx"
        assert len(review_idx) == C.shape[0], f"{rel}: N mismatch"

        # hiconf mask, exactly as the reference built it
        X13 = npz["feature_matrix"]
        scores = model["clf"].predict_proba(
            model["scaler"].transform(X13))[:, 1]
        if rel in oof_by_sess:
            for full_idx, v in oof_by_sess[rel].items():
                scores[full_idx] = v
            src = "oof"
        else:
            src = "deployed"
        hi = scores[review_idx] >= HICONF

        fps = footprints_from_sparse(A, d1, d2)
        X = features.compute_v2b_features(C, fps, Cn, hi)

        ref = pinned[key + "__X"]
        ok = np.allclose(X, ref, rtol=RTOL, atol=0)
        md = float(np.max(np.abs(X - ref))) if X.size else 0.0
        if md > worst[0]:
            bad_col = int(np.unravel_index(np.argmax(np.abs(X - ref)),
                                           X.shape)[1]) if X.size else -1
            worst = (md, rel, names[bad_col] if bad_col >= 0 else "-")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            per_col = np.max(np.abs(X - ref), axis=0)
            bad = {names[j]: float(per_col[j])
                   for j in range(len(names)) if per_col[j] > 0}
            print(f"  FAIL {rel} (hiconf={src}): max|diff| per bad col {bad}")

    print(f"Phase 1: {n_pass} pass, {n_fail} fail; "
          f"worst max|diff| = {worst[0]:.3g} ({worst[1]}, {worst[2]})")
    return n_fail == 0


def find_pending(limit=2):
    out = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            if ((sd / "ROIs_candidates.jpg").exists()
                    and (sd / "candidate_features.npz").exists()
                    and (sd / "A.txt").exists()
                    and (sd / "spatial_footprints.mat").exists()
                    and not (sd / "labels.mat").exists()
                    and not (sd / "ROIs.jpg").exists()):
                out.append(sd)
                if len(out) >= limit:
                    return out
    return out


def phase2_loader_orientation():
    pend = find_pending()
    if not pend:
        print("Phase 2: NO pending session available — orientation unverified!")
        return False
    print(f"Phase 2: loader-orientation check on {len(pend)} pending session(s)")
    # A.txt is dlmwrite text (~5 significant digits) while
    # spatial_footprints.mat holds full doubles, so tiny round-off diffs are
    # expected under the CORRECT (column-major) reshape.  The invariant is
    # geometric: order="F" must agree to text precision while order="C"
    # (transposed geometry) must not.
    ok_all = True
    for sd in pend:
        fps_loader = features.load_spatial(sd)          # (N, H, W)
        A = np.loadtxt(str(sd / "A.txt"))               # (pixels, N)
        n = A.shape[1]
        d1, d2 = fps_loader.shape[1], fps_loader.shape[2]
        assert fps_loader.shape[0] == n, f"{sd.name}: N mismatch"
        scale = float(np.max(np.abs(fps_loader))) or 1.0
        md_f = md_c = 0.0
        for k in range(n):
            img_f = A[:, k].reshape((d1, d2), order="F")
            img_c = A[:, k].reshape((d1, d2), order="C")
            md_f = max(md_f, float(np.max(np.abs(img_f - fps_loader[k]))))
            md_c = max(md_c, float(np.max(np.abs(img_c - fps_loader[k]))))
        traces = features.load_traces(sd)
        ok = (md_f / scale < 1e-3 and md_c > 100 * md_f
              and traces.shape[0] == n)
        ok_all &= ok
        print(f"  {sd.parent.name}/{sd.name}: N={n}, rel diff "
              f"order-F={md_f/scale:.2e} vs order-C={md_c/scale:.2e}, "
              f"C_raw rows={traces.shape[0]} -> {'OK' if ok else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    p1 = phase1_v2b_parity()
    p2 = phase2_loader_orientation()
    print(f"\nRESULT: phase1={'PASS' if p1 else 'FAIL'}  "
          f"phase2={'PASS' if p2 else 'FAIL'}")
    sys.exit(0 if (p1 and p2) else 1)
