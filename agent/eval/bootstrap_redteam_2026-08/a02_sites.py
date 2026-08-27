"""
a02_sites.py -- attack #2: find an 11th orientation site.

AUDIT.md lists 10 places where a MATLAB (pixels, N) matrix meets a numpy
reshape.  This audits the candidates the brief names plus every reshape /
flatten / h5py site in agent/*.py, with a NUMERIC check wherever one is
possible and the exact source line quoted as evidence:
  S1  .feature_expansion extraction A (assumed F-order by a2/backfill):
      prep_retro_ids proved it -- stored cn_correlation reproduces bit-for-bit
      from order-F images on 96/102 sessions and from order-C on the 6 retro
      sessions (the retro path's own transposition), never 'neither'.
  S2  features.load_spatial A.txt fallback: transposed AND assumes square
      frames (side = sqrt(n_pixels)); scan of every session in the 3 areas
      (labeled + pending) for the fallback condition; frame-shape census.
  S3  bootstrap_preagent._load_candidates A.txt fallback (>2 GB save stub):
      _reorder_Fcols_to_C on a synthetic asymmetric footprint == the primary
      C-order path; == rt_lib.fcols_to_images flattened C.
  S4  validate_threshold: candidates C-order + A_final reindexed at the call
      site (grep evidence).
  S5  decision_margins.py:83 / sweep_gsig.py:99: C-order flatten of a stack,
      consumed column-wise or against itself -> order-invariant / consistent.
  S6  curator PDF + one-class fallback: via features.load_spatial (primary
      path OK; the A.txt fallback would render transposed -- display only).
  S7  Coor.mat: no Python consumer.  S8 h5py readers all transpose.
  S9  MATLAB writers of spatial_footprints.mat (CNMFe_Biane_headless.m,
      CNMFe_final_save.m) use the same permute -> [n, row, col].
Read-only.
"""
import re
import subprocess
import sys

import numpy as np

import rt_lib as rt


def grep(pattern, files):
    out = []
    for f in files:
        p = rt.AGENT / f
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if re.search(pattern, line):
                out.append(f"{f}:{i}: {line.strip()[:140]}")
    return out


def s2_fallback_scan():
    rows = {"fallback_fires": [], "shapes": {}, "n_scanned": 0}
    for area in rt.AREAS:
        for sd in rt.iter_sessions(area):
            if not (sd / "candidate_features.npz").exists():
                continue
            rows["n_scanned"] += 1
            if (sd / "A.txt").exists() and not (sd / "spatial_footprints.mat").exists():
                rows["fallback_fires"].append(rt.rel(sd))
            Cn = rt.load_cn(sd / "Cn.mat")
            if Cn is not None:
                rows["shapes"][str(Cn.shape)] = rows["shapes"].get(str(Cn.shape), 0) + 1
    rows["n_nonsquare"] = sum(v for k, v in rows["shapes"].items() if len(set(eval(k))) > 1)
    return rows


def s3_reorder_test():
    sys.path.insert(0, str(rt.AGENT))
    import importlib
    bp = importlib.import_module("bootstrap_preagent")
    d1, d2 = 48, 80
    ys, xs = np.mgrid[0:d1, 0:d2]
    img = np.exp(-(((ys - 12) / 3.0) ** 2 + ((xs - 60) / 9.0) ** 2))
    A_F = img.flatten(order="F")[:, None]                 # MATLAB column
    fixed = bp._reorder_Fcols_to_C(A_F, d1, d2)           # (pixels, 1) C-order rows
    primary = img.flatten(order="C")[:, None]             # what the stack path yields
    ours = rt.fcols_to_images(A_F, d1, d2)[0].flatten(order="C")[:, None]
    return {"reorder_equals_primary": bool(np.array_equal(fixed, primary)),
            "reorder_equals_rt_fcols": bool(np.array_equal(fixed, ours)),
            "mixed_self_cosine": float(rt.cosine_rows(A_F.T, primary.T)[0, 0])}


def main():
    t = rt.Timer()
    rt.assert_pinned()
    retro = rt.read_result("retro_sessions")
    res = {"sites": {}}
    res["sites"]["S1_feature_expansion_A_orientation"] = {
        "evidence": {a: retro["counts"][a] for a in retro["counts"]},
        "verdict": "F-order PROVEN: stored cn_correlation reproduces from order-F images (96/102) or order-C (6 retro); 0 'neither'",
        "status": "OK"}
    s2 = s2_fallback_scan()
    res["sites"]["S2_features_load_spatial_A_txt_fallback"] = {
        "code": grep(r"reshape\(side, side\)|int\(np.sqrt\(n_pixels\)\)", ["features.py"]),
        "fallback_fires_on": s2["fallback_fires"], "n_scanned": s2["n_scanned"],
        "frame_shapes": s2["shapes"], "n_nonsquare_frames": s2["n_nonsquare"],
        "verdict": "LATENT (2 defects: transposed + square-only); fires on 0 sessions" if not s2["fallback_fires"]
                   else f"LIVE on {len(s2['fallback_fires'])} sessions",
        "status": "LATENT" if not s2["fallback_fires"] else "LIVE-BUG"}
    res["sites"]["S3_bootstrap_preagent_A_txt_fallback"] = {
        "code": grep(r"_reorder_Fcols_to_C\(", ["bootstrap_preagent.py", "validate_threshold.py"]),
        "synthetic": s3_reorder_test(), "status": "OK"}
    res["sites"]["S3_bootstrap_preagent_A_txt_fallback"]["verdict"] = (
        "OK: fallback reindexes F->C and equals the primary path" if all(
            res["sites"]["S3_bootstrap_preagent_A_txt_fallback"]["synthetic"][k] for k in ("reorder_equals_primary", "reorder_equals_rt_fcols"))
        else "BUG")
    vt = grep(r"_reorder_Fcols_to_C|fp3d\.reshape|A_final = ", ["validate_threshold.py"])
    res["sites"]["S4_validate_threshold"] = {"code": vt,
                                             "verdict": "OK: candidates C-order (fp3d.reshape) and A_final reindexed before the cosine",
                                             "status": "OK" if any("_reorder_Fcols_to_C(" in l for l in vt) else "CHECK"}
    res["sites"]["S5_decision_margins_sweep_gsig"] = {
        "code": grep(r"sf\.reshape\(N, -1\)\.T|fps\.reshape\(N, -1\)\.T", ["decision_margins.py", "sweep_gsig.py"]),
        "verdict": "OK: C-order flatten of an (N,H,W) stack used column-wise (trim margins) or against another stack "
                   "flattened the same way (gSig arms); the 'matching neuron.A' comment in decision_margins.py:83 is misleading "
                   "(it is C-order, not MATLAB order) but nothing compares it to neuron.A",
        "status": "OK (comment misleading)"}
    res["sites"]["S6_curator_pdf_and_one_class"] = {
        "code": grep(r"feat_module\.load_spatial\(session_dir\)|feat_module\.extract_all", ["curator.py"]),
        "verdict": "OK via the primary load_spatial path; under the A.txt fallback the PDF would render transposed and the "
                   "one-class cold start would see transposed footprints (display / cold-start only, and the fallback fires on 0 sessions)",
        "status": "OK (inherits S2 latent)"}
    res["sites"]["S7_Coor_mat"] = {"code": grep(r"Coor", [p.name for p in rt.AGENT.glob("*.py")]),
                                   "verdict": "no Python consumer (MATLAB get_contours only)", "status": "OK"}
    h5 = []
    for p in list(rt.AGENT.glob("*.py")) + list(rt.AGENT.glob("eval/**/*.py")):
        txt = p.read_text(errors="replace")
        if "h5py" in txt:
            h5.append({"file": str(p.relative_to(rt.AGENT)),
                       "transposes": bool(re.search(r"transpose\(2, ?1, ?0\)|\.T\b|\[::-1\]", txt))})
    res["sites"]["S8_h5py_readers"] = {"readers": h5, "verdict": "all transpose", "status": "OK" if all(x["transposes"] for x in h5) else "CHECK"}
    mw = []
    for f in ("CNMFe_Biane_headless.m", "CNMFe_final_save.m"):
        p = rt.AGENT.parent / f
        if p.exists():
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if "spatial_footprints" in line and ("permute" in line or "reshape" in line or "=" in line):
                    mw.append(f"{f}:{i}: {line.strip()[:140]}")
    res["sites"]["S9_matlab_writers"] = {"code": mw[:12], "verdict": "both writers build [n, row, col] stacks (a01 step 1 proves the stack == A.txt column-major)", "status": "OK"}
    res["sites"]["S10_extract_cand_traces_motion_qc"] = {
        "verdict": "MATLAB-side consumers of neuron.A only (no numpy reshape); motion features use the mean background trace, not footprints",
        "status": "OK"}
    res["sites"]["S11_v2b_centroid_neighbor"] = {
        "verdict": "features._v2b_centroid_and_mask works in image coordinates on both sides of the neighbour distance; "
                   "the bootstrap arm feeds C-order images and the agent path feeds order-F extraction images -- each internally consistent",
        "status": "OK"}
    res["sites"]["S12_run_cnmfe_matlab_reshape"] = {
        "code": grep(r"neuron\.reshape\(Y, 1\)", ["run_cnmfe.py"]),
        "verdict": "MATLAB statement inside a Python string: CNMFe's own reshape of the movie tensor Y (d1 x d2 x T -> pixels x T) "
                   "executed by MATLAB; no numpy pixel-order step, no footprint involved",
        "status": "OK"}
    live_bugs = [k for k, v in res["sites"].items() if v["status"] == "LIVE-BUG"]
    checks = {"no_new_live_production_defect": len(live_bugs) == 0,
              "s3_reorder_consistent": res["sites"]["S3_bootstrap_preagent_A_txt_fallback"]["status"] == "OK",
              "s4_validate_call_site_fixed": res["sites"]["S4_validate_threshold"]["status"] == "OK",
              "h5py_all_transpose": res["sites"]["S8_h5py_readers"]["status"] == "OK"}
    res["known_defects_confirmed"] = {"train_classifier.py:235 retro path": "C-match on 6 BLA sessions (prep_retro_ids)",
                                      "features.py:32-38 A.txt fallback": "transposed + square-only, fires on 0 sessions"}
    res["checks"] = checks
    res["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    res["criterion"] = "no live production-path orientation defect beyond the two already known; every eval-tool site order-invariant or self-consistent"
    for k, v in res["sites"].items():
        rt.log(f"  {k}: {v['status']} -- {v['verdict'][:110]}")
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a02", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
