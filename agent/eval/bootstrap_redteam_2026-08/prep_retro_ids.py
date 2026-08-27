"""
prep_retro_ids.py -> results/retro_sessions.json

Attack #4 identification (and attack #2's extraction-orientation check): for
every labeled agent session (BLA 79, vCA1 23) recompute cn_correlation for
the review-set rows from the .feature_expansion extraction A (MATLAB columns)
under BOTH reshapes -- order='F' (correct) and order='C' (the retro-path bug
at train_classifier.py:235) -- against the extraction's Cn and the session's
Cn.mat, and compare with the stored column 12 of the live npz.  A session is
  F-match : stored == order-F recomputation  (prospective, correct)
  C-match : stored == order-C recomputation  (retro-labeled, transposed)
  neither : investigate.
Also checks extraction Cn == session Cn.mat and that the extraction row count
equals the review set.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

TOL = 1e-6


def main():
    t = rt.Timer()
    rt.assert_pinned()
    out = {"sessions": {}, "counts": {}}
    for area in ("BLA", "vCA1"):
        rows = []
        for sd in rt.labeled_sessions(area):
            if rt.is_bootstrap(sd):
                continue
            r = rt.load_record(sd, area)
            ext = rt.load_extraction(area, r["name"])
            if ext is None:
                rows.append({"rel": r["name"], "status": "no_extraction"})
                continue
            rev_idx = np.flatnonzero(r["reviewed"])
            stored = r["X"][rev_idx, rt.CN_COL]
            A, d1, d2 = ext["A"], ext["d1"], ext["d2"]
            if A.shape[1] != len(rev_idx):
                rows.append({"rel": r["name"], "status": "N_mismatch", "ext_N": int(A.shape[1]),
                             "review_N": int(len(rev_idx))})
                continue
            cn_sess = rt.load_cn(sd / "Cn.mat")
            cn_ext = ext["Cn"]
            cn_same = bool(cn_sess is not None and cn_ext is not None and cn_sess.shape == cn_ext.shape
                           and np.max(np.abs(cn_sess - cn_ext)) == 0.0)
            imgF = rt.fcols_to_images(A, d1, d2)
            imgC = rt.ccols_to_images(A, d1, d2)
            def col(imgs, Cn):
                return np.array([rt.F.cn_features(imgs[i], Cn)["cn_correlation"] for i in range(len(imgs))])
            F_sess = col(imgF, cn_sess)
            C_sess = col(imgC, cn_sess)
            F_ext = col(imgF, cn_ext)
            dF, dC, dFe = (float(np.max(np.abs(stored - F_sess))), float(np.max(np.abs(stored - C_sess))),
                           float(np.max(np.abs(stored - F_ext))))
            if dF < TOL:
                status = "F-match"
            elif dC < TOL:
                status = "C-match"
            else:
                status = "neither"
            rows.append({"rel": r["name"], "status": status, "n_rows": int(len(r["y"])),
                         "n_reviewed": int(len(rev_idx)), "n_pos": int(r["y"].sum()),
                         "auto_rejected": int(len(r["auto_rejected"])),
                         "max_abs_diff_F_sessCn": dF, "max_abs_diff_C_sessCn": dC,
                         "max_abs_diff_F_extCn": dFe, "cn_ext_equals_session": cn_same,
                         "stored_vs_F_spearman": float(np.corrcoef(np.argsort(np.argsort(stored)),
                                                                   np.argsort(np.argsort(F_sess)))[0, 1]),
                         "frac_rows_changed_gt_0.1_if_corrected": float(np.mean(np.abs(stored - F_sess) > 0.1)),
                         "cv_eligible": bool(r["y"].sum() >= rt.MIN_POS)})
            rt.log(f"  {area} {r['name']}: {status}  |stored-F| {dF:.1e} |stored-C| {dC:.1e} "
                   f"cn_ext==sess {cn_same}  N {len(rev_idx)} pos {int(r['y'].sum())}")
        out["sessions"][area] = rows
        c = {s: sum(1 for x in rows if x["status"] == s) for s in ("F-match", "C-match", "neither", "N_mismatch", "no_extraction")}
        cm = [x for x in rows if x["status"] == "C-match"]
        out["counts"][area] = {**c, "C_match_rows": sum(x["n_rows"] for x in cm),
                               "C_match_pos": sum(x["n_pos"] for x in cm),
                               "C_match_cv_eligible": sum(1 for x in cm if x["cv_eligible"]),
                               "C_match_sessions": [x["rel"] for x in cm]}
        rt.log(f"{area}: {out['counts'][area]}")
    rt.write_result("retro_sessions", out, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
