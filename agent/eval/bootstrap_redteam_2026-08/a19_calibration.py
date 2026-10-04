"""
a19_calibration.py -- attack #19: is a pooled DG model usable at a fixed T?

On attack #18's DG_AL OOF vectors (fixtures/a18_oof.npz): the score-scale
shift between Cond A (own-only) and Cond B (pooled) -- quantile map, junk
caught at T=0.05 vs at matched false-AR per seed -- and a leak-free
calibrator: isotonic and Platt fitted INSIDE the training folds (nested
grouped CV on the training rows' own OOF) and applied to the held-out DG rows,
then the operating point at false-AR <= 1% / <= 3% vs own-only.  DG_AL runs
THRESHOLD_OVERRIDE = 0 today, so ranking is the operative metric until a
human calibrates a threshold.  Read-only.
"""
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import rt_lib as rt

T = 0.05


def op_at_far(scores, y, far_max):
    """Largest threshold with false-AR <= far_max -> (T, junk caught %)."""
    pos, neg = y == 1, y == 0
    grid = np.arange(0.001, 0.9, 0.001)
    best = None
    for t in grid:
        far = (scores[pos] < t).mean() * 100
        if far <= far_max:
            best = (float(t), float((scores[neg] < t).mean() * 100))
    return best or (None, 0.0)


def nested_calibrated(pools_oofB, y, groups, seed, kind):
    """Calibrate Cond B scores with a mapping fitted on OTHER groups' OOF scores
    (leak-free: the row being calibrated never contributes to its own map)."""
    out = np.full(len(y), np.nan)
    ug = np.unique(groups)
    if len(ug) >= 3:
        splits = list(StratifiedGroupKFold(n_splits=min(5, len(ug)), shuffle=True, random_state=seed).split(pools_oofB[:, None], y, groups))
    else:
        splits = [(np.flatnonzero(groups != k), np.flatnonzero(groups == k)) for k in ug]
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        if kind == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip").fit(pools_oofB[tr], y[tr])
            out[te] = m.predict(pools_oofB[te])
        else:
            lr = LogisticRegression().fit(np.log(pools_oofB[tr] / (1 - pools_oofB[tr] + 1e-9) + 1e-9)[:, None], y[tr])
            out[te] = lr.predict_proba(np.log(pools_oofB[te] / (1 - pools_oofB[te] + 1e-9) + 1e-9)[:, None])[:, 1]
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    f = np.load(rt.FIXTURES / "a18_oof.npz", allow_pickle=True)
    res = {"groupings": {}}
    for grouping in ("session", "animal", "fov"):
        y = f[f"DG_AL_{grouping}_y"]
        groups = f[f"DG_AL_{grouping}_groups"]
        g = {}
        for sw in ("1.0", "0.3"):
            A = f[f"DG_AL_{grouping}_oofA_{sw}"]
            B = f[f"DG_AL_{grouping}_oofB_{sw}"]
            ok = ~np.isnan(A[0])
            yy, gg = y[ok], groups[ok]
            qa = np.percentile(A[:, ok], [10, 25, 50, 75, 90], axis=1).mean(axis=1)
            qb = np.percentile(B[:, ok], [10, 25, 50, 75, 90], axis=1).mean(axis=1)
            junk_A = [rt.far_junk(a[ok], yy, T)[1] for a in A]
            junk_B = [rt.far_junk(b[ok], yy, T)[1] for b in B]
            far_A = [rt.far_junk(a[ok], yy, T)[0] for a in A]
            far_B = [rt.far_junk(b[ok], yy, T)[0] for b in B]
            mop = rt.matched_operating_point(A[:, ok], B[:, ok], yy, T)
            ops = {"A": {"far1": [], "far3": []}, "B_raw": {"far1": [], "far3": []},
                   "B_isotonic": {"far1": [], "far3": []}, "B_platt": {"far1": [], "far3": []}}
            aucs = {"A": [], "B_raw": [], "B_isotonic": [], "B_platt": []}
            for si, seed in enumerate(rt.SEEDS):
                cal = {"A": A[si, ok], "B_raw": B[si, ok],
                       "B_isotonic": nested_calibrated(B[si, ok], yy, gg, seed, "isotonic"),
                       "B_platt": nested_calibrated(B[si, ok], yy, gg, seed, "platt")}
                for k, s in cal.items():
                    m = ~np.isnan(s)
                    aucs[k].append(rt.auc(yy[m], s[m]))
                    for fm, key in ((1.0, "far1"), (3.0, "far3")):
                        ops[k][key].append(op_at_far(s[m], yy[m], fm)[1])
            g[sw] = {"score_quantiles_A_p10_p90": [float(x) for x in qa], "score_quantiles_B_p10_p90": [float(x) for x in qb],
                     "junk_at_0.05": {"A": rt.summarize(junk_A), "B": rt.summarize(junk_B)},
                     "far_at_0.05": {"A": rt.summarize(far_A), "B": rt.summarize(far_B)},
                     "junk_B_at_matched_far": mop["junk_new_at_matched_far_mean"], "junk_A_ref": mop["junk_ref_mean"],
                     "auc": {k: rt.summarize(v) for k, v in aucs.items()},
                     "junk_at_far_le_1pct": {k: rt.summarize(v["far1"]) for k, v in ops.items()},
                     "junk_at_far_le_3pct": {k: rt.summarize(v["far3"]) for k, v in ops.items()}}
            e = g[sw]
            rt.log(f"  [{grouping:7s} w={sw}] junk@0.05 A {e['junk_at_0.05']['A']['mean']:.1f}% B {e['junk_at_0.05']['B']['mean']:.1f}% "
                   f"(FAR A {e['far_at_0.05']['A']['mean']:.2f}% B {e['far_at_0.05']['B']['mean']:.2f}%) | junk at matched FAR "
                   f"{e['junk_A_ref']:.1f}% -> {e['junk_B_at_matched_far']:.1f}% | junk@FAR<=1%: A {e['junk_at_far_le_1pct']['A']['mean']:.1f} "
                   f"B_raw {e['junk_at_far_le_1pct']['B_raw']['mean']:.1f} iso {e['junk_at_far_le_1pct']['B_isotonic']['mean']:.1f} "
                   f"platt {e['junk_at_far_le_1pct']['B_platt']['mean']:.1f} | AUC A {e['auc']['A']['mean']:.4f} B {e['auc']['B_raw']['mean']:.4f} "
                   f"iso {e['auc']['B_isotonic']['mean']:.4f}")
        res["groupings"][grouping] = g
    a = res["groupings"]["animal"]["0.3"]
    res["usable"] = ("usable-with-calibration" if (a["junk_at_far_le_1pct"]["B_isotonic"]["mean"] >= a["junk_at_far_le_1pct"]["A"]["mean"] - 2
                                                  and a["auc"]["B_isotonic"]["mean"] >= a["auc"]["A"]["mean"])
                     else "not-usable-at-fixed-T" if a["junk_at_0.05"]["B"]["mean"] < 0.5 * a["junk_at_0.05"]["A"]["mean"]
                     else "inconclusive")
    res["verdict"] = "PASS"
    res["criterion"] = ("descriptive: the raw pooled scale collapse at T=0.05 is quantified; 'usable-with-calibration' if a "
                        "leak-free isotonic map restores junk-caught at false-AR <= 1% to within 2 pp of own-only and keeps AUC; "
                        "n = DG_AL reals (limiting)")
    rt.log(f"usable: {res['usable']}")
    rt.write_result("a19", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
