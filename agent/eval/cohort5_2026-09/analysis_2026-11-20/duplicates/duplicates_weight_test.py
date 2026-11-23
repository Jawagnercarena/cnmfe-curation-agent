"""
duplicates_weight_test.py -- does it matter that second copies of a kept cell
train as "not a cell"?

A reviewer who sees two candidates on one cell keeps one and deletes the other.
The deleted copy enters training as an agent-weight (5.0) negative although it
looks like a cell.  train_classifier.py zero-weights such duplicates for
bootstrap sessions only.  This compares, on the vCA1 pool of 2026-11-20
(same harness, folds and seeds as threshold/threshold_oof.py, session-grouped):

  A  as deployed: duplicate deletes are negatives at the agent weight
  B  duplicate deletes get training weight 0 (they stay in the test folds)

duplicate = a reviewed delete, not motion-tagged, whose support (> 0.2 x max)
has IoU >= 0.5 with a candidate KEPT in the same session.  The flag needs every
candidate's footprint, i.e. <session>/.pre_ingest/spatial_footprints.mat, which
exists only for sessions ingested since 2026-10-12; other sessions are
unflagged in both arms.

Read-only.  Usage: python duplicates_weight_test.py [--seeds 8] [--threads 8] [--cache DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "bootstrap_redteam_2026-08"))
import rt_lib as rt  # noqa: E402

AREA = "vCA1"
WIDTH = 35
H = W = 512
TS = [0.01, 0.02, 0.03, 0.04]


def dup_flags(rec, cache):
    """bool (n_cand,) or None when the all-candidate footprints are not available."""
    sd = rec["session_dir"]
    n = rec["n_cand"]
    p = sd / ".pre_ingest" / "spatial_footprints.mat"
    cp = cache / (sd.name + "_masks.npz") if cache else None
    if cp is not None and cp.exists():
        z = np.load(str(cp))
        c, yy, xx = z["c"], z["y"], z["x"]
    elif p.exists():
        A = sio.loadmat(str(p))["spatial_footprints"]
        if A.shape != (n, H, W):
            return None
        mx = A.reshape(n, -1).max(1)
        c, yy, xx = np.nonzero(A > (0.2 * mx[:, None, None]))
        if cp is not None:
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(cp), c=c.astype(np.int32), y=yy.astype(np.int16), x=xx.astype(np.int16))
    else:
        return None
    M = sp.csr_matrix((np.ones(len(c), np.float32), (c, yy.astype(int) * W + xx.astype(int))), shape=(n, H * W))
    d = sio.loadmat(str(sd / "labels.mat"))
    lab = d["labels"].ravel().astype(int)
    mot = d["motion_delete"].ravel().astype(int) if "motion_delete" in d else np.zeros(len(lab), int)
    rows = np.where(rec["reviewed"])[0] if len(lab) != n else np.arange(n)
    full = np.full(n, -1)
    full[rows] = lab
    fmot = np.zeros(n, int)
    fmot[rows] = mot
    keep = np.where(full == 1)[0]
    dl = np.where((full == 0) & (fmot == 0))[0]
    out = np.zeros(n, bool)
    if len(keep) and len(dl):
        inter = (M[dl] @ M[keep].T).toarray()
        J = inter / np.maximum(np.asarray(M[dl].sum(1)) + np.asarray(M[keep].sum(1)).T - inter, 1)
        out[dl[J.max(1) >= 0.5]] = True
    return out


def run(X, y, g, w_ag, Xb, yb, wb, seed, n_jobs):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, g):
        X_tr = np.vstack([X[tr], Xb])
        y_tr = np.concatenate([y[tr], yb])
        w_tr = np.concatenate([w_ag[tr], wb])
        oof[te] = rt.fit_predict(X_tr, y_tr, w_tr, X[te], True, 42, n_jobs=n_jobs)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    seeds = rt.SEEDS[:a.seeds]
    cache = Path(a.cache) if a.cache else None

    recs = rt.load_pool(AREA, width=WIDTH)
    cv, rest, bs = rt.split_pool(recs)
    X, y, g, rev, animals, names = rt.stack_agent(cv)
    Xb, yb, wb = rt.stack_bootstrap(bs, width=WIDTH)
    flags, have = [], []
    for r in cv:
        f = dup_flags(r, cache)
        have.append(f is not None)
        flags.append(f if f is not None else np.zeros(r["n_cand"], bool))
    dup = np.concatenate(flags)
    has = np.concatenate([[h] * r["n_cand"] for h, r in zip(have, cv)])
    print("pool: %d sessions (%d agent testable, %d bootstrap); agent rows %d, kept %d" % (
        len(recs), len(cv), len(bs), len(y), int(y.sum())))
    print("agent sessions with all-candidate footprints (flag computable): %d of %d; rows %d" % (
        sum(have), len(cv), int(has.sum())))
    print("duplicate deletes flagged: %d = %.1f%% of the %d reviewed deletes in those sessions (%.1f%% of all %d agent negatives)" % (
        int(dup.sum()), 100.0 * dup.sum() / max(((y == 0) & rev & has).sum(), 1), int(((y == 0) & rev & has).sum()),
        100.0 * dup.sum() / max((y == 0).sum(), 1), int((y == 0).sum())))
    by_an = {}
    for an in sorted(set(animals[has])):
        m = has & (animals == an)
        by_an[an] = (int((dup & m).sum()), int(((y == 1) & m).sum()))
    print("   per animal (duplicates / kept): " + ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in by_an.items()))

    agw = float(rt.AGENT_WEIGHT_OVERRIDE[AREA])
    wA = np.full(len(y), agw)
    wB = wA.copy()
    wB[dup] = 0.0
    res = {"seeds": seeds, "n_dup": int(dup.sum())}
    rows_out = []
    for s in seeds:
        oA = run(X, y, g, wA, Xb, yb, wb, s, a.threads)
        oB = run(X, y, g, wB, Xb, yb, wb, s, a.threads)
        r = {"seed": s}
        for tag, o in (("A", oA), ("B", oB)):
            r["auc_rev_" + tag] = rt.auc(y, o, rev)
            r["auc_rev_nodup_" + tag] = rt.auc(y, o, rev & ~dup)
            r["auc_rev_has_nodup_" + tag] = rt.auc(y, o, rev & has & ~dup)
            r["dup_median_" + tag] = float(np.median(o[dup])) if dup.any() else float("nan")
            r["kept_median_" + tag] = float(np.median(o[(y == 1) & has]))
            for T in TS:
                r["miss_%s_%.2f" % (tag, T)] = int(((o < T) & (y == 1)).sum())
                r["junk_%s_%.2f" % (tag, T)] = 100.0 * ((o < T) & (y == 0) & ~dup).sum() / max(((y == 0) & ~dup).sum(), 1)
                r["dupbelow_%s_%.2f" % (tag, T)] = int(((o < T) & dup).sum())
        rows_out.append(r)
    res["per_seed"] = rows_out
    n_pos = int(y.sum())

    def col(k):
        return np.array([r[k] for r in rows_out], float)

    print("\nsession-grouped OOF, %d seeds, same folds in both arms (mean; B - A range over seeds)" % len(seeds))
    print("  %-52s %10s %10s %22s" % ("", "A deployed", "B dup w=0", "B - A (min .. max)"))
    for label, k in (("AUC reviewed rows (duplicates count as deletes)", "auc_rev_"),
                     ("AUC reviewed rows, duplicates left out", "auc_rev_nodup_"),
                     ("AUC, flagged sessions only, duplicates left out", "auc_rev_has_nodup_"),
                     ("median score of the duplicate deletes", "dup_median_"),
                     ("median score of kept cells (flagged sessions)", "kept_median_")):
        A_, B_ = col(k + "A"), col(k + "B")
        print("  %-52s %10.4f %10.4f %+10.4f .. %+.4f" % (label, A_.mean(), B_.mean(), (B_ - A_).min(), (B_ - A_).max()))
    for T in TS:
        A_, B_ = col("miss_A_%.2f" % T), col("miss_B_%.2f" % T)
        print("  %-52s %9.2f%% %9.2f%% %+10.1f .. %+.1f cells of %d" % (
            "false auto-reject at %.2f" % T, 100 * A_.mean() / n_pos, 100 * B_.mean() / n_pos,
            (B_ - A_).min(), (B_ - A_).max(), n_pos))
    for T in TS:
        A_, B_ = col("junk_A_%.2f" % T), col("junk_B_%.2f" % T)
        print("  %-52s %9.1f%% %9.1f%% %+10.2f .. %+.2f pts" % (
            "junk auto-rejected at %.2f (non-duplicate deletes)" % T, A_.mean(), B_.mean(), (B_ - A_).min(), (B_ - A_).max()))
    for T in TS:
        A_, B_ = col("dupbelow_A_%.2f" % T), col("dupbelow_B_%.2f" % T)
        print("  %-52s %10.1f %10.1f   of %d" % ("duplicates scored below %.2f" % T, A_.mean(), B_.mean(), int(dup.sum())))
    (HERE / "duplicates_weight_test.json").write_text(json.dumps(res, indent=1), encoding="ascii")
    print("wrote duplicates_weight_test.json")


if __name__ == "__main__":
    main()
