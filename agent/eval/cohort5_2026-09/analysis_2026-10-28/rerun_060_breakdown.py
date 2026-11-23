"""
rerun_060_breakdown.py -- what the min_corr 0.60 re-run of 090526-pnb105 (2026-10-28,
168 candidates, curated at the deployed 0.01) added over its 0.70 run (44 candidates,
archived by copy in <session>/.prev_mincorr070_2026-10-28/).

Per candidate: Cn peak = max of the 3x3-mean 512-res Cn over the footprint core
(A > 0.5 max, Analysis A definition), score = deployed vCA1 joblib (scaler + clf) on the saved 35-col
feature matrix; auto_rejected = the curator's flag as saved in candidate_features.npz.
The 44 old candidates are matched to the new set by support Jaccard (>= 0.5).

Read-only on the session folder.  Output: rerun_060_breakdown.txt next to this file.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[2]
sys.path.insert(0, str(AGENT))
import local_config  # noqa: E402

NAME = "AVG5x-TSeries-090526-pnb105-520um-32z-000"
SD = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew" / NAME
PREV = SD / ".prev_mincorr070_2026-10-28"
MODEL = AGENT / "model" / "vCA1" / "classifier.joblib"
BANDS = [(-1.0, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 9.0)]


def load_set(d):
    A = np.asarray(loadmat(str(d / "spatial_footprints.mat"))["spatial_footprints"], dtype=float)
    z = np.load(str(d / "candidate_features.npz"), allow_pickle=True)
    X = np.asarray(z["feature_matrix"], dtype=float)
    n = A.shape[0]
    raw = np.asarray(z["auto_rejected"]).ravel()
    if raw.size == n and raw.dtype == bool:
        ar = raw
    else:                                   # the curator saves the rejected candidates' indices
        idx = raw.astype(int)
        assert idx.size == 0 or idx.max() < n, (idx.min(), idx.max(), n)
        ar = np.zeros(n, dtype=bool)
        ar[idx] = True
    assert A.shape[0] == X.shape[0] == ar.size, (A.shape, X.shape, ar.shape)
    return A, X, ar


def supports(A):
    out = np.zeros(A.shape, dtype=bool)
    for i, a in enumerate(A):
        m = a.max()
        if m > 0:
            out[i] = a > 0.3 * m
    return out


def cn_peak(A, Cn):
    """Analysis A's definition (kept_by_cn_band.py): max of the 3x3-mean Cn over the footprint core (A > 0.5 max)."""
    sm = uniform_filter(Cn, 3)
    out = np.full(A.shape[0], np.nan)
    for i, a in enumerate(A):
        m = a.max()
        if m > 0:
            out[i] = sm[a > 0.5 * m].max()
    return out


def band_label(lo, hi):
    if lo < 0:
        return "< 0.60"
    if hi > 1:
        return ">= 0.80"
    return f"{lo:.2f}-{hi:.2f}"


def main():
    lines = []
    say = lines.append
    o = joblib.load(str(MODEL))
    say(f"model: {MODEL.name}  n_sessions {o.get('n_sessions')}  n_features {o.get('n_features')}  "
        f"reject_threshold {o.get('reject_threshold')}  feature_version {o.get('feature_version')}")
    Cn = np.nan_to_num(np.asarray(loadmat(str(SD / "Cn.mat"))["Cn"], dtype=float))
    Cn_prev = np.nan_to_num(np.asarray(loadmat(str(PREV / "Cn.mat"))["Cn"], dtype=float))
    say(f"Cn live {Cn.shape} vs archived {Cn_prev.shape}: "
        f"{'identical' if Cn.shape == Cn_prev.shape and np.allclose(Cn, Cn_prev) else 'DIFFERENT'}")

    A_new, X_new, ar_new = load_set(SD)
    A_old, X_old, ar_old = load_set(PREV)
    s_new = o["clf"].predict_proba(o["scaler"].transform(X_new))[:, 1]
    s_old = o["clf"].predict_proba(o["scaler"].transform(X_old))[:, 1]
    thr = float(o["reject_threshold"])
    say(f"new set: {len(s_new)} candidates, curator auto_rejected {ar_new.sum()}, rescored < {thr}: {(s_new < thr).sum()}")
    say(f"old set: {len(s_old)} candidates, curator auto_rejected {ar_old.sum()}, rescored < {thr}: {(s_old < thr).sum()}")

    sup_new, sup_old = supports(A_new), supports(A_old)
    cn_new, cn_old = cn_peak(A_new, Cn), cn_peak(A_old, Cn)

    # match old -> new by support Jaccard
    flat_new = sup_new.reshape(len(sup_new), -1)
    flat_old = sup_old.reshape(len(sup_old), -1)
    inter = flat_old.astype(np.int32) @ flat_new.astype(np.int32).T
    union = flat_old.sum(1)[:, None] + flat_new.sum(1)[None, :] - inter
    jac = inter / np.maximum(union, 1)
    best = jac.argmax(1)
    bestj = jac[np.arange(len(best)), best]
    matched = bestj >= 0.5
    say("")
    say(f"old 44 recovered in the new set (support Jaccard >= 0.5): {matched.sum()}/{len(best)}; "
        f"Jaccard median {np.median(bestj):.2f}, min {bestj.min():.2f}")
    new_is_old = np.zeros(len(s_new), dtype=bool)
    new_is_old[best[matched]] = True
    say(f"new candidates not in the old set: {(~new_is_old).sum()}")

    say("")
    say("old candidates, old run (44 alone) vs new run (same cell among 168):")
    say("  old auto-rejected -> new auto-rejected: "
        f"{(ar_old & ar_new[best]).sum()} stay rejected, {(ar_old & ~ar_new[best]).sum()} now pass, "
        f"{(~ar_old & ar_new[best]).sum()} now rejected, {(~ar_old & ~ar_new[best]).sum()} stay in review")
    d = s_new[best] - s_old
    say(f"  score change (new - old): median {np.median(d):+.3f}, "
        f"p10 {np.percentile(d, 10):+.3f}, p90 {np.percentile(d, 90):+.3f}; "
        f">= 0.5 old {(s_old >= 0.5).sum()} -> new {(s_new[best] >= 0.5).sum()}")
    flipped = np.where(~ar_old & ar_new[best])[0]
    for i in flipped:
        say(f"    old #{i + 1}: Cn peak {cn_old[i]:.2f}, score {s_old[i]:.3f} -> {s_new[best[i]]:.3f} (new #{best[i] + 1})")

    say("")
    say("old set (44, seeded at 0.70) by the same Cn peak proxy: " + ", ".join(
        f"{band_label(lo, hi)} {((cn_old >= lo) & (cn_old < hi)).sum()}" for lo, hi in BANDS)
        + f"; min {np.nanmin(cn_old):.2f}, median {np.nanmedian(cn_old):.2f}")
    say("new set by Cn peak band (all 168):")
    say("  band       |   n | auto-rej | score med | >=0.5 | >=0.2 | 0.01-0.2 | of which new")
    for lo, hi in BANDS:
        m = (cn_new >= lo) & (cn_new < hi)
        if not m.any():
            say(f"  {band_label(lo, hi):10s} |   0 |")
            continue
        s = s_new[m]
        say(f"  {band_label(lo, hi):10s} | {m.sum():3d} | {ar_new[m].sum():8d} | {np.median(s):9.3f} | "
            f"{(s >= 0.5).sum():5d} | {(s >= 0.2).sum():5d} | {((s >= thr) & (s < 0.2)).sum():8d} | {(m & ~new_is_old).sum():3d}")
    m = ~new_is_old
    s = s_new[m]
    say("")
    say(f"the {m.sum()} added candidates: auto-rejected {ar_new[m].sum()}, score median {np.median(s):.3f}, "
        f">= 0.5: {(s >= 0.5).sum()}, 0.2-0.5: {((s >= 0.2) & (s < 0.5)).sum()}, "
        f"0.01-0.2: {((s >= thr) & (s < 0.2)).sum()}, < 0.01: {(s < thr).sum()}; "
        f"Cn peak median {np.nanmedian(cn_new[m]):.2f}, >= 0.70: {(cn_new[m] >= 0.70).sum()}")
    say(f"review set {(~ar_new).sum()}: old-set cells {(~ar_new & new_is_old).sum()}, added cells {(~ar_new & ~new_is_old).sum()}")
    say("")
    say("pilot precedent (LOG.md 2026-10-28 Analysis A): kept-rate for Cn peak 0.60-0.70 was 0/33 pooled over the 12 pilots "
        "(threshold-0 reviews); 0.70-0.75 1/74; 0.75-0.80 10/221; 0.80-0.85 35/407.")
    text = "\n".join(lines)
    print(text)
    (HERE / "rerun_060_breakdown.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
