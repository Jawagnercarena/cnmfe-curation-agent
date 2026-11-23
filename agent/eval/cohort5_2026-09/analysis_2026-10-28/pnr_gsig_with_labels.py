"""
pnr_gsig_with_labels.py -- Analyses B (min_pnr) and C (gSig/gSiz) of
docs/HANDOFF_2026-10-06_COHORT5_REASSESSMENT.md, scored against Taylor's labels.

Read-only.  12 cohort-5 pilot sessions reviewed at threshold 0.

B. PNR gate: per candidate, the peak of the full-movie PNR image (pnr.mat, 256 res,
   upsampled x2) inside the footprint core, minus the session's min_pnr (estimated
   on the 1000-frame pre-pass; the seeding applies it to the full-movie image).
   Kept-rate by margin band and by absolute PNR; kept cells within 1 and 2 units of
   the gate (a binding gate shows kept cells piling up at it).

C. gSig arms: the 2026-09-25 sweep arms (.sweep/<session>/gSig07, gSig11; gSiz =
   4 x gSig; each arm at its own estimated thresholds, read from
   agent/logs/sweep_<session>.log) vs the live 9/36 set.  Recall of Taylor's kept
   cells by each arm (cosine >= 0.45), restricted to kept cells whose live Cn peak
   is >= 0.85 so the arms' higher seeds do not confound the comparison; arm
   candidates with no live counterpart; kept-cell core radius per animal at the
   processing resolution (ssub 2) next to gSig.

Usage: python pnr_gsig_with_labels.py
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from local_config import DATA_PARENT   # noqa: E402

TASK = DATA_PARENT / "vCA1" / "6odorDualDiffRew"
SWEEP = DATA_PARENT / "vCA1" / ".sweep"
LOGS = HERE.parents[2] / "logs"
PILOTS = [
    "AVG5x-TSeries-082626-pnb102-479um-3z-000", "AVG5x-TSeries-082726-pnb102-497um-3z-000",
    "AVG5x-TSeries-082626-pnb104-404um-3z-000", "AVG5x-TSeries-082726-pnb104-416um-3z-000",
    "AVG5x-TSeries-082626-pnb105-397um-35z-000", "AVG5x-TSeries-082726-pnb105-391um-35z-000",
    "AVG5x-TSeries-082626-pnb106-205um-4z-000", "AVG5x-TSeries-082726-pnb106-201um-4z-000",
    "AVG5x-TSeries-082626-pnb109-398um-26z-000", "AVG5x-TSeries-082726-pnb109-405um-25z-000",
    "AVG5x-TSeries-082626-pnb99-399um-3z-000", "AVG5x-TSeries-082826-pnb99-436um-3z-000-000",
]
SET_DIR = {s: ".rerun_mincorr070_2026-10-06" for s in PILOTS}
SET_DIR["AVG5x-TSeries-082726-pnb105-391um-35z-000"] = ".rerun_mincorr060_2026-10-18"
MATCH = 0.45


def foot(p):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A


def cores(A):
    return [a > 0.5 * a.max() for a in A]


def norm_rows(A):
    X = A.reshape(len(A), -1)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def arm_params(s):
    """(min_corr, min_pnr) of the gSig07 and gSig11 arms from the sweep log (two Final lines, in that order)."""
    for cand in (LOGS / f"sweep_{s}.log", LOGS / f"sweep_cohort5_{s[14:27]}.log"):
        if cand.exists():
            t = cand.read_text(encoding="utf-8", errors="replace")
            m = re.findall(r"Final parameters: \{.*?'min_corr': ([0-9.]+), 'min_pnr': ([0-9.]+)", t)
            if len(m) >= 2:
                return {"gSig07": (float(m[0][0]), float(m[0][1])), "gSig11": (float(m[1][0]), float(m[1][1]))}
    return {}


def band(v, edges):
    return int(np.searchsorted(edges, v, side="right") - 1)


def tab(vals, kept, edges, labels):
    out = []
    for b in range(len(edges) - 1):
        m = np.array([band(v, edges) == b for v in vals])
        n, k = int(m.sum()), int(kept[m].sum())
        out.append("%s: %d/%d%s" % (labels[b], k, n, " (%.0f%%)" % (100.0 * k / n) if n else ""))
    return " | ".join(out)


def main():
    rows = []
    print("=== B. PNR gate (full-movie PNR peak in the footprint core minus the session's min_pnr)")
    print("session | min_pnr | kept/N | kept-rate by margin band | kept within 1 / 2 units of the gate | kept PNR median")
    M_EDGES = [-99, 0, 1, 2, 4, 8, 1e9]
    M_LAB = ["<0", "0-1", "1-2", "2-4", "4-8", ">=8"]
    for s in PILOTS:
        sd = TASK / s
        A = foot(sd / SET_DIR[s] / "spatial_footprints.mat")
        L = np.asarray(loadmat(str(sd / "labels.mat"))["labels"]).ravel()
        kept = (L == 1).astype(int)
        if kept.size != len(A):
            print("  %s: labels %d != candidates %d -- skipped" % (s, kept.size, len(A)))
            continue
        log = (sd / "agent_run.log").read_text(encoding="utf-8", errors="replace")
        mp = float(re.findall(r"min_pnr=(\d+(?:\.\d+)?)", log)[-1])
        pnr = np.nan_to_num(np.asarray(loadmat(str(sd / "pnr.mat"))["pnr"], dtype=float))
        if pnr.shape != (512, 512):
            pnr = np.kron(pnr, np.ones((2, 2)))
        Cn = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float))
        sm = ndi.uniform_filter(Cn, 3)
        C = cores(A)
        pk = np.array([pnr[c].max() for c in C])
        cn = np.array([sm[c].max() for c in C])
        rad = np.array([np.sqrt(c.sum() / np.pi) for c in C])
        margin = pk - mp
        animal = re.search(r"-(pnb\d+)-", s).group(1)
        rows.append({"s": s, "animal": animal, "A": A, "kept": kept, "pk": pk, "cn": cn, "rad": rad, "mp": mp, "margin": margin})
        k1 = int(kept[(margin >= 0) & (margin < 1)].sum())
        k2 = int(kept[(margin >= 0) & (margin < 2)].sum())
        print("  %s | %.1f | %d/%d | %s | %d / %d | %.1f" % (s[14:], mp, int(kept.sum()), len(A), tab(margin, kept, M_EDGES, M_LAB), k1, k2, np.median(pk[kept == 1])))
    allm = np.concatenate([r["margin"] for r in rows]); allk = np.concatenate([r["kept"] for r in rows]); allp = np.concatenate([r["pk"] for r in rows])
    print("POOLED margin bands: " + tab(allm, allk, M_EDGES, M_LAB))
    P_EDGES = [0, 6, 8, 10, 15, 20, 30, 1e9]
    P_LAB = ["<6", "6-8", "8-10", "10-15", "15-20", "20-30", ">=30"]
    print("POOLED absolute PNR bands: " + tab(allp, allk, P_EDGES, P_LAB))
    nk = int(allk.sum())
    print("kept cells within 1 unit of the gate: %d of %d (%.1f%%); within 2: %d (%.1f%%); kept PNR median %.1f, 5th pct %.1f; gates 5.7-8.4" % (
        int(allk[(allm >= 0) & (allm < 1)].sum()), nk, 100.0 * allk[(allm >= 0) & (allm < 1)].sum() / nk,
        int(allk[(allm >= 0) & (allm < 2)].sum()), 100.0 * allk[(allm >= 0) & (allm < 2)].sum() / nk,
        np.median(allp[allk == 1]), np.percentile(allp[allk == 1], 5)))
    print()
    print("=== C. gSig arms vs the live 9/36 set, scored on Taylor's kept cells")
    print("session | arm (min_corr/min_pnr) | arm N | recall of kept cells with live Cn >= 0.85 | recall of all kept | arm candidates with no live counterpart")
    agg = {"gSig07": [0, 0, 0, 0, 0], "gSig11": [0, 0, 0, 0, 0]}   # hits85, n85, hits_all, n_all, extras
    for r in rows:
        s = r["s"]; A = r["A"]; kept = r["kept"]; cn = r["cn"]
        params = arm_params(s)
        An = norm_rows(A)
        for arm in ("gSig07", "gSig11"):
            p = SWEEP / s / arm / "spatial_footprints.mat"
            if not p.exists():
                print("  %s | %s | no footprints" % (s[14:], arm))
                continue
            B = foot(p); Bn = norm_rows(B); S = Bn @ An.T          # arm x live
            best_live = S.max(0)                                   # per live candidate: best arm match
            k_all = kept == 1; k85 = k_all & (cn >= 0.85)
            hits_all = int((best_live[k_all] >= MATCH).sum()); hits85 = int((best_live[k85] >= MATCH).sum())
            extras = int((S.max(1) < MATCH).sum())
            a = agg[arm]; a[0] += hits85; a[1] += int(k85.sum()); a[2] += hits_all; a[3] += int(k_all.sum()); a[4] += extras
            mc, mp_ = params.get(arm, (float("nan"), float("nan")))
            print("  %s | %s (%.2f/%.1f) | %d | %d/%d | %d/%d | %d" % (s[14:], arm, mc, mp_, len(B), hits85, int(k85.sum()), hits_all, int(k_all.sum()), extras))
    for arm, a in agg.items():
        print("POOLED %s: recall of kept cells with Cn >= 0.85: %d/%d (%.1f%%); all kept: %d/%d (%.1f%%); extras without a live counterpart: %d over 12 sessions" % (
            arm, a[0], a[1], 100.0 * a[0] / max(a[1], 1), a[2], a[3], 100.0 * a[2] / max(a[3], 1), a[4]))
    print()
    print("Kept-cell core radius (A > 0.5 max) per animal, px at 512 and at the processing resolution (ssub 2); gSig is 9 at the processing resolution, gSiz 36:")
    for an in sorted(set(r["animal"] for r in rows), key=lambda x: int(x[3:])):
        rad = np.concatenate([r["rad"][r["kept"] == 1] for r in rows if r["animal"] == an])
        print("  %-7s n %3d | radius median %.1f px (IQR %.1f-%.1f) at 512 = %.1f px at 256; diameter at 256 ~ %.0f px" % (
            an, rad.size, np.median(rad), *np.percentile(rad, [25, 75]), np.median(rad) / 2, np.median(rad)))


if __name__ == "__main__":
    main()
