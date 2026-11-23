"""
kept_by_cn_band.py -- Analysis A of docs/HANDOFF_2026-10-06_COHORT5_REASSESSMENT.md.

Read-only.  For each of the 12 cohort-5 pilot sessions reviewed at threshold 0:
per candidate, the peak Cn inside the footprint core (A > 0.5 max) on the
session's full-movie Cn image; Taylor's label (1 = kept).  Tabulates kept-rate by
absolute Cn band, pooled / per animal / per session, the kept cells that would be
lost at a higher seed (Cn peak < 0.75, < 0.80), and the net gain over the first
(est85) run: kept cells with no counterpart in .prev_est85_2026-10-05/ (cosine
< 0.45, the sweep's rule).  082726-pnb105 is the 0.60 re-run (its candidate set is
.rerun_mincorr060_2026-10-18/); the other eleven are the 0.70 sets in
.rerun_mincorr070_2026-10-06/.  Labels are read from the local session when
ingested, otherwise from inbox/Taylor (read-only).

Usage: python kept_by_cn_band.py [--out result.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from local_config import DATA_PARENT, EXCHANGE_ROOT   # noqa: E402

TASK = Path("vCA1") / "6odorDualDiffRew"
LOCAL = DATA_PARENT / TASK
INBOX = Path(EXCHANGE_ROOT) / "inbox" / "Taylor" / TASK
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
OLD_DIR = ".prev_est85_2026-10-05"
EDGES = [0.60, 0.70, 0.75, 0.80, 0.85, 1.01]
MATCH = 0.45


def foot(p: Path):
    A = np.asarray(loadmat(str(p))["spatial_footprints"], dtype=float)
    return A.reshape(A.shape[0], 512, 512) if A.ndim == 2 else A


def labels_for(s: str):
    for p in (LOCAL / s / "labels.mat", INBOX / s / "labels.mat"):
        if p.exists():
            L = np.asarray(loadmat(str(p))["labels"]).ravel()
            return (L == 1).astype(int), ("local" if p.parent == LOCAL / s else "inbox")
    return None, None


def band_index(v):
    return int(np.searchsorted(EDGES, v, side="right") - 1)


def band_label(b):
    hi = EDGES[b + 1]
    return "%.2f-%.2f" % (EDGES[b], hi) if hi <= 1.0 else ">=%.2f" % EDGES[b]


def tabulate(cn, kept):
    out = {}
    for b in range(len(EDGES) - 1):
        m = np.array([band_index(v) == b for v in cn])
        out[band_label(b)] = (int(m.sum()), int(kept[m].sum()))
    return out


def fmt(tab):
    return " | ".join("%s: %d/%d%s" % (k, v[1], v[0], (" (%.0f%%)" % (100.0 * v[1] / v[0])) if v[0] else "") for k, v in tab.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    print("Per session: candidate set | labels source | kept/N | kept-rate by Cn-peak band | kept with Cn < 0.75 / < 0.80 | kept with no est85 counterpart")
    for s in PILOTS:
        sd = LOCAL / s
        A = foot(sd / SET_DIR[s] / "spatial_footprints.mat")
        kept, src = labels_for(s)
        if kept is None:
            print("  %s: NO LABELS -- skipped" % s)
            continue
        if kept.size != A.shape[0]:
            print("  %s: labels %d != candidates %d -- skipped" % (s, kept.size, A.shape[0]))
            continue
        Cn = np.nan_to_num(np.asarray(loadmat(str(sd / "Cn.mat"))["Cn"], dtype=float))
        sm = ndi.uniform_filter(Cn, 3)
        cn = np.array([sm[a > 0.5 * a.max()].max() for a in A])
        tab = tabulate(cn, kept)
        lost75 = int(kept[cn < 0.75].sum())
        lost80 = int(kept[cn < 0.80].sum())
        new_kept = None
        old_p = sd / OLD_DIR / "spatial_footprints.mat"
        if old_p.exists():
            A0 = foot(old_p)
            An = A.reshape(len(A), -1)
            An = An / (np.linalg.norm(An, axis=1, keepdims=True) + 1e-12)
            On = A0.reshape(len(A0), -1)
            On = On / (np.linalg.norm(On, axis=1, keepdims=True) + 1e-12)
            best = (An @ On.T).max(1)
            new_mask = best < MATCH
            new_kept = int(kept[new_mask].sum())
            new_kept_cn = cn[new_mask & (kept == 1)]
        animal = re.search(r"-(pnb\d+)-", s).group(1)
        rows.append({"session": s, "animal": animal, "set": SET_DIR[s], "labels_from": src, "n": int(A.shape[0]),
                     "kept": int(kept.sum()), "cn": cn.tolist(), "kept_vec": kept.tolist(), "bands": tab,
                     "kept_cn_lt_075": lost75, "kept_cn_lt_080": lost80, "new_vs_est85_kept": new_kept,
                     "new_vs_est85_kept_cn_median": (float(np.median(new_kept_cn)) if new_kept and len(new_kept_cn) else None)})
        print("  %s | %s | %s | %d/%d | %s | %d / %d | %s" % (
            s[14:], SET_DIR[s][7:19], src, int(kept.sum()), A.shape[0], fmt(tab), lost75, lost80,
            "-" if new_kept is None else "%d (median Cn %.2f)" % (new_kept, np.median(new_kept_cn))))
    print()
    print("Per animal (kept/N by band):")
    for an in sorted(set(r["animal"] for r in rows), key=lambda x: int(x[3:])):
        rr = [r for r in rows if r["animal"] == an]
        cn = np.concatenate([r["cn"] for r in rr]); kept = np.concatenate([r["kept_vec"] for r in rr])
        print("  %-7s %d sessions | %s" % (an, len(rr), fmt(tabulate(cn, kept))))
    print()
    cn = np.concatenate([r["cn"] for r in rows]); kept = np.concatenate([r["kept_vec"] for r in rows])
    pooled = tabulate(cn, kept)
    print("POOLED %d sessions, %d candidates, %d kept (%.1f%%)" % (len(rows), cn.size, int(kept.sum()), 100.0 * kept.mean()))
    print("  " + fmt(pooled))
    r7075 = pooled["0.70-0.75"]; r8085 = pooled["0.80-0.85"]
    rate = lambda v: (v[1] / v[0]) if v[0] else float("nan")
    print("  kept-rate 0.70-0.75 = %.1f%% vs 0.80-0.85 = %.1f%% (ratio %.2f; handoff rule asks >= 0.33)" % (
        100 * rate(r7075), 100 * rate(r8085), rate(r7075) / rate(r8085) if r8085[0] and r7075[0] else float("nan")))
    gains = [r["new_vs_est85_kept"] for r in rows if r["new_vs_est85_kept"] is not None]
    print("  kept cells with no est85 counterpart: total %d, per session median %.1f, range %d-%d (handoff rule asks >= ~5 per session)" % (
        sum(gains), float(np.median(gains)), min(gains), max(gains)))
    print("  kept cells that a 0.75 seed would not have seeded (Cn peak < 0.75): %d of %d; at 0.80: %d" % (
        sum(r["kept_cn_lt_075"] for r in rows), int(kept.sum()), sum(r["kept_cn_lt_080"] for r in rows)))
    # the 0.60 session on its own
    r60 = [r for r in rows if r["set"].startswith(".rerun_mincorr060")]
    if r60:
        r = r60[0]
        print("  0.60 session %s: %s" % (r["session"][14:], fmt(r["bands"])))
    if a.out:
        slim = [{k: v for k, v in r.items() if k not in ("cn", "kept_vec")} for r in rows]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"rows": slim, "pooled": pooled, "edges": EDGES}, indent=1), encoding="utf-8")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
