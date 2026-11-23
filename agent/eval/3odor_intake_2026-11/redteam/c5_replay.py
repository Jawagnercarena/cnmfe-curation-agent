"""C5.  Does the first-pass replay stand in for ACORN's real seeding?

Real seeds come from the SEEDTRACE shadow copy of greedyROI_endoscope.m, which records every
seed attempt (pass, pixel, value, outcome):
  - agent/eval/missed_somata_2026-11/seedtrace/<session>/seedtrace.mat: the six 090826
    sessions, full production pipeline, production thresholds (2026-11-04/05)
  - redteam/cache/matlab/c5_<session>.mat: initialisation stage only, at min_corr 0.60 / 0.70 /
    0.80 with the production min_pnr (this red team, run_matlab_rt.py c5)
For each (session, threshold): real pass-1 attempts (= the local maxima of the first
while-iteration), components initialised over all passes, and the replay on (a) the Cn0 /
PNR0 the MATLAB run itself returned, (b) this port's images from the tif.  Also whether the
replayed seed PIXELS are the real ones.  bd: the run used 10 at 256 resolution.
Usage: c5_replay.py > c5_replay.txt
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

OLD = rc.HERE.parents[1] / "missed_somata_2026-11" / "seedtrace"


def compare(name, mc, mp, bd, att, k, cn_m, pnr_m, src):
    p1 = att[att[:, 0] == 1]
    real_pix = set(int(i) - 1 for i in p1[:, 1])                       # MATLAB linear index, column-major
    ok5 = int((p1[:, 3] == 5).sum())
    rep_m = rc.first_pass(cn_m, pnr_m, mc, mp, bd=bd, return_mask=True)
    ry, rx = np.nonzero(rep_m)
    rep_pix = set(int(x * 256 + y) for y, x in zip(ry, rx))
    line = "  %s | %s | min_corr %.2f min_pnr %.1f bd %d | real pass-1 attempts %d (initialised in pass 1: %d), all passes: " \
           "attempts %d, initialised %d | replay on MATLAB's Cn0/PNR0 %d (same pixels: %d)" % (
               name[:22], src, mc, mp, bd, len(p1), ok5, len(att), k, int(rep_m.sum()), len(real_pix & rep_pix))
    port = np.nan
    f = rc.CACHE / (name + ".npz")
    if f.exists():
        z = np.load(f)
        port = rc.first_pass(z["cn"].astype(float), z["pnr"].astype(float), mc, mp, bd=bd)
        line += " | replay on this port %d | max |Cn0 - port| %.4f" % (port, np.abs(np.nan_to_num(cn_m) - z["cn"]).max())
    print(line)
    return dict(session=name, src=src, mc=mc, real=len(p1), init=k, rep=int(rep_m.sum()), port=port)


def main():
    rows = []
    print("== production SEEDTRACE runs (2026-11-04/05) ==")
    for d in sorted(OLD.iterdir()):
        f = d / "seedtrace.mat"
        if not f.exists():
            continue
        z = loadmat(str(f), squeeze_me=True, struct_as_record=False)
        st = [s for s in np.atleast_1d(z["ST_SEEDS"]) if s.stage == "init"][0]
        rows.append(compare(rc.short(d.name), float(st.min_corr), float(st.min_pnr), int(st.bd), np.atleast_2d(st.att),
                            int(st.k), np.asarray(st.Cn0, float), np.asarray(st.PNR0, float), "production"))
    print("\n== initialisation-only runs at three thresholds (this red team) ==")
    for f in sorted((rc.CACHE / "matlab").glob("c5_*.mat")):
        z = loadmat(str(f), squeeze_me=True, struct_as_record=False)
        name = f.stem[3:]
        for o in np.atleast_1d(z["out"]):
            rows.append(compare(name, float(o.min_corr), float(o.min_pnr), 10, np.atleast_2d(o.att), int(o.k),
                                np.asarray(z["Cn0"], float), np.asarray(z["PNR0"], float), "init-only"))
    if not rows:
        return
    e = np.array([abs(r["rep"] - r["real"]) / max(r["real"], 1) for r in rows])
    print("\nreplay (on MATLAB's own images) vs real pass-1 attempts: max relative error %.3f over %d (session, threshold) cases; "
          "within 10%% in %d" % (e.max(), len(rows), int((e <= 0.10).sum())))
    pr = [r for r in rows if np.isfinite(r["port"])]
    if pr:
        e = np.array([abs(r["port"] - r["real"]) / max(r["real"], 1) for r in pr])
        print("replay (on this port's images) vs real pass-1 attempts: max relative error %.3f over %d cases; within 10%% in %d" % (
            e.max(), len(pr), int((e <= 0.10).sum())))
    print("initialised components / pass-1 attempts: median %.2f [%.2f-%.2f]; Spearman(pass-1 attempts, initialised) %.2f (N %d)" % (
        np.median([r["init"] / r["real"] for r in rows]), min(r["init"] / r["real"] for r in rows),
        max(r["init"] / r["real"] for r in rows), spearmanr([r["real"] for r in rows], [r["init"] for r in rows])[0], len(rows)))
    by = {}
    for r in rows:
        by.setdefault(r["session"], []).append(r)
    print("how the count moves with min_corr inside a session (init-only runs): pass-1 attempts and initialised components")
    for s, rr in by.items():
        rr = sorted([r for r in rr if r["src"] == "init-only"], key=lambda r: r["mc"])
        if len(rr) >= 2:
            print("  %s: %s" % (s[:22], "; ".join("%.2f: %d attempts -> %d components" % (r["mc"], r["real"], r["init"]) for r in rr)))


if __name__ == "__main__":
    main()
