"""C1.  Does the port reproduce ACORN's seeding images, and were the two tasks measured alike?

For each of the 40 6odor sessions (cache/<session>.npz from pass_all.py vs the session's
own Cn.mat / pnr.mat, read-only):
  - pixel agreement: PNR port vs pnr.mat (256), imresize(Cn port) vs Cn.mat (512)
  - pk25 / pk50 / pk100 three ways: port at 256 | 2 x 2 mean of Cn.mat (how the earlier
    analysis measured 6odor) | the earlier analysis's own value (floor/floor_sessions.tsv)
For the 71 3odor sessions: pk50 of this port vs the earlier port (floor_sessions.tsv).
Then f = median (0.70 - floor) / (pk50 - floor) over the 31 calibration sessions, with
pk50 from Cn.mat (as before) and from the port (as applied to 3odor), and the 3odor rule
values under each.
Usage: c1_port_vs_mat.py > c1_port_vs_mat.txt
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402

PREV = rc.HERE.parent / "floor" / "floor_sessions.tsv"
FINAL_RE = re.compile(r"Final parameters: \{.*'min_corr': ([0-9.]+), 'min_pnr': ([0-9.]+)")


def final_params(sd):
    mc = mp = None
    for ln in open(sd / "agent_run.log", encoding="utf-8", errors="replace"):
        m = FINAL_RE.search(ln)
        if m:
            mc, mp = float(m.group(1)), float(m.group(2))
    return mc, mp


def prev_floor(T):
    # the earlier analysis: nearest simulated T of (500, 2400, 6400, 8445), floor at min_pnr 3
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


def main():
    prev = {r["session"]: r for r in csv.DictReader(open(PREV), delimiter="\t")}
    rows = []
    print("== 6odor: port vs the session's Cn.mat / pnr.mat ==")
    print("session | T | max|PNR diff| | max|Cn diff| at 512, interior | pk25 port / mat / prev | pk50 port / mat / prev | "
          "pk100 port / mat / prev | pk50 port - mat")
    for g, name, tif, sd, an in rc.sessions():
        f = rc.CACHE / (name + ".npz")
        if not f.exists():
            print("  NOT IN CACHE: %s" % name)
            continue
        z = np.load(f)
        cn, pnr, T = z["cn"].astype(float), z["pnr"].astype(float), int(z["T"])
        r = dict(group=g, session=name, animal=an, T=T, pk50_port=rc.pk(cn, 50), pk25_port=rc.pk(cn, 25),
                 pk100_port=rc.pk(cn, 100), pk10_port=rc.pk(cn, 10))
        p = prev.get(name)
        r["pk50_prev"] = float(p["pk50"]) if p else np.nan
        if g == "6odor":
            C = np.nan_to_num(rc.loadmat_img(sd / "Cn.mat", "Cn"))
            P = np.nan_to_num(rc.loadmat_img(sd / "pnr.mat", "pnr"))
            if C.shape != (512, 512) or P.shape != (256, 256):
                print("  UNEXPECTED SHAPES %s %s %s" % (name, C.shape, P.shape))
                continue
            Cb = rc.block2(C)
            dP = np.abs(pnr - P).max()
            dC = np.abs(rc.matlab_resize2x(cn) - C)[20:-20, 20:-20].max()
            r.update(pk25_mat=rc.pk(Cb, 25), pk50_mat=rc.pk(Cb, 50), pk100_mat=rc.pk(Cb, 100), dP=dP, dC=dC)
            r["mc_run"], r["mp_run"] = final_params(sd)
            print("  %s | %d | %.4f | %.4f | %.3f / %.3f / %s | %.4f / %.4f / %s | %.3f / %.3f / %s | %+.4f" % (
                name[:26], T, dP, dC, r["pk25_port"], r["pk25_mat"], p["pk25"][:5] if p else "-",
                r["pk50_port"], r["pk50_mat"], p["pk50"][:6] if p else "-", r["pk100_port"], r["pk100_mat"],
                p["pk100"][:5] if p else "-", r["pk50_port"] - r["pk50_mat"]))
        rows.append(r)
    six = [r for r in rows if r["group"] == "6odor" and "pk50_mat" in r]
    d = np.array([r["pk50_port"] - r["pk50_mat"] for r in six])
    print("\n6odor N %d: pk50 port - block-mean(Cn.mat): median %+.4f, min %+.4f, max %+.4f; |diff| > 0.01 in %d" % (
        len(six), np.median(d), d.min(), d.max(), int((np.abs(d) > 0.01).sum())))
    for k in ("pk25", "pk100"):
        dd = np.array([r[k + "_port"] - r[k + "_mat"] for r in six])
        print("          %s port - mat: median %+.4f, min %+.4f, max %+.4f" % (k, np.median(dd), dd.min(), dd.max()))
    print("          worst pixel disagreement: PNR %.4f, Cn (512, 20-px border removed) %.4f" % (
        max(r["dP"] for r in six), max(r["dC"] for r in six)))
    dprev = np.array([r["pk50_mat"] - r["pk50_prev"] for r in six if np.isfinite(r["pk50_prev"])])
    print("          block-mean(Cn.mat) pk50 - earlier analysis's 6odor pk50: max abs %.4f (N %d)" % (
        np.abs(dprev).max(), len(dprev)))
    three = [r for r in rows if r["group"] == "3odor" and np.isfinite(r["pk50_prev"])]
    d3 = np.array([r["pk50_port"] - r["pk50_prev"] for r in three])
    print("3odor N %d: pk50 this port - earlier port: median %+.4f, min %+.4f, max %+.4f; |diff| > 0.01 in %d" % (
        len(three), np.median(d3), d3.min(), d3.max(), int((np.abs(d3) > 0.01).sum())))

    hand = lambda r: r["animal"] == "pnb105" or r["session"].startswith("090526-pnb99")
    cal = [r for r in six if not hand(r) and r["mc_run"] is not None and abs(r["mc_run"] - 0.70) < 1e-6]
    print("\n== f, the rule's constant: median over %d calibration sessions of (0.70 - floor) / (pk50 - floor) ==" % len(cal))
    for label, key in (("pk50 from Cn.mat (as calibrated before)", "pk50_mat"), ("pk50 from the port (as applied to 3odor)", "pk50_port")):
        fv = np.array([(0.70 - prev_floor(r["T"])) / (r[key] - prev_floor(r["T"])) for r in cal])
        f = float(np.median(fv))
        rule3 = np.array([round(prev_floor(r["T"]) + f * (r["pk50_port"] - prev_floor(r["T"])), 2)
                          for r in rows if r["group"] == "3odor"])
        rulec = np.array([prev_floor(r["T"]) + f * (r[key] - prev_floor(r["T"])) for r in cal])
        print("  %s: f = %.4f (per-session f min %.3f, max %.3f) | calibration sessions land at %.3f-%.3f | "
              "3odor rule median %.2f [%.2f-%.2f], < 0.65 in %d of %d" % (
                  label, f, fv.min(), fv.max(), rulec.min(), rulec.max(), np.median(rule3), rule3.min(), rule3.max(),
                  int((rule3 < 0.65).sum()), len(rule3)))
    out = rc.HERE / "c1_sessions.tsv"
    keys = ["group", "session", "animal", "T", "pk10_port", "pk25_port", "pk50_port", "pk100_port", "pk50_mat",
            "pk50_prev", "mc_run", "mp_run"]
    with open(out, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
