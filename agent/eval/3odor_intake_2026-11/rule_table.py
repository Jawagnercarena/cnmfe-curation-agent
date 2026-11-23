"""Read-only.  A per-session min_corr computed from the session itself, and what it does.

  floor(T)  lowest min_corr at which a pure-noise movie of the session's length
            (matched spatial noise correlation, seedfloor.py null) gives zero
            first-pass seeds even at min_pnr 3: 0.55 full length, 0.575 at day-0
            length (interpolated by nearest simulated T)
  pk50      Cn of the session's 50th strongest Cn peak (11x11 local maxima, border 10)
  rule      min_corr = floor + f * (pk50 - floor), rounded to 0.01.  f is the one
            constant: the median over the 6odor sessions that ran at 0.70 (minus the
            hand-set pnb105 / 090526-pnb99 ones) of (0.70 - floor) / (pk50 - floor),
            i.e. the rule keeps 0.70's position between noise and the field's strong
            peaks, and moves with the field.
  min_pnr   kept at the current estimate (85th percentile of a 1000-frame block, clip
            5-15; for 3odor the mid-session block stands in for the pre-pass); the
            table also shows the top-50 peaks' PNR against the noise PNR for that T.
Prints per-session values for 3odor and 6odor and writes floor/rule_sessions.tsv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import floor_report as fr   # noqa: E402
import video_compare as vc  # noqa: E402

K = 50


def top_peak_pnr(cn, pnr, k=K):
    m = (cn == ndi.maximum_filter(cn, size=11))
    m[:10, :] = m[-10:, :] = m[:, :10] = m[:, -10:] = False
    ys, xs = np.nonzero(m)
    o = np.argsort(cn[ys, xs])[::-1][:k]
    v = pnr[ys[o], xs[o]]
    return float(np.median(v)), float(np.percentile(v, 10))


def main():
    nulls = fr.load_null()
    flo = {T: fr.floor_c(z, 3.0) for T, z in nulls.items()}
    pn = {T: float(np.percentile(z["pnr"][10:-10, 10:-10], 99.9)) for T, z in nulls.items()}
    Ts = np.array(sorted(flo))
    near = lambda T: int(Ts[np.argmin(np.abs(Ts - T))])
    rows = list(csv.DictReader(open(HERE / "floor" / "floor_sessions.tsv"), delimiter="\t"))
    num = lambda v: float(v) if v not in ("", "nan", "None") else np.nan
    for r in rows:
        for k in ("T", "mc_now", "mp_now", "n_cand", "pk50", "seeds_now"):
            r[k] = num(r[k])
        r["fl"] = flo[near(int(r["T"]))]
        r["pnoise"] = pn[near(int(r["T"]))]
    hand = lambda r: r["animal"] == "pnb105" or r["session"].startswith("090526-pnb99")
    cal = [r for r in rows if r["group"] == "6odor" and not hand(r) and abs(r["mc_now"] - 0.70) < 1e-6]
    f = float(np.median([(0.70 - r["fl"]) / (r["pk50"] - r["fl"]) for r in cal]))
    print("floor by T (zero noise seeds at min_pnr 3): %s" % ", ".join("T %d: %.3f" % (T, flo[T]) for T in Ts))
    print("noise PNR p99.9 by T: %s" % ", ".join("T %d: %.2f" % (T, pn[T]) for T in Ts))
    print("f = %.3f (from %d 6odor sessions run at 0.70)\n" % (f, len(cal)))
    vr = {r["session"]: r for r in csv.DictReader(open(HERE / "vfull" / "video_compare.tsv"), delimiter="\t")}
    out = []
    for r in rows:
        r["rule"] = round(r["fl"] + f * (r["pk50"] - r["fl"]), 2)
        name = r["session"]
        if r["group"] == "3odor":
            z = np.load(HERE / "vfull" / "img" / (name + ".npz"))
            cn, pnr = z["full_cn"], z["full_pnr"]
        else:
            sd = [x for x in vc.tif_list([name]) if x[0] == "6odor"][0][3]
            C = vc.mat_img(sd / "Cn.mat", "Cn")
            cn = C.reshape(256, 2, 256, 2).mean(axis=(1, 3)) if C.shape == (512, 512) else C
            pnr = vc.mat_img(sd / "pnr.mat", "pnr")
        r["seeds_rule"] = fr.seeds_at(cn, pnr, r["rule"], r["mp_now"])
        r["sd_rule"] = fr.seedable(cn, pnr, K, [(r["rule"], r["mp_now"])])[0]
        r["sd_now"] = fr.seedable(cn, pnr, K, [(r["mc_now"], r["mp_now"])])[0]
        r["pk_pnr_med"], r["pk_pnr_p10"] = top_peak_pnr(cn, pnr)
        r["noise_rel"] = num(vr[name]["noise_rel"])
        out.append(r)
    for g in ("3odor", "6odor"):
        print("== %s: session | sheet note | T | noise_rel | pk50 | min_corr now -> rule | min_pnr now | "
              "first-pass seeds now -> rule | top-50 seedable now -> rule | top-50 PNR median / p10 "
              "(noise p99.9) ==" % g)
        sub = sorted([r for r in out if r["group"] == g], key=lambda r: r["rule"])
        for r in sub:
            print("  %s | %s | %d | %.3f | %.3f | %.2f -> %.2f | %.1f | %d -> %d | %.2f -> %.2f | %.1f / %.1f (%.1f)" % (
                r["session"][:26], r["cat"][:22], r["T"], r["noise_rel"], r["pk50"], r["mc_now"], r["rule"],
                r["mp_now"], r["seeds_now"], r["seeds_rule"], r["sd_now"], r["sd_rule"], r["pk_pnr_med"],
                r["pk_pnr_p10"], r["pnoise"]))
        v = np.array([r["rule"] for r in sub])
        print("  rule min_corr: median %.2f [%.2f-%.2f]; < 0.65 in %d of %d; > 0.72 in %d\n" % (
            np.median(v), v.min(), v.max(), (v < 0.65).sum(), len(v), (v > 0.72).sum()))
    keys = ["group", "session", "animal", "cat", "T", "noise_rel", "fl", "pk50", "mc_now", "rule", "mp_now",
            "seeds_now", "seeds_rule", "sd_now", "sd_rule", "pk_pnr_med", "pk_pnr_p10", "pnoise", "n_cand"]
    with open(HERE / "floor" / "rule_sessions.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in out:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("wrote", HERE / "floor" / "rule_sessions.tsv")


if __name__ == "__main__":
    sys.exit(main())
