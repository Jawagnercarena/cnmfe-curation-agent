"""Read-only.  Noise floor of ACORN's seeding vs the real sessions.

Inputs (all under the scratchpad): floor/null_T*_m.npz (seedfloor.py null),
vfull/video_compare.tsv + vfull/img/*.npz (3odor full-movie Cn/PNR port),
6odor Cn.mat / pnr.mat / agent_run.log (read-only), the 3odor imaging sheet.
Prints:
  1. the null: Cn / PNR distribution of pure noise per movie length, and the
     lowest min_corr at which noise gives zero first-pass seeds
  2. replay check: first-pass seeds at each 6odor session's actual (min_corr,
     min_pnr) vs the candidates ACORN produced
  3. per session: where the session's strong Cn peaks sit relative to the floor,
     and first-pass seeds at the current rule vs at floor-based thresholds
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seedfloor as sf      # noqa: E402
import summarize as sm      # noqa: E402
import video_compare as vc  # noqa: E402

NULL_T = (500, 2400, 6400, 8445)
EXCLUDE = ("100926-pnb109-427um-25z-000",)
OVR = {"pnb99": 0.70, "pnb102": 0.70, "pnb104": 0.70, "pnb105": 0.60, "pnb106": 0.70, "pnb109": 0.70}
FINAL_RE = re.compile(r"Final parameters: \{.*'min_corr': ([0-9.]+), 'min_pnr': ([0-9.]+)")


def seeds_at(cn, pnr, c, p):
    return int(sf.seed_grid(cn, pnr, [c], [p])[0, 0])


def peaks(cn, pnr, floor_c):
    """Local maxima of Cn (11x11, border 10) above the noise floor, with their Cn and PNR."""
    m = (cn == ndi.maximum_filter(cn, size=11)) & (cn > floor_c)
    m[:10, :] = m[-10:, :] = m[:, :10] = m[:, -10:] = False
    return cn[m], pnr[m]


def seedable(cn, pnr, k, combos):
    """Of the k strongest Cn peaks (11x11 local maxima, border 10), the fraction whose 5x5
    window has >= 13 of 25 pixels with Cn >= c and PNR >= p, i.e. that survive greedyROI's
    medfilt2 at (c, p).  Label-free: says how many of the session's own strongest
    correlation peaks a threshold pair lets through."""
    m = (cn == ndi.maximum_filter(cn, size=11))
    m[:10, :] = m[-10:, :] = m[:, :10] = m[:, -10:] = False
    ys, xs = np.nonzero(m)
    o = np.argsort(cn[ys, xs])[::-1][:k]
    ys, xs = ys[o], xs[o]
    out = []
    for c, p in combos:
        ok = ((cn >= c) & (pnr >= p)).astype(np.int32)
        cnt = ndi.uniform_filter(ok.astype(float), size=5, mode="constant") * 25
        out.append(float((cnt[ys, xs] >= 12.5).mean()) if len(ys) else np.nan)
    return out


def load_null():
    out = {}
    for T in NULL_T:
        p = HERE / "floor" / ("null_T%d_m.npz" % T)
        if p.exists():
            out[T] = np.load(p)
    return out


def floor_c(z, p):
    g, cg, pg = z["grid"], z["c_grid"], z["p_grid"]
    j = int(np.argmin(np.abs(pg - p)))
    ok = np.where(g[:, j] == 0)[0]
    return float(cg[ok[0]]) if len(ok) else float("nan")


def main():
    nulls = load_null()
    print("== 1. pure noise (matched spatial correlation 0.18 x / 0.13 y), 256 x 256, border 10 ==")
    print("T(half rate) | Cn p50 / p99.9 / max | PNR p50 / p99.9 / max | lowest min_corr with 0 noise seeds at min_pnr 3 / 5")
    fl = {}
    for T, z in nulls.items():
        c = z["cn"][10:-10, 10:-10]
        p = z["pnr"][10:-10, 10:-10]
        fl[T] = dict(cmax=float(c.max()), pmax=float(p.max()), c3=floor_c(z, 3.0), c5=floor_c(z, 5.0))
        print("%5d | %.3f / %.3f / %.3f | %.2f / %.2f / %.2f | %.3f / %.3f   (measured noise corr %.3f / %.3f)" % (
            T, np.median(c), np.percentile(c, 99.9), c.max(), np.median(p), np.percentile(p, 99.9), p.max(),
            fl[T]["c3"], fl[T]["c5"], float(z["rx1"]), float(z["ry1"])))
    Ts = np.array(sorted(fl))

    def near(T):
        return int(Ts[np.argmin(np.abs(Ts - T))])

    vr = {r["session"]: r for r in csv.DictReader(open(HERE / "vfull" / "video_compare.tsv"), delimiter="\t")}
    notes = sm.sheet_notes()
    rows = []
    for group, stem, tif, sd in vc.tif_list():
        name = stem.replace("AVG5x-TSeries-", "")
        if name in EXCLUDE or name not in vr:
            continue
        v = vr[name]
        an = v["animal"]
        T = int(v["frames"]) // 2
        r = dict(group=group, session=name, animal=an, T=T)
        if group == "3odor":
            z = np.load(HERE / "vfull" / "img" / (name + ".npz"))
            cn, pnr = z["full_cn"], z["full_pnr"]
            r["cat"] = notes.get((an, name[:6]), ("", "", "NO SHEET ROW", 0))[2]
            r["mc_now"] = OVR[an]
            r["mp_now"] = float(np.clip(float(v["blk_pnr_p85"]), 5.0, 15.0))   # pre-pass estimate stand-in
            r["n_cand"] = np.nan
        else:
            C = vc.mat_img(sd / "Cn.mat", "Cn")
            cn = C.reshape(256, 2, 256, 2).mean(axis=(1, 3)) if C.shape == (512, 512) else C
            pnr = vc.mat_img(sd / "pnr.mat", "pnr")
            mc = mp = None
            for ln in open(sd / "agent_run.log", encoding="utf-8", errors="replace"):
                m = FINAL_RE.search(ln)
                if m:
                    mc, mp = float(m.group(1)), float(m.group(2))
            r["cat"] = "6odor"
            r["mc_now"], r["mp_now"] = mc, mp
            r["n_cand"] = float(v["n_cand"]) if v.get("n_cand") else np.nan
        Tn = near(T)
        r["fc"] = fl[Tn]["c5"]
        r["fp"] = 5.0       # current clamp minimum; at fc noise gives zero seeds for any min_pnr >= 4-5
        r["seeds_now"] = seeds_at(cn, pnr, r["mc_now"], r["mp_now"]) if r["mc_now"] else np.nan
        r["seeds_070"] = seeds_at(cn, pnr, 0.70, r["mp_now"])
        r["seeds_060"] = seeds_at(cn, pnr, 0.60, r["mp_now"])
        r["seeds_floor"] = seeds_at(cn, pnr, r["fc"], r["fp"])
        pc, pp = peaks(cn, pnr, r["fc"])
        strong = np.sort(pc)[::-1]
        r["n_peaks_floor"] = len(pc)
        for k in (25, 50, 100):
            r["pk%d" % k] = float(strong[k - 1]) if len(strong) >= k else np.nan
        r["sd070"], r["sd060"], r["sdfloor"] = seedable(cn, pnr, 50, [(0.70, r["mp_now"]), (0.60, r["mp_now"]),
                                                                     (r["fc"], r["fp"])])
        rows.append(r)
        print("%s %s T %d now %.2f/%.1f seeds now %s 0.70 %d 0.60 %d floor(%.3f/%.1f) %d | Cn of 25th/50th/100th "
              "strongest peak %.3f/%.3f/%.3f" % (group, name, T, r["mc_now"] or np.nan, r["mp_now"] or np.nan,
                                                 r["seeds_now"], r["seeds_070"], r["seeds_060"], r["fc"], r["fp"],
                                                 r["seeds_floor"], r["pk25"], r["pk50"], r["pk100"]), flush=True)

    six = [r for r in rows if r["group"] == "6odor" and np.isfinite(r["n_cand"])]
    rho, pv = spearmanr([r["seeds_now"] for r in six], [r["n_cand"] for r in six])
    print("\n== 2. replay check, 6odor N %d: Spearman(first-pass seeds at the run's own thresholds, candidates) "
          "= %.2f (p %.2g); median ratio candidates / seeds %.2f ==" % (
              len(six), rho, pv, np.median([r["n_cand"] / max(r["seeds_now"], 1) for r in six])))

    print("\n== 3. groups: median [min-max] ==")
    cats = [("6odor", lambda r: r["group"] == "6odor")]
    for c in sm.CATS:
        cats.append(("3odor " + c, lambda r, c=c: r["group"] == "3odor" and r["cat"] == c))
    cats.append(("3odor pnb99 (all)", lambda r: r["group"] == "3odor" and r["animal"] == "pnb99"))
    keys = ["pk25", "pk50", "pk100", "sd070", "sd060", "sdfloor", "seeds_070", "seeds_060", "seeds_floor",
            "n_peaks_floor"]
    print("%-30s %4s | %s" % ("group", "N", " | ".join(keys)))
    for nm, f in cats:
        sub = [r for r in rows if f(r)]
        if not sub:
            continue
        cells = []
        for k in keys:
            v = np.array([r[k] for r in sub], float)
            v = v[np.isfinite(v)]
            fmt = "%.3f [%.3f-%.3f]" if k.startswith("pk") else (
                "%.2f [%.2f-%.2f]" if k.startswith("sd") else "%.0f [%.0f-%.0f]")
            cells.append(fmt % (np.median(v), v.min(), v.max()) if len(v) else "-")
        print("%-30s %4d | %s" % (nm, len(sub), " | ".join(cells)))
    out = HERE / "floor" / "floor_sessions.tsv"
    ks = ["group", "session", "animal", "cat", "T", "mc_now", "mp_now", "n_cand", "seeds_now", "seeds_070",
          "seeds_060", "fc", "fp", "seeds_floor", "n_peaks_floor", "pk25", "pk50", "pk100", "sd070", "sd060",
          "sdfloor"]
    with open(out, "w") as fh:
        fh.write("\t".join(ks) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
