"""C6.  Does pk50 measure how bright a field is, or how many cells it has?  (cache only)

  A. Cell count alone: pk50 of the same session computed on half of the field (4 halves) and
     on a quarter (4 quadrants).  Brightness is unchanged, the number of cells is roughly
     halved / quartered.  Rule shift = f * (pk50 sub-field - pk50 full).  Same for pk10, pk25.
  B. Same-field pairs (own registration AND the earlier table agree it is one field): does the
     pk50 difference follow the dimness difference (noise_rel)?  Pooled and within field.
  C. Labelled pilots (12, all bright): pk50 against the number of cells the reviewer kept.
  D. Tracked cells: the reviewer's kept cells of each pilot, carried by the registration shift
     into every other session of the same field.  Per target session: the share of those cells
     that can seed from their own core (dilated 1 px) at 0.70, at the rule, at the floor, and
     their peak Cn.  Bright same-field targets give the day-to-day baseline (cells that are
     simply silent that day); dim targets show what dimming adds.
Usage: c6_pk50.py > c6_pk50.txt
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import rt_fov as rf  # noqa: E402
import rt_labels as rl  # noqa: E402

F_PREV = 0.444
PAIRS = rc.HERE.parent / "fov" / "pairs.tsv"
RULES = rc.HERE.parent / "floor" / "rule_sessions.tsv"
MY_SCORE = 0.30


def floor_prev(T):
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


def rule_prev(pk50, T):
    fl = floor_prev(T)
    return fl + F_PREV * (pk50 - fl)


def pk_region(ys, xs, v, k, sel):
    w = v[sel(ys, xs)]
    return float(w[k - 1]) if len(w) >= k else np.nan


def main():
    S = {}
    for g, name, tif, sd, an in rc.sessions():
        f = rc.CACHE / (name + ".npz")
        if f.exists():
            z = np.load(f)
            S[name] = dict(group=g, sd=sd, animal=an, T=int(z["T"]), cn=z["cn"].astype(float), pnr=z["pnr"].astype(float),
                           mean=z["mean256"].astype(float), noise_rel=float(z["noise_rel"]), est_mp=float(z["est_mp"]))
    cats = {r["session"]: r["cat"] for r in csv.DictReader(open(RULES), delimiter="\t")}
    for name, s in S.items():
        s["cat"] = cats.get(name, "?")
        ys, xs, v = rc.cn_peaks(s["cn"])
        s["peaks"] = (ys, xs, v)
        for k in (10, 25, 50, 100):
            s["pk%d" % k] = float(v[k - 1]) if len(v) >= k else np.nan
        s["rule"] = rule_prev(s["pk50"], s["T"])
        s["n_pk_floor"] = int((v > floor_prev(s["T"])).sum())
    print("sessions in cache: %d" % len(S))

    # ---------------------------------------------------------------- A
    print("\n== A. cell count alone: the statistic on a sub-field minus on the full field ==")
    regions = {
        "half": [lambda y, x: x < 128, lambda y, x: x >= 128, lambda y, x: y < 128, lambda y, x: y >= 128],
        "quarter": [lambda y, x: (x < 128) & (y < 128), lambda y, x: (x >= 128) & (y < 128),
                    lambda y, x: (x < 128) & (y >= 128), lambda y, x: (x >= 128) & (y >= 128)],
    }
    for label, sel_s in (("all sessions", lambda s: True), ("6odor", lambda s: s["group"] == "6odor"),
                         ("3odor dim/hazy noted", lambda s: s["cat"] == "dim/hazy noted")):
        sub = [s for s in S.values() if sel_s(s)]
        for rn, fns in regions.items():
            for k in (10, 25, 50):
                d = []
                for s in sub:
                    ys, xs, v = s["peaks"]
                    for fn in fns:
                        d.append(pk_region(ys, xs, v, k, fn) - s["pk%d" % k])
                d = np.array(d)
                d = d[np.isfinite(d)]
                print("  %-22s %-7s pk%-3d: N %3d sub-fields | diff median %+.3f [p10 %+.3f, p90 %+.3f] | rule shift "
                      "(x %.3f) median %+.3f; |shift| > 0.02 in %.0f%%" % (
                          label, rn, k, len(d), np.median(d), np.percentile(d, 10), np.percentile(d, 90), F_PREV,
                          F_PREV * np.median(d), 100.0 * (np.abs(F_PREV * d) > 0.02).mean()))

    # ---------------------------------------------------------------- B
    print("\n== B. same-field pairs ==")
    prev_pairs = {}
    for r in csv.DictReader(open(PAIRS), delimiter="\t"):
        prev_pairs[frozenset((r["s1"], r["s2"]))] = (float(r["score"]), int(r["dy"]), int(r["dx"]), r["zoom_diff"])
    names = sorted(S)
    reg = {}
    agree = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if S[a]["animal"] != S[b]["animal"]:
                continue
            sc, dy, dx = rf.register(S[a]["mean"], S[b]["mean"])
            reg[(a, b)] = (sc, dy, dx)
            reg[(b, a)] = (sc, -dy, -dx)
            pp = prev_pairs.get(frozenset((a, b)))
            if pp is not None:
                agree.append((sc, pp[0], pp[3]))
    ag = np.array([(a, b) for a, b, zd in agree])
    same_prev = ag[:, 1] > 0.354
    print("  within-animal pairs registered: %d; also in the earlier table: %d" % (len(reg) // 2, len(ag)))
    print("  earlier score > 0.354 (its same-field cut): %d pairs; of those own score >= %.2f: %d" % (
        int(same_prev.sum()), MY_SCORE, int((ag[same_prev, 0] >= MY_SCORE).sum())))
    print("  earlier score < 0.202 (its new-field cut): %d pairs; of those own score >= %.2f: %d" % (
        int((ag[:, 1] < 0.202).sum()), MY_SCORE, int((ag[ag[:, 1] < 0.202, 0] >= MY_SCORE).sum())))
    print("  Spearman(own score, earlier score) = %.2f" % spearmanr(ag[:, 0], ag[:, 1])[0])

    def same_field(a, b):
        pp = prev_pairs.get(frozenset((a, b)))
        return reg[(a, b)][0] >= MY_SCORE and pp is not None and pp[0] > 0.354

    pairs = [(a, b) for (a, b) in reg if a < b and same_field(a, b)]
    for label, sel in (("all same-field pairs", lambda a, b: True),
                       ("3odor-3odor", lambda a, b: S[a]["group"] == S[b]["group"] == "3odor"),
                       ("6odor-3odor", lambda a, b: S[a]["group"] != S[b]["group"]),
                       ("pairs with |noise_rel diff| >= 0.05", lambda a, b: abs(S[a]["noise_rel"] - S[b]["noise_rel"]) >= 0.05)):
        pp = [(a, b) for a, b in pairs if sel(a, b)]
        if len(pp) < 5:
            continue
        dn = np.array([S[a]["noise_rel"] - S[b]["noise_rel"] for a, b in pp])
        line = "  %-36s N %3d pairs (%d sessions):" % (label, len(pp), len(set(x for p in pp for x in p)))
        for k in ("pk10", "pk25", "pk50", "pk100"):
            dk = np.array([S[a][k] - S[b][k] for a, b in pp])
            ok = np.isfinite(dk)
            line += " Spearman(d %s, d noise_rel) %+.2f |" % (k, spearmanr(dk[ok], dn[ok])[0])
        print(line)
    # fields = connected components
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    comp = defaultdict(list)
    for n in names:
        comp[find(n)].append(n)
    fields = [sorted(v) for v in comp.values() if len(v) >= 4]
    print("  fields with >= 4 sessions: %d (sizes %s)" % (len(fields), sorted(len(f) for f in fields)))
    dn, dk, rhos = [], [], []
    for fl in fields:
        nr = np.array([S[n]["noise_rel"] for n in fl])
        pk = np.array([S[n]["pk50"] for n in fl])
        dn += list(nr - nr.mean())
        dk += list(pk - pk.mean())
        if np.ptp(nr) > 0:
            rhos.append(spearmanr(nr, pk)[0])
        print("    %s x%d: noise_rel %.3f-%.3f, pk50 %.3f-%.3f, within-field Spearman(pk50, noise_rel) %+.2f" % (
            S[fl[0]]["animal"], len(fl), nr.min(), nr.max(), pk.min(), pk.max(), spearmanr(nr, pk)[0]))
    if dn:
        print("  pooled within-field (field means removed): Spearman(pk50, noise_rel) = %+.2f, N %d sessions in %d fields; "
              "median of per-field rho %+.2f" % (spearmanr(dn, dk)[0], len(dn), len(fields), np.median(rhos)))

    # ---------------------------------------------------------------- C
    print("\n== C. labelled pilots: pk50 vs the number of kept cells ==")
    pil = []
    cands = {}
    for g, name, tif, sd, an in rl.pilots():
        if name not in S:
            continue
        cd = rl.candidates(sd)
        cands[name] = cd
        mc, mp = rl.run_params(sd)
        s = S[name]
        s["kept"] = int(cd["labels"].sum())
        s["mc_run"], s["mp_run"] = mc, mp
        pil.append(s)
        print("  %s: kept %3d of %3d | noise_rel %.3f | pk10 %.3f pk25 %.3f pk50 %.3f pk100 %.3f | rule %.3f | ran at %.2f" % (
            name[:22], s["kept"], len(cd["labels"]), s["noise_rel"], s["pk10"], s["pk25"], s["pk50"], s["pk100"], s["rule"], mc))
    if len(pil) >= 6:
        kept = [s["kept"] for s in pil]
        for k in ("pk10", "pk25", "pk50", "pk100"):
            print("  Spearman(%s, kept count) = %+.2f (p %.3f); Spearman(%s, noise_rel) = %+.2f, N %d" % (
                k, *spearmanr([s[k] for s in pil], kept), k, spearmanr([s[k] for s in pil], [s["noise_rel"] for s in pil])[0], len(pil)))

    # ---------------------------------------------------------------- D
    print("\n== D. the pilots' kept cells tracked into other sessions of the same field ==")
    print("pilot -> target | target group / note | own score, shift | noise_rel | pk50 | rule | min_pnr (exact pre-pass) | "
          "kept cells inside the frame | can seed at 0.70 / rule / floor (min_pnr as estimated) | same with min_pnr 3 | "
          "median peak Cn of the tracked cells")
    out_rows = []
    st = ndi.generate_binary_structure(2, 2)
    for pname, cd in cands.items():
        kc = cd["core"][cd["labels"] == 1]
        targets = [(b, reg[(pname, b)]) for b in names if b != pname and (pname, b) in reg and same_field(pname, b)]
        targets = [(pname, (1.0, 0, 0))] + sorted(targets, key=lambda t: S[t[0]]["noise_rel"])
        for b, (sc, dy, dx) in targets:
            t = S[b]
            # register(a = pilot, b = target): pilot[y, x] ~ target[y - dy, x - dx]
            m = rf.shift_masks(kc, -dy, -dx)
            inside = np.array([mm.any() and not (mm[:10].any() or mm[-10:].any() or mm[:, :10].any() or mm[:, -11:].any()) for mm in m])
            m = np.array([ndi.binary_dilation(mm, st) for mm in m[inside]])
            if len(m) < 5:
                continue
            fl = floor_prev(t["T"])
            res = []
            for p in (t["est_mp"], 3.0):
                ce, pkc = rl.cell_ceiling(m, t["cn"], t["pnr"], p)
                res.append((float((ce >= 0.70).mean()), float((ce >= round(t["rule"], 2)).mean()), float((ce >= fl).mean())))
            row = dict(pilot=pname, target=b, group=t["group"], cat=t["cat"], score=sc, dy=dy, dx=dx, noise_rel=t["noise_rel"],
                       pk50=t["pk50"], rule=round(t["rule"], 2), est_mp=t["est_mp"], n=len(m), s070=res[0][0], srule=res[0][1],
                       sfloor=res[0][2], s070_p3=res[1][0], srule_p3=res[1][1], sfloor_p3=res[1][2], med_peak=float(np.median(pkc)))
            out_rows.append(row)
            print("  %s -> %s | %s %s | %.2f (%+d, %+d) | %.3f | %.3f | %.2f | %.1f | %d | %.2f / %.2f / %.2f | "
                  "%.2f / %.2f / %.2f | %.3f" % (
                      pname[:13], b[:22], t["group"], t["cat"][:14], sc, dy, dx, t["noise_rel"], t["pk50"], row["rule"],
                      t["est_mp"], len(m), *res[0], *res[1], row["med_peak"]))
    tr = [r for r in out_rows if r["pilot"] != r["target"]]
    if tr:
        for label, sel in (("bright targets (noise_rel < 0.15)", lambda r: r["noise_rel"] < 0.15),
                           ("middle (0.15-0.19)", lambda r: 0.15 <= r["noise_rel"] < 0.19),
                           ("dim targets (noise_rel >= 0.19)", lambda r: r["noise_rel"] >= 0.19),
                           ("dim/hazy noted", lambda r: r["cat"] == "dim/hazy noted")):
            sub = [r for r in tr if sel(r)]
            if not sub:
                continue
            print("  %-34s N %2d pilot-target pairs: seedable at 0.70 median %.2f | at the rule %.2f | at the floor %.2f | "
                  "(min_pnr 3: %.2f / %.2f / %.2f) | rule median %.2f" % (
                      label, len(sub), np.median([r["s070"] for r in sub]), np.median([r["srule"] for r in sub]),
                      np.median([r["sfloor"] for r in sub]), np.median([r["s070_p3"] for r in sub]),
                      np.median([r["srule_p3"] for r in sub]), np.median([r["sfloor_p3"] for r in sub]),
                      np.median([r["rule"] for r in sub])))
    keys = ["pilot", "target", "group", "cat", "score", "dy", "dx", "noise_rel", "pk50", "rule", "est_mp", "n", "s070",
            "srule", "sfloor", "s070_p3", "srule_p3", "sfloor_p3", "med_peak"]
    with open(rc.HERE / "c6_tracked.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in out_rows:
            fh.write("\t".join(("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    keys = ["group", "animal", "cat", "T", "noise_rel", "est_mp", "pk10", "pk25", "pk50", "pk100", "rule", "n_pk_floor"]
    with open(rc.HERE / "c6_sessions.tsv", "w") as fh:
        fh.write("session\t" + "\t".join(keys) + "\n")
        for n in names:
            fh.write(n + "\t" + "\t".join(("%.4f" % S[n][k]) if isinstance(S[n][k], float) else str(S[n][k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
