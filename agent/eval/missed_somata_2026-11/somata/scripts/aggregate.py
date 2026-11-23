"""Pool the per-session diagnostic outputs under miss/. Read-only on everything except its own report."""
import json, re, sys
import numpy as np
from pathlib import Path
from scipy.stats import fisher_exact, wilcoxon, binomtest

OUT = Path(__file__).parent / "miss"
sess = []
for C in sorted(OUT.iterdir()):
    if not (C / "done.txt").exists(): continue
    run = (C / "run.txt").read_text()
    mc = float(re.search(r"min_corr ([\d.]+)", run).group(1)); mp = float(re.search(r"min_pnr ([\d.]+)", run).group(1))
    b = json.load(open(C / "blobs.json")); r = json.load(open(C / "reds.json"))
    date, animal = C.name.split("-")
    # keep only red somata the guide outlines (bleed_flag False); recompute red_soma for GCaMP somata
    G = Path(r"D:\retro_tdTomato_Alignment\results") / animal / date / run.splitlines()[0].strip() / "red_guide"
    rb = np.atleast_1d(np.genfromtxt(G / "red_blobs.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"))
    flag = {(int(row["row"]), int(row["col"])): str(row["bleed_flag"]).strip() == "True" for row in rb} if rb.size else {}
    n_flag = sum(flag.values())
    r = [x for x in r if not flag.get((x["r"], x["c"]), False)]
    keep = np.array([(rr, cc) for (rr, cc), f in flag.items() if not f], float).reshape(-1, 2)
    for x in b:
        x["red_soma"] = bool(keep.size and np.min(np.hypot(keep[:, 0] - x["r"], keep[:, 1] - x["c"])) <= 10)
    if n_flag: print(f"{C.name}: {n_flag} bleed-flagged red blob(s) excluded")
    for x in b: x["mc"] = mc; x["mp"] = mp; x["tag"] = C.name; x["animal"] = animal
    for x in r: x["mc"] = mc; x["tag"] = C.name; x["animal"] = animal
    sess.append(dict(tag=C.name, animal=animal, mc=mc, mp=mp, b=b, r=r))
print(f"sessions pooled: {len(sess)}")
B = [x for s in sess for x in s["b"]]; R = [x for s in sess for x in s["r"]]
cats = ["review", "autorej", "none"]
pct = lambda a, b_: f"{100*a/b_:.0f}%" if b_ else "-"

print("\n## Per session (GCaMP mean-image somata)")
print("| session | min_corr | somata | review | auto-rej | none | none fail Cn | none fail PNR | red somata | red: review/autorej/none | red none visible in GCaMP |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in sess:
    b, r = s["b"], s["r"]; n = len(b); c = {k: sum(x["cat"] == k for x in b) for k in cats}
    nn = [x for x in b if x["cat"] == "none"]
    rc = {k: sum(x["cat"] == k for x in r) for k in cats}
    rnv = sum(x["dog_z"] >= 3 for x in r if x["cat"] == "none")
    print(f"| {s['tag']} | {s['mc']:.2f} | {n} | {c['review']} ({pct(c['review'],n)}) | {c['autorej']} ({pct(c['autorej'],n)}) | "
          f"{c['none']} ({pct(c['none'],n)}) | {sum(x['cn_max']<s['mc'] for x in nn)} | {sum(x['pnr_max']<s['mp'] for x in nn)} | "
          f"{len(r)} | {rc['review']}/{rc['autorej']}/{rc['none']} | {rnv}/{rc['none']} |")

def q(sel, k):
    a = np.array([x[k] for x in sel], float); a = a[np.isfinite(a)]
    return "%.2f [%.2f-%.2f]" % (np.median(a), *np.percentile(a, [25, 75])) if a.size else "-"

print("\n## Pooled GCaMP somata by category")
n = len(B)
for k in cats:
    sel = [x for x in B if x["cat"] == k]
    print(f"- {k}: n={len(sel)} ({pct(len(sel), n)}); DoG z {q(sel,'dog_z')}; Cn max - min_corr "
          f"{q([dict(v=x['cn_max']-x['mc']) for x in sel],'v')}; np-corr trace PNR {q(sel,'pnr_np_corr')}; "
          f"frac frames >3sd {q(sel,'frac_3sig')}; frames at 255 max {max(x['sat_frac'] for x in sel):.4f}")
nn = [x for x in B if x["cat"] == "none"]
fail = [x for x in nn if not x["gate"]]
print(f"- none: fail a gate {len(fail)}/{len(nn)}; of those Cn<min_corr {sum(x['cn_max']<x['mc'] for x in fail)}, "
      f"PNR<min_pnr {sum(x['pnr_max']<x['mp'] for x in fail)}, only-PNR {sum((x['pnr_max']<x['mp']) and (x['cn_max']>=x['mc']) for x in fail)}")
pg = [x for x in nn if x["gate"]]
print(f"- none but passes both gates: {len(pg)}; Cn margin {q([dict(v=x['cn_max']-x['mc']) for x in pg],'v')}; "
      f"within 22 px of a candidate centroid {sum(x.get('d_cand', 1e9) <= 22 for x in pg)}/{len(pg)}; np-corr trace PNR {q(pg,'pnr_np_corr')}")

print("\n## Coverage by mean-image contrast (pooled)")
for lo, hi in [(3, 5), (5, 8), (8, 15), (15, 1e9)]:
    sel = [x for x in B if lo <= x["dog_z"] < hi]; m = len(sel)
    c = {k: sum(x["cat"] == k for x in sel) for k in cats}
    print(f"- DoG z {lo}-{'inf' if hi > 1e8 else hi}: n={m}; review {pct(c['review'],m)}, auto-rej {pct(c['autorej'],m)}, none {pct(c['none'],m)}")

print("\n## Coverage by min_corr")
for mc in sorted({x["mc"] for x in B}):
    sel = [x for x in B if x["mc"] == mc]; m = len(sel); c = {k: sum(x["cat"] == k for x in sel) for k in cats}
    print(f"- min_corr {mc}: sessions {sum(s['mc']==mc for s in sess)}, somata {m}; review {pct(c['review'],m)}, auto-rej {pct(c['autorej'],m)}, none {pct(c['none'],m)}")

print("\n## Per animal")
for a in sorted({s["animal"] for s in sess}):
    sel = [x for x in B if x["animal"] == a]; m = len(sel); c = {k: sum(x["cat"] == k for x in sel) for k in cats}
    rr = [x for x in R if x["animal"] == a]; rn = sum(x["cat"] == "none" for x in rr)
    print(f"- {a}: sessions {sum(s['animal']==a for s in sess)}, somata {m}; review {pct(c['review'],m)}, auto-rej {pct(c['autorej'],m)}, "
          f"none {pct(c['none'],m)}; red somata {len(rr)}, no candidate {rn} ({pct(rn,len(rr))})")

print("\n## Red overlap of GCaMP somata")
rv = [x for x in B if x["cat"] == "review"]; rn_ = [x for x in B if x["cat"] == "none"]; ra = [x for x in B if x["cat"] == "autorej"]
a1, a2 = sum(x["red_soma"] for x in rn_), sum(x["red_soma"] for x in rv)
print(f"- on a red soma (<=10 px): none {a1}/{len(rn_)} ({pct(a1,len(rn_))}), review {a2}/{len(rv)} ({pct(a2,len(rv))}), "
      f"auto-rej {sum(x['red_soma'] for x in ra)}/{len(ra)} ({pct(sum(x['red_soma'] for x in ra),len(ra))})")
print(f"  Fisher none vs review: OR {fisher_exact([[a1, len(rn_)-a1],[a2, len(rv)-a2]])[0]:.2f}, p {fisher_exact([[a1, len(rn_)-a1],[a2, len(rv)-a2]])[1]:.3f} (pooled, ignores session)")
d = []
for s in sess:
    x1 = [x["red_z"] for x in s["b"] if x["cat"] == "none"]; x2 = [x["red_z"] for x in s["b"] if x["cat"] == "review"]
    if len(x1) >= 3 and len(x2) >= 3: d.append(np.median(x1) - np.median(x2))
d = np.array(d)
print(f"- per-session median red z (none minus review), sessions with >=3 each: n={d.size}, higher in none {int((d>0).sum())}, "
      f"median diff {np.median(d):.2f}, sign test p {binomtest(int((d>0).sum()), d.size).pvalue:.3f}, Wilcoxon p {wilcoxon(d).pvalue:.3f}")

print("\n## Red somata (from the red guides)")
m = len(R); c = {k: sum(x["cat"] == k for x in R) for k in cats}
print(f"- n={m}; review {c['review']} ({pct(c['review'],m)}), auto-rej {c['autorej']} ({pct(c['autorej'],m)}), none {c['none']} ({pct(c['none'],m)})")
for k in cats:
    sel = [x for x in R if x["cat"] == k]
    if sel:
        print(f"  - {k}: visible GCaMP soma (DoG z>=3) {sum(x['dog_z']>=3 for x in sel)}/{len(sel)}; DoG z {q(sel,'dog_z')}; "
              f"pass gates {sum(x['gate'] for x in sel)}/{len(sel)}; np-corr trace PNR {q(sel,'pnr_c')}; frac >3sd {q(sel,'frac3')}")
rnone = [x for x in R if x["cat"] == "none"]
print(f"- red, no candidate, fails a gate: {sum(not x['gate'] for x in rnone)}/{len(rnone)} "
      f"(Cn {sum(x['cn_max']<x['mc'] for x in rnone if not x['gate'])})")

print("\n## Highly visible GCaMP somata (DoG z >= 8) that are not review candidates")
hi = [x for x in B if x["dog_z"] >= 8]
hn = [x for x in hi if x["cat"] != "review"]
print(f"- {len(hn)}/{len(hi)} ({pct(len(hn),len(hi))}): auto-rej {sum(x['cat']=='autorej' for x in hn)}, none {sum(x['cat']=='none' for x in hn)}; "
      f"on a red soma {sum(x['red_soma'] for x in hn)} (vs {sum(x['red_soma'] for x in hi if x['cat']=='review')}/{len(hi)-len(hn)} of visible review)")
print(f"  np-corr trace PNR: these {q(hn,'pnr_np_corr')} vs visible review {q([x for x in hi if x['cat']=='review'],'pnr_np_corr')}")
top = sorted(hn, key=lambda x: -x["dog_z"])[:12]
for x in top:
    print(f"   {x['tag']} ({x['r']},{x['c']}) {x['cat']:7s} z {x['dog_z']:5.1f} Cn {x['cn_max']:.2f} gate {x['gate']!s:5s} "
          f"tracePNR {x['pnr_np_corr']:4.1f} red_soma {x['red_soma']} red_z {x['red_z']:.1f}")
