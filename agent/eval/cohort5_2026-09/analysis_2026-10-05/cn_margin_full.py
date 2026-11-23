"""Join the MATLAB-exported per-candidate Cn/PNR maxima (review set = all
candidates for these threshold-0 sessions) with the human labels and the
session's seeding thresholds.  Read-only."""
import os
import re
import csv
import numpy as np
from scipy.io import loadmat

ROOT = r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew"
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_cnmax.csv")

by = {}
with open(CSV, newline="") as fh:
    for r in csv.DictReader(fh):
        by.setdefault(r["session"], []).append((int(r["k"]), float(r["cnmax"]), float(r["pnrmax"])))

rows = []
print("session | N review | labels | kept | min_corr/min_pnr | cand within 0.03 (kept) | within 0.05 (kept) | kept median Cn margin | rejected median Cn margin")
for name, items in sorted(by.items()):
    sd = os.path.join(ROOT, name)
    items.sort()
    cn = np.array([c for _, c, _ in items])
    pn = np.array([p for _, _, p in items])
    N = len(items)
    t = open(os.path.join(sd, "agent_run.log"), encoding="utf-8", errors="replace").read()
    m1 = re.search(r"min_corr=(\d+(?:\.\d+)?), min_pnr=(\d+(?:\.\d+)?)", t)
    mc, mp = float(m1.group(1)), float(m1.group(2))
    L = loadmat(os.path.join(sd, "labels.mat"))["labels"].ravel().astype(int)
    if L.size != N:
        print("SKIP %s: %d review candidates vs %d labels" % (name, N, L.size))
        continue
    cm, pm = cn - mc, pn - mp
    kp = L == 1
    w3, w5 = cm < 0.03, cm < 0.05
    print(" | ".join([name, str(N), str(L.size), str(int(kp.sum())), "%.2f / %.1f" % (mc, mp),
                      "%d (%d)" % (int(w3.sum()), int((w3 & kp).sum())),
                      "%d (%d)" % (int(w5.sum()), int((w5 & kp).sum())),
                      "%.3f" % np.median(cm[kp]) if kp.any() else "-",
                      "%.3f" % np.median(cm[~kp]) if (~kp).any() else "-"]))
    for a, b, l in zip(cm, pm, L):
        rows.append((a, b, int(l)))

arr = np.array(rows)
cm, pm, lb = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)
print()
print("POOLED: %d candidates, %d kept (%.1f%%)" % (len(arr), int((lb == 1).sum()), 100.0 * (lb == 1).mean()))
print("Cn margin above min_corr | n cand | kept | kept-rate")
edges = [-9, 0.0, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 9]
names = ["<0", "0-.02", ".02-.03", ".03-.05", ".05-.10", ".10-.15", ".15-.20", ">=.20"]
for lo, hi, nm in zip(edges[:-1], edges[1:], names):
    sel = (cm >= lo) & (cm < hi)
    k = int((lb[sel] == 1).sum())
    print("%s | %d | %d | %s" % (nm, int(sel.sum()), k, "-" if not sel.any() else "%.0f%%" % (100.0 * k / sel.sum())))
print("PNR margin above min_pnr | n cand | kept | kept-rate")
pedges = [-99, 0, 1, 2, 4, 8, 16, 99]
pnames = ["<0", "0-1", "1-2", "2-4", "4-8", "8-16", ">=16"]
for lo, hi, nm in zip(pedges[:-1], pedges[1:], pnames):
    sel = (pm >= lo) & (pm < hi)
    k = int((lb[sel] == 1).sum())
    print("%s | %d | %d | %s" % (nm, int(sel.sum()), k, "-" if not sel.any() else "%.0f%%" % (100.0 * k / sel.sum())))
print()
print("Cumulative from the threshold upward (what a lower threshold would be adding at the margin):")
for cut in (0.02, 0.03, 0.05, 0.08, 0.10):
    sel = cm < cut
    k = int((lb[sel] == 1).sum())
    print("  candidates with Cn margin < %.2f: %d of %d (%.1f%% of all), kept %d (kept-rate %s)" % (
        cut, int(sel.sum()), len(arr), 100.0 * sel.mean(), k, "-" if not sel.any() else "%.1f%%" % (100.0 * k / sel.sum())))
