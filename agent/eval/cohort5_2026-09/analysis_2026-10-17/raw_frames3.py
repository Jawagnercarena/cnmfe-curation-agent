"""Read-only.  Raw count level day 0 (082626, 13-bit) vs day 1 (082726, 12-bit) for all
five animals imaged both days, plus pnb99 (082626 vs 082826, both 13-bit).  Pages 100
and 300 of every 15th cycle."""
import os, numpy as np, tifffile
ROOT = "X:/Jeremy/2p data/6odorDualDiffRew"
PAIRS = [("pnb102", "082626", "TSeries-082626-pnb102-479um-3z-000", 90, "082726", "TSeries-082726-pnb102-497um-3z-000", 150),
         ("pnb104", "082626", "TSeries-082626-pnb104-404um-3z-000", 90, "082726", "TSeries-082726-pnb104-416um-3z-000", 150),
         ("pnb105", "082626", "TSeries-082626-pnb105-397um-35z-000", 90, "082726", "TSeries-082726-pnb105-391um-35z-000", 150),
         ("pnb106", "082626", "TSeries-082626-pnb106-205um-4z-000", 90, "082726", "TSeries-082726-pnb106-201um-4z-000", 150),
         ("pnb109", "082626", "TSeries-082626-pnb109-398um-26z-000", 90, "082726", "TSeries-082726-pnb109-405um-25z-000", 150),
         ("pnb99", "082626", "TSeries-082626-pnb99-399um-3z-000", 90, "082826", "TSeries-082826-pnb99-436um-3z-000-000", 150)]
def stats(day, ts, ncyc):
    d = os.path.join(ROOT, day, ts); vals = []
    for cyc in range(1, ncyc + 1, 15):
        p = os.path.join(d, "%s_Cycle%05d_Ch2_000001.ome.tif" % (ts, cyc))
        if not os.path.exists(p): continue
        with tifffile.TiffFile(p, is_ome=False) as tf:
            for fr in (100, 300):
                if fr < len(tf.pages):
                    im = tf.pages[fr].asarray().astype(np.int32); vals.append((im.mean(), np.percentile(im, 50), np.percentile(im, 99), (im == 0).mean()))
    v = np.array(vals); return len(v), v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), 100 * v[:, 3].mean()
print("animal | day0: n, mean, p50, p99, %zero | day1: n, mean, p50, p99, %zero | mean ratio day1/day0 | p99 ratio")
for a, d0, t0, n0, d1, t1, n1 in PAIRS:
    s0 = stats(d0, t0, n0); s1 = stats(d1, t1, n1)
    print("%s | %d, %.0f, %.0f, %.0f, %.1f | %d, %.0f, %.0f, %.0f, %.1f | %.2f | %.2f" % (a, *s0, *s1, s1[1] / s0[1], s1[3] / s0[3]))
