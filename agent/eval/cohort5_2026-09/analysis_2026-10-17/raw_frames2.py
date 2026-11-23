"""Read-only.  Raw Bruker OME-TIFF (one multi-page file per cycle/trial) on the server:
intensity range and fraction of pixels at the ADC ceiling, pnb105 day 0 (13-bit) vs
day 1 (12-bit) and a day-1 reference.  Pages 50/200/400 of every 10th cycle."""
import os, numpy as np, tifffile
ROOT = "X:/Jeremy/2p data/6odorDualDiffRew"
S = [("082626", "TSeries-082626-pnb105-397um-35z-000", 90), ("082726", "TSeries-082726-pnb105-391um-35z-000", 150), ("082726", "TSeries-082726-pnb109-405um-25z-000", 150)]
for day, ts, ncyc in S:
    d = os.path.join(ROOT, day, ts)
    p1 = os.path.join(d, "%s_Cycle00001_Ch2_000001.ome.tif" % ts)
    with tifffile.TiffFile(p1, is_ome=False) as tf:
        npg = len(tf.pages); pg = tf.pages[0]; print("== %s | file size %.0f MB | pages %d | %s %s" % (ts, os.path.getsize(p1) / 1e6, npg, pg.shape, pg.dtype))
    vals = []; c12 = c13 = tot = 0; mx = 0
    for cyc in range(1, ncyc + 1, 10):
        p = os.path.join(d, "%s_Cycle%05d_Ch2_000001.ome.tif" % (ts, cyc))
        if not os.path.exists(p): print("   missing cycle", cyc); continue
        with tifffile.TiffFile(p, is_ome=False) as tf:
            for fr in (50, 200, 400):
                if fr >= len(tf.pages): continue
                im = tf.pages[fr].asarray().astype(np.int32)
                tot += im.size; c12 += int((im >= 4095).sum()); c13 += int((im >= 8191).sum()); mx = max(mx, int(im.max()))
                vals.append((im.mean(), np.percentile(im, 50), np.percentile(im, 99), np.percentile(im, 99.9), (im == 0).mean()))
    v = np.array(vals)
    print("   %d raw frames | mean %.0f | p50 %.0f | p99 %.0f | p99.9 %.0f | max %d | px >= 4095: %.4f%% | px >= 8191: %.4f%% | px == 0: %.2f%%" % (
        len(v), v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 3].mean(), mx, 100.0 * c12 / tot, 100.0 * c13 / tot, 100 * v[:, 4].mean()))
