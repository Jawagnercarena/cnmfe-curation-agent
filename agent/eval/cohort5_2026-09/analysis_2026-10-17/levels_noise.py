"""Read-only.  uint8 gray-level usage and temporal noise vs mean (shot-noise slope)
for pnb105 day 0 / day 1 and two references, from 400 consecutive frames mid-session."""
import sys, numpy as np
from pathlib import Path
import tifffile
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
LOCAL = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
S = ["AVG5x-TSeries-082626-pnb105-397um-35z-000", "AVG5x-TSeries-082726-pnb105-391um-35z-000",
     "AVG5x-TSeries-082626-pnb109-398um-26z-000", "AVG5x-TSeries-082726-pnb109-405um-25z-000"]
print("session | levels used (of 256) | p1 / p50 / p99 / max | %px==255 | temporal noise (MAD of diff) at mean 60-80 | at 100-120 | var/mean slope")
for s in S:
    tif = LOCAL / s / (s + ".tif")
    with tifffile.TiffFile(str(tif)) as t: nf = int(t.imagej_metadata.get("images", 0)); off = t.pages[0].dataoffsets[0]
    i0 = nf // 2
    with open(tif, "rb") as fh:
        fh.seek(off + i0 * 512 * 512); Y = np.frombuffer(fh.read(400 * 512 * 512), dtype=np.uint8).reshape(400, 512, 512).astype(np.float32)
    h = np.bincount(Y.astype(np.uint8).ravel(), minlength=256); levels = int((h > 0).sum())
    m = Y.mean(0); v = Y.var(0); d = np.median(np.abs(np.diff(Y, axis=0)), axis=0) / 0.6745 / np.sqrt(2)
    sel = lambda lo, hi: (m >= lo) & (m < hi)
    A = np.vstack([m.ravel(), np.ones(m.size)]).T; k = (m.ravel() > 20) & (m.ravel() < 200)
    slope = np.linalg.lstsq(A[k], v.ravel()[k], rcond=None)[0][0]
    print("%s | %d | %d / %d / %d / %d | %.2f | %.2f | %.2f | %.2f" % (s[14:], levels, np.percentile(Y, 1), np.percentile(Y, 50), np.percentile(Y, 99), Y.max(),
          100 * (Y == 255).mean(), np.median(d[sel(60, 80)]) if sel(60, 80).any() else -1, np.median(d[sel(100, 120)]) if sel(100, 120).any() else -1, slope))
