"""Read-only.  Per 1000-frame block of each tif: brightness, CNMF-E-style correlation
image (zero-mean Gaussian filter, gSig 9 at 256 res, 8-neighbour local correlation),
PNR proxy (peak of filtered trace / noise from the MAD of the first difference),
single-frame jitter vs the block mean, and block-mean similarity to block 0."""
import sys, os, time, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
import tifffile
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
LOCAL = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
S = sys.argv[1:] or ["AVG5x-TSeries-082626-pnb105-397um-35z-000", "AVG5x-TSeries-082726-pnb105-391um-35z-000"]
BLK = 1000; GSIG = 9; BOX = 4 * GSIG + 1
def read_block(fh, off, i0, n):
    fh.seek(off + i0 * 512 * 512); raw = np.frombuffer(fh.read(n * 512 * 512), dtype=np.uint8).reshape(n, 512, 512)
    return raw.reshape(n, 256, 2, 256, 2).mean(axis=(2, 4), dtype=np.float32)        # ssub 2
def filt(Y):
    out = np.empty_like(Y)
    for k in range(len(Y)):
        out[k] = ndi.gaussian_filter(Y[k], GSIG, truncate=2.0) - ndi.uniform_filter(Y[k], BOX)
    return out
def local_corr(F):
    Z = F - F.mean(0); Z /= (Z.std(0) + 1e-6)
    acc = np.zeros(F.shape[1:], np.float32); cnt = np.zeros(F.shape[1:], np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0: continue
            sh = np.roll(np.roll(Z, dr, 1), dc, 2); v = (Z * sh).mean(0)
            m = np.ones(F.shape[1:], bool); m[:1 if dr == 1 else None, :] = m if dr != 1 else False
            acc += v; cnt += 1
    cn = acc / cnt; cn[:2, :] = cn[-2:, :] = cn[:, :2] = cn[:, -2:] = 0
    return cn
def shift(a, b):
    a = a - a.mean(); b = b - b.mean(); F = np.fft.fft2(a) * np.conj(np.fft.fft2(b)); F /= (np.abs(F) + 1e-9)
    r = np.fft.ifft2(F).real; p = np.unravel_index(np.argmax(r), r.shape)
    return np.hypot(*[(x if x <= 128 else x - 256) for x in p])
for s in S:
    sd = LOCAL / s; tif = sd / (s + ".tif")
    with tifffile.TiffFile(str(tif)) as t: nf = int(t.imagej_metadata.get("images", 0)); off = t.pages[0].dataoffsets[0]
    print("== %s: %d frames, %d blocks of %d" % (s[14:], nf, nf // BLK, BLK), flush=True)
    print("block | frames | mean | corr(blockmean, block0) | Cn: %px>0.7 | %px>0.8 | maxima>0.7 (11px win) | PNR p90 | %px Cn>0.7 & PNR>6 | frame jitter >=1px (of 20) | med frame-vs-blockmean corr")
    ref = None; t0 = time.time()
    with open(tif, "rb") as fh:
        for b in range(nf // BLK):
            Y = read_block(fh, off, b * BLK, BLK); mimg = Y.mean(0)
            if ref is None: ref = mimg
            F = filt(Y); cn = local_corr(F)
            noise = np.median(np.abs(np.diff(F, axis=0)), axis=0) / 0.6745 / np.sqrt(2) + 1e-6
            pnr = (F.max(0) - np.median(F, axis=0)) / noise
            inner = (slice(12, 244), slice(12, 244))
            c = cn[inner]; p = pnr[inner]
            mx = (c == ndi.maximum_filter(c, size=11)) & (c > 0.7)
            idx = np.linspace(0, BLK - 1, 20).astype(int)
            jit = sum(shift(mimg, Y[i]) >= 1 for i in idx)
            fc = np.median([np.corrcoef(mimg[inner].ravel(), Y[i][inner].ravel())[0, 1] for i in idx])
            print("%2d | %5d-%5d | %5.1f | %.3f | %5.1f | %5.1f | %3d | %5.1f | %5.1f | %2d | %.3f" % (
                b, b * BLK, (b + 1) * BLK - 1, mimg.mean(), np.corrcoef(ref[inner].ravel(), mimg[inner].ravel())[0, 1],
                100 * (c > 0.7).mean(), 100 * (c > 0.8).mean(), int(mx.sum()), np.percentile(p, 90),
                100 * ((c > 0.7) & (p > 6)).mean(), jit, fc), flush=True)
    print("   (%.1f min)" % ((time.time() - t0) / 60), flush=True)
