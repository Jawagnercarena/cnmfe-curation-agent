"""Read-only.  pnb105 day 1 (082626, 266 cand) vs day 2 (082726, 79 cand), same FOV A,
same laser/PMT per the imaging sheet.  (1) are the two mean images the same field
(phase-correlation shift + correlation after shift)?  (2) residual motion within each
session: shift of 24 block means (40 consecutive frames) vs the session mean.
(3) soma contrast on the mean image at Taylor's kept cells (day 2) / at the 0.70
candidates (day 1).  Two reference sessions for scale."""
import sys, numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy import ndimage as ndi
import tifffile
sys.path.insert(0, "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent"); import local_config
DP = Path(local_config.DATA_PARENT); LOCAL = DP / "vCA1" / "6odorDualDiffRew"
S = ["AVG5x-TSeries-082626-pnb105-397um-35z-000", "AVG5x-TSeries-082726-pnb105-391um-35z-000",
     "AVG5x-TSeries-082726-pnb109-405um-25z-000", "AVG5x-TSeries-082826-pnb99-436um-3z-000-000"]
def frames(sd, name, idx):
    tif = sd / (name + ".tif")
    with tifffile.TiffFile(str(tif)) as t:
        nf = int(t.imagej_metadata.get("images", 0)); off = t.pages[0].dataoffsets[0]
    out = np.zeros((len(idx), 512, 512), np.float32)
    with open(tif, "rb") as fh:
        for j, i in enumerate(idx):
            fh.seek(off + int(i) * 512 * 512); out[j] = np.frombuffer(fh.read(512 * 512), dtype=np.uint8).reshape(512, 512)
    return out, nf
def shift(a, b):
    """integer shift (dr, dc) that best aligns b onto a, by phase correlation on the central 400x400"""
    a = a[56:456, 56:456] - a[56:456, 56:456].mean(); b = b[56:456, 56:456] - b[56:456, 56:456].mean()
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b)); F /= (np.abs(F) + 1e-9)
    r = np.fft.ifft2(F).real; p = np.unravel_index(np.argmax(r), r.shape)
    return [(x if x <= 200 else x - 400) for x in p]
def corr(a, b, d):
    b2 = np.roll(np.roll(b, d[0], 0), d[1], 1); m = slice(80, 432)
    return np.corrcoef(a[m, m].ravel(), b2[m, m].ravel())[0, 1]
means = {}
for s in S:
    sd = LOCAL / s
    with tifffile.TiffFile(str(sd / (s + ".tif"))) as t: nf = int(t.imagej_metadata.get("images", 0))
    mimg, _ = frames(sd, s, np.linspace(0, nf - 1, 300).astype(int)); mimg = mimg.mean(0); means[s] = mimg
    shifts = []
    for b0 in np.linspace(0, nf - 41, 24).astype(int):
        blk, _ = frames(sd, s, np.arange(b0, b0 + 40)); d = shift(mimg, blk.mean(0)); shifts.append(np.hypot(*d))
    shifts = np.array(shifts)
    # sharpness: gradient energy of the mean image, normalised by its std
    g = np.hypot(ndi.sobel(mimg, 0), ndi.sobel(mimg, 1))[40:472, 40:472]
    print("%s | frames %d | mean %.1f | block-vs-mean shift: median %.1f px, max %.1f px, blocks>=3px: %d/24 | sharpness %.3f" % (
        s[14:], nf, mimg.mean(), np.median(shifts), shifts.max(), int((shifts >= 3).sum()), g.mean() / mimg[40:472, 40:472].std()))
a, b = means[S[0]], means[S[1]]
d = shift(a, b); print("day2 -> day1 mean-image shift (dr, dc) = %s px; correlation after shift %.3f (before %.3f)" % (d, corr(a, b, d), corr(a, b, [0, 0])))
d2 = shift(means[S[2]], b); print("control: day2 pnb105 vs 082726-pnb109: shift %s, corr after %.3f" % (d2, corr(means[S[2]], b, d2)))
np.save("C:/Users/julia/AppData/Local/Temp/claude/c--code-CNMF-E-LEGACY-BIANE-CLAUDE/b4949d41-93ee-415a-9a29-2d42de916c5d/scratchpad/pnb105_means.npy", np.stack([a, b]))
