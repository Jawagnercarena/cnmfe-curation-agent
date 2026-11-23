"""Own field registration between two sessions' mean images (256 x 256, from the cache).

register(a, b) -> (score, dy, dx): the best integer translation, such that
a[y, x] ~ b[y - dy, x - dx], i.e. np.roll(b, (dy, dx), (0, 1)) lines up with a.
Images are band-passed at cell scale (Gaussian 1 px minus 4 px), edge-zeroed, Hann-windowed
and normalised; score is the normalised cross-correlation at the best shift (|shift| <= maxs).
Translation only: fields imaged at another zoom are not handled (score stays low).
"""
import numpy as np
from scipy import ndimage as ndi

WIN = np.outer(np.hanning(256), np.hanning(256))


def prep(img):
    img = np.asarray(img, float)
    d = ndi.gaussian_filter(img, 1.0) - ndi.gaussian_filter(img, 4.0)
    d[:8, :] = d[-8:, :] = 0
    d[:, :8] = d[:, -8:] = 0
    d = d * WIN
    d = d - d.mean()
    return d / (np.sqrt((d * d).sum()) + 1e-12)


def register(a, b, maxs=90):
    A, B = prep(a), prep(b)
    fa = np.fft.rfft2(A, s=(512, 512))
    fb = np.fft.rfft2(B, s=(512, 512))
    cc = np.fft.irfft2(fa * np.conj(fb), s=(512, 512))          # cc[d] = sum_x A[x] B[x - d]
    cc = np.roll(cc, (maxs, maxs), axis=(0, 1))[:2 * maxs + 1, :2 * maxs + 1]
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    return float(cc[iy, ix]), int(iy - maxs), int(ix - maxs)


def shift_masks(masks, dy, dx):
    """Move boolean masks (N, 256, 256) by (dy, dx) without wrap-around."""
    out = np.zeros_like(masks)
    h, w = masks.shape[1:]
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[:, ys, xs] = masks[:, yd, xd]
    return out


if __name__ == "__main__":                       # self-test of the sign convention
    rng = np.random.default_rng(0)
    b = ndi.gaussian_filter(rng.standard_normal((256, 256)), 2)
    a = np.roll(b, (7, -12), axis=(0, 1))
    print("expect (score ~1, 7, -12):", register(a, b))
    m = np.zeros((1, 256, 256), bool)
    m[0, 100, 100] = True
    print("expect (107, 88):", np.argwhere(shift_masks(m, 7, -12)[0])[0])
