"""Red-team core, 2026-11-11.  Independent port of ACORN's seeding statistics.

Written from the MATLAB sources only (greedyROI_endoscope.m, correlation_image_endoscope.m,
correlation_image.m, get_noise_fft.m, dsData.m, downSample.m); nothing is imported from
the scripts one folder up.  Read-only on data.

Conventions: movies are (T, 256, 256) float32, i.e. numpy [t, row, col] = MATLAB (row, col, t).

  load_ds(tif)                 ssub 2 / tsub 2 movie exactly as Sources2D.load_data + dsData
  seeding_images(Y)            Cn0 / PNR0 of greedyROI_endoscope.m (one patch, sig 3)
  prepass_images(Y)            Cn / PNR of correlation_image_endoscope.m on the 1000
                               linspace frames (3 x 3 patches combined by max, sig 5)
  estimate_params(cn, pnr)     params.py:estimate_min_corr_min_pnr (85th pct, clamps)
  first_pass(cn, pnr, c, p)    first while-iteration seed screen of greedyROI_endoscope.m
  matlab_resize2x(img)         imresize(img, [512 512]) (bicubic), as CNMFe_Biane_headless.m:238
"""
import re
import sys
from pathlib import Path

import numpy as np
import scipy.fft as sfft
import tifffile
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))          # agent/, for local_config only
from local_config import DATA_PARENT           # noqa: E402

ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
DIR_3ODOR = Path(r"D:/2p data/3odor")
DIR_6ODOR = Path(DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
CACHE = HERE / "cache"
GSIG, GSIZ = 5, 18                              # ceil(9 / 2), ceil(36 / 2)  (downSample.m)
NW = 12                                         # fft workers


def short(name):
    return name.replace("AVG5x-TSeries-", "")


def sessions(only=None):
    """[(group, short name, tif path, session dir or None)] for the six animals."""
    out = []
    pat = re.compile(r"-(pnb\d+)-")
    for t in sorted(DIR_3ODOR.glob("AVG5x-TSeries-*.tif")):
        m = pat.search(t.name)
        if m and m.group(1) in ANIMALS:
            out.append(("3odor", short(t.stem), t, None, m.group(1)))
    for d in sorted(p for p in DIR_6ODOR.iterdir() if p.is_dir() and p.name.startswith("AVG5x")):
        m = pat.search(d.name)
        if m and m.group(1) in ANIMALS:
            t = d / (d.name + ".tif")
            if t.exists():
                out.append(("6odor", short(d.name), t, d, m.group(1)))
    if only:
        out = [x for x in out if any(o in x[1] for o in only)]
    return out


# ---------------------------------------------------------------- movie
def open_movie(tif):
    """Memory map of the uint8 (N, 512, 512) stack (contiguous ImageJ tif)."""
    mm = tifffile.memmap(str(tif), mode="r")
    if mm.dtype != np.uint8 or mm.ndim != 3 or mm.shape[1:] != (512, 512):
        raise RuntimeError("unexpected tif layout %s %s" % (mm.dtype, mm.shape))
    return mm


def ds_block(raw):
    """dsData.m: mean of consecutive pairs in time (odd tail dropped), then
    imresize(., 1/2, 'box') = 2 x 2 block mean."""
    n = (raw.shape[0] // 2) * 2
    y = raw[:n].astype(np.float32)
    y = y.reshape(n // 2, 2, 512, 512).mean(axis=1)
    return y.reshape(n // 2, 256, 2, 256, 2).mean(axis=(2, 4))


def load_ds(tif, chunk=1024, mm=None):
    """Whole movie at 256 x 256, half rate.  load_data reads Tbatch = 1024 frames at a
    time and keeps floor(N / 2) output frames."""
    mm = open_movie(tif) if mm is None else mm
    n = (mm.shape[0] // 2) * 2
    out = np.empty((n // 2, 256, 256), np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s // 2:e // 2] = ds_block(np.asarray(mm[s:e]))
    return out


# ---------------------------------------------------------------- filter
def fspecial_gaussian(n, sig):
    h = (n - 1) / 2.0
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sig * sig))
    g[g < np.finfo(float).eps * g.max()] = 0
    return g / g.sum()


def centered_psf(gsiz=GSIZ, gsig=GSIG):
    psf = fspecial_gaussian(int(round(gsiz)), gsig)
    nz = psf >= psf[:, 0].max()
    psf = psf - psf[nz].mean()
    psf[~nz] = 0
    return psf


PSF = centered_psf()


def imfilter_replicate(Y, k=PSF, batch=256):
    """imfilter(Y, k, 'replicate'): correlation, same size.  MATLAB puts the kernel
    origin at floor((size + 1) / 2) (1-based), so an 18-wide kernel covers offsets
    -8 .. +9.  Done with an FFT on an edge-padded stack."""
    kh, kw = k.shape
    ch, cw = (kh + 1) // 2 - 1, (kw + 1) // 2 - 1          # 0-based origin
    pad = ((0, 0), (ch, kh - 1 - ch), (cw, kw - 1 - cw))
    T, H, W = Y.shape
    ph, pw = H + kh - 1, W + kw - 1
    fh, fw = sfft.next_fast_len(ph, real=True), sfft.next_fast_len(pw, real=True)
    kk = np.zeros((fh, fw), np.float32)
    kk[:kh, :kw] = k
    Kf = np.conj(sfft.rfft2(kk))
    out = np.empty((T, H, W), np.float32)
    for i in range(0, T, batch):
        P = np.pad(Y[i:i + batch], pad, mode="edge")
        F = sfft.rfft2(P, s=(fh, fw), workers=NW)
        F *= Kf
        out[i:i + batch] = sfft.irfft2(F, s=(fh, fw), workers=NW)[:, :H, :W]
    return out


# ---------------------------------------------------------------- noise, Cn
def sub_median(X):
    """X is (T, P); subtract each pixel's median over time, in place."""
    for j in range(0, X.shape[1], 8192):
        X[:, j:j + 8192] -= np.median(X[:, j:j + 8192], axis=0)[None]
    return X


def noise_fft(X):
    """get_noise_fft.m with the defaults (noise_range [0.25 0.5], 'logmexp').  X is (T, P)."""
    N = X.shape[0]
    ff = np.arange(0, N // 2 + 1) / N
    ind = (ff > 0.25) & (ff <= 0.5)
    sn = np.empty(X.shape[1], np.float64)
    for j in range(0, X.shape[1], 8192):
        xd = sfft.rfft(X[:, j:j + 8192], axis=0, workers=NW)
        psd = (xd.real.astype(np.float64) ** 2 + xd.imag.astype(np.float64) ** 2) / N
        psd[1:-1] *= 2                                   # psdx(:, 2:end-1) = 2 * ...
        with np.errstate(divide="ignore"):
            sn[j:j + 8192] = np.sqrt(np.exp(np.mean(np.log(psd[ind] / 2), axis=0)))
    return sn


def corr_image(X, h, w):
    """correlation_image(X, [1 2], h, w): mean over the 8 neighbours of the temporal
    correlation.  X is (T, P) and is centred / normalised IN PLACE."""
    T = X.shape[0]
    for j in range(0, X.shape[1], 8192):
        b = X[:, j:j + 8192]
        b -= b.mean(axis=0)[None]
        s = np.sqrt(np.einsum("tp,tp->p", b, b, dtype=np.float64) / T)
        s[s == 0] = 1
        b /= s[None].astype(np.float32)
    Z = X.reshape(T, h, w)
    acc = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.float64)
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        a = Z[:, max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
        b = Z[:, max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
        c = np.einsum("tij,tij->ij", a, b, dtype=np.float64) / T
        acc[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] += c
        acc[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)] += c
        cnt[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] += 1
        cnt[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)] += 1
    return acc / cnt


def filtered_traces(Y):
    """HY (T, P) after imfilter + median subtraction, its max and the fft noise."""
    T, h, w = Y.shape
    X = imfilter_replicate(Y).reshape(T, h * w)
    sub_median(X)
    return X, X.max(axis=0).astype(np.float64), noise_fft(X)


def seeding_images(Y, sig=3.0, want_traces=False):
    """Cn0, PNR0 of greedyROI_endoscope.m (whole field as one patch)."""
    T, h, w = Y.shape
    X, hmax, sn = filtered_traces(Y)
    pnr = (hmax / sn).reshape(h, w)
    keep = X.copy() if want_traces else None
    X[X < (sn * sig)[None].astype(np.float32)] = 0
    cn = corr_image(X, h, w)
    if want_traces:
        return cn, pnr, keep, sn
    return cn, pnr


def prepass_frames(T):
    """round(linspace(1, T, min(T, 1000))) as 0-based indices (MATLAB round: half away)."""
    v = np.linspace(1, T, min(T, 1000))
    return (np.floor(v + 0.5).astype(int) - 1)


def prepass_images(Y, sig=5.0):
    """correlation_image_endoscope.m on the pre-pass frames of Y (the FULL half-rate movie)."""
    Ys = Y[prepass_frames(Y.shape[0])]
    T, d1, d2 = Ys.shape
    r0p = np.floor(np.linspace(1, d1, 4) + 0.5).astype(int)
    c0p = np.floor(np.linspace(1, d2, 4) + 0.5).astype(int)
    bd = int(round(GSIZ))
    Cn = np.zeros((d1, d2))
    PNR = np.zeros((d1, d2))
    for mr in range(3):
        r0 = max(1, r0p[mr] - bd)
        r1 = min(d1, r0p[mr + 1] + bd - 1)
        for mc in range(3):
            c0 = max(1, c0p[mc] - bd)
            c1 = min(d2, c0p[mc + 1] + bd - 1)
            P = Ys[:, r0 - 1:r1, c0 - 1:c1]
            nr, nc = P.shape[1:]
            X = imfilter_replicate(P).reshape(T, nr * nc)
            sub_median(X)
            hmax = X.max(axis=0).astype(np.float64)
            sn = noise_fft(X)
            PNR[r0 - 1:r1, c0 - 1:c1] = np.maximum(PNR[r0 - 1:r1, c0 - 1:c1], (hmax / sn).reshape(nr, nc))
            X[X < (sn * sig)[None].astype(np.float32)] = 0
            Cn[r0 - 1:r1, c0 - 1:c1] = np.maximum(Cn[r0 - 1:r1, c0 - 1:c1], corr_image(X, nr, nc))
    return Cn, PNR


def estimate_params(cn, pnr, pct=85, raw=False):
    """params.py:estimate_min_corr_min_pnr (non-bootstrap)."""
    c = cn[np.isfinite(cn)].ravel()
    p = pnr[np.isfinite(pnr)].ravel()
    c = c[c > 0.1]
    p = p[p > 1.0]
    rc, rp = float(np.percentile(c, pct)), float(np.percentile(p, pct))
    if raw:
        return rc, rp
    return round(float(np.clip(rc, 0.6, 0.92)), 2), round(float(np.clip(rp, 5.0, 15.0)), 1)


# ---------------------------------------------------------------- seeding screen
MED = int(np.floor(GSIZ / 4 + 0.5))             # round(gSiz / 4) = 5
LMAX = 2 * MED + 1                              # 11


def border_mask(shape, bd=10):
    m = np.zeros(shape, bool)
    m[:bd, :] = True
    m[shape[0] - bd:, :] = True
    m[:, :bd] = True
    m[:, shape[1] - bd - 1:] = True             # ind_bd(:, (end-bd):end): bd + 1 columns
    return m


def first_pass(cn, pnr, c, p, bd=10, return_mask=False):
    """Seeds of the FIRST while-iteration of greedyROI_endoscope.m (lines 120-123, 176-181,
    218-219): v = Cn .* PNR zeroed below (c, p); medfilt2 5 x 5 (zero padded); re-zero the
    screened pixels; local max over 11 x 11 (taken before the border is zeroed); border zeroed."""
    cn = np.nan_to_num(np.asarray(cn, float))
    pnr = np.nan_to_num(np.asarray(pnr, float))
    off = (cn < c) | (pnr < p)
    v = cn * pnr
    v[off] = 0
    v = ndi.median_filter(v, size=MED, mode="constant", cval=0.0)
    v[off] = 0
    vmax = ndi.maximum_filter(v, size=LMAX, mode="constant", cval=0.0)
    v[border_mask(v.shape, bd)] = 0
    seeds = (v == vmax) & (vmax > 0) & (v >= c * p)
    return seeds if return_mask else int(seeds.sum())


def cn_peaks(cn, bd=10, size=LMAX):
    """Local maxima of Cn (size x size), symmetric border bd removed; sorted strongest first."""
    cn = np.nan_to_num(np.asarray(cn, float))
    m = cn == ndi.maximum_filter(cn, size=size)
    m[:bd, :] = m[-bd:, :] = False
    m[:, :bd] = m[:, -bd:] = False
    ys, xs = np.nonzero(m)
    o = np.argsort(cn[ys, xs])[::-1]
    return ys[o], xs[o], cn[ys[o], xs[o]]


def pk(cn, k):
    v = cn_peaks(cn)[2]
    return float(v[k - 1]) if len(v) >= k else float("nan")


def pass_count(cn, pnr, c, p):
    """Per pixel: how many of the 25 pixels of its 5 x 5 window pass (Cn >= c and PNR >= p)."""
    ok = ((np.nan_to_num(cn) >= c) & (np.nan_to_num(pnr) >= p)).astype(np.float64)
    return np.rint(ndi.uniform_filter(ok, size=MED, mode="constant", cval=0.0) * MED * MED).astype(int)


def seedable_at(cn, pnr, ys, xs, c, p):
    """True where the pixel itself passes and >= 13 of its 25 window pixels pass, i.e. the
    5 x 5 median of the screened v is non-zero there and the pixel is not re-zeroed."""
    n = pass_count(cn, pnr, c, p)
    self_ok = (np.nan_to_num(cn)[ys, xs] >= c) & (np.nan_to_num(pnr)[ys, xs] >= p)
    return (n[ys, xs] >= 13) & self_ok


# ---------------------------------------------------------------- MATLAB imresize x2
def _cubic(x):
    ax = np.abs(x)
    return np.where(ax <= 1, 1.5 * ax ** 3 - 2.5 * ax ** 2 + 1,
                    np.where(ax <= 2, -0.5 * ax ** 3 + 2.5 * ax ** 2 - 4 * ax + 2, 0.0))


def _resize_axis(a, n_out, axis):
    n_in = a.shape[axis]
    scale = n_out / n_in
    u = (np.arange(1, n_out + 1)) / scale + 0.5 * (1 - 1 / scale)      # 1-based input coordinate
    left = np.floor(u - 2).astype(int)
    idx = left[:, None] + np.arange(0, 6)[None]                        # P = 4 + 2 taps
    w = _cubic(u[:, None] - idx)
    w = w / w.sum(axis=1, keepdims=True)
    idx = np.clip(idx, 1, n_in) - 1
    a = np.moveaxis(a, axis, 0)
    out = np.einsum("ok,ok...->o...", w, a[idx])
    return np.moveaxis(out, 0, axis)


def matlab_resize2x(img, n=512):
    """imresize(img, [n n]) for an upscale: bicubic, no antialiasing, replicate edges."""
    return _resize_axis(_resize_axis(np.asarray(img, float), n, 0), n, 1)


def block2(img):
    """512 -> 256 by 2 x 2 mean (how the earlier analysis brought Cn.mat back to 256)."""
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def loadmat_img(path, key):
    from scipy.io import loadmat
    return np.asarray(loadmat(str(path))[key], dtype=float)
