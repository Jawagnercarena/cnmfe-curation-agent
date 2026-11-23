"""C7.  Degrade labelled bright sessions artificially and recompute the seeding images.

For each (session, variant, level) the tif is re-read at full resolution and full rate and
  noise   + sigma * Gaussian noise (lag-1 spatial correlation built like the measured one)
  blur    Gaussian blur (sigma --blur px at 512) of every frame, then the same noise
  shot    + noise whose variance is proportional to the pixel value (signal-dependent)
is applied, rounded to integers (NOT clipped to 0..255: the real dim tifs do not clip either,
and clipping a symmetric additive noise would), then downsampled exactly like the pipeline.
sigma is set so that noise_rel (noise / p1-p99 range of the mean image, measured on the
1000 mid-session frames, as in the session table) reaches --levels; the achieved value is
measured on the degraded block and stored.
Output cache/degrade/<session>__<variant>__<level>.npz: cn, pnr (seeding port), pcn, ppnr
(pre-pass port), est_mc / est_mp (params.py estimate), sigma, achieved noise_rel.
Reads the tif only.  Usage: c7_degrade.py --only <substr ...> [--variants noise blur] [--levels ...]
"""
import argparse
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_core as rc  # noqa: E402
import noise_null as nn  # noqa: E402

OUT = rc.CACHE / "degrade"
KX, KY = nn.taps512(0.18), nn.taps512(0.13)
KNORM = float(np.sqrt((KX ** 2).sum() * (KY ** 2).sum()))


def noise_rel_of(B):
    bm = B.mean(0)
    d = np.median(np.abs(np.diff(B, axis=0)), axis=0) / 0.6745 / np.sqrt(2)
    nu = float(np.median(d[24:-24, 24:-24]))
    rng = float(np.percentile(bm[24:-24, 24:-24], 99) - np.percentile(bm[24:-24, 24:-24], 1))
    return nu, rng


def degrade(raw, variant, sigma, blur, gen):
    y = raw.astype(np.float32)
    if variant == "blur":
        y = ndi.gaussian_filter(y, (0, blur, blur), mode="nearest")
    z = gen.standard_normal(y.shape, dtype=np.float32)
    z = nn.smooth(z, KX, KY) / KNORM
    if variant == "shot":
        y = y + sigma * np.sqrt(np.maximum(y, 0)) * z          # sigma is per sqrt(count) here
    else:
        y = y + sigma * z
    return np.rint(y)


def one(job):
    """One session: the tif is read ONCE into RAM (uint8), then every (variant, level)."""
    (group, name, tif, sd, an), variants, levels, blur = job
    todo = [(v, lv) for v in variants for lv in levels
            if not (OUT / ("%s__%s__%.2f.npz" % (name, v, lv))).exists()]
    if not todo:
        return name + " (all cached)"
    mm = rc.open_movie(tif)
    nf = mm.shape[0]
    raw = np.empty(mm.shape, np.uint8)
    for s in range(0, nf, 2048):
        raw[s:s + 2048] = mm[s:s + 2048]
    del mm
    st = max(0, nf // 2 - 500)
    B = raw[st:st + min(1000, nf - st)].astype(np.float32)
    nu0, rng0 = noise_rel_of(B)
    msgs = []
    for variant, level in todo:
        t0 = time.time()
        dst = OUT / ("%s__%s__%.2f.npz" % (name, variant, level))
        Bb = ndi.gaussian_filter(B, (0, blur, blur), mode="nearest") if variant == "blur" else B
        nu, rng = noise_rel_of(Bb)
        want = level * rng
        if variant == "shot":
            med = float(np.median(Bb.mean(0)[24:-24, 24:-24]))
            sigma = float(np.sqrt(max(want ** 2 - nu ** 2, 0.0) / max(med, 1e-6)))
        else:
            sigma = float(np.sqrt(max(want ** 2 - nu ** 2, 0.0)))
        gen = np.random.default_rng([zlib.crc32(name.encode()), int(round(level * 100)), len(variant)])
        got_nu, got_rng = noise_rel_of(degrade(B, variant, sigma, blur, np.random.default_rng(7)))
        n = (nf // 2) * 2
        Y = np.empty((n // 2, 256, 256), np.float32)
        for s in range(0, n, 512):
            e = min(s + 512, n)
            Y[s // 2:e // 2] = rc.ds_block(degrade(raw[s:e], variant, sigma, blur, gen))
        cn, pnr = rc.seeding_images(Y)
        pcn, ppnr = rc.prepass_images(Y)
        T = Y.shape[0]
        del Y
        emc, emp = rc.estimate_params(pcn, ppnr)
        rmc, rmp = rc.estimate_params(pcn, ppnr, raw=True)
        tmp = OUT / ("%s__%s__%.2f.tmp.npz" % (name, variant, level))
        np.savez_compressed(tmp, cn=cn.astype(np.float32), pnr=pnr.astype(np.float32), pcn=pcn.astype(np.float32),
                            ppnr=ppnr.astype(np.float32), est_mc=emc, est_mp=emp, raw_mc=rmc, raw_mp=rmp, sigma=sigma,
                            level=level, variant=variant, blur=blur, T=T, noise_rel_before=nu0 / rng0,
                            noise_rel_got=got_nu / got_rng, session=name)
        tmp.replace(dst)
        msg = "%s %s level %.2f: sigma %.2f, noise_rel %.3f -> %.3f, pk50 %.3f, est %.2f/%.1f, %.0fs" % (
            name, variant, level, sigma, nu0 / rng0, got_nu / got_rng, rc.pk(cn, 50), emc, emp, time.time() - t0)
        print(msg, flush=True)
        msgs.append(msg)
    return "%s done (%d conditions)" % (name, len(msgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["noise", "blur"])
    ap.add_argument("--levels", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--blur", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rc.NW = 6
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.variants, a.levels, a.blur) for s in rc.sessions(a.only) if s[0] == "6odor"]
    print("%d sessions x %d conditions" % (len(jobs), len(a.variants) * len(a.levels)), flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
