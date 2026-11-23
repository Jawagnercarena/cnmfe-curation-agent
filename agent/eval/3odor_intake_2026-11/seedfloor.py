"""Read-only.  Where is the NOISE floor of ACORN's seeding, per session?

Cn and PNR are both normalised to each pixel's own noise (PNR = max / sigma, Cn is
computed after zeroing everything below 3 sigma), so pure noise should reach the
same Cn / PNR whatever its amplitude; what it reaches depends on the movie length
T and on the noise's spatial structure.  This script measures that floor.

  noise   per session, the spatial correlation of the frame-to-frame residual
          r(t) = y(t) - (y(t-1) + y(t+1)) / 2 on a 1000-frame mid block at 256 x 256
          (ssub 2, tsub 2): lag-1 / lag-2 in x and y.  White noise gives 0.
  null    Gaussian noise movie, 256 x 256 x T, optionally given the measured lag-1
          correlation in x and y (3-tap kernel per axis), through the port of greedyROI_endoscope.m in
          video_compare.py (psf gSig 5 / gSiz 18, median, get_noise_fft, 3 sigma, Cn).
  seeds   first-pass replay of greedyROI's seed screen on (Cn, PNR) images:
          v = Cn.*PNR, zero where Cn < c or PNR < p; medfilt2 round(gSiz/4) = 5x5
          (zero padded); re-zero; local max over 2*round(gSiz/4)+1 = 11x11; border
          bd 10 zeroed; seed = local max with v >= c*p.  Counted on a (c, p) grid.

Subcommands:
  noise --only <substr...> --out <tsv>
  null  --T <frames at half rate> --rhox <lag-1 x> --rhoy <lag-1 y> --seed <int> --out <npz>
  real  --out <dir>        seed grids for every session image in vfull/img (+ 6odor Cn.mat/pnr.mat)
Writes only under --out.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_compare as vc   # noqa: E402

C_GRID = np.round(np.arange(0.30, 0.951, 0.025), 3)
P_GRID = np.round(np.arange(3.0, 15.01, 0.5), 2)
BD = 10


def seed_grid(cn, pnr, cg=C_GRID, pg=P_GRID):
    cn = np.nan_to_num(cn.astype(np.float64))
    pnr = np.nan_to_num(pnr.astype(np.float64))
    out = np.zeros((len(cg), len(pg)), np.int32)
    for i, c in enumerate(cg):
        for j, p in enumerate(pg):
            v = cn * pnr
            off = (cn < c) | (pnr < p)
            v[off] = 0
            v = ndi.median_filter(v, size=5, mode="constant", cval=0.0)
            v[off] = 0
            vmax = ndi.maximum_filter(v, size=11, mode="constant", cval=0.0)
            v[:BD, :] = v[-BD:, :] = v[:, :BD] = v[:, -BD - 1:] = 0
            out[i, j] = int(((v == vmax) & (vmax > 0) & (v >= c * p)).sum())
    return out


def residual_corr(Y):
    r = Y[1:-1] - 0.5 * (Y[:-2] + Y[2:])
    r = r - r.mean(0)
    r = r / (r.std(0) + 1e-9)
    r = r[:, 10:-10, 10:-10]
    return {"rx1": float((r[:, :, 1:] * r[:, :, :-1]).mean()), "ry1": float((r[:, 1:] * r[:, :-1]).mean()),
            "rx2": float((r[:, :, 2:] * r[:, :, :-2]).mean()), "ry2": float((r[:, 2:] * r[:, :-2]).mean())}


def cmd_noise(a):
    rows = []
    for group, stem, tif, sd in vc.tif_list(a.only):
        nf, off = vc.layout(tif)
        st = max(0, nf // 2 - 500)
        Y = vc.downsample(vc.read_block(tif, off, st, min(1000, nf - st)))
        d = residual_corr(Y)
        d.update(group=group, session=stem.replace("AVG5x-TSeries-", ""))
        rows.append(d)
        print(d, flush=True)
    keys = ["group", "session", "rx1", "ry1", "rx2", "ry2"]
    with open(a.out, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for d in rows:
            fh.write("\t".join(("%.4f" % d[k]) if isinstance(d[k], float) else d[k] for k in keys) + "\n")


def tap(rho):
    # 3-tap kernel [a, 1, a]: lag-1 autocorrelation 2a / (1 + 2a^2) = rho
    return 0.0 if rho <= 0 else float((1.0 - np.sqrt(1.0 - 2.0 * rho * rho)) / (2.0 * rho))


def cmd_null(a):
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    ax, ay = tap(a.rhox), tap(a.rhoy)
    Y = np.empty((a.T, 256, 256), np.float32)
    for i in range(0, a.T, 500):
        n = min(500, a.T - i)
        z = rng.standard_normal((n, 256 + 16, 256 + 16)).astype(np.float32)
        if ax > 0:
            z = ndi.correlate1d(z, np.array([ax, 1.0, ax], np.float32), axis=2, mode="wrap")
        if ay > 0:
            z = ndi.correlate1d(z, np.array([ay, 1.0, ay], np.float32), axis=1, mode="wrap")
        Y[i:i + n] = z[:, 8:-8, 8:-8]
    rc = residual_corr(Y[:1000])
    cn, pnr = vc.cn_pnr_ds(Y)
    del Y
    g = seed_grid(cn, pnr)
    np.savez_compressed(a.out, cn=cn, pnr=pnr, grid=g, c_grid=C_GRID, p_grid=P_GRID, T=a.T, rhox=a.rhox,
                        rhoy=a.rhoy, **rc)
    c = cn[10:-10, 10:-10]
    p = pnr[10:-10, 10:-10]
    print("null T %d target rx/ry %.2f/%.2f | measured %.3f/%.3f | Cn p50 %.3f p99 %.3f p99.9 %.3f max %.3f | "
          "PNR p50 %.2f p99 %.2f p99.9 %.2f max %.2f | %.0fs" % (
              a.T, a.rhox, a.rhoy, rc["rx1"], rc["ry1"], *np.percentile(c, [50, 99, 99.9]), c.max(),
              *np.percentile(p, [50, 99, 99.9]), p.max(), time.time() - t0), flush=True)
    for cc in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        i = int(np.argmin(np.abs(C_GRID - cc)))
        print("   c %.2f seeds at p 4/5/6/7/8: %s" % (cc, [int(g[i, np.argmin(np.abs(P_GRID - pp))])
                                                       for pp in (4, 5, 6, 7, 8)]), flush=True)


def cmd_real(a):
    a.out.mkdir(parents=True, exist_ok=True)
    img = Path(__file__).resolve().parent / "vfull" / "img"
    for group, stem, tif, sd in vc.tif_list():
        name = stem.replace("AVG5x-TSeries-", "")
        dst = a.out / (name + ".npz")
        if dst.exists():
            continue
        if group == "3odor":
            f = img / (name + ".npz")
            if not f.exists():
                print("no image file for", name, flush=True)
                continue
            z = np.load(f)
            cn, pnr = z["full_cn"], z["full_pnr"]
        else:
            C = vc.mat_img(sd / "Cn.mat", "Cn")
            P = vc.mat_img(sd / "pnr.mat", "pnr")
            cn = C.reshape(256, 2, 256, 2).mean(axis=(1, 3)) if C.shape == (512, 512) else C
            pnr = P
        nf, _ = vc.layout(tif)
        g = seed_grid(cn, pnr)
        np.savez_compressed(dst, grid=g, c_grid=C_GRID, p_grid=P_GRID, T=nf // 2, group=group)
        print(group, name, nf // 2, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("noise")
    n.add_argument("--only", nargs="*", default=None)
    n.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("null")
    u.add_argument("--T", type=int, required=True)
    u.add_argument("--rhox", type=float, default=0.0)
    u.add_argument("--rhoy", type=float, default=0.0)
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("real")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    return {"noise": cmd_noise, "null": cmd_null, "real": cmd_real}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
