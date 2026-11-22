"""Re-register sample raw trials with suite2p under several settings; score each; validate 'prod'.

Scratch only. Run in the suite2p env.

Reference image: built the way production built it -- suite2p's compute_reference on the 300
frames it takes from the WHOLE session (scratch/raw/<session>/ref_frames.npy, fetched read-only
from the server) -- so 'prod' reproduces the delivered movie for the sample trials. The 5-frame
groups are aligned to the global frame grid of the session, so reproduced frames can be compared
one-to-one with frames of the delivered AVG5x tif (--validate).

Scores on the 5-frame-averaged output (each variant against its own clean template):
  bad<0.5 / <0.75  fraction of averaged frames whose correlation with the template is below
                   0.5 / 0.75 of the variant's quiet-frame level (p90)
  d>2px            fraction with a rigid residual above 2 px
  nr>2px           fraction with a non-rigid remainder (inner blocks) above 2 px
  crisp            sqrt(mean |grad(mean image)|^2) of the registered sample
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion_metrics as mm  # noqa: E402

from suite2p import default_ops  # noqa: E402
from suite2p.registration import register  # noqa: E402

DATA = Path(r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew")
B128 = {"block_size": [128, 128], "maxregshiftNR": 5}
VARIANTS = {
    "prod":             {},
    "rigid":            {"nonrigid": False},
    "mrs20":            {"maxregshift": 0.2},
    "mrs30":            {"maxregshift": 0.3},
    "mrs30_rigid":      {"maxregshift": 0.3, "nonrigid": False},
    "mrs30_blk128":     {"maxregshift": 0.3, **B128},
    "mrs30_t1":         {"maxregshift": 0.3, "smooth_sigma_time": 1.0},
    "mrs30_blk128_t1":  {"maxregshift": 0.3, "smooth_sigma_time": 1.0, **B128},
    "mrs30_sm2":        {"maxregshift": 0.3, "smooth_sigma": 2.0},
    "mrs30_blk128_sm2": {"maxregshift": 0.3, "smooth_sigma": 2.0, **B128},
    "blk128":           B128,
}


def base_ops() -> dict:
    ops = default_ops()
    ops.update(np.load(Path.home() / ".suite2p" / "ops" / "ops_user.npy", allow_pickle=True).item())
    ops["reg_tif"] = False
    ops["do_bidiphase"] = False
    ops["frames_include"] = -1
    return ops


def crispness(img: np.ndarray, m: int = 60) -> float:
    gy, gx = np.gradient(img.astype(np.float64))
    return float(np.sqrt((gy[m:-m, m:-m] ** 2 + gx[m:-m, m:-m] ** 2).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--variants", nargs="+", default=["prod"])
    ap.add_argument("--two-step", nargs="*", default=[], help="variants to ALSO run with a second pass")
    ap.add_argument("--recenter", nargs="*", default=[],
                    help="variants to ALSO run with the reference rebuilt at the modal field position")
    ap.add_argument("--validate", action="store_true", help="compare 'prod' with the delivered AVG5x tif")
    ap.add_argument("--factor", type=int, default=5)
    args = ap.parse_args()

    src = HERE / "raw" / args.session
    files = sorted(src.glob("trial*.npy"))
    tnum = [int(re.search(r"trial(\d+)", f.name).group(1)) for f in files]
    nfr = np.load(HERE / "out_xml" / (args.session + "_nfr.npy"))
    gstart = np.concatenate([[0], np.cumsum(nfr)])
    trials = [np.load(f) for f in files]
    nper = [t.shape[0] for t in trials]
    for t, n in zip(tnum, nper):
        if n != nfr[t - 1]:
            raise SystemExit(f"trial {t}: {n} frames on disk, XML says {nfr[t - 1]}")
    raw = np.concatenate(trials).astype(np.int16)
    del trials
    ref_frames = np.load(src / "ref_frames.npy").astype(np.int16)
    print(f"{args.session}: {len(files)} trials, {raw.shape[0]} raw frames; reference from "
          f"{ref_frames.shape[0]} full-session frames", flush=True)
    dst = HERE / "reg2" / args.session
    dst.mkdir(parents=True, exist_ok=True)

    # 5-frame groups on the session's global grid: (start in sample array, AVG5x frame index)
    groups = []
    a = 0
    for t, n in zip(tnum, nper):
        g0 = gstart[t - 1]
        skip = (-g0) % args.factor
        k = (n - skip) // args.factor
        for j in range(k):
            groups.append((a + skip + j * args.factor, (g0 + skip) // args.factor + j))
        a += n
    gidx = np.array([g[1] for g in groups])

    jobs = ([(v, "") for v in args.variants] + [(v, "2step") for v in args.two_step]
            + [(v, "recenter") for v in args.recenter])
    print(f"{'variant':20s} {'sec':>4s} | raw: |off-med| p50   p95   p99  f2f95  cXY50  cXY05 clip% "
          f"med(y,x) >51px% | AVG5x: bad<.5 bad<.75  d>2px nr>2px   r50   r05  crisp", flush=True)
    for vname, mode in jobs:
        two = mode == "2step"
        tag = vname + ("_" + mode if mode else "")
        ops = base_ops()
        ops.update(VARIANTS[vname])
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            ref = register.compute_reference(ref_frames.copy(), ops=ops)
            if mode == "recenter":
                # where do the 300 reference frames sit relative to suite2p's reference?
                tmp = np.zeros_like(ref_frames)
                r0 = register.compute_reference_and_register_frames(ref_frames.copy(), f_align_out=tmp,
                                                                    refImg=ref, ops={**ops, "nonrigid": False})
                y0, x0, c0 = [np.asarray(v) for v in r0[4]]
                good = c0 >= np.median(c0)
                my, mx = int(np.round(np.median(y0[good]))), int(np.round(np.median(x0[good])))
                tol = 5
                near = good & (np.abs(y0 - my) <= tol) & (np.abs(x0 - mx) <= tol)
                if near.sum() < 15:
                    tol = 10
                    near = good & (np.abs(y0 - my) <= tol) & (np.abs(x0 - mx) <= tol)
                # frames near the modal position, each moved the few px onto it, then averaged
                acc = np.zeros(ref.shape, np.float64)
                for i in np.where(near)[0]:
                    acc += np.roll(ref_frames[i], (-(int(y0[i]) - my), -(int(x0[i]) - mx)), axis=(0, 1))
                ref = (acc / near.sum()).astype(np.int16)
                recenter_note = (f"    recenter: modal offset of the 300 reference frames vs suite2p's reference "
                                 f"= (y {my:+d}, x {mx:+d}); new reference from {int(near.sum())} frames within {tol} px")
            out = np.zeros_like(raw)
            res = register.compute_reference_and_register_frames(raw.copy(), f_align_out=out, refImg=ref, ops=ops)
            if two:
                # second pass: reference = mean of the better-registered half of the first pass
                cxy1 = np.asarray(res[4][2])
                ref = out[cxy1 >= np.median(cxy1)].mean(axis=0).astype(np.int16)
                out = np.zeros_like(raw)
                res = register.compute_reference_and_register_frames(raw.copy(), f_align_out=out, refImg=ref, ops=ops)
        _, rmin, rmax, mean_img, rig, nonrig, _ = res
        yoff, xoff, cxy = [np.asarray(v, dtype=np.float32) for v in rig]
        lim = round(ops["maxregshift"] * 512)
        clip = float(((np.abs(yoff) >= lim - 1) | (np.abs(xoff) >= lim - 1)).mean())
        dmag = np.hypot(yoff - np.median(yoff), xoff - np.median(xoff))
        f2f = np.hypot(np.diff(yoff), np.diff(xoff))
        avg = np.stack([out[s:s + args.factor].mean(axis=0) for s, _ in groups])
        crisp = crispness(out.mean(axis=0, dtype=np.float64))
        tif = dst / f"{tag}.tif"
        tifffile.imwrite(tif, np.clip(avg, 0, 65535).astype(np.uint16), imagej=True)
        with contextlib.redirect_stdout(io.StringIO()):
            mm.main([str(tif), "--out", str(dst)])
        z = np.load(dst / f"{tag}.npz")
        d = np.hypot(z["dy"], z["dx"])
        inner = np.hypot(z["bdy"], z["bdx"])[:, 1:3, 1:3].reshape(len(d), -1).max(axis=1)
        q = np.percentile(z["r_raw"], 90)
        extra = {}
        if ops["nonrigid"]:
            extra = dict(y1=np.asarray(nonrig[0], np.float32), x1=np.asarray(nonrig[1], np.float32))
        np.savez_compressed(dst / f"{tag}_s2p.npz", yoff=yoff, xoff=xoff, corrXY=cxy, ref=ref,
                            mean_img=mean_img, nper=np.array(nper), gidx=gidx, **extra)
        far = float(((np.abs(yoff) > 51) | (np.abs(xoff) > 51)).mean())
        medtxt = f"({np.median(yoff):+.0f},{np.median(xoff):+.0f})"
        print(f"{tag:20s} {time.time() - t0:4.0f} | {np.percentile(dmag, 50):18.1f} {np.percentile(dmag, 95):5.1f} "
              f"{np.percentile(dmag, 99):5.1f} {np.percentile(f2f, 95):6.1f} {np.percentile(cxy, 50):6.4f} "
              f"{np.percentile(cxy, 5):6.4f} {100 * clip:5.1f} {medtxt:>8s} {100 * far:6.1f} | {(z['r_raw'] < 0.5 * q).mean():13.3f} "
              f"{(z['r_raw'] < 0.75 * q).mean():7.3f} {(d > 2).mean():6.3f} {(inner > 2).mean():6.3f} "
              f"{np.percentile(z['r_raw'], 50):5.3f} {np.percentile(z['r_raw'], 5):5.3f} {crisp:6.2f}", flush=True)

        if mode == "recenter":
            print(recenter_note, flush=True)
        if args.validate and tag == "prod":
            name = "AVG5x-" + args.session
            deliv = tifffile.memmap(DATA / name / (name + ".tif"), mode="r")
            rr = np.zeros(len(gidx), np.float32)
            for i, g in enumerate(gidx):
                a_ = avg[i, 30:-30, 30:-30].astype(np.float64).ravel()
                b_ = np.asarray(deliv[g, 30:-30, 30:-30], dtype=np.float64).ravel()
                rr[i] = np.corrcoef(a_, b_)[0, 1]
            # the same comparison one AVG frame off, as a null
            null = np.array([np.corrcoef(avg[i, 30:-30, 30:-30].ravel().astype(np.float64),
                                         np.asarray(deliv[min(g + 1, deliv.shape[0] - 1), 30:-30, 30:-30],
                                                    dtype=np.float64).ravel())[0, 1]
                             for i, g in list(enumerate(gidx))[::7]])
            print(f"    VALIDATION prod vs delivered tif ({len(gidx)} averaged frames): pixel r median "
                  f"{np.median(rr):.4f}, p05 {np.percentile(rr, 5):.4f}, min {rr.min():.4f}, "
                  f"frames with r > 0.98: {(rr > 0.98).mean():.1%}   [null, next frame: median {np.median(null):.4f}]",
                  flush=True)
            np.save(dst / "validation_r.npy", rr)
            # same comparison with the 5-frame groups started 1 or 2 raw frames early / late
            txt = []
            for off in (-2, -1, 1, 2):
                vals = []
                for i in range(0, len(groups), 7):
                    s0 = groups[i][0] + off
                    if s0 < 0 or s0 + args.factor > out.shape[0]:
                        continue
                    a_ = out[s0:s0 + args.factor].mean(axis=0)[30:-30, 30:-30].astype(np.float64).ravel()
                    b_ = np.asarray(deliv[gidx[i], 30:-30, 30:-30], dtype=np.float64).ravel()
                    vals.append(np.corrcoef(a_, b_)[0, 1])
                txt.append(f"{off:+d}: {np.median(vals):.4f}")
            print("    VALIDATION with the groups shifted by N raw frames: " + "  ".join(txt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
