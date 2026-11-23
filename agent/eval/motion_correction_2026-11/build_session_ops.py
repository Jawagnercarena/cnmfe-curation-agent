"""Build the per-session suite2p ops file for a full-session re-registration (job A of the handoff).

Run in the suite2p env. Reads a LOCAL copy of a raw TSeries folder; writes only into --out-dir.

    python build_session_ops.py <local TSeries folder> --out-dir <dir> [--maxregshift 0.2]
                                [--smooth-sigma S] [--no-recenter]

What it does:
  1. Lists the tifs the way suite2p does (natsorted) and counts the pages of each.
  2. Reads the frames suite2p would take for its reference image:
     np.linspace(0, n_frames, 1 + nimg_init, dtype=int)[:-1]   (register.py:396)
     and converts them the way suite2p's tif reader does (uint16 -> // 2 -> int16, io/tiff.py:131),
     so the reference is on the scale of suite2p's binary. This matters: with norm_frames on,
     every frame is clipped to the 1st..99th percentile of the reference.
  3. suite2p's own reference from those frames (register.compute_reference, new ops).
  4. Recentred reference, as in reg_experiment2.py: rigid-register the reference frames to
     suite2p's reference, take the median offset of the better-correlated half, average the
     frames within 5 px of it (10 px if fewer than 15), each rolled onto it.
  5. Writes ops_<session>.npy = ops_user.npy + the overrides + refImg + force_refImg True
     (run_s2p.py:110 uses ops["refImg"] only when force_refImg is set), and
     ref_<session>.npz with both references, the offsets and the frame indices.

Never writes ops_user.npy.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted

from suite2p import default_ops
from suite2p.registration import register

USER_OPS = Path.home() / ".suite2p" / "ops" / "ops_user.npy"


def to_s2p_int16(im: np.ndarray) -> np.ndarray:
    """Same conversion as suite2p.io.tiff.read_tiff."""
    if im.dtype.type == np.uint16:
        return (im // 2).astype(np.int16)
    if im.dtype.type != np.int16:
        return im.astype(np.int16)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--maxregshift", type=float, default=0.2)
    ap.add_argument("--smooth-sigma", type=float, default=None)
    ap.add_argument("--no-recenter", action="store_true",
                    help="only the overrides; suite2p builds its own reference")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--recipe", choices=["single", "iter"], default="single",
                    help="single = the handoff's one-pass recipe (median of the better-correlated half); "
                         "iter = median of all but the worst 10%%, repeated until within 2 px")
    args = ap.parse_args()

    session = args.session_dir.resolve()
    name = session.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("_" + args.tag) if args.tag else ""
    ops_out = args.out_dir / f"ops_{name}{suffix}.npy"
    ref_out = args.out_dir / f"ref_{name}{suffix}.npz"
    if ops_out.exists() or ref_out.exists():
        print(f"ERROR: {ops_out.name} or {ref_out.name} already exists; not overwriting")
        return 1

    saved = np.load(USER_OPS, allow_pickle=True).item()
    overrides = {"maxregshift": float(args.maxregshift)}
    if args.smooth_sigma is not None:
        overrides["smooth_sigma"] = float(args.smooth_sigma)
    new_ops = dict(saved)
    new_ops.update(overrides)

    if args.no_recenter:
        np.save(ops_out, new_ops, allow_pickle=True)
        print(f"{name}: overrides {overrides}, no forced reference -> {ops_out}")
        return 0

    # ---- frames suite2p would take for its reference
    files = natsorted(str(p) for p in session.glob("*.tif"))
    if not files:
        print(f"ERROR: no tif files in {session}")
        return 1
    t0 = time.time()
    npages = np.zeros(len(files), np.int64)
    for i, f in enumerate(files):
        with tifffile.TiffFile(f, is_ome=False) as tf:
            npages[i] = len(tf.pages)
    n_frames = int(npages.sum())
    nimg = int(min(saved.get("nimg_init", 300), n_frames))
    idx = np.linspace(0, n_frames, 1 + nimg, dtype=int)[:-1]
    edges = np.concatenate([[0], np.cumsum(npages)])
    fidx = np.searchsorted(edges, idx, side="right") - 1
    page = idx - edges[fidx]
    frames = None
    for fi in np.unique(fidx):
        with tifffile.TiffFile(files[fi], is_ome=False) as tf:
            for k in np.where(fidx == fi)[0]:
                im = to_s2p_int16(tf.pages[int(page[k])].asarray())
                if frames is None:
                    frames = np.zeros((nimg,) + im.shape, np.int16)
                    raw_dtype = str(tf.pages[int(page[k])].dtype)
                frames[k] = im
    print(f"{name}: {len(files)} tifs, {n_frames} frames, {nimg} reference frames "
          f"(raw dtype {raw_dtype}, mean after conversion {frames.mean():.1f}) in {time.time() - t0:.0f}s",
          flush=True)

    # ---- suite2p's reference, then the recentred one (reg_experiment2.py lines 121-139)
    ops = default_ops()
    ops.update(new_ops)

    def offsets(ref_img):
        with contextlib.redirect_stdout(io.StringIO()):
            out = register.compute_reference_and_register_frames(
                frames.copy(), f_align_out=np.zeros_like(frames), refImg=ref_img,
                ops={**ops, "nonrigid": False, "reg_tif": False})
        return [np.asarray(v) for v in out[4]]

    with contextlib.redirect_stdout(io.StringIO()):
        ref0 = register.compute_reference(frames.copy(), ops=ops)
    y0, x0, c0 = offsets(ref0)

    if args.recipe == "iter":
        # The phase-correlation peak falls with the size of the shift (less overlap), so the
        # 'better-correlated half' of the single-pass recipe is biased toward suite2p's own
        # reference position (090826-pnb104: all-frame median (+14,+45), better half (+13,+24)).
        # Here: median over all frames but the worst-correlated 10% (out-of-plane frames),
        # rebuild at that position, and repeat until the median offset is within 2 px.
        lim_old = round(float(saved["maxregshift"]) * min(ref0.shape))
        ref, y, x, c = ref0, y0, x0, c0
        total = np.array([0, 0])
        steps = []
        n_used = 0
        for it in range(4):
            keep = c >= np.percentile(c, 10)
            my, mx = int(np.round(np.median(y[keep]))), int(np.round(np.median(x[keep])))
            # first pass: leave suite2p's reference alone within 5 px; later passes: done within 2 px
            if max(abs(my), abs(mx)) <= (2 if steps else 5):
                break
            for tol in (5, 10):
                near = keep & (np.abs(y - my) <= tol) & (np.abs(x - mx) <= tol)
                if near.sum() >= 15:
                    break
            if near.sum() < 5:
                print(f"ERROR: only {int(near.sum())} frames near (y {my:+d}, x {mx:+d}) in pass {it + 1}")
                return 1
            acc = np.zeros(ref0.shape, np.float64)
            for i in np.where(near)[0]:
                acc += np.roll(frames[i], (-(int(y[i]) - my), -(int(x[i]) - mx)), axis=(0, 1))
            ref = (acc / near.sum()).astype(np.int16)
            n_used = int(near.sum())
            total += (my, mx)
            steps.append((my, mx, n_used, tol))
            y, x, c = offsets(ref)
        keep = c >= np.percentile(c, 10)
        res_y, res_x = float(np.median(y[keep])), float(np.median(x[keep]))
        far0 = float(((np.abs(y0) > lim_old) | (np.abs(x0) > lim_old)).mean())
        far1 = float(((np.abs(y) > lim_old) | (np.abs(x) > lim_old)).mean())
        print("  iterated recentring: " + ("; ".join(
            f"pass {i + 1} moved (y {s[0]:+d}, x {s[1]:+d}) using {s[2]} frames within {s[3]} px"
            for i, s in enumerate(steps)) or "suite2p's reference already central"))
        print(f"  total move (y {total[0]:+d}, x {total[1]:+d}); remaining median offset of the reference frames "
              f"(y {res_y:+.0f}, x {res_x:+.0f}); mean |x| {np.abs(x).mean():.1f} px (was {np.abs(x0).mean():.1f})")
        print(f"  reference frames more than {lim_old} px away: {100 * far0:.1f}% from suite2p's reference, "
              f"{100 * far1:.1f}% from the recentred one")
        if steps:
            new_ops["refImg"] = ref
            new_ops["force_refImg"] = True
        np.save(ops_out, new_ops, allow_pickle=True)
        np.savez_compressed(ref_out, ref_suite2p=ref0, ref_recentred=ref, y0=y0, x0=x0, c0=c0,
                            y_final=y, x_final=x, c_final=c, my=int(total[0]), mx=int(total[1]),
                            steps=np.array(steps), n_near=n_used, idx=idx, npages=npages,
                            overrides=np.array(sorted(overrides.items()), dtype=object))
        print(f"  overrides {overrides}" + (" + refImg + force_refImg" if steps else "") + f" -> {ops_out}")
        return 0

    good = c0 >= np.median(c0)
    my, mx = int(np.round(np.median(y0[good]))), int(np.round(np.median(x0[good])))
    tol = 5
    near = good & (np.abs(y0 - my) <= tol) & (np.abs(x0 - mx) <= tol)
    if near.sum() < 15:
        tol = 10
        near = good & (np.abs(y0 - my) <= tol) & (np.abs(x0 - mx) <= tol)
    if near.sum() < 5:
        print(f"ERROR: only {int(near.sum())} reference frames within {tol} px of the modal offset "
              f"(y {my:+d}, x {mx:+d}); no reference built")
        return 1
    acc = np.zeros(ref0.shape, np.float64)
    for i in np.where(near)[0]:
        acc += np.roll(frames[i], (-(int(y0[i]) - my), -(int(x0[i]) - mx)), axis=(0, 1))
    ref = (acc / near.sum()).astype(np.int16)

    lim_old = round(float(saved["maxregshift"]) * max(ref.shape))
    far = float(((np.abs(y0) > lim_old) | (np.abs(x0) > lim_old)).mean())
    far_new = float(((np.abs(y0 - my) > lim_old) | (np.abs(x0 - mx) > lim_old)).mean())
    print(f"  modal offset of the reference frames vs suite2p's reference: (y {my:+d}, x {mx:+d}); "
          f"new reference from {int(near.sum())} frames within {tol} px")
    print(f"  reference frames more than {lim_old} px from suite2p's reference: {100 * far:.1f}%; "
          f"from the recentred position: {100 * far_new:.1f}%")

    new_ops["refImg"] = ref
    new_ops["force_refImg"] = True
    np.save(ops_out, new_ops, allow_pickle=True)
    np.savez_compressed(ref_out, ref_suite2p=ref0, ref_recentred=ref, y0=y0, x0=x0, c0=c0,
                        my=my, mx=mx, tol=tol, n_near=int(near.sum()), idx=idx, npages=npages,
                        overrides=np.array(sorted(overrides.items()), dtype=object))
    print(f"  overrides {overrides} + refImg + force_refImg -> {ops_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
