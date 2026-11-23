"""Score one full-session re-registration (job A of the handoff). Read-only on its inputs.

    python score_full.py --label <text> --out <scores dir>
                         [--session-dir <TSeries folder with suite2p/plane0/ops.npy>]
                         [--ops-file <ops file handed to process_2p>] [--tif <AVG5x tif>]

Two independent parts; each runs when its input is given:
  --session-dir  from the kept ops.npy: share of raw frames within 1 px of the rigid search limit,
                 share shifted by more than 51 px (the old limit; np.roll wraps that strip),
                 median offset, corrXY, suite2p's own badframes. With --ops-file: was the forced
                 reference used, and do the frames in suite2p's raw binary equal the frames the
                 reference was built from (scale check).
  --tif          motion_metrics.py on the averaged movie: bad = share of frames whose correlation
                 with the movie's own template is below 0.5 x the quiet level (90th percentile),
                 degraded = below 0.75 x. Also per quarter of the session.
Writes <out>/<label>.json and the motion_metrics npz.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motion_metrics as mm  # noqa: E402

OLD_LIMIT = 51


def pct(x, q) -> float:
    return float(np.percentile(x, q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session-dir", type=Path)
    ap.add_argument("--ops-file", type=Path)
    ap.add_argument("--tif", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"label": args.label}

    if args.session_dir:
        plane = args.session_dir / "suite2p" / "plane0"
        ops = np.load(plane / "ops.npy", allow_pickle=True).item()
        yoff = np.asarray(ops["yoff"], np.float64)
        xoff = np.asarray(ops["xoff"], np.float64)
        cxy = np.asarray(ops["corrXY"], np.float64)
        n = len(yoff)
        ly, lx = int(ops["Ly"]), int(ops["Lx"])
        lim = int(min(round(float(ops["maxregshift"]) * min(ly, lx)), min(ly, lx) // 2))   # rigid.py: lcorr
        at_lim = (np.abs(yoff) >= lim - 1) | (np.abs(xoff) >= lim - 1)
        far = (np.abs(yoff) > OLD_LIMIT) | (np.abs(xoff) > OLD_LIMIT)
        dmag = np.hypot(yoff - np.median(yoff), xoff - np.median(xoff))
        bad_s2p = np.asarray(ops.get("badframes", np.zeros(n, bool)), bool)
        res.update(dict(
            nframes=int(ops["nframes"]), n_offsets=n, n_files=int(len(ops.get("frames_per_file", []))),
            maxregshift=float(ops["maxregshift"]), smooth_sigma=float(ops["smooth_sigma"]),
            nonrigid=bool(ops["nonrigid"]), force_refImg=bool(ops.get("force_refImg", False)),
            limit_px=lim, at_limit=float(at_lim.mean()), beyond_51px=float(far.mean()),
            med_y=float(np.median(yoff)), med_x=float(np.median(xoff)),
            dev_p50=pct(dmag, 50), dev_p95=pct(dmag, 95), dev_p99=pct(dmag, 99),
            max_abs_y=float(np.abs(yoff).max()), max_abs_x=float(np.abs(xoff).max()),
            corrXY_p50=pct(cxy, 50), corrXY_p05=pct(cxy, 5), corrXY_p01=pct(cxy, 1),
            s2p_badframes=float(bad_s2p.mean()),
            yrange=[int(v) for v in ops.get("yrange", [])], xrange=[int(v) for v in ops.get("xrange", [])],
        ))
        print(f"[{args.label}] suite2p: {res['nframes']} frames in {res['n_files']} files; maxregshift "
              f"{res['maxregshift']} (limit {lim} px), smooth_sigma {res['smooth_sigma']}, "
              f"force_refImg {res['force_refImg']}")
        print(f"  frames within 1 px of the limit: {100 * res['at_limit']:.2f}%   shifted more than "
              f"{OLD_LIMIT} px: {100 * res['beyond_51px']:.2f}%   suite2p badframes: {100 * res['s2p_badframes']:.2f}%")
        print(f"  median offset (y, x) = ({res['med_y']:+.0f}, {res['med_x']:+.0f}); |offset - median| "
              f"p50 {res['dev_p50']:.1f}  p95 {res['dev_p95']:.1f}  p99 {res['dev_p99']:.1f} px; max |y| "
              f"{res['max_abs_y']:.0f}, max |x| {res['max_abs_x']:.0f}")
        print(f"  corrXY p50 {res['corrXY_p50']:.4f}  p05 {res['corrXY_p05']:.4f}  p01 {res['corrXY_p01']:.4f}; "
              f"valid range y {res['yrange']} x {res['xrange']}")
        # quarters of the session: where do the limit hits and the far shifts sit
        q = np.minimum((np.arange(n) * 4) // n, 3)
        res["at_limit_quarters"] = [float(at_lim[q == k].mean()) for k in range(4)]
        res["beyond_51px_quarters"] = [float(far[q == k].mean()) for k in range(4)]
        res["med_y_quarters"] = [float(np.median(yoff[q == k])) for k in range(4)]
        res["med_x_quarters"] = [float(np.median(xoff[q == k])) for k in range(4)]
        print("  per quarter: median (y, x) " + "  ".join(
            f"({res['med_y_quarters'][k]:+.0f},{res['med_x_quarters'][k]:+.0f})" for k in range(4))
            + "; more than 51 px " + "  ".join(f"{100 * v:.1f}%" for v in res["beyond_51px_quarters"]))
        np.savez_compressed(args.out / (args.label + "_s2p.npz"), yoff=yoff, xoff=xoff, corrXY=cxy,
                            badframes=bad_s2p, refImg=np.asarray(ops["refImg"]),
                            meanImg=np.asarray(ops["meanImg"]))

        if args.ops_file:
            given = np.load(args.ops_file, allow_pickle=True).item()
            if "refImg" in given:
                same = bool(np.array_equal(np.asarray(given["refImg"]), np.asarray(ops["refImg"])))
                res["forced_ref_used"] = same
                print(f"  reference in ops.npy equals the forced reference: {same}")
                ref_npz = args.ops_file.with_name(args.ops_file.name.replace("ops_", "ref_", 1)).with_suffix(".npz")
                raw_bin = plane / "data_raw.bin"
                if ref_npz.is_file() and raw_bin.is_file():
                    z = np.load(ref_npz, allow_pickle=True)
                    idx = z["idx"]
                    mov = np.memmap(raw_bin, dtype=np.int16, mode="r", shape=(int(ops["nframes"]), ly, lx))
                    npages = z["npages"]
                    same_counts = bool(np.array_equal(np.asarray(ops["frames_per_file"]).astype(np.int64),
                                                      npages.astype(np.int64)))
                    res["frames_per_file_match"] = same_counts
                    print(f"  frames per file equal the page counts used for the reference: {same_counts}")
                    # suite2p 0.14.6 writes the rigid shift back into data_raw.bin (register_frames
                    # shifts a writable memmap view in place), so the binary frame must equal the
                    # tif frame, converted like suite2p does, rolled by (-yoff, -xoff). That checks
                    # the frame indexing and the intensity scale of the reference frames at once.
                    import tifffile
                    from natsort import natsorted
                    files = natsorted(str(p) for p in args.session_dir.glob("*.tif"))
                    if same_counts and len(files) == len(npages):
                        edges = np.concatenate([[0], np.cumsum(npages)])
                        hit = tot = 0
                        for k in range(0, len(idx), 5):
                            g = int(idx[k])
                            fi = int(np.searchsorted(edges, g, side="right") - 1)
                            with tifffile.TiffFile(files[fi], is_ome=False) as tf:
                                im = tf.pages[g - int(edges[fi])].asarray()
                            if im.dtype.type == np.uint16:
                                im = im // 2
                            rolled = np.roll(im.astype(np.int16), (-int(yoff[g]), -int(xoff[g])), axis=(0, 1))
                            hit += bool(np.array_equal(np.asarray(mov[g]), rolled))
                            tot += 1
                        res["ref_frames_checked"] = tot
                        res["ref_frames_match_binary"] = hit
                        print(f"  reference frames identical to suite2p's binary (after its rigid shift): "
                              f"{hit} of {tot} checked")

    if args.tif:
        with contextlib.redirect_stdout(io.StringIO()):
            mm.main([str(args.tif), "--out", str(args.out)])
        z = np.load(args.out / (args.tif.stem + ".npz"))
        r = z["r_raw"]
        nt = len(r)
        quiet = pct(r, 90)
        bad = r < 0.5 * quiet
        deg = r < 0.75 * quiet
        d = np.hypot(z["dy"], z["dx"])
        q = np.minimum((np.arange(nt) * 4) // nt, 3)
        res.update(dict(
            tif=str(args.tif), avg_frames=int(nt), quiet_r=quiet, bad=float(bad.mean()),
            degraded=float(deg.mean()), r_p50=pct(r, 50), r_p05=pct(r, 5),
            resid_gt2px=float((d > 2).mean()), resid_p95=pct(d, 95),
            bad_quarters=[float(bad[q == k].mean()) for k in range(4)],
            degraded_quarters=[float(deg[q == k].mean()) for k in range(4)],
        ))
        print(f"[{args.label}] movie {args.tif.name}: {nt} frames; bad {100 * res['bad']:.1f}%  degraded "
              f"{100 * res['degraded']:.1f}%  (quiet r {quiet:.3f}, r p50 {res['r_p50']:.3f}, p05 {res['r_p05']:.3f}); "
              f"rigid residual > 2 px {100 * res['resid_gt2px']:.1f}%")
        print("  bad per quarter: " + "  ".join(f"{100 * v:.1f}%" for v in res["bad_quarters"])
              + "; degraded per quarter: " + "  ".join(f"{100 * v:.1f}%" for v in res["degraded_quarters"]))

    (args.out / (args.label + ".json")).write_text(json.dumps(res, indent=1), encoding="ascii")
    return 0


if __name__ == "__main__":
    sys.exit(main())
