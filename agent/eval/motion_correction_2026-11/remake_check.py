"""Does a full re-made session agree with its raw-sample test?

    python remake_check.py --session <TSeries folder kept by process_2p --keep-tseries>
                           --new-tif <AVG5x tif written by that run>
                           --delivered-tif <old AVG5x tif>
                           --sample-dir <scratch dir written by raw_sample_test.py>

Read-only. Checks, for the frames the raw-sample test registered with the installed
rule (scores.npz: new_yoff / new_xoff per sampled raw frame, gidx = averaged-frame
indices of new_avg.tif):
  1. frame counts: new tif, delivered tif and sum(XML frames) // 5 agree;
  2. rigid shifts: the full run's ops.npy yoff / xoff at the same raw frames equal the
     sample's (same reference -> same shifts);
  3. pixels: per-frame Pearson r between the sample's averaged frames and the same
     frames of the new movie (8-bit scaling differs, so r, not equality).
Run in the suite2p env (raw_sample_test imports suite2p).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from raw_sample_test import FACTOR, xml_frames  # noqa: E402


def frame_r(a, b):
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--new-tif", type=Path, required=True)
    ap.add_argument("--delivered-tif", type=Path, required=True)
    ap.add_argument("--sample-dir", type=Path, required=True)
    a = ap.parse_args()

    nfr = xml_frames(a.session)
    gstart = np.concatenate([[0], np.cumsum(nfr)]).astype(int)
    new = tifffile.memmap(a.new_tif, mode="r")
    old = tifffile.memmap(a.delivered_tif, mode="r")
    n_xml = int(gstart[-1]) // FACTOR
    print(f"{a.session.name}")
    print(f"  frames: new {new.shape[0]}, delivered {old.shape[0]}, XML // {FACTOR} = {n_xml}, "
          f"raw {int(gstart[-1])} in {len(nfr)} cycles; new dtype {new.dtype}, shape {new.shape[1:]}")
    ok = new.shape[0] == old.shape[0] == n_xml

    z = np.load(a.sample_dir / "scores.npz")
    ops = np.load(a.session / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()
    yoff, xoff = np.asarray(ops["yoff"]), np.asarray(ops["xoff"])
    if len(yoff) != int(gstart[-1]):
        print(f"  !! ops.npy has {len(yoff)} rigid shifts, XML says {int(gstart[-1])} raw frames")
        ok = False
    cycles = sorted(set(z["gcyc"].tolist()))
    raw_idx = np.concatenate([np.arange(gstart[c - 1], gstart[c]) for c in cycles])
    if len(raw_idx) != len(z["new_yoff"]):
        print(f"  !! sampled cycles {cycles} hold {len(raw_idx)} raw frames, "
              f"the sample has {len(z['new_yoff'])}")
        ok = False
    else:
        dy = np.abs(yoff[raw_idx] - z["new_yoff"])
        dx = np.abs(xoff[raw_idx] - z["new_xoff"])
        same = np.mean((dy == 0) & (dx == 0))
        print(f"  rigid shifts at the {len(raw_idx)} sampled raw frames (cycles {cycles[0]}-{cycles[-1]}, "
              f"{len(cycles)} cycles): identical {100 * same:.2f}%, max |dy| {dy.max():.0f}, "
              f"max |dx| {dx.max():.0f}")
        lim = int(round(ops["maxregshift"] * max(ops["Ly"], ops["Lx"])))
        at = np.mean((np.abs(yoff) >= lim - 1) | (np.abs(xoff) >= lim - 1))
        print(f"  whole session: maxregshift {ops['maxregshift']} (+/-{lim} px), "
              f"raw frames within 1 px of the limit {100 * at:.3f}%, "
              f"median offset (y {np.median(yoff):+.0f}, x {np.median(xoff):+.0f})")

    samp = tifffile.imread(a.sample_dir / "new_avg.tif")
    gidx = z["gidx"]
    if len(samp) != len(gidx):
        print(f"  !! new_avg.tif has {len(samp)} frames, gidx {len(gidx)}")
        ok = False
    else:
        r = frame_r(np.asarray(new[gidx]), samp)
        print(f"  pixels, {len(gidx)} sampled averaged frames: r min {r.min():.4f}, "
              f"p1 {np.percentile(r, 1):.4f}, median {np.median(r):.4f}")
    print("  frame counts agree" if ok else "  !! CHECK FAILED (see above)")


if __name__ == "__main__":
    main()
