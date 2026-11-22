"""Copy a few raw trials of one TSeries from the server into local scratch as .npy.

Read-only on the server (opens the ome.tif files for reading, nothing else). One .npy per
trial under scratch/raw/<session>/trialNNN.npy (uint16, frames x 512 x 512).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

RAW = Path(r"X:\Jeremy\2p data\6odorDualDiffRew")
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("session")                      # TSeries-090826-pnb104-425um-3z-000
ap.add_argument("--trials", type=int, nargs="+", required=True, help="1-based cycle numbers")
args = ap.parse_args()

day = args.session.split("-")[1]
src = RAW / day / args.session
dst = HERE / "raw" / args.session
dst.mkdir(parents=True, exist_ok=True)
for tr in args.trials:
    out = dst / f"trial{tr:03d}.npy"
    if out.is_file():
        print(f"have {out.name}")
        continue
    hits = sorted(src.glob(f"*_Cycle{tr:05d}_Ch2_*.ome.tif"))
    if len(hits) != 1:
        print(f"trial {tr}: expected 1 file, found {len(hits)}")
        continue
    t0 = time.time()
    with tifffile.TiffFile(hits[0], is_ome=False) as tf:
        arr = np.stack([p.asarray() for p in tf.pages])
    np.save(out, arr)
    print(f"trial {tr}: {arr.shape} {arr.dtype} min {arr.min()} max {arr.max()} mean {arr.mean():.1f} "
          f"({hits[0].stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
sys.exit(0)
