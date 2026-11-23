"""Fetch exactly the frames suite2p uses to build its reference image for a full session.

suite2p takes frames np.linspace(0, n_frames, 1 + nimg_init, dtype=int)[:-1] of the whole movie
(register.compute_reference_and_register_frames). Read-only on the server: opens each trial's
ome.tif and reads the needed pages. Writes scratch/raw/<session>/ref_frames.npy (+ indices).
"""
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

RAW = Path(r"X:\Jeremy\2p data\6odorDualDiffRew")
HERE = Path(__file__).resolve().parent
NIMG = 300

session = sys.argv[1]
day = session.split("-")[1]
src = RAW / day / session
nfr = np.load(HERE / "out_xml" / (session + "_nfr.npy"))
n_frames = int(nfr.sum())
edges = np.concatenate([[0], np.cumsum(nfr)])
idx = np.linspace(0, n_frames, 1 + min(NIMG, n_frames), dtype=int)[:-1]
trial = np.searchsorted(edges, idx, side="right") - 1
page = idx - edges[trial]
out = np.zeros((len(idx), 512, 512), np.uint16)
t0 = time.time()
for tr in np.unique(trial):
    hits = sorted(src.glob(f"*_Cycle{tr + 1:05d}_Ch2_*.ome.tif"))
    if len(hits) != 1:
        raise SystemExit(f"trial {tr + 1}: expected 1 file, found {len(hits)}")
    with tifffile.TiffFile(hits[0], is_ome=False) as tf:
        if len(tf.pages) != nfr[tr]:
            raise SystemExit(f"trial {tr + 1}: {len(tf.pages)} pages, XML says {nfr[tr]}")
        for k in np.where(trial == tr)[0]:
            out[k] = tf.pages[int(page[k])].asarray()
dst = HERE / "raw" / session
dst.mkdir(parents=True, exist_ok=True)
np.save(dst / "ref_frames.npy", out)
np.save(dst / "ref_frames_idx.npy", idx)
print(f"{session}: {len(idx)} reference frames of {n_frames} ({len(np.unique(trial))} trials) "
      f"in {time.time() - t0:.0f}s; mean {out.mean():.0f}")
