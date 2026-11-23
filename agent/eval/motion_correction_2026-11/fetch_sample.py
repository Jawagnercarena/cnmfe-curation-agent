"""Fetch a raw sample of one TSeries from ANY task folder on the server (read-only there).

    python fetch_sample.py <server TSeries folder> --scratch <local dir> [--n-trials 12]

Generalises xml_info.py + fetch_raw_trials.py + fetch_ref_frames.py, which are tied to
6odorDualDiffRew. Writes, in the layout reg_experiment2.py expects under --scratch:
    out_xml/<session>_nfr.npy          frames per sequence, from the Prairie XML
    raw/<session>/trialNNN.npy         --n-trials evenly spaced trials (uint16)
    raw/<session>/ref_frames.npy       the 300 frames suite2p takes for its reference
    raw/<session>/ref_frames_idx.npy
Opens the server files for reading only.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

NIMG = 300


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--n-trials", type=int, default=12)
    args = ap.parse_args()
    src = args.src
    name = src.name
    xml = src / (name + ".xml")
    if not xml.is_file():
        print(f"ERROR: no {xml.name} in {src}")
        return 1
    text = xml.read_text(encoding="utf-8-sig", errors="replace")
    starts = [m.start() for m in re.finditer(r"<Sequence ", text)]
    bounds = starts + [len(text)]
    nfr = np.array([text.count("<Frame ", bounds[i], bounds[i + 1]) for i in range(len(starts))])
    del text
    ntr = len(nfr)
    n_frames = int(nfr.sum())
    (args.scratch / "out_xml").mkdir(parents=True, exist_ok=True)
    np.save(args.scratch / "out_xml" / (name + "_nfr.npy"), nfr)
    vals, cnts = np.unique(nfr, return_counts=True)
    print(f"{name}: {ntr} sequences, {n_frames} frames ("
          + ", ".join(f"{v} x{c}" for v, c in zip(vals, cnts)) + ")", flush=True)

    dst = args.scratch / "raw" / name
    dst.mkdir(parents=True, exist_ok=True)

    def trial_files(tr: int) -> list[Path]:
        """One multi-page file per cycle (current Prairie View) or one file per frame (older)."""
        hits = sorted(src.glob(f"*_Cycle{tr:05d}_Ch2_*.ome.tif"))
        if len(hits) != 1 and len(hits) != nfr[tr - 1]:
            raise SystemExit(f"trial {tr}: {len(hits)} files, XML says {nfr[tr - 1]} frames")
        return hits

    def read_page(tr: int, page: int) -> np.ndarray:
        hits = trial_files(tr)
        if len(hits) == 1:
            with tifffile.TiffFile(hits[0], is_ome=False) as tf:
                return tf.pages[page].asarray()
        with tifffile.TiffFile(hits[page], is_ome=False) as tf:
            return tf.pages[0].asarray()

    # ---- evenly spaced trials
    step = max(1, ntr // args.n_trials)
    trials = list(range(min(5, ntr), ntr + 1, step))[:args.n_trials]
    t0 = time.time()
    arr = None
    for tr in trials:
        out = dst / f"trial{tr:03d}.npy"
        if out.is_file():
            continue
        hits = trial_files(tr)
        if len(hits) == 1:
            with tifffile.TiffFile(hits[0], is_ome=False) as tf:
                arr = np.stack([p.asarray() for p in tf.pages])
        else:
            frames = []
            for h in hits:
                with tifffile.TiffFile(h, is_ome=False) as tf:
                    frames.append(tf.pages[0].asarray())
            arr = np.stack(frames)
        if arr.shape[0] != nfr[tr - 1]:
            raise SystemExit(f"trial {tr}: {arr.shape[0]} pages, XML says {nfr[tr - 1]}")
        np.save(out, arr)
    if arr is not None:
        print(f"  trials {trials} ({arr.shape[1]}x{arr.shape[2]} {arr.dtype}, "
              f"{'one file per cycle' if len(hits) == 1 else 'one file per frame'}) in {time.time() - t0:.0f}s",
              flush=True)

    # ---- the reference frames (register.py:396)
    if not (dst / "ref_frames.npy").is_file():
        t0 = time.time()
        edges = np.concatenate([[0], np.cumsum(nfr)])
        idx = np.linspace(0, n_frames, 1 + min(NIMG, n_frames), dtype=int)[:-1]
        trial = np.searchsorted(edges, idx, side="right") - 1
        page = idx - edges[trial]
        ref = None
        for k in range(len(idx)):
            im = read_page(int(trial[k]) + 1, int(page[k]))
            if ref is None:
                ref = np.zeros((len(idx),) + im.shape, np.uint16)
            ref[k] = im
        np.save(dst / "ref_frames.npy", ref)
        np.save(dst / "ref_frames_idx.npy", idx)
        print(f"  {len(idx)} reference frames from {len(np.unique(trial))} trials in {time.time() - t0:.0f}s; "
              f"mean {ref.mean():.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
