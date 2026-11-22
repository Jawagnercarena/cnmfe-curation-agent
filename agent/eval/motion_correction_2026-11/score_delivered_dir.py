"""Score every AVG5x movie in a folder with the pipeline rule's QC metric (read-only on the movies).

    python score_delivered_dir.py <folder> --out <csv> [--match SUBSTR ...] [--also <tif> ...]

bad / degraded = share of frames whose band-passed correlation with the movie's own template is
below 0.5 x / 0.75 x the movie's quiet level (90th percentile).  The function is
mc_rule.movie_quality from the job-B draft (D:/2p data/mc_rerun_2026-11/scripts_proposed), the
same definition as r_raw in motion_metrics.py.  The share of raw frames at suite2p's search limit
cannot be measured on a delivered movie (the shift record was not kept), so this is half of the
rule's QC line.

One movie at a time: D: is a single hard disk.  Rows already in the csv are skipped, so an
interrupted run can be continued.  --also scores extra tifs first (known-answer check).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

RULE_DIR = Path(r"D:\2p data\mc_rerun_2026-11\scripts_proposed")
FIELDS = ["session", "frames", "bad_pct", "degraded_pct", "quiet_r", "seconds", "tif_bytes", "tif_written", "path"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", nargs="*", default=[], help="substrings; a movie needs any one (default: all)")
    ap.add_argument("--also", nargs="*", type=Path, default=[], help="extra tifs scored first")
    args = ap.parse_args()

    sys.path.insert(0, str(RULE_DIR))
    import mc_rule
    import tifffile

    tifs = list(args.also) + sorted(p for p in args.folder.glob("AVG*.tif")
                                    if not args.match or any(m in p.name for m in args.match))
    done = set()
    if args.out.is_file():
        with open(args.out, newline="", encoding="ascii") as fh:
            done = {row["path"] for row in csv.DictReader(fh)}
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="ascii") as fh:
            csv.DictWriter(fh, FIELDS).writeheader()
    todo = [p for p in tifs if str(p) not in done]
    print(f"{len(tifs)} movie(s), {len(tifs) - len(todo)} already scored, {len(todo)} to do", flush=True)
    failed = 0
    for i, tif in enumerate(todo, start=1):
        t0 = time.time()
        try:
            mm = tifffile.memmap(str(tif), mode="r")
            frames = int(mm.shape[0])
            del mm
            bad, degraded, quiet = mc_rule.movie_quality(tif)
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(todo)}] FAILED {tif.name}: {type(e).__name__}: {e}", flush=True)
            continue
        st = tif.stat()
        row = {"session": tif.stem, "frames": frames, "bad_pct": f"{100 * bad:.2f}",
               "degraded_pct": f"{100 * degraded:.2f}", "quiet_r": f"{quiet:.4f}",
               "seconds": f"{time.time() - t0:.0f}", "tif_bytes": st.st_size,
               "tif_written": f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}", "path": str(tif)}
        with open(args.out, "a", newline="", encoding="ascii") as fh:
            csv.DictWriter(fh, FIELDS).writerow(row)
        print(f"[{i}/{len(todo)}] {tif.stem}: bad {row['bad_pct']}%  degraded {row['degraded_pct']}%  "
              f"({frames} frames, {row['seconds']} s)", flush=True)
    print(f"finished: {len(todo) - failed} scored, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
