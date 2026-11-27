"""One-off: re-average a kept registration, 8-bit range from the first fully lit frame.

    <suite2p env python> reaverage_display_range.py --scripts-dir "D:/2p data/scripts"
        --task-dir <scratch task dir> --session <TSeries name> --log <file> [--go]

Why (2026-11-24, 091526-pnb99): the recording starts with ~360 dark raw frames
(PMT/shutter off). avg_project.ijm converts to 8-bit with a plain run("8-bit"),
which takes the display range from averaged frame 1 -- dark -- so every later
frame clipped at 255 (old movie and first re-make alike). The user decided:
re-average with the range from after the dark start, keeping every frame so
frame numbers stay aligned with behaviour.

Steps (dry run prints 1 only):
  1. Dark start from the registered 16-bit frames (reg_tif): a raw frame is lit
     when its mean exceeds 0.25 x the median of frame means sampled across the
     session. range slice = first averaged frame (5 raw frames, after the
     macro's tail trim) whose raw frames are ALL lit; also reports lit/dark
     counts in the session-wide sample.
  2. The existing AVG tif in the task dir (the clipped re-make, scratch) is
     MOVED to <task>/_clipped_range_slice1/ (refuses if something is there).
  3. Fiji with avg_project_rerange.ijm (same as production except the range
     slice) on hard links of reg_tif, exactly as process_2p.run_average stages.
  4. process_2p.verify_avg_tif, then mc_rule.qc() -> row appended to
     <task>/mc_qc.csv. The shift record from the first averaging stays (same
     registration); save_record is not called again.
Nothing is deleted except this script's own staging links/work dir (as in
process_2p.run_average). Refuses a UNC or X: task dir.
"""
import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MACRO = HERE / "avg_project_rerange.ijm"
FACTOR = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", type=Path, required=True)
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    task = a.task_dir.resolve()
    s = str(task)
    if s.startswith("\\\\") or s[:2].upper() == "X:":
        sys.exit(f"refusing a server path: {task}")
    sys.path.insert(0, str(a.scripts_dir))
    import avg_project as avg
    import mc_rule
    import process_2p
    import tifffile

    session = task / a.session
    reg = avg.reg_tif_dir(session, 0)
    files = avg.ordered_reg_tifs(reg, 0)
    if not files:
        sys.exit(f"no reg_tif in {reg}")
    total = avg.expected_frames(session, 0)
    keep = (total // FACTOR) * FACTOR
    out_tif = task / f"AVG{FACTOR}x-{a.session}.tif"
    aside_dir = task / "_clipped_range_slice1"

    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        if a.go:
            with open(a.log, "a", encoding="ascii", errors="replace") as fh:
                fh.write(line + "\n")

    # ---- 1. dark start ----
    sample_means = []
    for f in files:
        with tifffile.TiffFile(f) as t:
            n = len(t.pages)
            for k in range(0, n, 50):
                sample_means.append(float(t.pages[k].asarray().mean()))
    sample_means = np.array(sample_means)
    thresh = 0.25 * float(np.median(sample_means))
    lit_sample = sample_means > thresh
    # exact per-frame means from the start until 50 consecutive lit frames
    means, offset = [], 0
    for f in files:
        x = tifffile.imread(f)
        means.extend(x.reshape(len(x), -1).mean(axis=1).tolist())
        m = np.array(means) > thresh
        if len(m) >= 50 and np.any(np.convolve(m, np.ones(50), "valid") == 50):
            break
    m = np.array(means) > thresh
    first_lit = int(np.argmax(m))
    j = math.ceil(first_lit / FACTOR)
    while not m[j * FACTOR:j * FACTOR + FACTOR].all():
        j += 1
    range_slice = j + 1
    log(f"{a.session}: {len(files)} reg_tif files, {total} raw frames, keep {keep} -> "
        f"{keep // FACTOR} averaged; median sampled frame mean {np.median(sample_means):.1f}, "
        f"lit threshold {thresh:.1f}")
    log(f"  session-wide sample: {lit_sample.sum()} lit / {(~lit_sample).sum()} dark of "
        f"{len(lit_sample)} sampled frames; dark sampled frame indices (x50 within file): "
        f"{np.flatnonzero(~lit_sample)[:20].tolist()}")
    log(f"  first lit raw frame {first_lit} (mean {means[first_lit]:.1f}; frame before "
        f"{means[first_lit - 1] if first_lit else float('nan'):.1f}); dark raw frames at the start "
        f"{first_lit}; range from averaged frame {j} (0-based) = slice {range_slice}, raw "
        f"{j * FACTOR}..{j * FACTOR + FACTOR - 1}")
    if not a.go:
        print(f"DRY RUN: would move {out_tif.name} to {aside_dir}, run Fiji with {MACRO.name} "
              f"(range slice {range_slice}), verify, QC")
        return 0

    # ---- 2. move the clipped re-make aside ----
    if out_tif.exists():
        dst = aside_dir / out_tif.name
        if dst.exists():
            sys.exit(f"refusing: {dst} already exists")
        aside_dir.mkdir(exist_ok=True)
        out_tif.rename(dst)
        log(f"  moved the clipped re-make to {dst}")

    # ---- 3. Fiji ----
    stage_dir = task / f"_avgstage_{a.session}"
    work_dir = task / f"_avgwork_{a.session}"
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec_path = work_dir / "spec.txt"
    status_path = work_dir / "status.txt"

    def ij(p):
        return str(p).replace("\\", "/")

    try:
        avg.stage_links(files, stage_dir)
        # avg_project.run_fiji's spec plus line 5 (range slice); same Fiji call.
        spec_path.write_text("\n".join([ij(stage_dir), ij(out_tif), str(FACTOR), str(total),
                                        ij(status_path), str(range_slice)]), encoding="ascii")
        result = subprocess.run([str(avg.FIJI_BAT), "--headless", "--console", "-batch",
                                 str(MACRO), str(spec_path)], capture_output=True, text=True)
        status = (status_path.read_text(encoding="utf-8", errors="replace")
                  if status_path.exists() else "")
        ok = "STATUS=OK" in status and out_tif.is_file()
        if not ok:
            tail = ((result.stdout or "") + (result.stderr or "")).strip()[-2000:]
            status += f"\nfiji exit code = {result.returncode}\n--- fiji console ---\n{tail}"
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    for line in status.splitlines():
        log("  fiji " + line)
    if not ok:
        if out_tif.is_file():
            out_tif.rename(out_tif.with_name(out_tif.stem + ".failed.tif"))
        log(f"  FIJI FAILED after {time.time() - t0:.0f} s")
        return 1
    log(f"  Fiji OK in {time.time() - t0:.0f} s")

    # ---- 4. verify + QC ----
    vok, detail = process_2p.verify_avg_tif(out_tif, total // FACTOR)
    log(f"  verify: {'OK' if vok else 'FAILED'} {detail}")
    if not vok:
        return 1
    row = mc_rule.qc(task, session, out_tif, 0)
    log(f"  QC row: bad {row['bad_pct']}% degraded {row['degraded_pct']}% quiet r "
        f"{row['quiet_r']} at limit {row['at_limit_pct']}% flag '{row['flag']}'")
    return 0 if not row["flag"] else 2


if __name__ == "__main__":
    sys.exit(main())
