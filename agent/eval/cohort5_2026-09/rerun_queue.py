"""
rerun_queue.py -- re-run cohort-5 pilot sessions in place with the current
parameters (agent/config/param_overrides.json) and curate at threshold 0 (or at
the deployed joblib's reject_threshold with --threshold model), outside the
watcher.

Mirrors watcher_vCA1.py's setup (config_vCA1 installed as `config`, own log
file under agent/logs/) and calls watcher.process_session for each session, so
the pipeline is exactly the watcher's: pre-pass -> parameter estimate (with
the override) -> headless CNMFe -> curator -> launcher -> background weights.

Why not the watcher itself: the running watcher imported params.py before the
override file existed, and it runs sessions one at a time.  Two of these
runners side by side halve the wall-clock; two headless MATLAB runs is this
machine's limit (one is ~75 GB), so do not stage new tifs for the watcher
while two runners are active.

Safety: the watcher ignores these sessions while they run because neuron.mat
and review_report.pdf stay in place until the new run overwrites them (its
retry scans key on those files).  The old candidate set is preserved by COPY
into <session>/.prev_est85_2026-10-05/ before the run (dot folder, invisible
to every scanner).  Nothing is deleted or moved.  review_assigned.txt is not
touched.  Dry run by default; --go runs.
"""
import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path

AGENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))
import config_vCA1                      # noqa: E402
sys.modules["config"] = config_vCA1
import watcher                          # noqa: E402
import curator                          # noqa: E402
import params                           # noqa: E402
import local_config                     # noqa: E402

ARCHIVE = ".prev_est85_2026-10-05"
TASK_DIR = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
# Two balanced lists: each holds three short day-1 sessions and three long day-2 sessions.
LIST_A = [
    "AVG5x-TSeries-082626-pnb102-479um-3z-000",
    "AVG5x-TSeries-082726-pnb102-497um-3z-000",
    "AVG5x-TSeries-082626-pnb105-397um-35z-000",
    "AVG5x-TSeries-082726-pnb105-391um-35z-000",
    "AVG5x-TSeries-082626-pnb109-398um-26z-000",
    "AVG5x-TSeries-082726-pnb109-405um-25z-000",
]
LIST_B = [
    "AVG5x-TSeries-082626-pnb104-404um-3z-000",
    "AVG5x-TSeries-082726-pnb104-416um-3z-000",
    "AVG5x-TSeries-082626-pnb106-205um-4z-000",
    "AVG5x-TSeries-082726-pnb106-201um-4z-000",
    "AVG5x-TSeries-082626-pnb99-399um-3z-000",
    "AVG5x-TSeries-082826-pnb99-436um-3z-000-000",
]
KEEP_COPY = ["candidate_features.npz", "spatial_footprints.mat", "review_neuron.mat",
             "review_summary.txt", "agent_run.log", "Cn.mat", "pnr.mat", "ROIs_candidates.jpg"]
REVIEWED = ["labels.mat", "ROIs.jpg", "labels_provenance.txt"]


def setup_log(tag: str):
    for h in list(watcher.logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            watcher.logger.removeHandler(h)
    fh = logging.FileHandler(watcher.LOG_DIR / f"rerun_cohort5_{tag}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    watcher.logger.addHandler(fh)


def check(sd: Path):
    name = sd.name
    if not sd.is_dir():
        return "folder missing"
    if not (sd / f"{name}.tif").exists():
        return "no .tif inside the session folder"
    if not (sd / f"{name}.mat").exists():
        return "converted .mat movie missing"
    if not (sd / "neuron.mat").exists():
        return "no neuron.mat (never ran, or a run is in progress)"
    hit = [m for m in REVIEWED if (sd / m).exists()]
    if hit:
        return "looks reviewed: " + ", ".join(hit)
    return None


def archive_copy(sd: Path, log, archive: str):
    dst = sd / archive
    if dst.exists():
        log(f"[RERUN] {sd.name}: SKIP (archive folder {archive}/ already exists -- refusing to overwrite it)")
        return False
    dst.mkdir()
    n = 0
    for f in KEEP_COPY:
        p = sd / f
        if p.exists():
            shutil.copy2(str(p), str(dst / f))
            n += 1
    log(f"[RERUN] {sd.name}: {n} file(s) copied to {archive}/ (originals left in place)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", choices=["A", "B"], help="which half of the 12 pilot sessions")
    ap.add_argument("--sessions", nargs="*", help="explicit session folder names instead of --list")
    ap.add_argument("--go", action="store_true", help="run (default: dry run)")
    ap.add_argument("--archive", default=ARCHIVE,
                    help="dot-folder name for the pre-run copy of the current outputs; a session is skipped if it already has one")
    ap.add_argument("--threshold", default="0",
                    help="curator reject threshold: a number (default 0, the pilots' threshold-0 reviews) "
                         "or 'model' for the deployed joblib's reject_threshold")
    args = ap.parse_args()
    if args.sessions:
        names, tag = args.sessions, "custom"
    elif args.list:
        names, tag = (LIST_A if args.list == "A" else LIST_B), args.list
    else:
        sys.exit("ERROR: give --list A|B or --sessions <names>")
    setup_log(tag)
    log = watcher.log
    curator.THRESHOLD_OVERRIDE = None if args.threshold.lower() == "model" else float(args.threshold)
    mode = "GO" if args.go else "DRY RUN"
    log(f"[RERUN] {mode} list {tag}: {len(names)} session(s); curator threshold override "
        f"{curator.THRESHOLD_OVERRIDE}; overrides file {params.OVERRIDES_PATH} "
        f"(exists: {params.OVERRIDES_PATH.exists()}); task folder {TASK_DIR}; archive folder {args.archive}/")
    for name in names:
        sd = TASK_DIR / name
        why = check(sd)
        if why:
            log(f"[RERUN] {name}: SKIP ({why})")
            continue
        if (sd / args.archive).exists():
            log(f"[RERUN] {name}: SKIP (archive folder {args.archive}/ already exists -- refusing to overwrite it)")
            continue
        if not args.go:
            log(f"[RERUN] {name}: would copy {len([f for f in KEEP_COPY if (sd / f).exists()])} "
                f"file(s) to {args.archive}/ and re-run in place")
            continue
        if not archive_copy(sd, log, args.archive):
            continue
        t0 = time.time()
        old_mtime = (sd / "neuron.mat").stat().st_mtime
        try:
            watcher.process_session(TASK_DIR, sd / f"{name}.tif")
        except Exception as e:                      # keep the queue moving
            import traceback
            log(f"[RERUN] {name}: ERROR {e}\n{traceback.format_exc()}")
        new = (sd / "neuron.mat").exists() and (sd / "neuron.mat").stat().st_mtime > old_mtime
        rs = sd / "review_summary.txt"
        rs_text = rs.read_text(encoding="utf-8", errors="replace") if rs.exists() else ""
        m_thr = re.search(r"threshold: ([0-9.]+)", rs_text)
        thr_used = m_thr.group(1) if m_thr else "?"
        n_cand = "?"
        try:
            import numpy as np
            z = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
            n_cand = int(np.asarray(z["n_candidates"]).ravel()[0])
        except Exception:
            pass
        log(f"[RERUN] {name}: finished in {(time.time() - t0) / 60:.1f} min; new neuron.mat: {new}; "
            f"candidates: {n_cand}; review threshold: {thr_used}")
    log(f"[RERUN] list {tag} {'finished' if args.go else 'dry run complete'}")


if __name__ == "__main__":
    main()
