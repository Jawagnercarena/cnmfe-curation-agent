"""rerun_new_movie.py -- job C of docs/HANDOFF_2026-11-20_MOTION_CORRECTION.md: put a
re-registered AVG5x movie into an existing, not yet reviewed ACORN session and re-run
the session in place, outside the watcher.

    python rerun_new_movie.py prepare <session> [<session> ...] [--go]
    python rerun_new_movie.py run     <session> [<session> ...] [--go]

Both steps are dry runs unless --go is given.  Local only: everything happens inside
the session folders under DATA_PARENT/vCA1/6odorDualDiffRew; the new movie is read
from --new-dir.  Nothing is removed: old files are copied or renamed into an archive
folder inside the session folder.

prepare, per session (every check for every session first, then the changes):
  1. <session>/<ARCHIVE>/ is created (refused when it exists).
  2. Every top-level file except the movie .tif / .mat and review_assigned.txt is
     COPIED there and checked by size + SHA-256.  The originals stay in place until
     the new run overwrites them, because the watcher keys on them: neuron.mat
     (find_new_tifs), review_report.pdf (find_curator_retries) and Ybg_weights.mat
     (find_precompute_pending).  If one went missing the watcher would start its own
     job on the folder.
  3. <session>_neurons/ (per-candidate PNGs), the old .tif and the old movie .mat are
     RENAMED into the archive folder (same volume).
  4. The new .tif is copied in as <session>.tif.partial, checked against its source
     by SHA-256, then renamed to <session>.tif.
  5. ARCHIVE_README.txt in the archive folder records what was done, with hashes.
  The old and the new .tif have the same size (same frame count, 8-bit), so "the new
  movie differs from the old one" and every copy check use SHA-256, never the size.

run, per session, one after the other:
  rebuilds <session>.mat from the new .tif (agent/movie_mat.py -> rebuild_movie_mat.m:
  never overwrites, checks sampled frames against the .tif), then calls
  watcher.process_session the way agent/eval/cohort5_2026-09/rerun_queue.py does
  (vCA1 config, curator at the deployed model's reject_threshold, log file
  agent/logs/rerun_cohort5_mc_new_movie.log).  review_assigned.txt is not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[1]
COHORT5 = AGENT / "eval" / "cohort5_2026-09"
ARCHIVE = ".prev_maxregshift01_2026-11-21"
README = "ARCHIVE_README.txt"
NEW_DIR = Path(r"D:\2p data\mc_rerun_2026-11\movies\new")
REVIEWED = ["labels.mat", "ROIs.jpg", "labels_provenance.txt"]
WATCHER_KEYS = ["neuron.mat", "review_report.pdf", "Ybg_weights.mat"]
NOT_COPIED = ["review_assigned.txt"]
NOTE = ("The delivered movie (suite2p maxregshift 0.1) was replaced by a re-registered one "
        "(maxregshift 0.2 + recentred reference; docs/HANDOFF_2026-11-20_MOTION_CORRECTION.md). "
        "This folder holds the old movie and the outputs of the run on it.")


def say(msg: str):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            block = fh.read(8 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def tif_shape(p: Path):
    import tifffile
    with tifffile.TiffFile(str(p)) as tf:
        s = tf.series[0]
        return tuple(int(v) for v in s.shape), str(s.dtype)


def stamp(p: Path) -> str:
    return f"{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}"


def default_task_dir() -> Path:
    sys.path.insert(0, str(AGENT))
    import local_config
    return Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"


# --------------------------------------------------------------------------- prepare

def plan_one(sd: Path, new_dir: Path, min_quiet_min: float):
    """(plan, problems).  Reads only."""
    name = sd.name
    if not sd.is_dir():
        return None, ["session folder missing"]
    old_tif, old_mat = sd / f"{name}.tif", sd / f"{name}.mat"
    new_tif, arch, neurons = new_dir / f"{name}.tif", sd / ARCHIVE, sd / f"{name}_neurons"
    probs = []
    if not old_tif.is_file():
        probs.append(f"no {name}.tif in the session folder")
    if not new_tif.is_file():
        probs.append(f"no new movie at {new_tif}")
    for k in WATCHER_KEYS:
        if not (sd / k).is_file():
            probs.append(f"{k} missing (the watcher would start its own job on this folder)")
    hit = [m for m in REVIEWED if (sd / m).exists()]
    if hit:
        probs.append("looks reviewed: " + ", ".join(hit))
    if arch.exists():
        probs.append(f"archive folder {ARCHIVE}/ already exists")
    if list(sd.glob("*.partial")):
        probs.append("a .partial file is left over from an earlier attempt")
    extra = [p.name for p in sd.iterdir() if p.is_dir() and p != neurons]
    if extra:
        probs.append("unexpected folder(s): " + ", ".join(sorted(extra)))
    if probs:
        return None, probs
    newest = max(p.stat().st_mtime for p in sd.iterdir() if p.is_file() and p.name not in NOT_COPIED)
    quiet = (time.time() - newest) / 60
    if quiet < min_quiet_min:
        probs.append(f"a file in this folder was written {quiet:.0f} min ago (limit {min_quiet_min:.0f})")
    shape_old, shape_new = tif_shape(old_tif), tif_shape(new_tif)
    if shape_old != shape_new:
        probs.append(f"the new movie is {shape_new}, the old one {shape_old}")
    if probs:
        return None, probs
    h_old, h_new = sha256(old_tif), sha256(new_tif)
    if h_old == h_new:
        probs.append("the new movie is byte-identical to the one already in the session folder")
        return None, probs
    copy = sorted(p for p in sd.iterdir()
                  if p.is_file() and p not in (old_tif, old_mat) and p.name not in NOT_COPIED)
    move = [p for p in (neurons, old_tif, old_mat) if p.exists()]
    return {"sd": sd, "name": name, "arch": arch, "old_tif": old_tif, "new_tif": new_tif,
            "copy": copy, "move": move, "h_old": h_old, "h_new": h_new, "shape": shape_old}, []


def do_one(plan: dict):
    sd, name, arch = plan["sd"], plan["name"], plan["arch"]
    lines = []
    arch.mkdir()
    for p in plan["copy"]:
        q = arch / p.name
        shutil.copy2(str(p), str(q))
        hp, hq = sha256(p), sha256(q)
        if p.stat().st_size != q.stat().st_size or hp != hq:
            raise RuntimeError(f"copy of {p.name} into {ARCHIVE}/ does not match the original")
        lines.append(f"copied  {hp}  {p.stat().st_size:>14,d}  {stamp(p)}  {p.name}")
    say(f"  {len(plan['copy'])} file(s) copied to {ARCHIVE}/ and checked (originals left in place)")
    for p in plan["move"]:
        q = arch / p.name
        if p.is_dir():
            n = sum(1 for _ in p.iterdir())
            desc = f"folder, {n} item(s)"
            size, when = 0, stamp(p)
        else:
            desc = plan["h_old"] if p == plan["old_tif"] else "(not hashed)"
            size, when = p.stat().st_size, stamp(p)
        os.rename(str(p), str(q))
        if p.exists() or not q.exists():
            raise RuntimeError(f"renaming {p.name} into {ARCHIVE}/ did not take effect")
        if q.is_file() and q.stat().st_size != size:
            raise RuntimeError(f"{q} changed size during the rename")
        lines.append(f"moved   {desc:<64}  {size:>14,d}  {when}  {p.name}")
        say(f"  moved {p.name} into {ARCHIVE}/")
    part = sd / f"{name}.tif.partial"
    shutil.copy2(str(plan["new_tif"]), str(part))
    if sha256(part) != plan["h_new"]:
        raise RuntimeError(f"the copy of the new movie does not match its source; left at {part}")
    final = sd / f"{name}.tif"
    if final.exists():
        raise RuntimeError(f"{final.name} reappeared in the session folder; new movie left at {part}")
    os.rename(str(part), str(final))
    say(f"  new movie in place: {final.name} (sha256 {plan['h_new'][:16]}..., was {plan['h_old'][:16]}...)")
    text = [f"Archive written {datetime.now():%Y-%m-%d %H:%M} by agent/eval/motion_correction_2026-11/"
            f"rerun_new_movie.py prepare.",
            f"Session: {name}", "", NOTE, "",
            f"Old movie sha256 {plan['h_old']}",
            f"New movie sha256 {plan['h_new']}  (copied from {plan['new_tif']})",
            f"Frames x height x width, pixel type: {plan['shape'][0]}, {plan['shape'][1]} (both movies)",
            "review_assigned.txt was left in the session folder and not copied.", "",
            "action, sha256, bytes, last written, name:"] + lines
    (arch / README).write_text("\n".join(text) + "\n", encoding="ascii")


def cmd_prepare(args) -> int:
    task = args.task_dir or default_task_dir()
    say(f"prepare ({'GO' if args.go else 'DRY RUN'}): task folder {task}; new movies from {args.new_dir}; "
        f"archive folder {ARCHIVE}/")
    plans, bad = [], 0
    for name in args.sessions:
        plan, probs = plan_one(task / name, args.new_dir, args.min_quiet_min)
        if probs:
            bad += 1
            for why in probs:
                say(f"{name}: REFUSED ({why})")
            continue
        plans.append(plan)
        mb = sum(p.stat().st_size for p in plan["copy"]) / 1e6
        say(f"{name}: ok -- copy {len(plan['copy'])} file(s) ({mb:.0f} MB), move "
            f"{', '.join(p.name for p in plan['move'])}; new movie {plan['shape'][0]} {plan['shape'][1]}, "
            f"sha256 {plan['h_new'][:16]}... (old {plan['h_old'][:16]}...)")
    if bad:
        say(f"{bad} session(s) refused; nothing was changed")
        return 1
    if not args.go:
        say("dry run complete; nothing was changed")
        return 0
    for plan in plans:
        say(f"{plan['name']}: archiving and swapping the movie")
        do_one(plan)
    say("prepare finished")
    return 0


# ------------------------------------------------------------------------------- run

def prepared(sd: Path):
    if not sd.is_dir():
        return "folder missing"
    if not (sd / ARCHIVE / README).is_file():
        return f"not prepared (no {ARCHIVE}/{README})"
    if not (sd / f"{sd.name}.tif").is_file():
        return "no .tif inside the session folder"
    hit = [m for m in REVIEWED if (sd / m).exists()]
    if hit:
        return "looks reviewed: " + ", ".join(hit)
    if list(sd.glob("*.partial")):
        return "a .partial file is left over"
    return None


def cmd_run(args) -> int:
    sys.path.insert(0, str(COHORT5))
    import rerun_queue as rq            # installs config_vCA1 as `config`, imports watcher / curator / params
    import movie_mat
    rq.setup_log("mc_new_movie")
    log = rq.watcher.log
    rq.curator.THRESHOLD_OVERRIDE = None        # the deployed model's reject_threshold
    log(f"[MC-RERUN] {'GO' if args.go else 'DRY RUN'}: {len(args.sessions)} session(s); curator threshold = "
        f"the model's; overrides file {rq.params.OVERRIDES_PATH}; task folder {rq.TASK_DIR}")
    failed = 0
    for name in args.sessions:
        sd = rq.TASK_DIR / name
        why = prepared(sd)
        if why:
            failed += 1
            log(f"[MC-RERUN] {name}: SKIP ({why})")
            continue
        has_mat = (sd / f"{name}.mat").exists()
        if not args.go:
            log(f"[MC-RERUN] {name}: would {'use the existing' if has_mat else 'rebuild the'} movie .mat "
                f"and re-run in place")
            continue
        t0 = time.time()
        try:
            movie_mat.ensure(sd, log=log)
        except Exception as e:
            failed += 1
            log(f"[MC-RERUN] {name}: SKIP (movie rebuild failed: {e})")
            continue
        why = rq.check(sd)
        if why:
            failed += 1
            log(f"[MC-RERUN] {name}: SKIP ({why})")
            continue
        old_mtime = (sd / "neuron.mat").stat().st_mtime
        try:
            rq.watcher.process_session(rq.TASK_DIR, sd / f"{name}.tif")
        except Exception as e:                      # keep the queue moving
            import traceback
            log(f"[MC-RERUN] {name}: ERROR {e}\n{traceback.format_exc()}")
        new = (sd / "neuron.mat").exists() and (sd / "neuron.mat").stat().st_mtime > old_mtime
        rs = sd / "review_summary.txt"
        rs_text = rs.read_text(encoding="utf-8", errors="replace") if rs.exists() else ""
        m_thr = re.search(r"threshold: ([0-9.]+)", rs_text)
        n_cand = "?"
        try:
            import numpy as np
            z = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
            n_cand = int(np.asarray(z["n_candidates"]).ravel()[0])
        except Exception:
            pass
        if not new:
            failed += 1
        log(f"[MC-RERUN] {name}: finished in {(time.time() - t0) / 60:.1f} min; new neuron.mat: {new}; "
            f"candidates: {n_cand}; review threshold: {m_thr.group(1) if m_thr else '?'}")
    log(f"[MC-RERUN] {'finished' if args.go else 'dry run complete'}; {failed} session(s) skipped or failed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--go", action="store_true", help="make the changes (default: dry run)")
    p.add_argument("--task-dir", type=Path, help="task folder (default: DATA_PARENT/vCA1/6odorDualDiffRew)")
    p.add_argument("--new-dir", type=Path, default=NEW_DIR, help="folder holding <session>.tif, the new movie")
    p.add_argument("--min-quiet-min", type=float, default=60.0,
                   help="refuse a session folder written to more recently than this many minutes ago")
    r = sub.add_parser("run")
    r.add_argument("sessions", nargs="+")
    r.add_argument("--go", action="store_true", help="run (default: dry run)")
    args = ap.parse_args()
    return cmd_prepare(args) if args.cmd == "prepare" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
