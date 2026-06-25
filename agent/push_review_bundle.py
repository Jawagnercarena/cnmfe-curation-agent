"""
push_review_bundle.py - stage CNMFe review bundles on the lab server so remote
reviewers can pull them, run run_final_review.m, and push the curated folder
back (ingest_returns.py brings it home for retraining).

Single session:
  python push_review_bundle.py vCA1\\3odor\\AVG5x-...-000
  python push_review_bundle.py vCA1\\3odor\\AVG5x-...-000 --dry-run

Batch (every session awaiting review, skipping ones already out for review):
  python push_review_bundle.py --all
  python push_review_bundle.py --all --area vCA1 --task 3odor
  python push_review_bundle.py --all --dry-run

Each staged session is marked "out for review" via a review_assigned.txt marker
in the LOCAL session folder, so the watcher's REVIEW_QUEUE.md lists it as out and
a later --all run will not send it again. To re-open a session, delete that
marker file (the scripts never delete it for you).

Outbound bundle = the minimal set CNMFe_final_save.m needs:
  {SESSION}.mat (raw video), review_neuron.mat, Cn.mat, pnr.mat, Ybg_weights.mat,
  review_report.pdf, review_summary.txt, plus a freshly generated, self-locating
  run_final_review.m (so it works on any machine regardless of the original path).

Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.

SAFETY: this script only COPIES files to the server outbox and WRITES a local
assignment marker / launcher. It never deletes or removes anything, on the server
or locally. There is no delete/move call anywhere in this file.
"""
import argparse
import shutil
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT, REPO_ROOT
from review_prep import write_launcher, write_assignment, read_assignment

REQUIRED = ["review_neuron.mat"]
OPTIONAL = ["Cn.mat", "pnr.mat", "Ybg_weights.mat",
            "review_report.pdf", "review_summary.txt"]
# {SESSION}.mat (raw video) is handled separately; run_final_review.m is generated
# fresh into the bundle (never copied from the session, which may be stale).


def resolve_session(arg: str) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else (DATA_PARENT / arg)


def session_rel(session_dir: Path) -> Path:
    try:
        return session_dir.relative_to(DATA_PARENT)
    except ValueError:
        return Path(session_dir.name)


def is_awaiting(session_dir: Path) -> bool:
    """Agent-prepared, not yet human-reviewed: has the review set, no final ROIs."""
    return ((session_dir / "ROIs_candidates.jpg").exists()
            and (session_dir / "review_neuron.mat").exists()
            and not (session_dir / "ROIs.jpg").exists())


def find_awaiting(data_parent: Path, area=None, task=None):
    areas = ([data_parent / area] if area else
             [d for d in sorted(data_parent.iterdir())
              if d.is_dir() and not d.name.startswith(".")])
    for area_dir in areas:
        if not area_dir.is_dir():
            continue
        tasks = ([area_dir / task] if task else
                 [d for d in sorted(area_dir.iterdir())
                  if d.is_dir() and not d.name.startswith(".")])
        for task_dir in tasks:
            if not task_dir.is_dir():
                continue
            for sd in sorted(task_dir.iterdir()):
                if sd.is_dir() and is_awaiting(sd):
                    yield sd


def collect_files(session_dir: Path):
    """List the existing bundle files. Raises ValueError if a required file is missing."""
    files = []
    video = session_dir / f"{session_dir.name}.mat"
    if video.exists():
        files.append(video)
    else:
        print(f"  WARNING: raw video {video.name} not found - reviewer needs it for the video pass.")
    for name in REQUIRED:
        f = session_dir / name
        if not f.exists():
            raise ValueError(f"required file missing: {f}")
        files.append(f)
    for name in OPTIONAL:
        f = session_dir / name
        if f.exists():
            files.append(f)
        else:
            print(f"  (optional) missing: {name}")
    return files


def stage_one(session_dir: Path, exchange: str, dry_run: bool, assignee):
    """Stage one session's bundle. Returns total bytes (would-be in dry-run)."""
    dest = Path(exchange) / "outbox" / session_rel(session_dir)
    files = collect_files(session_dir)
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in files:
        size = f.stat().st_size
        if dry_run:
            print(f"  would copy {f.name} ({size/1e6:.1f} MB)")
            total += size
            continue
        target = dest / f.name
        if target.exists() and target.stat().st_size == size:
            print(f"  skip (already staged): {f.name}")
            continue
        print(f"  copy {f.name} ({size/1e6:.1f} MB) ...")
        shutil.copy2(str(f), str(target))
        total += size
    if dry_run:
        print("  would generate run_final_review.m (self-locating)")
        print(f"  -> {len(files) + 1} items, {total/1e6:.1f} MB, to {dest}")
    else:
        write_launcher(dest, REPO_ROOT)           # fresh, self-locating launcher
        write_assignment(session_dir, assignee)   # mark out-for-review (local marker)
        print("  generated run_final_review.m; marked out-for-review")
        print(f"  -> staged {total/1e6:.1f} MB to {dest}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?",
                    help="session dir (absolute) or area\\task\\session relative to DATA_PARENT")
    ap.add_argument("--all", action="store_true",
                    help="batch: stage every awaiting session not already out for review")
    ap.add_argument("--area", help="(with --all) limit to this brain area")
    ap.add_argument("--task", help="(with --all) limit to this task")
    ap.add_argument("--assignee", help="optional reviewer name, recorded in the marker")
    ap.add_argument("--force", action="store_true",
                    help="(with --all) include sessions already out for review")
    ap.add_argument("--exchange", default=EXCHANGE_ROOT,
                    help="exchange root (default: CNMFE_EXCHANGE_ROOT / .env)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be staged, without copying or marking anything")
    args = ap.parse_args()

    if not args.exchange:
        sys.exit("ERROR: exchange root not set. Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.")

    if args.all:
        sessions = list(find_awaiting(DATA_PARENT, args.area, args.task))
        if not args.force:
            sessions = [s for s in sessions if read_assignment(s) is None]
        if not sessions:
            print("No awaiting sessions to stage (all reviewed or already out for review).")
            return
        print(f"{'DRY RUN: ' if args.dry_run else ''}{len(sessions)} session(s) to stage:\n")
        staged, grand, failures = 0, 0, []
        for sd in sessions:
            print(f"[{session_rel(sd)}]")
            try:
                grand += stage_one(sd, args.exchange, args.dry_run, args.assignee)
                staged += 1
            except Exception as e:
                failures.append((sd, e))
                print(f"  ERROR: {e}")
            print()
        verb = "would stage" if args.dry_run else "staged"
        print(f"Done: {verb} {staged}/{len(sessions)} session(s), {grand/1e6:.1f} MB total.")
        if failures:
            print(f"{len(failures)} failed:")
            for sd, e in failures:
                print(f"  - {session_rel(sd)}: {e}")
        return

    # single session
    if not args.session:
        sys.exit("ERROR: provide a session (area\\task\\session) or use --all.")
    session_dir = resolve_session(args.session)
    if not session_dir.is_dir():
        sys.exit(f"ERROR: session folder not found: {session_dir}")
    existing = read_assignment(session_dir)
    if existing and not args.dry_run:
        print(f"NOTE: already out for review (since {existing.get('assigned', '?')}); re-staging anyway.")
    print(f"[{session_rel(session_dir)}]")
    try:
        stage_one(session_dir, args.exchange, args.dry_run, args.assignee)
    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
