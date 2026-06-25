"""
push_review_bundle.py - stage a session's review bundle on the lab server so a
remote reviewer can pull it, run run_final_review.m in MATLAB, and push the
curated folder back (ingest_returns.py brings it home for retraining).

Outbound bundle = the minimal set CNMFe_final_save.m needs to run a review:
  {SESSION}.mat (raw video), review_neuron.mat, Cn.mat, pnr.mat,
  Ybg_weights.mat, run_final_review.m, review_report.pdf, review_summary.txt

The bundle is copied to  <exchange>/outbox/{area}/{task}/{session}/ .

Usage:
  python push_review_bundle.py <session_dir>                  # absolute path
  python push_review_bundle.py vCA1\\3odor\\AVG5x-...-000      # relative to DATA_PARENT
  python push_review_bundle.py vCA1\\3odor\\AVG5x-...-000 --dry-run
Set CNMFE_EXCHANGE_ROOT (or agent/.env) to the server exchange folder first,
or pass --exchange.

SAFETY: this script only COPIES files to the server outbox (creating the session
subfolder if needed). It never deletes or removes anything, on the server or
locally. There is no delete/move call anywhere in this file.
"""
import argparse
import shutil
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT

# Required for the reviewer to even start; abort if missing.
REQUIRED = ["review_neuron.mat", "run_final_review.m"]
# Helpful but non-fatal (review still runs, just slower / less guidance).
OPTIONAL = ["Cn.mat", "pnr.mat", "Ybg_weights.mat",
            "review_report.pdf", "review_summary.txt"]
# {SESSION}.mat (raw video) is handled separately; it is named after the folder.


def resolve_session(arg: str) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else (DATA_PARENT / arg)


def session_rel(session_dir: Path) -> Path:
    """area/task/session relative to DATA_PARENT (defines the outbox layout)."""
    try:
        return session_dir.relative_to(DATA_PARENT)
    except ValueError:
        return Path(session_dir.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?",
                    help="session dir (absolute) or area\\task\\session relative to DATA_PARENT")
    ap.add_argument("--exchange", default=EXCHANGE_ROOT,
                    help="exchange root (default: CNMFE_EXCHANGE_ROOT / .env)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the files that would be staged, without copying anything")
    args = ap.parse_args()

    if not args.exchange:
        sys.exit("ERROR: exchange root not set. Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.")
    if not args.session:
        sys.exit("ERROR: provide a session dir or area\\task\\session path.")

    session_dir = resolve_session(args.session)
    if not session_dir.is_dir():
        sys.exit(f"ERROR: session folder not found: {session_dir}")

    dest = Path(args.exchange) / "outbox" / session_rel(session_dir)

    files = []
    video = session_dir / f"{session_dir.name}.mat"
    if video.exists():
        files.append(video)
    else:
        print(f"WARNING: raw video {video.name} not found - the reviewer needs it for the video pass.")
    for name in REQUIRED:
        f = session_dir / name
        if not f.exists():
            sys.exit(f"ERROR: required file missing: {f}")
        files.append(f)
    for name in OPTIONAL:
        f = session_dir / name
        if f.exists():
            files.append(f)
        else:
            print(f"  (optional) missing: {name}")

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for f in files:
        size = f.stat().st_size
        if args.dry_run:
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
    if args.dry_run:
        print(f"\nDRY RUN - nothing copied. Bundle would contain {len(files)} files "
              f"({total/1e6:.1f} MB), destined for:\n  {dest}")
    else:
        print(f"\nBundle staged at: {dest}\n  newly copied: {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
