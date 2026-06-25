"""
ingest_returns.py - bring curated session folders back from the lab server into
the canonical DATA_PARENT tree, so the watcher's auto-retrain picks up the new
labels.mat.

It mirrors  <exchange>/inbox/{area}/{task}/{session}/  ->  DATA_PARENT/{area}/{task}/{session}/
copying files that are new or changed. By default the unchanged multi-GB raw
video already present on this machine is NOT re-copied (matched by size); use
--force to copy every file regardless. Either way, the full curated folder ends
up archived on this machine.

Usage:
  python ingest_returns.py                        # ingest every session found in inbox
  python ingest_returns.py vCA1\\3odor\\AVG5x-...  # one session
  python ingest_returns.py --force                # copy all files, even same-size ones
  python ingest_returns.py --dry-run
Set CNMFE_EXCHANGE_ROOT (or agent/.env) first, or pass --exchange.

SAFETY: this script only READS from the server inbox and writes copies into the
LOCAL data tree. It never writes to, modifies, or deletes anything on the server.
There is no delete/move call anywhere in this file.
"""
import argparse
import shutil
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT


def iter_sessions(inbox: Path):
    """Yield session dirs under inbox (those holding a curated result)."""
    for p in inbox.rglob("*"):
        if p.is_dir() and ((p / "labels.mat").exists() or (p / "neuron.mat").exists()):
            yield p


def copy_session(src: Path, dst: Path, force: bool, dry: bool):
    copied = skipped = 0
    bytes_copied = 0
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        target = dst / rel
        size = f.stat().st_size
        if not force and target.exists() and target.stat().st_size == size:
            skipped += 1
            continue
        if dry:
            print(f"    would copy {rel} ({size/1e6:.1f} MB)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(target))
        copied += 1
        bytes_copied += size
    return copied, skipped, bytes_copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?",
                    help="area\\task\\session to ingest (default: all found in inbox)")
    ap.add_argument("--exchange", default=EXCHANGE_ROOT,
                    help="exchange root (default: CNMFE_EXCHANGE_ROOT / .env)")
    ap.add_argument("--force", action="store_true",
                    help="copy every file, even if a same-size copy already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.exchange:
        sys.exit("ERROR: exchange root not set. Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.")
    inbox = Path(args.exchange) / "inbox"
    if not inbox.is_dir():
        print(f"No inbox yet at {inbox} - nothing to ingest.")
        return

    sessions = [inbox / args.session] if args.session else list(iter_sessions(inbox))
    if not sessions:
        print("Nothing to ingest.")
        return

    for src in sessions:
        if not src.is_dir():
            print(f"SKIP (not found): {src}")
            continue
        rel = src.relative_to(inbox)
        dst = DATA_PARENT / rel
        print(f"\nIngest {rel}")
        print(f"  {src}  ->  {dst}")
        c, s, b = copy_session(src, dst, args.force, args.dry_run)
        print(f"  copied {c} files ({b/1e6:.1f} MB), skipped {s} unchanged")
        if not args.dry_run and (dst / "labels.mat").exists():
            print("  labels.mat present -> watcher will auto-retrain on its next poll.")
    print("\nDone.")


if __name__ == "__main__":
    main()
