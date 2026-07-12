"""
ingest_returns.py - bring curated session folders back from the lab server into
the canonical DATA_PARENT tree, so the watcher's auto-retrain picks up the new
labels.mat.

It mirrors  <exchange>/inbox/<reviewer>/{area}/{task}/{session}/  ->  DATA_PARENT/{area}/{task}/{session}/
copying files that are new or changed. Reviewers push their finished folder into
their own inbox/<name>/ subfolder (mirroring outbox/<name>/), so each reviewer's
"done" pile sits in one place.

Destination resolution is by SESSION NAME, not by path shape: central always
already has the session folder (it generated the outbound bundle), so we find the
existing local DATA_PARENT/{area}/{task}/{session} by name and repatriate into it.
This is robust to a reviewer dropping the {area} level when they copy into their
inbox (e.g. inbox/<name>/{task}/{session} instead of .../{area}/{task}/...): a
blind "last three path parts" would otherwise invent a phantom area folder named
after the reviewer/task and strand the labels away from candidate_features.npz.
If no existing folder matches, we fall back to the path-derived {area}/{task}/{session}
ONLY when that area already exists on disk; otherwise the session is SKIPPED with a
loud warning rather than silently creating a bogus area. By default the unchanged
multi-GB raw video already present on this machine is NOT re-copied (matched by
size); use --force to copy every file regardless. Either way, the full curated
folder ends up archived on this machine.

Usage:
  python ingest_returns.py                              # ingest every session found in inbox
  python ingest_returns.py Alisia\\vCA1\\3odor\\AVG5x-...  # one session (inbox-relative path)
  python ingest_returns.py --force                      # copy all files, even same-size ones
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


def _is_real_session(d: Path) -> bool:
    """A central session folder we'd repatriate into (has features / the review set)."""
    return (d / "candidate_features.npz").exists() or (d / "review_neuron.mat").exists()


def find_local_session(session_name: str):
    """Existing DATA_PARENT/{area}/{task}/{session_name} folders, searched by name.
    Restricted to the canonical 3-level depth so it stays cheap."""
    matches = []
    for area in DATA_PARENT.iterdir():
        if not area.is_dir() or area.name.startswith("."):
            continue
        for task in area.iterdir():
            if not task.is_dir() or task.name.startswith("."):
                continue
            cand = task / session_name
            if cand.is_dir():
                matches.append(cand)
    return matches


def resolve_dest(src: Path):
    """Return (dst, note) for a returned session, or (None, reason) to skip.
    Resolves by session name against the local tree first; falls back to the
    path-derived location only if that area already exists on disk."""
    matches = find_local_session(src.name)
    real = [m for m in matches if _is_real_session(m)]
    pick = real or matches
    if len(pick) == 1:
        return pick[0], "matched existing local session by name"
    if len(pick) > 1:
        listing = ", ".join(str(m.relative_to(DATA_PARENT)) for m in pick)
        return None, f"AMBIGUOUS: {len(pick)} existing folders named '{src.name}': {listing}"
    # No existing local folder -> derive {area}/{task}/{session} from the path,
    # but only trust it if the area already exists (guards against phantom areas).
    canon = Path(*src.parts[-3:])
    area = canon.parts[0]
    if (DATA_PARENT / area).is_dir():
        return DATA_PARENT / canon, f"new session, area '{area}' exists"
    return None, (
        f"cannot resolve destination: no existing folder named '{src.name}', and the "
        f"path-derived area '{area}' is not a known area under {DATA_PARENT}. The reviewer "
        f"likely dropped the {{area}} level when copying into the inbox ({src}). "
        f"Fix the inbox path to <reviewer>/<area>/<task>/<session> and re-run.")


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

    skipped_sessions = []
    for src in sessions:
        if not src.is_dir():
            print(f"SKIP (not found): {src}")
            continue
        dst, note = resolve_dest(src)
        if dst is None:
            print(f"\nSKIP {src.name}")
            print(f"  !! {note}")
            skipped_sessions.append((src, note))
            continue
        print(f"\nIngest {dst.relative_to(DATA_PARENT)}  ({note})")
        print(f"  {src}  ->  {dst}")
        c, s, b = copy_session(src, dst, args.force, args.dry_run)
        print(f"  copied {c} files ({b/1e6:.1f} MB), skipped {s} unchanged")
        if not args.dry_run and (dst / "labels.mat").exists():
            print("  labels.mat present -> watcher will auto-retrain on its next poll.")

    if skipped_sessions:
        print(f"\n{len(skipped_sessions)} session(s) SKIPPED (unresolved destination):")
        for src, note in skipped_sessions:
            print(f"  - {src.name}: {note}")
    print("\nDone.")


if __name__ == "__main__":
    main()
