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
    """Yield session dirs under inbox (those holding a curated result).

    Dot-prefixed folders are never sessions: a returned folder can carry a
    parked diagnostic tree from the central machine (first seen 2026-08-20,
    when returns mirrored from central included .gsig9_wrongparams/, whose
    parked neuron.mat made ingest treat the park folder itself as a session
    named '.gsig9_wrongparams' and fail resolution with a confusing
    "area 'DG AL' is not a known area" skip). The dot prefix means "set
    aside" everywhere else in this pipeline; honour it here too.
    """
    for p in inbox.rglob("*"):
        if not p.is_dir():
            continue
        if any(part.startswith(".") for part in p.relative_to(inbox).parts):
            continue
        if not ((p / "labels.mat").exists() or (p / "neuron.mat").exists()):
            continue
        if is_training_sandbox(p, inbox):
            print(f"SKIP (training sandbox, never ingested): {p.relative_to(inbox)}")
            continue
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


PROVENANCE_NAME = "labels_provenance.txt"

# A folder carrying this marker is a reviewer-TRAINING sandbox (see
# agent/push_training_bundle.py and docs/TRAINING.md).  Its labels.mat is a
# trainee's practice result on a session that already has real labels, and
# resolve_dest matches returns by session NAME, so a training folder dropped
# into inbox/ would otherwise land on the real session.  Never ingested.
TRAINING_MARKER = "TRAINING_SESSION.txt"


def is_training_sandbox(p: Path, root: Path) -> bool:
    """True if p or any ancestor between root and p carries TRAINING_SESSION.txt."""
    cur = root
    try:
        parts = p.relative_to(root).parts
    except ValueError:
        parts = ()
    for part in parts:
        cur = cur / part
        if (cur / TRAINING_MARKER).exists():
            return True
    return False

# Files that only the central machine may write, and that a return must never
# carry back over the local copy.  A reviewer bundle does not contain them
# (push_review_bundle stages review_neuron.mat + Cn/pnr/Ybg_weights/pdf/summary
# and nothing else), but a reviewer who mirrors whole central folders can, and
# copy_session copies any file whose size differs.
#
#   labels_provenance.txt   -- central metadata (who reviewed this session);
#                              a stale echo would overwrite the real record.
#   candidate_features.npz  -- the feature matrix curator.py wrote at curation
#                              time.  Scoring is positional and areas differ in
#                              feature-contract width (BLA runs the 35-column v2
#                              contract, vCA1/DG_AL 13), so a mirrored-back copy
#                              from before a contract swap would silently
#                              downgrade the session's row width and either
#                              corrupt the training corpus or trip the trainer's
#                              width guard.  Sizes differ across contracts, so
#                              the same-size skip below would NOT catch it.
CENTRAL_ONLY = {PROVENANCE_NAME, "candidate_features.npz"}


def read_provenance(session_dir: Path):
    """Reviewer whose labels this local session carries, or None if unrecorded."""
    f = session_dir / PROVENANCE_NAME
    if not f.exists():
        return None
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("reviewer:"):
            return line.split(":", 1)[1].strip() or None
    return None


def write_provenance(session_dir: Path, reviewer: str, src: Path):
    """Record which reviewer's labels the local session now carries.

    Written on every ingest that carries a labels.mat (even when the file was
    size-skipped as unchanged: identical bytes still mean the local labels are
    that reviewer's). copy2 preserves the reviewer's mtimes, so without this
    record there is no way to tell later WHO produced the labels -- which is
    exactly what made the 2026-08-20 duplicate-review near-miss hard to
    reconstruct (a session assigned to one reviewer was validly reviewed by
    another, and a later return from the assignee would have silently
    overwritten the ingested labels).
    """
    from datetime import datetime
    (session_dir / PROVENANCE_NAME).write_text(
        f"reviewer: {reviewer}\n"
        f"ingested: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"source: {src}\n", encoding="utf-8")


def read_label_counts(labels_path: Path):
    """
    Read a returned labels.mat and return (n_keep, n_delete, n_motion).
    n_motion is None for labels.mat written before the (m) motion-delete option
    existed (no 'motion_delete' variable). Returns None if the file can't be read.
    """
    try:
        import scipy.io as sio
        d = sio.loadmat(str(labels_path))
        labels = d["labels"].flatten()
        n_keep   = int((labels == 1).sum())
        n_delete = int((labels == 0).sum())
        n_motion = int(d["motion_delete"].flatten().sum()) if "motion_delete" in d else None
        return n_keep, n_delete, n_motion
    except Exception:
        return None


def copy_session(src: Path, dst: Path, force: bool, dry: bool, on_first_copy=None):
    """on_first_copy, if given, is called once just before the first file is
    copied (or would be, in a dry run) -- main() uses it to print the session
    header only for sessions that actually bring something new."""
    copied = skipped = 0
    bytes_copied = 0
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        # Never import dot-prefixed subtrees: those are parked/set-aside data
        # (e.g. a stale .gsig9_wrongparams/ mirrored back by a reviewer) and
        # must not be written into the clean local session.
        if any(part.startswith(".") for part in rel.parts):
            continue
        # Central-machine-only files (see CENTRAL_ONLY): a return that echoes
        # them back must never overwrite the local copy.
        if rel.name in CENTRAL_ONLY:
            continue
        target = dst / rel
        size = f.stat().st_size
        if not force and target.exists() and target.stat().st_size == size:
            skipped += 1
            continue
        if copied == 0 and on_first_copy is not None:
            on_first_copy()
        if dry:
            print(f"    would copy {rel} ({size/1e6:.1f} MB)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(target))
        copied += 1
        bytes_copied += size
    return copied, skipped, bytes_copied


def print_summary_table(rows, dry: bool):
    """One line per session seen in the inbox: NEW ones first, then the rest.
    rows = (reviewer, session, status, n_files, n_bytes, labels_cell)."""
    if not rows:
        return
    order = {"NEW": 0, "SKIPPED": 1, "NOT FOUND": 2, "unchanged": 3}
    rows = sorted(rows, key=lambda r: (order.get(r[2], 9), r[0].lower(), r[1].lower()))
    head = ("status", "reviewer", "session", "files", "MB", "keep/del/motion")
    body = [(st, rv, se, str(n) if n else "-", f"{b/1e6:.1f}" if n else "-", lab)
            for rv, se, st, n, b, lab in rows]
    w = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    fmt = "  ".join(f"{{:<{w[i]}}}" if i < 3 or i == 5 else f"{{:>{w[i]}}}"
                    for i in range(len(head)))
    n_new = sum(1 for r in rows if r[2] == "NEW")
    n_skip = sum(1 for r in rows if r[2] in ("SKIPPED", "NOT FOUND"))
    print(f"\nSummary{' (dry run)' if dry else ''}: {len(rows)} session(s) in inbox -- "
          f"{n_new} new, {len(rows) - n_new - n_skip} unchanged, {n_skip} skipped")
    print(fmt.format(*head))
    print("  ".join("-" * x for x in w))
    for line in body:
        print(fmt.format(*line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?",
                    help="area\\task\\session to ingest (default: all found in inbox)")
    ap.add_argument("--exchange", default=EXCHANGE_ROOT,
                    help="exchange root (default: CNMFE_EXCHANGE_ROOT / .env)")
    ap.add_argument("--force", action="store_true",
                    help="copy every file, even if a same-size copy already exists")
    ap.add_argument("--replace-labels", action="store_true",
                    help="allow a return to replace labels that a DIFFERENT reviewer "
                         "already provided for the same session (normally refused; "
                         "the provenance record is rewritten to the new reviewer)")
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
    rows = []                    # one summary-table row per session seen
    total_motion_tags = 0        # motion-tagged deletes across this ingest
    sessions_with_motion = 0     # sessions that carried >=1 motion tag
    sessions_with_field = 0      # sessions whose labels.mat has the motion_delete field
    for src in sessions:
        try:
            reviewer = src.relative_to(inbox).parts[0]
        except ValueError:
            reviewer = None
        if not src.is_dir():
            print(f"SKIP (not found): {src}")
            rows.append((reviewer or "-", src.name, "NOT FOUND", 0, 0, "-"))
            continue
        if is_training_sandbox(src, inbox):
            print(f"\nSKIP {src.name}")
            print("  !! training sandbox (TRAINING_SESSION.txt present): a trainee's practice "
                  "result, never ingested. Ask the trainee to return it to "
                  "training/<name>/returns/ instead.")
            skipped_sessions.append((src, "training sandbox"))
            rows.append((reviewer or "-", src.name, "SKIPPED", 0, 0, "-"))
            continue
        dst, note = resolve_dest(src)
        if dst is None:
            print(f"\nSKIP {src.name}")
            print(f"  !! {note}")
            skipped_sessions.append((src, note))
            rows.append((reviewer or "-", src.name, "SKIPPED", 0, 0, "-"))
            continue

        # Duplicate-review guard: refuse to let one reviewer's return replace
        # labels a DIFFERENT reviewer already provided for this session.
        # First case 2026-08-20: a session staged to Alisia had already been
        # validly reviewed by Taylor and ingested (and trained on); her return
        # arriving later would have silently overwritten his labels -- and the
        # rest of her review (neuron.mat, ROIs.jpg, traces) would have replaced
        # his, leaving a session that mixes two people's decisions. The whole
        # session is therefore skipped, not just labels.mat. Deliberate
        # replacement: --replace-labels.
        if reviewer and (src / "labels.mat").exists():
            prev = read_provenance(dst)
            if (prev and prev.lower() != reviewer.lower()
                    and not args.replace_labels):
                msg = (f"session already carries {prev}'s ingested labels; "
                       f"refusing {reviewer}'s duplicate review (whole session "
                       f"skipped). If the replacement is intentional, re-run "
                       f"with --replace-labels.")
                print(f"\nSKIP {src.name}")
                print(f"  !! {msg}")
                skipped_sessions.append((src, msg))
                rows.append((reviewer, str(dst.relative_to(DATA_PARENT)),
                             "SKIPPED", 0, 0, "-"))
                continue

        # Only sessions that bring something new get a detailed block; the
        # header is printed lazily, just before the first file is copied, so a
        # long multi-GB copy still announces itself up front. Sessions with
        # nothing to copy stay silent here and appear in the summary table.
        def _header(dst=dst, note=note, src=src):
            print(f"\nIngest {dst.relative_to(DATA_PARENT)}  ({note})")
            print(f"  {src}  ->  {dst}")
        c, s, b = copy_session(src, dst, args.force, args.dry_run, on_first_copy=_header)
        is_new = c > 0
        if is_new:
            verb = "would copy" if args.dry_run else "copied"
            print(f"  {verb} {c} files ({b/1e6:.1f} MB), skipped {s} unchanged")
        if reviewer and (src / "labels.mat").exists() and not args.dry_run:
            write_provenance(dst, reviewer, src)

        # Report the reviewer's label breakdown, including motion-delete tags.
        # Read from the source so this works in --dry-run too (nothing copied yet).
        labels_src = src / "labels.mat"
        labels_cell = "-"
        if labels_src.exists():
            counts = read_label_counts(labels_src)
            if counts is None:
                labels_cell = "unreadable"
                if is_new:
                    print("  labels.mat present but could not be read for a summary.")
            else:
                n_keep, n_delete, n_motion = counts
                if n_motion is None:
                    labels_cell = f"{n_keep}/{n_delete}/-"
                    if is_new:
                        print(f"  labels: {n_keep} keep / {n_delete} delete "
                              f"(no motion tags -- reviewed before the (m) option)")
                else:
                    labels_cell = f"{n_keep}/{n_delete}/{n_motion}"
                    sessions_with_field += 1
                    total_motion_tags += n_motion
                    if n_motion > 0:
                        sessions_with_motion += 1
                    if is_new:
                        print(f"  labels: {n_keep} keep / {n_delete} delete, "
                              f"of which {n_motion} tagged as motion deletes")
            if is_new and not args.dry_run:
                print("  labels.mat present -> watcher will auto-retrain on its next poll.")
        rows.append((reviewer or "-", str(dst.relative_to(DATA_PARENT)),
                     "NEW" if is_new else "unchanged", c, b, labels_cell))

    print_summary_table(rows, args.dry_run)
    if skipped_sessions:
        print(f"\n{len(skipped_sessions)} session(s) SKIPPED (not ingested):")
        for src, note in skipped_sessions:
            print(f"  - {src.name}: {note}")
    if sessions_with_field:
        print(f"\nMotion labels this ingest: {total_motion_tags} tag(s) across "
              f"{sessions_with_motion} session(s) "
              f"({sessions_with_field} session(s) reviewed with the (m) option).")
    print("\nDone.")


if __name__ == "__main__":
    main()
