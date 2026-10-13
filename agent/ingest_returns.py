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

RE-REVIEWS (2026-10-12).  A return for a session that ALREADY carries labels.mat
locally is never written over the live files.  It is PARKED, every review
output copied regardless of size, under
    DATA_PARENT/{area}/{task}/{session}/.rereview/<labels-save-stamp>_<reviewer>/
and the operator decides with agent/rereview.py (compare, adopt, keep-current).
A return carrying REREVIEW_SESSION.txt (staged by rereview.py request) is parked
even with --replace-labels; an unmarked return for a labelled session is parked
unless --replace-labels is given, which keeps the old overwrite-in-place
behaviour for deliberate replacements.

PRE-INGEST SNAPSHOT (2026-10-12).  Before the first file of a return is written,
every local file the ingest is about to overwrite (the headless neuron.mat with
all candidates, spatial_footprints.mat, Coor.mat, the trace text files, the
per-candidate PNGs; never the raw video) is copied to
    DATA_PARENT/{area}/{task}/{session}/.pre_ingest/<relpath>   (+ PRE_INGEST.txt)
once per session: an existing .pre_ingest/ is never rewritten, --force included.
A first-review return whose labels.mat length does not match the live review
set (candidate_features.npz minus auto-rejected) is SKIPPED with nothing written.
labels_provenance.txt is rewritten only when files were actually copied or no
record exists yet.  The reviewer of record is the inbox folder the return was
ingested from (user decision 2026-10-12); nothing here compares it with
review_assigned.txt or review_meta.txt.

SAFETY: this script only READS from the server inbox and writes copies into the
LOCAL data tree. It never writes to, modifies, or deletes anything on the server.
There is no delete/move call anywhere in this file.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT
from review_prep import REREVIEW_DIRNAME, REREVIEW_MARKER, REREVIEW_RETURNED, write_kv


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


def labels_length(labels_path: Path):
    """Number of entries in a labels.mat, or None if unreadable."""
    try:
        import scipy.io as sio
        return int(sio.loadmat(str(labels_path))["labels"].size)
    except Exception:
        return None


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


def plan_copies(src: Path, dst: Path, force: bool):
    """Every file copy_session would write, after all skip rules:
    [(src_file, rel, size)], plus the number of files skipped."""
    plan, skipped = [], 0
    for f in sorted(src.rglob("*")):
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
        # Bundle INPUTS (the frozen review set and its report) are only ever
        # produced here; a return carries central's copy or a STALE one.  On
        # 2026-10-12 a return reviewed from a 09-25 bundle overwrote the
        # 10-06 re-run's review_neuron.mat / review_summary.txt / report of
        # two live sessions because the sizes differed, leaving labels that
        # no longer line up with candidate_features.npz.  Never overwrite an
        # existing input, with or without --force; create it only if missing.
        if rel.name in BUNDLE_INPUTS and target.exists():
            if target.stat().st_size != size:
                print(f"    keep local {rel} (the returned copy differs in size: a stale "
                      f"bundle input, never written over the live review set)")
            skipped += 1
            continue
        if not force and target.exists() and target.stat().st_size == size:
            skipped += 1
            continue
        plan.append((f, rel, size))
    return plan, skipped


def snapshot_before_overwrite(dst: Path, items, src: Path, dry: bool) -> dict:
    """Copy every local file that this ingest is about to overwrite into
    dst/.pre_ingest/<rel> (copy2), once per session.

    The first ingest of a session replaces the headless outputs (neuron.mat
    with ALL candidates, spatial_footprints.mat, Coor.mat, the trace text
    files, the per-candidate PNGs) with the reviewer's kept set, and at a
    threshold above 0 the auto-rejected candidates' footprints and traces then
    survive nowhere (2026-10-12 incident).  The snapshot is WRITE-ONCE: an
    existing .pre_ingest/ is the pristine pre-review state and is never
    touched again, --force included.  items = [(rel, size)] of targets that
    exist and are in the copy plan (the raw video is excluded by the caller).
    Returns {'status': none|kept|would|taken, 'files', 'bytes'}."""
    if not items:
        return {"status": "none", "files": 0, "bytes": 0}
    snap = dst / SNAPSHOT_DIRNAME
    total = sum(size for _, size in items)
    if snap.exists():
        taken = ""
        try:
            for line in (snap / SNAPSHOT_NOTE).read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("taken:"):
                    taken = line.split(":", 1)[1].strip()
        except Exception:
            pass
        print(f"    pre-ingest snapshot kept as is ({SNAPSHOT_DIRNAME}/ exists"
              + (f", taken {taken}" if taken else "") + "); not rewritten")
        return {"status": "kept", "files": 0, "bytes": 0}
    if dry:
        print(f"    would snapshot {len(items)} file(s) ({total/1e6:.1f} MB) to "
              f"{SNAPSHOT_DIRNAME}/ before overwriting them")
        return {"status": "would", "files": len(items), "bytes": total}
    for rel, _ in items:
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(dst / rel), str(snap / rel))
    lines = [f"taken: {datetime.now():%Y-%m-%d %H:%M:%S}",
             "what: copies of the local files that the ingest below overwrote, "
             "made just before the first write (write-once; never updated)",
             f"before_ingest_of: {src}",
             f"had_labels: {'yes' if (dst / 'labels.mat').exists() else 'no'}",
             f"n_files: {len(items)}",
             f"bytes: {total}",
             "files:"] + [f"  {rel.as_posix()}" for rel, _ in items]
    (snap / SNAPSHOT_NOTE).write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    print(f"    pre-ingest snapshot: {len(items)} file(s) ({total/1e6:.1f} MB) -> {SNAPSHOT_DIRNAME}/")
    return {"status": "taken", "files": len(items), "bytes": total}


def copy_session(src: Path, dst: Path, force: bool, dry: bool, on_first_copy=None,
                 stats: dict = None):
    """on_first_copy, if given, is called once just before the first file is
    copied (or would be, in a dry run) -- main() uses it to print the session
    header only for sessions that actually bring something new.  stats, if
    given, receives {'snapshot': ...} from snapshot_before_overwrite()."""
    plan, skipped = plan_copies(src, dst, force)
    copied = bytes_copied = 0
    if plan and on_first_copy is not None:
        on_first_copy()
    if plan:
        to_save = [(rel, size) for f, rel, size in plan
                   if (dst / rel).exists() and not _is_raw_video(f, src.name)]
        snap = snapshot_before_overwrite(dst, to_save, src, dry)
        if stats is not None:
            stats["snapshot"] = snap
    for f, rel, size in plan:
        target = dst / rel
        if dry:
            print(f"    would copy {rel} ({size/1e6:.1f} MB)")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(target))
        copied += 1
        bytes_copied += size
    return copied, skipped, bytes_copied


# Pre-ingest snapshot folder inside a session (dot folder: every session
# scanner ignores it, like .rereview/ and .excluded/).
SNAPSHOT_DIRNAME = ".pre_ingest"
SNAPSHOT_NOTE = "PRE_INGEST.txt"


# Bundle INPUT files (what push_review_bundle / rereview.py stage).  When a
# re-review return is parked, these are skipped if the live session already has
# a same-size copy; every other returned file (the review OUTPUTS: labels.mat,
# neuron.mat, ROIs.jpg, traces, merge logs, review_meta.txt, ...) is copied
# regardless of size, because a redone labels.mat can be byte-for-byte the same
# size as the first one and the size check would silently drop it.
BUNDLE_INPUTS = {"review_neuron.mat", "Cn.mat", "pnr.mat", "Ybg_weights.mat",
                 "Ybg_mean.mat", "review_report.pdf", "review_summary.txt",
                 "run_final_review.m", "ROIs_candidates.jpg"}


def _is_raw_video(f: Path, session_name: str) -> bool:
    return f.name == f"{session_name}.mat" or f.suffix.lower() in (".tif", ".tiff")


def expected_n_review(session_dir: Path):
    """How many candidates the review set holds, from candidate_features.npz
    (all N minus the auto-rejected), or None if unreadable."""
    try:
        import numpy as np
        npz = np.load(str(session_dir / "candidate_features.npz"), allow_pickle=True)
        n = int(npz["n_candidates"].ravel()[0]) if "n_candidates" in npz.files \
            else int(len(npz["feature_matrix"]))
        auto = set(npz["auto_rejected"].ravel().astype(int).tolist())
        return n - len([i for i in range(n) if i in auto])
    except Exception:
        return None


def park_return(src: Path, dst: Path, reviewer: str, force: bool, dry: bool):
    """Park a return for an already-labelled session under
    dst/.rereview/<stamp>_<reviewer>/ instead of writing over the live files.

    Returns (parked_dir, n_copied, n_bytes, status) with status one of
    'parked', 'already parked', 'would park'.  Nothing in the live session
    folder is touched; the parked folder is new.  The stamp is the reviewer's
    labels.mat save time, so re-running ingest finds the same folder and
    reports it as already parked (use --force to copy again)."""
    labels_src = src / "labels.mat"
    stamp_src = labels_src if labels_src.exists() else src
    stamp = datetime.fromtimestamp(stamp_src.stat().st_mtime).strftime("%Y%m%d-%H%M")
    base = f"{stamp}_{reviewer}"
    parked = dst / REREVIEW_DIRNAME / base
    suffix = 1
    while (parked / REREVIEW_RETURNED).exists():
        # Same folder name: the same return seen again (idempotent re-run) or,
        # if the labels differ, a different return saved in the same minute.
        same = (not labels_src.exists() or not (parked / "labels.mat").exists()
                or labels_src.read_bytes() == (parked / "labels.mat").read_bytes())
        if same:
            if not force:
                return parked, 0, 0, "already parked"
            break
        suffix += 1
        parked = dst / REREVIEW_DIRNAME / f"{base}-{suffix}"
    copied = bytes_copied = 0
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.name in CENTRAL_ONLY:
            continue
        if _is_raw_video(f, src.name):
            continue            # the live session already holds the video
        size = f.stat().st_size
        live = dst / rel
        if rel.name in BUNDLE_INPUTS and live.exists() and live.stat().st_size == size:
            continue            # unchanged bundle input
        if dry:
            print(f"    would park {rel} ({size/1e6:.1f} MB)")
        else:
            (parked / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(f), str(parked / rel))
        copied += 1
        bytes_copied += size
    if dry:
        return parked, copied, bytes_copied, "would park"
    n_labels = None
    try:
        import scipy.io as sio
        n_labels = int(sio.loadmat(str(parked / "labels.mat"))["labels"].size)
    except Exception:
        pass
    n_exp = expected_n_review(dst)
    aligned = (n_labels is not None and n_exp is not None and n_labels == n_exp)
    rn = src / "review_neuron.mat"
    live_rn = dst / "review_neuron.mat"
    stale_bundle = (rn.exists() and live_rn.exists()
                    and rn.stat().st_size != live_rn.stat().st_size)
    write_kv(parked / REREVIEW_RETURNED, {
        "reviewer": reviewer,
        "source": str(src),
        "parked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "labels_saved": datetime.fromtimestamp(stamp_src.stat().st_mtime)
                                .strftime("%Y-%m-%d %H:%M:%S"),
        "marker_present": "yes" if (src / REREVIEW_MARKER).exists() else "no",
        "n_labels": "-" if n_labels is None else n_labels,
        "expected_n_review": "-" if n_exp is None else n_exp,
        "aligned": "yes" if aligned else "NO",
        "live_reviewer_at_park": read_provenance(dst) or "unknown",
        "stale_bundle": "yes" if stale_bundle else "no",
    })
    if not aligned:
        print(f"  !! WARNING: parked labels.mat has {n_labels} entries but the live "
              f"review set expects {n_exp}; rereview.py adopt will refuse it.")
    if stale_bundle:
        print("  !! WARNING: the returned review_neuron.mat differs in size from the live "
              "one -- the reviewer worked from a stale bundle.")
    return parked, copied, bytes_copied, "parked"


def print_summary_table(rows, dry: bool):
    """One line per session seen in the inbox: NEW ones first, then the rest.
    rows = (reviewer, session, status, n_files, n_bytes, labels_cell)."""
    if not rows:
        return
    order = {"NEW": 0, "PARKED": 0, "SKIPPED": 1, "NOT FOUND": 2, "unchanged": 3,
             "parked already": 3}
    rows = sorted(rows, key=lambda r: (order.get(r[2], 9), r[0].lower(), r[1].lower()))
    head = ("status", "reviewer", "session", "files", "MB", "keep/del/motion", "pre-ingest snapshot")
    body = [(r[2], r[0], r[1], str(r[3]) if r[3] else "-", f"{r[4]/1e6:.1f}" if r[3] else "-",
             r[5], r[6] if len(r) > 6 else "-")
            for r in rows]
    w = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    fmt = "  ".join(f"{{:<{w[i]}}}" if i < 3 or i >= 5 else f"{{:>{w[i]}}}"
                    for i in range(len(head)))
    n_new = sum(1 for r in rows if r[2] == "NEW")
    n_parked = sum(1 for r in rows if r[2] == "PARKED")
    n_skip = sum(1 for r in rows if r[2] in ("SKIPPED", "NOT FOUND"))
    print(f"\nSummary{' (dry run)' if dry else ''}: {len(rows)} session(s) in inbox -- "
          f"{n_new} new, {n_parked} parked (re-review), "
          f"{len(rows) - n_new - n_parked - n_skip} unchanged, {n_skip} skipped")
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

        # Re-review parking: a return for a session that already carries labels
        # never overwrites the live files (the first review would be lost and,
        # for the same reviewer, a same-size labels.mat would even be skipped
        # as "unchanged").  Marked returns (REREVIEW_SESSION.txt from
        # rereview.py request) are always parked; unmarked ones are parked
        # unless --replace-labels asks for the old overwrite-in-place path.
        marked = (src / REREVIEW_MARKER).exists()
        live_labelled = dst.is_dir() and (dst / "labels.mat").exists()
        if marked or (live_labelled and not args.replace_labels):
            who = reviewer or "unknown"
            if not (dst.is_dir() and (dst / "candidate_features.npz").exists()):
                msg = ("re-review return, but no live session with "
                       "candidate_features.npz to park it against")
                print(f"\nSKIP {src.name}")
                print(f"  !! {msg}")
                skipped_sessions.append((src, msg))
                rows.append((who, src.name, "SKIPPED", 0, 0, "-"))
                continue
            if marked and args.replace_labels:
                print(f"\n  NOTE: {src.name} carries {REREVIEW_MARKER}; --replace-labels "
                      f"does not apply, parking it instead.")
            # The return that IS the live version (its labels.mat byte-identical
            # to the local one) is not a second review: a whole-inbox re-run
            # must stay a no-op for it, not park it again.
            if live_labelled and (src / "labels.mat").exists() \
                    and (src / "labels.mat").read_bytes() == (dst / "labels.mat").read_bytes():
                prev = read_provenance(dst) or "unknown"
                if prev.lower() != who.lower():
                    print(f"\n  NOTE: {src.name}: this return's labels are already the live ones, "
                          f"but the provenance names {prev} and the return sits under "
                          f"inbox/{who}; the inbox folder is the reviewer of record -- edit "
                          f"{PROVENANCE_NAME} by hand if {prev} is wrong.")
                lab = "-"
                counts = read_label_counts(src / "labels.mat")
                if counts is not None:
                    k, d, m = counts
                    lab = f"{k}/{d}/{'-' if m is None else m}"
                rows.append((who, str(dst.relative_to(DATA_PARENT)), "unchanged", 0, 0, lab))
                continue
            print(f"\nPark {dst.relative_to(DATA_PARENT)}  (re-review return by {who}; "
                  f"live labels by {read_provenance(dst) or 'unknown'} untouched)")
            parked, c, b, status = park_return(src, dst, who, args.force, args.dry_run)
            print(f"  {src}  ->  {parked}")
            if status == "already parked":
                print("  already parked (same labels.mat save time); --force copies again")
                st_cell = "parked already"
            else:
                verb = "would park" if args.dry_run else "parked"
                print(f"  {verb} {c} files ({b/1e6:.1f} MB)")
                st_cell = "PARKED"
                if not args.dry_run:
                    print("  next: rereview.py compare / adopt / keep-current "
                          "(nothing retrains until a version is adopted)")
            lab = "-"
            if (src / "labels.mat").exists():
                counts = read_label_counts(src / "labels.mat")
                if counts is not None:
                    k, d, m = counts
                    lab = f"{k}/{d}/{'-' if m is None else m}"
                    print(f"  labels: {k} keep / {d} delete"
                          + ("" if m is None else f", {m} motion tags"))
            rows.append((who, str(dst.relative_to(DATA_PARENT)), st_cell, c, b, lab))
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

        # Stale-bundle guard for the first-review path (2026-10-12): a labels.mat
        # whose length does not match the live review set would land on the
        # session, the trainer would skip it with its size-mismatch WARN, and
        # the session would silently drop out of training.  Nothing is written.
        if (src / "labels.mat").exists() and dst.is_dir() \
                and (dst / "candidate_features.npz").exists():
            n_lab = labels_length(src / "labels.mat")
            n_exp = expected_n_review(dst)
            if n_lab is not None and n_exp is not None and n_lab != n_exp:
                msg = (f"labels.mat has {n_lab} entries but the live review set expects "
                       f"{n_exp} (reviewed from a stale bundle?); nothing written. Re-push "
                       f"the current bundle for a fresh review, or move the return aside.")
                print(f"\nSKIP {src.name}")
                print(f"  !! {msg}")
                skipped_sessions.append((src, msg))
                rows.append((reviewer or "-", str(dst.relative_to(DATA_PARENT)),
                             "SKIPPED", 0, 0, "-"))
                continue

        # Only sessions that bring something new get a detailed block; the
        # header is printed lazily, just before the first file is copied, so a
        # long multi-GB copy still announces itself up front. Sessions with
        # nothing to copy stay silent here and appear in the summary table.
        def _header(dst=dst, note=note, src=src):
            print(f"\nIngest {dst.relative_to(DATA_PARENT)}  ({note})")
            print(f"  {src}  ->  {dst}")
        stats = {}
        c, s, b = copy_session(src, dst, args.force, args.dry_run, on_first_copy=_header,
                               stats=stats)
        is_new = c > 0
        if is_new:
            verb = "would copy" if args.dry_run else "copied"
            print(f"  {verb} {c} files ({b/1e6:.1f} MB), skipped {s} unchanged")
        snap = stats.get("snapshot", {"status": "none", "files": 0, "bytes": 0})
        snap_cell = {"taken": f"{snap['files']} files/{snap['bytes']/1e6:.0f} MB",
                     "would": f"would {snap['files']} files/{snap['bytes']/1e6:.0f} MB",
                     "kept": "kept", "none": "-"}[snap["status"]]
        # Provenance is rewritten only when this ingest actually brought files
        # (or no record exists yet): on 2026-10-12 a no-copy whole-inbox run
        # replaced the 'ingested:' date of 108 untouched sessions.
        if reviewer and (src / "labels.mat").exists() and not args.dry_run \
                and (is_new or not (dst / PROVENANCE_NAME).exists()):
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
                     "NEW" if is_new else "unchanged", c, b, labels_cell, snap_cell))

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
