"""
push_review_bundle.py - stage CNMFe review bundles on the lab server so remote
reviewers can pull them, run run_final_review.m, and push the curated folder
back (ingest_returns.py brings it home for retraining).

Sessions are routed to a PER-REVIEWER subfolder so each machine only pulls its
own work:  outbox/<reviewer>/{area}/{task}/{session}/ . --assignee names the
reviewer and is REQUIRED (that name is the folder, and is recorded in the marker).

Single session:
  python push_review_bundle.py vCA1\\3odor\\AVG5x-...-000 --assignee Alisia
  python push_review_bundle.py vCA1\\3odor\\AVG5x-...-000 --assignee Alisia --dry-run

Batch (every session awaiting review, skipping ones already out for review):
  python push_review_bundle.py --all --assignee Alisia
  python push_review_bundle.py --all --assignee Alisia,Julian      # split round-robin
  python push_review_bundle.py --all --area vCA1 --task 3odor --assignee Alisia
  python push_review_bundle.py --all --animal bla8,bla12 --assignee Alisia   # by animal/FOV
  python push_review_bundle.py --all --assignee Alisia --dry-run

To RE-assign sessions already out for review (e.g. re-deal Alisia's pile across
several reviewers), add --force so already-marked sessions are re-staged and the
marker is rewritten to the new reviewer. Delete the stale outbox/<old>/ folder on
the server first (that server delete is a manual operator step - this script and
Claude never delete from the server).

Each staged session is also marked "out for review" via a review_assigned.txt
marker in the LOCAL session folder, so the watcher's REVIEW_QUEUE.md lists it as
out (and to whom) and a later --all run will not send it again. To re-open a
session, delete that marker file (the scripts never delete it for you).

Outbound bundle = the minimal set CNMFe_final_save.m needs:
  {SESSION}.mat (raw video), review_neuron.mat, Cn.mat, pnr.mat, Ybg_weights.mat,
  review_report.pdf, review_summary.txt, plus a freshly generated, self-locating
  run_final_review.m (so it works on any machine regardless of the original path).

Red guide (retro-labelled animals): for a pnb* session the push first runs the
retro project's hook (python -m retro_align guide-for-push <session> [--dry-run],
from the retro project folder), which reuses the session's tdTomato
red_guide.png if it is current and rebuilds it otherwise. The guide is then
looked up (CNMFE_RETRO_RESULTS, see local_config.py) and added only when it
covers the live review set. Never required: a hook that fails, times out or
cannot start leaves the session pushed without a guide. CNMFE_RETRO_GUIDE_BUILD=0
skips the hook. Re-review bundles (rereview.py) only look the guide up.

Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.

SAFETY: this script only COPIES files to the server outbox and WRITES a local
assignment marker / launcher. It never deletes or removes anything, on the server
or locally. There is no delete/move call anywhere in this file. The one other
local write: a finished session whose movie .mat was pruned gets it rebuilt from
its .tif first (movie_mat.py, via MATLAB), inside the local session folder.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import (DATA_PARENT, EXCHANGE_ROOT, PYTHON_EXE, REPO_ROOT,
                          RETRO_GUIDE_BUILD, RETRO_RESULTS)
from review_prep import write_launcher, write_assignment, read_assignment
import movie_mat

REQUIRED = ["review_neuron.mat"]
OPTIONAL = ["Cn.mat", "pnr.mat", "Ybg_weights.mat",
            "review_report.pdf", "review_summary.txt"]
# {SESSION}.mat (raw video) is handled separately; run_final_review.m is generated
# fresh into the bundle (never copied from the session, which may be stale).
RED_GUIDE = "red_guide.png"
# The red guide is not a session file: it lives in the retro project's results
# tree (read-only here) and find_red_guide() adds it.
GUIDE_HOOK_TIMEOUT = 900        # s; a first build took 48 s, a current guide ~2.5 s
COMPARE_CHUNK = 8 << 20


def safe_name(name: str) -> str:
    """Folder-safe reviewer name, so outbox/<reviewer>/ is always a clean path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("_")


ANIMAL_RE = re.compile(r"^[a-z]+\d+$", re.IGNORECASE)


def animal_of(session_name: str):
    """The animal token in a session name (e.g. 'bla12', 'pnb88'), or None.
    It is the hyphen-delimited segment that is letters-then-digits."""
    for seg in session_name.split("-"):
        if ANIMAL_RE.match(seg):
            return seg.lower()
    return None


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


def collect_files(session_dir: Path, dry_run=False, build_red_guide=False):
    """List the existing bundle files. Raises ValueError if a required file is missing.
    build_red_guide=True (the push) runs the retro guide hook first; the default
    (rereview.py) only looks an existing guide up."""
    files = []
    for name in REQUIRED:
        f = session_dir / name
        if not f.exists():
            raise ValueError(f"required file missing: {f}")
        files.append(f)
    # The raw video goes first.  A finished session's movie may have been pruned
    # (prune_movie_mats.py); it is then rebuilt from the session .tif, and a
    # rebuild that fails raises rather than ship a bundle with no video.
    video = movie_mat.ensure(session_dir, dry_run=dry_run)
    if video is not None:
        files.insert(0, video)
    elif not (dry_run and movie_mat.movie_tif(session_dir) is not None):
        print(f"  WARNING: raw video {movie_mat.movie_mat(session_dir).name} not found - "
              f"reviewer needs it for the video pass.")
    for name in OPTIONAL:
        f = session_dir / name
        if f.exists():
            files.append(f)
        else:
            print(f"  (optional) missing: {name}")
    guide = find_red_guide(session_dir, build=build_red_guide, dry_run=dry_run)
    if guide is not None:
        files.append(guide)
    return files


def find_red_guide(session_dir: Path, root=None, build=False, dry_run=False):
    """The retro project's red_guide.png for this session, or None.

    With build (the push, pnb* sessions, RETRO_GUIDE_BUILD on) the retro hook runs
    first and must report the guide ready (exit 0); any other outcome means no
    guide this time.  The lookup below stays the gate either way.

    Matched by exact session-folder name at <root>/<animal>/<MMDDYY>/<session>/
    red_guide/ (the MMDDYY level is 6 digits even where the session name uses 8,
    so the path is never rebuilt from the name).  Added only if the guide's
    candidates_shown.npz covers the live review set (same N, same auto-rejected
    indices): on 2026-11-02 090526-pnb99 was re-run (30 -> 76 candidates) after
    its guide was drawn, and a stale guide outlines the wrong candidates.
    Never required; raises ValueError only when more than one guide matches."""
    name = session_dir.name
    retro = (animal_of(name) or "").startswith("pnb")
    root = RETRO_RESULTS if root is None else root
    if not root or not Path(root).is_dir():
        if retro:
            print(f"  red guide: not found (retro results root '{root}' missing)")
        return None
    if build and retro and RETRO_GUIDE_BUILD:
        code, line = run_guide_hook(session_dir, Path(root).parent, dry_run)
        if code == 3:
            why = line.split("would be made", 2)[-1].strip(" :") or line
            print(f"  red guide: would be built at push {why}")
            return None
        print(f"  {line}")
        if code != 0:
            return None
    hits = []
    for animal_dir in sorted(Path(root).iterdir()):
        if not animal_dir.is_dir() or animal_dir.name.startswith("."):
            continue
        for date_dir in sorted(animal_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name.startswith("."):
                continue
            f = date_dir / name / "red_guide" / RED_GUIDE
            if f.is_file():
                hits.append(f)
    if not hits:
        if retro:
            print("  red guide: not found (run retro guide-for-push first)")
        return None
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} red guides match {name}, refusing to pick one: "
                         + ", ".join(str(h) for h in hits))
    problem = red_guide_mismatch(hits[0].parent / "candidates_shown.npz", session_dir)
    if problem:
        print(f"  red guide: STALE, NOT included -- {problem}. Run retro "
              f"guide-for-push on this session, then push it again.")
        return None
    print(f"  red guide: included ({hits[0]})")
    return hits[0]


def run_guide_hook(session_dir: Path, project: Path, dry_run: bool):
    """Run the retro project's guide-for-push hook for one session.  Returns
    (exit code, one ASCII line to print).  Codes: 0 guide ready, 1 not made,
    2 not applicable, 3 (dry run) would be made; -1 = the hook did not finish
    (timeout, or it could not start).  Never raises: the push goes on."""
    cmd = [PYTHON_EXE, "-m", "retro_align", "guide-for-push", str(session_dir)]
    if dry_run:
        cmd.append("--dry-run")
    print("  red guide: running retro guide-for-push ...", flush=True)
    try:
        r = subprocess.run(cmd, cwd=str(project), capture_output=True, encoding="utf-8",
                           errors="replace", timeout=GUIDE_HOOK_TIMEOUT,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    except subprocess.TimeoutExpired:
        return -1, (f"red guide: hook timed out after {GUIDE_HOOK_TIMEOUT} s; "
                    f"pushing without a guide")
    except OSError as e:
        return -1, (f"red guide: hook could not start ({type(e).__name__}: {e}); "
                    f"pushing without a guide")
    out = (r.stdout or "").strip().splitlines()
    line = out[-1] if out else ""
    if not line.startswith("red guide"):         # crashed, or not the hook at all
        err = (r.stderr or "").strip().splitlines()
        code = -1
        line = (f"red guide: hook exited {r.returncode} without a result line "
                f"({err[-1] if err else line or 'no output'}); pushing without a guide")
    else:
        code = r.returncode
    return code, line.encode("ascii", "replace").decode("ascii")


def red_guide_mismatch(shown: Path, session_dir: Path):
    """Why the guide's candidates_shown.npz does not match the session's live
    review set (candidate_features.npz), or None if it does."""
    try:
        import numpy as np
        g = np.load(str(shown), allow_pickle=False)
        live = np.load(str(session_dir / "candidate_features.npz"), allow_pickle=True)
        g_n = int(g["n_candidates"].ravel()[0])
        n = (int(live["n_candidates"].ravel()[0]) if "n_candidates" in live.files
             else int(len(live["feature_matrix"])))
        g_auto = set(g["auto_rejected"].ravel().astype(int).tolist())
        auto = set(live["auto_rejected"].ravel().astype(int).tolist())
    except Exception as e:
        return f"cannot check it against the live review set ({type(e).__name__}: {e})"
    if g_n != n or len(g_auto) != len(auto):
        return (f"drawn for {g_n} candidates / {len(g_auto)} auto-rejected, "
                f"the live set is {n} / {len(auto)}")
    if g_auto != auto:
        return f"same counts ({n} / {len(auto)}) but different auto-rejected candidates"
    return None


def same_contents(src: Path, target: Path) -> bool:
    """True if target exists and is byte-identical to src.  Size alone is not
    enough: a regenerated file can keep its size (a red guide redrawn after a
    re-run, a re-saved Cn.mat), and a size-only skip would leave the old copy in
    the bundle.  Reads both files, so an identical raw video is read back from
    the server in full."""
    if not target.is_file() or target.stat().st_size != src.stat().st_size:
        return False
    with open(src, "rb") as a, open(target, "rb") as b:
        while True:
            ca, cb = a.read(COMPARE_CHUNK), b.read(COMPARE_CHUNK)
            if ca != cb:
                return False
            if not ca:
                return True


def stage_one(session_dir: Path, exchange: str, dry_run: bool, assignee: str):
    """Stage one session's bundle into outbox/<assignee>/. Returns total bytes."""
    dest = Path(exchange) / "outbox" / assignee / session_rel(session_dir)
    files = collect_files(session_dir, dry_run=dry_run, build_red_guide=True)
    print(f"  assignee: {assignee}")
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
        if same_contents(f, target):
            print(f"  skip (already staged): {f.name}")
            continue
        note = " -- replaces a different copy" if target.exists() else ""
        print(f"  copy {f.name} ({size/1e6:.1f} MB){note} ...")
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
    ap.add_argument("--animal",
                    help="(with --all) limit to sessions for these animals (comma list, e.g. "
                         "bla8,bla12) - lets you give each reviewer consistent animals/FOVs.")
    ap.add_argument("--assignee",
                    help="reviewer name -> routes the bundle to outbox/<name>/ and records the "
                         "marker. REQUIRED. With --all you may pass a comma-separated list "
                         "(e.g. Alisia,Julian) to split the batch round-robin across reviewers.")
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
        assignees = [safe_name(a) for a in (args.assignee or "").split(",") if safe_name(a)]
        if not assignees:
            sys.exit("ERROR: --assignee is required (routes each session to outbox/<reviewer>/). "
                     "Pass one name, or several comma-separated to split the batch round-robin.")
        sessions = list(find_awaiting(DATA_PARENT, args.area, args.task))
        if args.animal:
            wanted = {a.strip().lower() for a in args.animal.split(",") if a.strip()}
            sessions = [s for s in sessions if animal_of(s.name) in wanted]
        if not args.force:
            sessions = [s for s in sessions if read_assignment(s) is None]
        if not sessions:
            print("No awaiting sessions to stage (all reviewed or already out for review).")
            return
        split = (f" across {len(assignees)} reviewers ({', '.join(assignees)})"
                 if len(assignees) > 1 else f" to {assignees[0]}")
        print(f"{'DRY RUN: ' if args.dry_run else ''}{len(sessions)} session(s) to stage{split}:\n")
        staged, grand, failures = 0, 0, []
        for i, sd in enumerate(sessions):
            who = assignees[i % len(assignees)]
            print(f"[{session_rel(sd)}]")
            try:
                grand += stage_one(sd, args.exchange, args.dry_run, who)
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
    who = safe_name((args.assignee or "").split(",")[0])
    if not who:
        sys.exit("ERROR: --assignee <reviewer> is required (routes the bundle to outbox/<reviewer>/).")
    session_dir = resolve_session(args.session)
    if not session_dir.is_dir():
        sys.exit(f"ERROR: session folder not found: {session_dir}")
    existing = read_assignment(session_dir)
    if existing and not args.dry_run:
        print(f"NOTE: already out for review (since {existing.get('assigned', '?')}, "
              f"to {existing.get('to', '-')}); re-staging anyway.")
    print(f"[{session_rel(session_dir)}]")
    try:
        stage_one(session_dir, args.exchange, args.dry_run, who)
    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
