"""
push_DG_AL_batch.py -- stage every awaiting DG_AL session to its assigned
reviewer in one command, using the assignment table from Jeremy's sheet
(confirmed 2026-08-16): Taylor = DG6D both planes, Alisia = DG6E plane A,
Aneesh = DG6E plane B.

Assignment is PER PLANE (the trailing A/B in the session name is the z-plane),
which push_review_bundle's --all --animal cannot express -- and its animal
regex cannot parse DG6D anyway -- so this script turns the twelve-command
batch into one.

Safety:
  - Sessions already out for review (review_assigned.txt) are skipped unless
    --force is given, exactly like push_review_bundle --all.
  - A session whose processed gSig (read from its agent_run.log) disagrees
    with the animal's current animal_params.json entry is SKIPPED LOUDLY: it
    was processed at the wrong parameters (first case: the 2026-08-18
    non-averaged TSeries batch, which ran at fallback gSig=9/36 because the
    positional animal parse broke on the missing AVGNx prefix) and pushing it
    would ship exactly the silent under-detection the calibrated parameters
    exist to prevent.  There is deliberately no flag to override this guard:
    re-run the session at the right parameters instead.
  - This script only reads local files and shells out to push_review_bundle.py,
    which is copy-only.  Nothing is deleted, locally or on the server.

Usage:
    python push_DG_AL_batch.py --plan       # print the mapping; touch nothing
    python push_DG_AL_batch.py --dry-run    # forward --dry-run to push script
    python push_DG_AL_batch.py              # stage for real (operator runs this)
    python push_DG_AL_batch.py --force      # re-stage even if already assigned
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT
import push_review_bundle as prb
from review_prep import read_assignment

AREA = "DG_AL"

# Per-plane reviewer assignment (Jeremy's sheet, "DG grp3" tab, 2026-08-16).
# Keys: (animal, plane).  Extend here when new DG animals arrive.
ASSIGNMENTS = {
    ("DG6D", "A"): "Taylor",
    ("DG6D", "B"): "Taylor",
    ("DG6E", "A"): "Alisia",
    ("DG6E", "B"): "Aneesh",
}

_PLANE_RE = re.compile(r"-\d+([A-Z])$")


def animal_and_plane(session_name: str):
    animal = next((seg for seg in session_name.split("-")
                   if seg in {a for a, _ in ASSIGNMENTS}), None)
    m = _PLANE_RE.search(session_name)
    plane = m.group(1) if m else None
    return animal, plane


def processed_gsig(session_dir: Path):
    """The gSig the session was actually processed with, from its run log."""
    log_file = session_dir / "agent_run.log"
    if not log_file.exists():
        return None
    hits = re.findall(r"'gSig': (\d+)", log_file.read_text(encoding="utf-8",
                                                           errors="replace"))
    return int(hits[-1]) if hits else None


def expected_gsig(animal: str):
    cfg_file = AGENT_DIR / "config" / "animal_params.json"
    try:
        entry = json.loads(cfg_file.read_text(encoding="utf-8")).get(animal)
        return int(entry["gSig"]) if entry else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true",
                    help="print the assignment plan and exit; touches nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="forward --dry-run to push_review_bundle.py")
    ap.add_argument("--force", action="store_true",
                    help="include sessions already marked out for review")
    args = ap.parse_args()

    to_push, skipped = [], []
    for sd in prb.find_awaiting(DATA_PARENT, area=AREA):
        name = sd.name
        animal, plane = animal_and_plane(name)
        who = ASSIGNMENTS.get((animal, plane))
        if who is None:
            skipped.append((name, f"no assignment for animal={animal} plane={plane} "
                                  f"-- extend ASSIGNMENTS in this script"))
            continue
        if not args.force and read_assignment(sd) is not None:
            skipped.append((name, "already out for review (--force to re-stage)"))
            continue
        got, want = processed_gsig(sd), expected_gsig(animal)
        if want is not None and got is not None and got != want:
            skipped.append((name, f"PARAM MISMATCH: processed at gSig={got}, "
                                  f"{animal} is calibrated at gSig={want} -- re-run "
                                  f"this session; pushing it would ship silent "
                                  f"under-detection"))
            continue
        to_push.append((sd, who, got))

    print(f"DG_AL push plan ({len(to_push)} to stage, {len(skipped)} skipped):\n")
    for sd, who, got in to_push:
        print(f"  {sd.name}  ->  {who}  (gSig={got})")
    if skipped:
        print()
        for name, why in skipped:
            print(f"  SKIP {name}\n       {why}")
    if args.plan or not to_push:
        if not to_push and not args.plan:
            print("\nNothing to stage.")
        return

    print()
    failures = []
    for sd, who, _ in to_push:
        rel = str(prb.session_rel(sd))
        cmd = [sys.executable, str(AGENT_DIR / "push_review_bundle.py"),
               rel, "--assignee", who]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"[{rel}] -> {who}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failures.append((rel, r.returncode))
        print()
    if failures:
        print(f"{len(failures)} push(es) FAILED:")
        for rel, rc in failures:
            print(f"  {rel} (exit {rc})")
        sys.exit(1)
    verb = "would stage" if args.dry_run else "staged"
    print(f"Done: {verb} {len(to_push)} session(s).")


if __name__ == "__main__":
    main()
