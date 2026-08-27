"""
plan_fov_assignment.py
Assign review sessions to reviewers so that each reviewer owns WHOLE fields of
view, then report what has to move and what an operator must delete.

Why FOV-whole matters: re-imaging one FOV across days is tracking, not extra
yield -- those sessions hold largely the same neurons.  Splitting a FOV across
two reviewers means the same physical cells are judged by two different people
on different days, and that inconsistency is baked into the training labels.

FOV is not a column.  For the pnb (retro-tdTomato) cohort the ROI column is
empty and the field is free text in Notes, e.g. "d24. FOVb. d3 of extinction",
"original FOVA", "new fovc", "FOVc (same as 6/19)".  Resolution rules, in order:

  1. A letter named in the row's own notes wins ("FOVb", "new fovc").
  2. "new FOV" with no letter starts its own group.  It is never merged with
     EARLIER days -- that is the whole point of the phrase -- but a later row
     may name it retroactively.
  3. Explicit back-references resolve the row they point at:
     "FOVc (same as 6/19)" names 6/19; "same FOV as D6 (FOVa)" names D6.
  4. Unnamed rows with no boundary between them and a resolved row inherit it
     (the field only changes when someone says it changed).
  5. FOV letters are TASK-SCOPED.  pnb98 FOVa in 6odor (~505um) and in 3odor
     (~465um) are different fields, so the group key is (animal, task, fov).
  6. CTA's two sessions are one field by design (operator statement, 2026-08-18).

SAFETY: this script never deletes anything, locally or on the server, and never
writes to the server itself.  --apply shells out to push_review_bundle.py, which
is copy-only.  Old bundles left behind under the wrong reviewer are printed as a
manual delete list -- clearing the lab server is a human action.

Usage:
    python plan_fov_assignment.py                  # plan only (read-only)
    python plan_fov_assignment.py --area vCA1 --animal pnb97,pnb98
    python plan_fov_assignment.py --apply          # stage the moves
"""
import argparse
import csv
import itertools
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT
import push_review_bundle as P
from review_prep import read_assignment

# Imaging sheets for the retro cohort, keyed by the CNMFe task folder name.
SHEETS = {
    "3odor":            r"D:\Analysis\3odor\2p imaging parameters 4.0 (retro) - 3odor.csv",
    "6odorDualDiffRew": r"D:\Analysis\6odor dual DR\2p imaging parameters 4.0 (retro) - 4_6odor DR.csv",
    "CTA":              r"D:\Analysis\CTA\2p imaging parameters 4.0 (retro) - CTA.csv",
}

# Tasks whose sessions are one field by design, regardless of what Notes say.
SINGLE_FOV_TASKS = {"CTA"}

NAMED_RE   = re.compile(r"\bfov\s*([a-d])\b", re.IGNORECASE)
NEW_ANY_RE = re.compile(r"\bnew\s+fov", re.IGNORECASE)
# "FOVc (same as 6/19)"  /  "same FOV as D6 (FOVa, best)"
REF_DATE_RE = re.compile(r"same\s+(?:as|fov\s+as)\s+(\d{1,2}/\d{1,2})", re.IGNORECASE)
REF_DAY_RE  = re.compile(r"same\s+fov\s+as\s+d(?:ay)?\s*(\d+)", re.IGNORECASE)
DAY_RE      = re.compile(r"^\s*(?:day|d)\s*(\d+)", re.IGNORECASE)

SESSION_RE = re.compile(r"-(\d{6}|\d{8})-([a-z]+\d+)-(\d+)um", re.IGNORECASE)


def parse_date(s):
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def parse_session_name(name):
    """(animal, date, depth) from AVG5x-TSeries-<MMDDYY>-<animal>-<depth>um-..."""
    m = SESSION_RE.search(name)
    if not m:
        return None
    for fmt in ("%m%d%y", "%m%d%Y"):
        try:
            return m.group(2).lower(), datetime.strptime(m.group(1), fmt).date(), int(m.group(3))
        except ValueError:
            pass
    return None


def load_sheet(task, path, animals):
    """Ordered rows for the animals of interest: [(animal, date, day, notes)]."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    hdr = rows[0]
    idx = {n.strip(): i for i, n in enumerate(hdr)}
    if "Notes" not in idx:
        return []
    out = []
    for r in rows[1:]:
        if len(r) < len(hdr):
            r = r + [""] * (len(hdr) - len(r))
        animal = r[idx["Animal ID"]].strip().lower()
        if animal not in animals:
            continue
        d = parse_date(r[idx["Date"]])
        if d is None:
            continue
        notes = r[idx["Notes"]].strip()
        dm = DAY_RE.match(notes)
        day = int(dm.group(1)) if dm else None
        out.append({"animal": animal, "date": d, "day": day, "notes": notes,
                    "fov": None, "why": ""})
    out.sort(key=lambda x: (x["animal"], x["date"]))
    return out


def resolve_fovs(task, rows):
    """Fill row['fov'] per the rules in the module docstring."""
    by_animal = defaultdict(list)
    for r in rows:
        by_animal[r["animal"]].append(r)

    for animal, seq in by_animal.items():
        # -- pass 1: names in the row's own notes, and boundaries --
        for i, r in enumerate(seq):
            notes = r["notes"]
            # strip a trailing back-reference so its letter is not read as this
            # row's own name when the row itself is otherwise unnamed
            own = re.sub(r"\(([^)]*same[^)]*)\)", " ", notes, flags=re.IGNORECASE)
            m = NAMED_RE.search(own)
            if m:
                r["fov"] = m.group(1).lower()
                r["why"] = "named in notes"
            elif NEW_ANY_RE.search(notes):
                r["fov"] = f"new@{r['date']:%m%d}"
                r["why"] = "unnamed 'new FOV' - own group"
            r["is_boundary"] = bool(NEW_ANY_RE.search(notes))

        # -- pass 2: explicit back-references name an earlier row --
        for r in seq:
            m = NAMED_RE.search(r["notes"])
            if not m:
                continue
            letter = m.group(1).lower()
            ref = REF_DATE_RE.search(r["notes"])
            if ref:
                mm, dd = ref.group(1).split("/")
                for t in seq:
                    if t["date"].month == int(mm) and t["date"].day == int(dd):
                        t["fov"] = letter
                        t["why"] = f"named retroactively by {r['date']:%m/%d}"
            refd = REF_DAY_RE.search(r["notes"])
            if refd:
                want = int(refd.group(1))
                for t in seq:
                    if t["day"] == want:
                        t["fov"] = letter
                        t["why"] = f"named retroactively by {r['date']:%m/%d} (D{want})"

        # -- pass 3: unnamed rows inherit across a stretch with no boundary --
        for i, r in enumerate(seq):
            if r["fov"]:
                continue
            for j in range(i + 1, len(seq)):          # look forward
                if seq[j].get("is_boundary"):
                    break
                if seq[j]["fov"]:
                    r["fov"] = seq[j]["fov"]
                    r["why"] = f"inherited from {seq[j]['date']:%m/%d} (no FOV change between)"
                    break
            if r["fov"]:
                continue
            # Then backward. A boundary row always carries a FOV from pass 1, so
            # the first hit going back is either that new FOV (correct: rows after
            # a "new FOV" belong to it) or the field in force before any change.
            for j in range(i - 1, -1, -1):
                if seq[j]["fov"]:
                    r["fov"] = seq[j]["fov"]
                    r["why"] = f"carried forward from {seq[j]['date']:%m/%d}"
                    break

    if task in SINGLE_FOV_TASKS:
        for r in rows:
            r["fov"] = "single"
            r["why"] = "task recorded at one FOV by design"
    return rows


def build_groups(area, animals):
    sessions = [s for s in P.find_awaiting(DATA_PARENT, area, None)
                if P.animal_of(s.name) in animals]
    lookup = {}
    for task, path in SHEETS.items():
        rows = resolve_fovs(task, load_sheet(task, path, animals))
        for r in rows:
            lookup[(task, r["animal"], r["date"])] = r

    groups = defaultdict(list)
    unresolved = []
    for sd in sorted(sessions):
        parsed = parse_session_name(sd.name)
        who = (read_assignment(sd) or {}).get("to")
        task = sd.parent.name
        if parsed is None:
            unresolved.append((sd, who, "session name unparsed"))
            continue
        animal, date, depth = parsed
        row = lookup.get((task, animal, date))
        if row is None or not row["fov"]:
            unresolved.append((sd, who, "no sheet row / FOV unresolved"))
            groups[(animal, task, f"?{sd.name}")].append((sd, who, depth, "UNRESOLVED"))
            continue
        groups[(animal, task, row["fov"])].append((sd, who, depth, row["why"]))
    return groups, unresolved


def choose_assignment(groups, reviewers):
    """Give whole groups to reviewers: balance session counts, then minimise moves."""
    keys = sorted(groups)
    best = None
    for combo in itertools.product(range(len(reviewers)), repeat=len(keys)):
        counts = [0] * len(reviewers)
        moves = 0
        for gi, ri in enumerate(combo):
            members = groups[keys[gi]]
            counts[ri] += len(members)
            moves += sum(1 for _, who, _, _ in members if who != reviewers[ri])
        spread = max(counts) - min(counts)
        score = (spread, moves, combo)
        if best is None or score < best[0]:
            best = (score, combo, counts)
    _, combo, counts = best
    return {keys[gi]: reviewers[ri] for gi, ri in enumerate(combo)}, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="vCA1")
    ap.add_argument("--animal", default="pnb97,pnb98")
    ap.add_argument("--reviewers", default="Taylor,Aneesh")
    ap.add_argument("--apply", action="store_true",
                    help="stage the moves via push_review_bundle.py (copies only)")
    args = ap.parse_args()

    animals = {a.strip().lower() for a in args.animal.split(",") if a.strip()}
    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]

    groups, unresolved = build_groups(args.area, animals)
    plan, counts = choose_assignment(groups, reviewers)

    print("=" * 92)
    print("FOV GROUPS  (a group is one field; it goes to exactly one reviewer)")
    print("=" * 92)
    for key in sorted(plan, key=lambda k: (k[0], k[1], str(k[2]))):
        animal, task, fov = key
        members = groups[key]
        depths = ", ".join(str(d) for _, _, d, _ in sorted(members, key=lambda m: m[2]))
        why = members[0][3]
        print(f"  {animal} / {task} / FOV {fov}   -> {plan[key]}   "
              f"({len(members)} sessions; {depths} um)")
        print(f"        basis: {why}")

    print()
    print("resulting load: " + ", ".join(f"{r}={c}" for r, c in zip(reviewers, counts)))

    moves = []
    for key, who in plan.items():
        for sd, cur, depth, _ in groups[key]:
            if cur != who:
                moves.append((sd, cur, who))

    print()
    print("=" * 92)
    print(f"MOVES  ({len(moves)} session(s) change reviewer)")
    print("=" * 92)
    total = 0
    for sd, cur, who in sorted(moves, key=lambda m: (m[2], m[0].name)):
        vid = sd / f"{sd.name}.mat"
        gb = vid.stat().st_size / 1e9 if vid.exists() else 0
        total += gb
        print(f"  {str(cur):<8} -> {who:<8} {gb:>5.1f} GB  {sd.parent.name}/{sd.name}")
    print(f"\n  re-copy volume: {total:.0f} GB")

    if unresolved:
        print()
        print("=" * 92)
        print(f"UNRESOLVED  ({len(unresolved)}) - each treated as its own group; "
              f"confirm before trusting")
        print("=" * 92)
        for sd, who, reason in unresolved:
            print(f"  [{who}] {sd.parent.name}/{sd.name}  - {reason}")

    # Stale server copies: re-staging copies, it never removes the old folder.
    outbox = Path(EXCHANGE_ROOT) / "outbox" if EXCHANGE_ROOT else None
    if outbox and outbox.exists() and moves:
        print()
        print("=" * 92)
        print("OPERATOR MUST DELETE BY HAND (re-staging does not remove the old copy)")
        print("=" * 92)
        for sd, cur, who in sorted(moves, key=lambda m: m[0].name):
            old = outbox / cur / P.session_rel(sd)
            if old.exists():
                print(f"  {old}")
        print("\n  This script deletes nothing. Removing files from the lab server is manual.")

    if not args.apply:
        print("\n(plan only - nothing staged. re-run with --apply to copy the moves.)")
        return

    print()
    print("=" * 92)
    print("APPLYING")
    print("=" * 92)
    failed = []
    for sd, cur, who in moves:
        rel = P.session_rel(sd)
        cmd = [sys.executable, str(AGENT_DIR / "push_review_bundle.py"),
               str(rel), "--assignee", who]
        print(f"\n$ {' '.join(cmd[1:])}")
        rc = subprocess.run(cmd, cwd=str(AGENT_DIR)).returncode
        if rc != 0:
            failed.append((rel, rc))
    print()
    if failed:
        print(f"{len(failed)} move(s) FAILED:")
        for rel, rc in failed:
            print(f"  {rel}  (exit {rc})")
    else:
        print(f"all {len(moves)} move(s) staged.")
    print("Old copies under the previous reviewer are still on the server - "
          "delete them by hand using the list above.")


if __name__ == "__main__":
    main()
