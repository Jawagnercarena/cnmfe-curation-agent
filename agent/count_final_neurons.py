"""
count_final_neurons.py - report how many neurons were finally extracted from each
finalized CNMFe session under a folder.

A session is "finalized" once run_final_review.m has written its outputs. The final
extracted set is the KEPT neurons; MATLAB overwrites the session's files with just
that set, so the count is simply:

  - C_raw.txt   -> one row per final neuron  (the authoritative count; see
                   CNMFe_final_save.m -> dlmwrite('C_raw.txt', full(neuron.C_raw)))
  - labels.mat  -> keep / delete / motion breakdown (cross-check + motion tally)

Usage:
  python count_final_neurons.py                    # sweep DATA_PARENT (all areas)
  python count_final_neurons.py <folder>           # sweep one folder (recursively)
  python count_final_neurons.py <folder> --csv out.csv   # also write a CSV table

Read-only: it never writes into the data tree. The only file it may create is the
CSV you name with --csv.
"""
import argparse
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
try:
    from local_config import DATA_PARENT
except Exception:
    DATA_PARENT = None


def count_rows(c_raw: Path):
    """Number of rows (= final neurons) in C_raw.txt, counted from newlines.
    Robust to a missing trailing newline. Returns None if unreadable."""
    try:
        data = c_raw.read_bytes()
    except Exception:
        return None
    if not data:
        return 0
    n = data.count(b"\n")
    if not data.endswith(b"\n"):
        n += 1
    return n


def label_breakdown(labels_mat: Path):
    """(n_keep, n_delete, n_motion) from labels.mat, or None if unreadable.
    n_motion is None when the file predates the (m) motion-delete option."""
    try:
        import scipy.io as sio
        d = sio.loadmat(str(labels_mat))
        labels = d["labels"].flatten()
        n_keep = int((labels == 1).sum())
        n_delete = int((labels == 0).sum())
        n_motion = int(d["motion_delete"].flatten().sum()) if "motion_delete" in d else None
        return n_keep, n_delete, n_motion
    except Exception:
        return None


def find_sessions(root: Path):
    """Every finalized session dir under root (has C_raw.txt or labels.mat)."""
    seen = set()
    for name in ("C_raw.txt", "labels.mat"):
        for hit in root.rglob(name):
            sd = hit.parent
            if sd not in seen:
                seen.add(sd)
    return sorted(seen)


def rel_label(session_dir: Path, root: Path) -> str:
    try:
        return str(session_dir.relative_to(root))
    except ValueError:
        return session_dir.name


def main():
    ap = argparse.ArgumentParser(description="Count finally-extracted neurons per session.")
    ap.add_argument("root", nargs="?", default=None,
                    help="folder to sweep recursively (default: DATA_PARENT, all areas)")
    ap.add_argument("--csv", help="also write a CSV table to this path")
    args = ap.parse_args()

    root = Path(args.root) if args.root else DATA_PARENT
    if root is None:
        sys.exit("ERROR: no folder given and DATA_PARENT is unset. Pass a folder to sweep.")
    root = root.resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: not a folder: {root}")

    sessions = find_sessions(root)
    if not sessions:
        print(f"No finalized sessions found under {root}")
        return

    rows = []          # (label, final, keep, delete, motion, flag)
    total_final = 0
    total_motion = 0
    motion_sessions = 0
    print(f"Finalized sessions under {root}:\n")
    header = f"{'final':>6}  {'keep':>5}  {'del':>5}  {'motion':>6}   session"
    print(header)
    print("-" * len(header))
    for sd in sessions:
        c_raw = sd / "C_raw.txt"
        final = count_rows(c_raw) if c_raw.exists() else None
        brk = label_breakdown(sd / "labels.mat") if (sd / "labels.mat").exists() else None
        keep = deleted = motion = None
        if brk is not None:
            keep, deleted, motion = brk
        # authoritative final count: C_raw rows; fall back to labels keep count.
        count = final if final is not None else keep
        flag = ""
        if final is not None and keep is not None and final != keep:
            flag = f"  (!) C_raw={final} != labels keep={keep}"
        if count is not None:
            total_final += count
        m_disp = "-" if motion is None else str(motion)
        if motion:
            total_motion += motion
            motion_sessions += 1
        print(f"{'-' if count is None else count:>6}  "
              f"{'-' if keep is None else keep:>5}  "
              f"{'-' if deleted is None else deleted:>5}  "
              f"{m_disp:>6}   {rel_label(sd, root)}{flag}")
        rows.append((rel_label(sd, root),
                     "" if count is None else count,
                     "" if keep is None else keep,
                     "" if deleted is None else deleted,
                     "" if motion is None else motion))

    print("-" * len(header))
    print(f"{len(sessions)} session(s), {total_final} neurons finally extracted "
          f"in total; {total_motion} motion tag(s) across {motion_sessions} session(s).")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "final_neurons", "keep", "delete", "motion"])
            w.writerows(rows)
        print(f"\nCSV written: {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()
