"""Put re-made AVG5x movies in place of the delivered ones in a local task folder.

    python swap_in_remade_movies.py --new-dir <task dir of the re-make run>
        --task-dir "D:/2p data/3odor" --archive .prev_maxregshift01_2026-11-22
        [--go] NAME [NAME ...]

NAME is an AVG5x tif file name (AVG5x-TSeries-...-000.tif). For each NAME:
  - <task-dir>/NAME (the delivered movie) and <new-dir>/NAME (the re-made one) must
    exist, the re-made run's QC row must be unflagged, and nothing may already sit
    at <task-dir>/<archive>/NAME, <task-dir>/NAME.part or the record destination;
  - the new movie is copied to <task-dir>/NAME.part and compared with its source
    (SHA-256);
  - then the delivered movie is MOVED (renamed, same volume) to
    <task-dir>/<archive>/NAME and NAME.part is renamed to NAME;
  - the shift record <new-dir>/mc_record/<stem>_mc.npz is copied to
    <task-dir>/mc_record/ and the session's row of <new-dir>/mc_qc.csv is appended to
    <task-dir>/mc_qc.csv, as process_2p.py would have written them there;
  - one paragraph per movie goes into <task-dir>/<archive>/README.txt (sizes,
    SHA-256 of both versions, dates).
Never overwrites, never deletes. Local folders only; refuses a UNC or X: path.
Dry run by default; --go acts.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

PREFIX = "AVG5x-"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def qc_row(csv_path, session):
    with open(csv_path, newline="", encoding="ascii") as fh:
        rows = [r for r in csv.DictReader(fh) if r["session"] == session]
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--archive", required=True,
                    help="subfolder of the task dir for the delivered movies")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    task = a.task_dir.resolve()
    new_dir = a.new_dir.resolve()
    for p in (task, new_dir):
        s = str(p)
        if s.startswith("\\\\") or s[:2].upper() == "X:":
            sys.exit(f"refusing a server path: {p}")
        if not p.is_dir():
            sys.exit(f"not a folder: {p}")
    if not a.archive or "/" in a.archive or "\\" in a.archive or a.archive in (".", ".."):
        sys.exit(f"archive must be one plain folder name: {a.archive!r}")
    arch_dir = task / a.archive
    new_csv = new_dir / "mc_qc.csv"
    if not new_csv.is_file():
        sys.exit(f"no QC file in the re-make folder: {new_csv}")

    todo, refused = [], []
    for name in dict.fromkeys(a.names):
        stem = name[:-4] if name.lower().endswith(".tif") else name
        name = stem + ".tif"
        session = stem[len(PREFIX):] if stem.startswith(PREFIX) else None
        old, new = task / name, new_dir / name
        part, arch = task / (name + ".part"), arch_dir / name
        rec_src = new_dir / "mc_record" / f"{stem}_mc.npz"
        rec_dst = task / "mc_record" / rec_src.name
        row = qc_row(new_csv, session) if session else None
        why = None
        if session is None:
            why = f"not an {PREFIX} name"
        elif not old.is_file():
            why = f"no delivered movie {old}"
        elif not new.is_file() or new.stat().st_size == 0:
            why = f"no re-made movie {new}"
        elif arch.exists():
            why = f"already archived: {arch}"
        elif part.exists():
            why = f"leftover {part}; look at it first"
        elif not rec_src.is_file():
            why = f"no shift record {rec_src}"
        elif rec_dst.exists():
            why = f"record already in place: {rec_dst}"
        elif row is None:
            why = f"no QC row for {session} in {new_csv}"
        elif row["flag"]:
            why = f"QC flagged: {row['flag']}"
        if why:
            refused.append((name, why))
        else:
            todo.append((name, session, old, new, part, arch, rec_src, rec_dst, row))

    print(f"{'GO' if a.go else 'DRY RUN'}: {len(todo)} movie(s); task {task}; archive {arch_dir}")
    for name, session, old, new, part, arch, rec_src, rec_dst, row in todo:
        print(f"  swap {name}")
        print(f"    delivered {old.stat().st_size} bytes -> {arch}")
        print(f"    re-made   {new.stat().st_size} bytes -> {old} (via .part)")
        print(f"    record    {rec_src.name} -> {rec_dst.parent}")
        print(f"    QC row    bad {row['bad_pct']}%  degraded {row['degraded_pct']}%  "
              f"at limit {row['at_limit_pct']}%  ref offset ({row['ref_offset_y']}, {row['ref_offset_x']})")
    for name, why in refused:
        print(f"  REFUSED {name}: {why}")
    if refused:
        sys.exit(f"{len(refused)} refused; nothing done.")
    if not a.go:
        return

    with open(new_csv, newline="", encoding="ascii") as fh:
        fields = csv.DictReader(fh).fieldnames
    arch_dir.mkdir(exist_ok=True)
    (task / "mc_record").mkdir(exist_ok=True)
    for name, session, old, new, part, arch, rec_src, rec_dst, row in todo:
        t0 = time.time()
        shutil.copy2(str(new), str(part))
        h_new = sha256(new)
        if sha256(part) != h_new:
            sys.exit(f"!! copy of {name} does not match its source; left as {part}, nothing moved")
        h_old = sha256(old)
        old_stat = old.stat()
        if arch.exists():
            sys.exit(f"!! {arch} appeared during the run; nothing moved")
        os.rename(old, arch)
        os.rename(part, old)
        if rec_dst.exists():
            sys.exit(f"!! {rec_dst} appeared during the run; movie swapped, record not copied")
        shutil.copy2(str(rec_src), str(rec_dst))
        qc_dst = task / "mc_qc.csv"
        first = not qc_dst.exists()
        with open(qc_dst, "a", newline="", encoding="ascii") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if first:
                w.writeheader()
            w.writerow(row)
        with open(arch_dir / "README.txt", "a", encoding="ascii") as fh:
            fh.write(
                f"{name}\n"
                f"  archived here {time.strftime('%Y-%m-%d %H:%M')}: the movie delivered "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(old_stat.st_mtime))} "
                f"(suite2p maxregshift 0.1, suite2p's own reference), "
                f"{old_stat.st_size} bytes, SHA-256 {h_old}\n"
                f"  replaced in {task} by the re-made movie (installed rule: maxregshift 0.2 + "
                f"recentred reference), {new.stat().st_size} bytes, SHA-256 {h_new}; "
                f"QC bad {row['bad_pct']}% degraded {row['degraded_pct']}% "
                f"at limit {row['at_limit_pct']}%\n"
                f"  shift record: {rec_dst}\n\n")
        print(f"  swapped {name} in {time.time() - t0:.0f} s (old SHA-256 {h_old[:12]}..., "
              f"new {h_new[:12]}...)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
