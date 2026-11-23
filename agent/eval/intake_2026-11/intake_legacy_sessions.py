"""
intake_legacy_sessions.py -- copy curated legacy (pre-ACORN) sessions from the
lab server into the ACORN data tree so bootstrap_preagent.py can match them.

Decided with the user on 2026-11-05 (memory: project_valence_unintegrated):
28 sessions whose manual curation and movie sit together in a server
"... CNMFed" folder, plus the movie for the already-present bla6 2/6/25
Valence session.

Contract:
  * READ-ONLY on the server.  Every write goes under --dest-root
    (default: DATA_PARENT from local_config).  Nothing is ever deleted.
  * A new session folder must not exist yet (no overwrite, no merge); the
    folder name must not exist anywhere in the area tree, dot folders
    included.  tif_only entries need the folder to exist, hold a
    neuron.mat, and have no movie yet.
  * The curated cell count (C_raw.txt rows) must equal the sheet's count.
  * Each file is copied to <name>.partial, hashed while copying, re-hashed
    from the local copy, and only then renamed to its final name.  The
    movie .tif goes last, after neuron.mat is already in place, so a
    watcher can never see a .tif without neuron.mat.
  * The converted movie .mat next to a source .tif is not copied: the
    bootstrap pre-pass regenerates it from the .tif.
  * Each session gets INTAKE_MANIFEST.txt (source, renames, md5 per file).

Usage:
  python intake_legacy_sessions.py --dry-run
  python intake_legacy_sessions.py [--only <substring>]
"""
import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_DIR))
try:
    from local_config import DATA_PARENT
except Exception:
    DATA_PARENT = None

HERE = Path(__file__).resolve().parent
CHUNK = 8 << 20

V = "valence 4-0 CNMFed"
O = "bla 3odor CNMFed"
O2 = "bla 3odor 2nd cohort CNMFed"
T = "2tones CNMFed"
D = "4odor diff outcomes CNMFed"

# area, task, destination session name, source folder (relative to the server
# root), sheet cell count, mode ("full" = curation + movie, "tif_only" = movie
# into an existing folder; cells is then the existing curation's count).
PLAN = [
    # BLA Valence (original battery)
    ("BLA", "Valence", "AVG5x-TSeries-02052025-bla2-470um-3z-000",
     f"{V}/AVG5x-TSeries-02052025-bla2-470um-3z-000", 24, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-02062025-bla2-523um-3z-000",
     f"{V}/ALT-AVG5x-TSeries-02062025-bla2-523um-3z-000-concatenated", 45, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-02052025-bla3-631um-26z-000",
     f"{V}/AVG5x-TSeries-02052025-bla3b-631um-26z-000", 40, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-02062025-bla3-695um-26z-000",
     f"{V}/AVG5x-TSeries-02062025-bla3-695um-26z-000", 69, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-02062025-bla4-670um-32z-000",
     f"{V}/AVG5x-TSeries-02062025-bla4-um-32z-000", 26, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-02052025-bla6-780um-32z-000",
     f"{V}/TSeries-02052025-bla4-780um-32z-000-concatenated", 18, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-07162025-bla13-665um-25z-000",
     f"{V}/AVG5x-TSeries-07162025-blay13-665um-25z-000", 31, "full"),
    ("BLA", "Valence", "AVG5x-TSeries-02062025-bla6-745um-3z-000",
     f"{V}/Julian_AVG5x-TSeries-02062025-bla6-745um-3z-000", 20, "tif_only"),
    # BLA 3odor
    ("BLA", "3odor", "AVG5x-TSeries-02202025-bla2-710um-27z-000",
     f"{O}/AVG5x-TSeries-02202025-BLA2-710um-27z-000", 20, "full"),
    ("BLA", "3odor", "AVG5x-TSeries-02202025-bla3-655um-26z-000",
     f"{O}/AVG5x-TSeries-02202025-BLA3-655um-26z-000", 53, "full"),
    ("BLA", "3odor", "AVG5x-TSeries-02202025-bla4-750um-28z-000",
     f"{O}/AVG5x-TSeries-02202025-BLA4-750um-28z-000-practice", 25, "full"),
    ("BLA", "3odor", "AVG5x-TSeries-02202025-bla5-523um-25z-000",
     f"{O}/AVG5x-TSeries-02202025-BLA5-523um-25z-000", 62, "full"),
    ("BLA", "3odor", "AVG5x-TSeries-02202025-bla6-730um-3z-000",
     f"{O}/AVG5x-TSeries-02202025-BLA6-730um-3z-000", 15, "full"),
    ("BLA", "3odor", "AVG5x-TSeries-03132025-bla5-653um-25z-000",
     f"{O}/AVG5x-TSeries-03132025-bla5b-653um-25z-000", 60, "full"),
    ("BLA", "3odor", "AVG5x-TSeries-050225-bla13-546um-27z-000",
     f"{O2}/AVG5x-TSeries-050225-bla13-546um-27z-000", 26, "full"),
    # vCA1 2tones
    ("vCA1", "2tones", "AVG2x-TSeries-093025-pnb88-145um-115um-35z-000-A",
     f"{T}/AVG2x-TSeries-093025-pnb88-145um-115um-35z-000-A", 9, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-100125-pnb88-128um-35z-000",
     f"{T}/AVG5x-TSeries-100125-pnb88-128um-35z-000", 36, "full"),
    ("vCA1", "2tones", "AVG2x-TSeries-093025-pnb74-650um-570um-22z-000-A",
     f"{T}/AVG2x-TSeries-093025-pnb74-650um-570um-22z-000-A", 2, "full"),
    ("vCA1", "2tones", "AVG2x-TSeries-093025-pnb74-650um-570um-22z-000-B",
     f"{T}/AVG2x-TSeries-093025-pnb74-650um-570um-22z-000-B", 4, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-100125-pnb74-535um-22z-000",
     f"{T}/AVG5x-TSeries-100125-pnb74-535um-22z-000", 15, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-100925-pnb74-452um-25z-000",
     f"{T}/AVG5x-TSeries-100925-pnb74-452um-25z-000", 17, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-100125-pnb76-531um-3z-000",
     f"{T}/AVG5x-TSeries-100125-pnb76-531um-3z-000", 4, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-100925-pnb76-613um-3z-000",
     f"{T}/AVG5x-TSeries-100925-pnb76-613um-3z-000", 5, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-101025-pnb76-517um-3z-000",
     f"{T}/AVG5x-TSeries-101025-pnb76-517um-3z-000", 6, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-101325-pnb76-521um-3z-000",
     f"{T}/AVG5x-TSeries-101325-pnb76-521um-3z-000", 6, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-101425-pnb76-518um-3z-000",
     f"{T}/AVG5x-TSeries-101425-pnb76-518um-3z-000", 5, "full"),
    ("vCA1", "2tones", "AVG5x-TSeries-101525-pnb76-520um-3z-000",
     f"{T}/AVG5x-TSeries-101525-pnb76-520um-3z-000", 3, "full"),
    # vCA1 6odorDualDiffRew
    ("vCA1", "6odorDualDiffRew", "AVG5x-TSeries-02092026-pnb74-767um-25z-000",
     f"{D}/AVG5x-TSeries-02092026-pnb74-767um-25z-000", 4, "full"),
]

MOVIE_EXT = (".tif", ".tiff")


class PlanError(Exception):
    pass


def c_raw_rows(p: Path) -> int:
    b = p.read_bytes()
    if not b:
        return 0
    return b.count(b"\n") + (0 if b.endswith(b"\n") else 1)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def names_in_area(area_root: Path) -> set:
    """Lower-cased session folder names anywhere in the area tree, dot folders included."""
    out = set()
    if not area_root.is_dir():
        return out
    for p in area_root.rglob("*"):
        if p.is_dir():
            out.add(p.name.lower())
    return out


def build(entry, src_root: Path, dst_root: Path, area_names: dict):
    """Validate one entry; return a list of (src, dst) copy steps, movie last."""
    area, task, dest, src_rel, cells, mode = entry
    src = src_root / src_rel
    dst = dst_root / area / task / dest
    if not src.is_dir():
        raise PlanError(f"source folder missing: {src}")
    if not (dst_root / area / task).is_dir():
        raise PlanError(f"task folder missing (create it by hand): {dst_root / area / task}")
    files = [f for f in src.iterdir() if f.is_file()]
    movies = [f for f in files if f.suffix.lower() in MOVIE_EXT]
    if len(movies) != 1:
        raise PlanError(f"expected exactly one movie in {src}, found {[m.name for m in movies]}")
    movie = movies[0]
    movie_mat = movie.with_suffix(".mat").name.lower()
    steps = []
    if mode == "full":
        if dst.exists():
            raise PlanError(f"destination already exists (no overwrite, no merge): {dst}")
        if dest.lower() in area_names[area]:
            raise PlanError(f"a folder named {dest} already exists somewhere under {area}")
        for need in ("neuron.mat", "C_raw.txt"):
            if not (src / need).exists():
                raise PlanError(f"no {need} in {src}")
        n = c_raw_rows(src / "C_raw.txt")
        if n != cells:
            raise PlanError(f"{src.name}: C_raw has {n} cells, the sheet says {cells}")
        for f in sorted(files):
            low = f.name.lower()
            if f == movie or low == movie_mat or low == ".dir.mat" or f.name.startswith("._"):
                continue
            steps.append((f, dst / f.name))
        for d in sorted(p for p in src.iterdir() if p.is_dir()):
            name = f"{dest}_neurons" if d.name.endswith("_neurons") else d.name
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                if f.name.startswith("._"):
                    continue
                steps.append((f, dst / name / f.relative_to(d)))
    elif mode == "tif_only":
        if not dst.is_dir():
            raise PlanError(f"tif_only needs an existing folder: {dst}")
        if not (dst / "neuron.mat").exists():
            raise PlanError(f"tif_only folder has no neuron.mat: {dst}")
        if [p for p in dst.iterdir() if p.is_file() and p.suffix.lower() in MOVIE_EXT]:
            raise PlanError(f"tif_only folder already has a movie: {dst}")
        if (dst / "INTAKE_MANIFEST.txt").exists():
            raise PlanError(f"already has INTAKE_MANIFEST.txt: {dst}")
        n = c_raw_rows(dst / "C_raw.txt")
        if n != cells:
            raise PlanError(f"{dst.name}: live C_raw has {n} cells, expected {cells}")
    else:
        raise PlanError(f"unknown mode {mode}")
    steps.append((movie, dst / f"{dest}{movie.suffix.lower()}"))
    return src, dst, steps


def copy_verified(src: Path, dst: Path) -> str:
    """Copy src -> dst via dst.partial, hashing the stream and the local copy."""
    if dst.exists():
        raise PlanError(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    h = hashlib.md5()
    with src.open("rb") as fi, tmp.open("xb") as fo:
        for c in iter(lambda: fi.read(CHUNK), b""):
            h.update(c)
            fo.write(c)
    shutil.copystat(str(src), str(tmp))
    got = md5_file(tmp)
    if got != h.hexdigest():
        raise PlanError(f"md5 mismatch after copy, left as {tmp}")
    tmp.rename(dst)
    return got


def run(plan, src_root: Path, dst_root: Path, dry: bool, out_dir: Path, only=None):
    plan = [e for e in plan if not only or only.lower() in (e[2] + " " + e[3]).lower()]
    area_names = {a: names_in_area(dst_root / a) for a in {e[0] for e in plan}}
    built, errors = [], []
    for e in plan:
        try:
            built.append((e, *build(e, src_root, dst_root, area_names)))
        except (PlanError, OSError) as ex:
            errors.append((e, str(ex)))
    total = 0
    for e, src, dst, steps in built:
        size = sum(s.stat().st_size for s, _ in steps)
        total += size
        renamed = "" if Path(e[3]).name == e[2] else f"  (from {Path(e[3]).name})"
        print(f"{e[0]}/{e[1]}/{e[2]}  [{e[5]}, {e[4]} cells, {len(steps)} files, "
              f"{size / 2**30:.2f} GB]{renamed}")
    for e, msg in errors:
        print(f"!! {e[2]}: {msg}")
    print(f"\n{len(built)} ready, {len(errors)} with problems, {total / 2**30:.1f} GB to copy")
    if errors:
        print("Nothing copied: fix the problems first.")
        return 1
    if dry:
        print("Dry run: nothing written.")
        return 0

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sessions = {}
    for e, src, dst, steps in built:
        print(f"\n== {dst}")
        lines = [f"intake: {stamp}", f"by: agent/eval/intake_2026-11/intake_legacy_sessions.py",
                 f"mode: {e[5]}", f"source: {src}", f"sheet_cells: {e[4]}",
                 f"renamed_from: {Path(e[3]).name}", "files:"]
        for s, d in steps:
            digest = copy_verified(s, d)
            lines.append(f"  {digest}  {d.relative_to(dst)}  <-  {s.relative_to(src)}")
            print(f"  ok {d.relative_to(dst)}")
        man = dst / "INTAKE_MANIFEST.txt"
        with man.open("x", encoding="ascii", errors="replace") as fh:
            fh.write("\n".join(lines) + "\n")
        sessions.setdefault(e[0], []).append(str(dst))
    out_dir.mkdir(parents=True, exist_ok=True)
    for area, dirs in sessions.items():
        p = out_dir / f"bootstrap_sessions_{area}.txt"
        mode = "a" if p.exists() else "w"
        with p.open(mode, encoding="ascii") as fh:
            fh.write(f"# intake {stamp}\n" + "\n".join(dirs) + "\n")
        print(f"\nsessions file: {p}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=r"X:\Jeremy")
    ap.add_argument("--dest-root", default=None)
    ap.add_argument("--only", default=None, help="substring of the destination or source name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dst_root = Path(args.dest_root) if args.dest_root else DATA_PARENT
    if dst_root is None:
        sys.exit("ERROR: no --dest-root and DATA_PARENT is unset")
    sys.exit(run(PLAN, Path(args.src_root), Path(dst_root), args.dry_run, HERE, args.only))


if __name__ == "__main__":
    main()
