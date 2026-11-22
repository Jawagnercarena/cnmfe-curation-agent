"""
stage_3odor_2026-11-21.py -- copy named 3odor tifs of the six cohort-5 animals
from the local staging folder into the vCA1 3odor task folder for the watcher.

Local to local, copy-only.  Each tif is copied to <dest>/<name>.tif.part, the
copy is compared with the source (size, then SHA-256 unless --no-verify), and
only when every requested file has passed are the .part files renamed to .tif.
The watcher matches *.tif / *.tiff only, so it can never start on a partial
file, and it sees the whole set at once instead of starting a run while the
remaining copies are still loading the disk.  Nothing is deleted or moved on
the source side; nothing existing is overwritten on the destination side.

Sessions are named explicitly on the command line (session name, with or
without the AVG5x-TSeries- prefix and the .tif suffix).  There is no "all".

Refused:
  - a session in NOT_RUN (superseded by a redo after the lens cleaning; user
    decision 2026-11-21);
  - a session whose folder or tif already exists in any task folder of the area;
  - a session with a leftover .part in the destination (look at it first);
  - with --expect-min-corr X: a session whose min_corr in
    agent/config/param_overrides.json does not resolve to X.  Use it for the
    labelled dim run, where a run at the wrong seed wastes the experiment.

Dry run by default; --go copies.  --src / --dest exist so the path logic can be
exercised against a scratch directory first.
"""
import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

AGENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))
from local_config import DATA_PARENT   # noqa: E402
import params                          # noqa: E402

SRC_DEFAULT = Path("D:/2p data/3odor")
DEST_DEFAULT = DATA_PARENT / "vCA1" / "3odor"
PREFIX = "AVG5x-TSeries-"
ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
NOT_RUN = {
    "AVG5x-TSeries-092926-pnb105-430um-35z-000": "redone as 100226-pnb105-373um",
    "AVG5x-TSeries-092926-pnb109-133um-25z-001": "redone as 100126-pnb109-440um",
    "AVG5x-TSeries-093026-pnb109-90um-25z-000": "redone as 100226-pnb109-310um",
}


def stem_of(arg):
    s = Path(arg).name
    for suf in (".tif", ".tiff"):
        if s.lower().endswith(suf):
            s = s[: -len(suf)]
    if not s.startswith(PREFIX):
        s = PREFIX + s
    return s


def resolved_min_corr(stem):
    """min_corr that param_overrides.json gives this session (None = no entry)."""
    animal = params.parse_animal_id(stem)
    entry = params.load_param_overrides().get(animal, {}) if animal else {}
    val = entry.get("min_corr")
    sess = entry.get("sessions", {}).get(stem, {})
    if "min_corr" in sess:
        return sess["min_corr"], "session"
    return val, "animal"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="session names to stage")
    ap.add_argument("--go", action="store_true", help="copy (default: dry run)")
    ap.add_argument("--expect-min-corr", type=float, default=None,
                    help="refuse any session whose override min_corr is not exactly this")
    ap.add_argument("--no-verify", action="store_true", help="compare sizes only, skip the SHA-256")
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    a = ap.parse_args()

    if not a.src.is_dir():
        sys.exit(f"source folder missing: {a.src}")
    if not a.dest.is_dir():
        sys.exit(f"task folder missing (not created here): {a.dest}")
    area = a.dest.parent

    todo, refused = [], []
    seen = set()
    for arg in a.sessions:
        s = stem_of(arg)
        if s in seen:
            continue
        seen.add(s)
        src = a.src / (s + ".tif")
        mc, level = resolved_min_corr(s)
        if not any(f"-{an}-" in s for an in ANIMALS):
            refused.append((s, "not one of the six cohort-5 animals"))
        elif s in NOT_RUN:
            refused.append((s, f"not to be run: {NOT_RUN[s]}"))
        elif not src.is_file():
            refused.append((s, f"no such tif in {a.src}"))
        elif any(area.glob(f"*/{s}")) or any(area.glob(f"*/{s}.tif")) or any(area.glob(f"*/{s}.tiff")):
            refused.append((s, f"already present under {area}"))
        elif (a.dest / (s + ".tif.part")).exists():
            refused.append((s, "leftover .tif.part in the destination; look at it first"))
        elif a.expect_min_corr is not None and mc != a.expect_min_corr:
            refused.append((s, f"override min_corr is {mc} ({level} level), expected {a.expect_min_corr}"))
        else:
            todo.append((s, src, mc, level))

    total = sum(src.stat().st_size for _, src, _, _ in todo)
    print(f"{'GO' if a.go else 'DRY RUN'}: {len(todo)} of {len(seen)} session(s), "
          f"{total / 1e9:.2f} GB, {a.src} -> {a.dest}")
    for s, src, mc, level in todo:
        print(f"  stage   {s}  {src.stat().st_size / 1e9:.2f} GB  min_corr {mc} ({level} level)")
    for s, why in refused:
        print(f"  REFUSED {s}: {why}")
    if refused:
        print(f"{len(refused)} session(s) refused; nothing is copied until the list is clean.")
        sys.exit(2)
    if not a.go:
        return

    t0 = time.time()
    parts = []
    for i, (s, src, _, _) in enumerate(todo, 1):
        part = a.dest / (s + ".tif.part")
        t1 = time.time()
        shutil.copy2(str(src), str(part))
        ok = part.stat().st_size == src.stat().st_size
        how = "size"
        if ok and not a.no_verify:
            ok = sha256(part) == sha256(src)
            how = "sha256"
        if not ok:
            print(f"  !! copy of {s} does not match its source ({how}); left as .part, nothing renamed",
                  flush=True)
            sys.exit(3)
        parts.append((part, a.dest / (s + ".tif")))
        print(f"  [{i}/{len(todo)}] copied + verified ({how}) {s}  "
              f"{src.stat().st_size / 1e9:.2f} GB in {(time.time() - t1) / 60:.1f} min", flush=True)
    for part, final in parts:
        part.rename(final)
        print(f"  ready   {final.name}", flush=True)
    print(f"done: {len(parts)} tif(s) staged in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
