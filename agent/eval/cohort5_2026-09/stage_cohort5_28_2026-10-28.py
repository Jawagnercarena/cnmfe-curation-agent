"""
stage_cohort5_28_2026-10-28.py -- copy the 28 remaining cohort-5 6odorDualDiffRew
tifs from the server (READ ONLY) into the vCA1 task folder for the watcher.

Copy-only: shutil.copy2 from X:/Jeremy/for CNMFe/6odorDualDiffRew/<session>.tif to
DATA_PARENT/vCA1/6odorDualDiffRew/<session>.tif.part, renamed to .tif when the copy
is complete, so the watcher (which only matches *.tif / *.tiff and waits for a
stable size) can never start on a partial file.  Skips a session whose folder or
tif already exists locally, and the 450-frame stub 082726-pnb99-440um (PrairieView
crashed mid-recording per the imaging sheet).  Nothing is deleted or moved on
either side.  Dry run by default; --go copies.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

AGENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT))
from local_config import DATA_PARENT   # noqa: E402

NAS = Path("X:/Jeremy/for CNMFe/6odorDualDiffRew")
TASK = DATA_PARENT / "vCA1" / "6odorDualDiffRew"
ANIMALS = ("pnb99", "pnb102", "pnb104", "pnb105", "pnb106", "pnb109")
SKIP = {"AVG5x-TSeries-082726-pnb99-440um-3z-000"}   # 450-frame stub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="copy (default: dry run)")
    a = ap.parse_args()
    if not NAS.is_dir():
        sys.exit(f"server folder not reachable: {NAS}")
    if not TASK.is_dir():
        sys.exit(f"task folder missing: {TASK}")
    todo = []
    for f in sorted(NAS.glob("*.tif")):
        s = f.stem
        if not any(f"-{an}-" in s for an in ANIMALS):
            continue
        if s in SKIP:
            print(f"skip (stub)     {s}")
            continue
        if (TASK / s).exists() or (TASK / f.name).exists():
            print(f"skip (local)    {s}")
            continue
        todo.append(f)
    total = sum(f.stat().st_size for f in todo)
    print(f"{'GO' if a.go else 'DRY RUN'}: {len(todo)} tif(s), {total / 1e9:.1f} GB -> {TASK}")
    t0 = time.time()
    done = 0
    for i, f in enumerate(todo, 1):
        sz = f.stat().st_size
        if not a.go:
            print(f"  would copy {f.name} ({sz / 1e9:.2f} GB)")
            continue
        part = TASK / (f.name + ".part")
        dst = TASK / f.name
        t1 = time.time()
        shutil.copy2(str(f), str(part))
        if part.stat().st_size != sz:
            print(f"  !! size mismatch after copy: {f.name} ({part.stat().st_size} vs {sz}); left as .part", flush=True)
            continue
        part.rename(dst)
        done += sz
        print(f"  [{i:2d}/{len(todo)}] {f.name}  {sz / 1e9:.2f} GB in {(time.time() - t1) / 60:.1f} min  "
              f"({done / 1e9:.0f} of {total / 1e9:.0f} GB, {(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
    if a.go:
        print(f"done: {len(todo)} file(s), {done / 1e9:.1f} GB in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
