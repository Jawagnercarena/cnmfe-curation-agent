"""
regen_launchers.py - rewrite existing run_final_review.m files to the current
self-locating launcher.

Sessions processed before the self-locating fix have a hardcoded
`session_dir = 'D:/Julian_CNMFe/...'` that only resolves on the central machine;
copied to a reviewer's machine it fails to find review_neuron.mat. This rewrites
every existing run_final_review.m in the local data tree so it self-locates.

Run on the central machine (writes only to the LOCAL data tree):
  python regen_launchers.py                 # all areas under DATA_PARENT
  python regen_launchers.py --area vCA1
  python regen_launchers.py --dry-run

SAFETY: only writes run_final_review.m files in the local data tree. No deletes.
"""
import argparse
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, REPO_ROOT
from review_prep import launcher_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", help="limit to this brain area (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    areas = ([DATA_PARENT / args.area] if args.area else
             [d for d in sorted(DATA_PARENT.iterdir())
              if d.is_dir() and not d.name.startswith(".")])
    text = launcher_text(REPO_ROOT)
    n = 0
    for area_dir in areas:
        if not area_dir.is_dir():
            print(f"SKIP (not found): {area_dir}")
            continue
        for f in sorted(area_dir.rglob("run_final_review.m")):
            n += 1
            if args.dry_run:
                print(f"  would rewrite {f}")
            else:
                f.write_text(text)
                print(f"  rewrote {f}")
    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"\n{verb} {n} run_final_review.m file(s).")


if __name__ == "__main__":
    main()
