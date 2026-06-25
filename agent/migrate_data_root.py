"""
migrate_data_root.py
One-time migration script.

Run this AFTER physically moving the Julian_CNMFe task folders into
Julian_CNMFe\\BLA\\ (step 3 of the reorganisation instructions).

What it does:
  - Finds every run_final_review.m under D:\\Julian_CNMFe\\BLA\\
  - Replaces the old path prefix  D:/Julian_CNMFe/  with  D:/Julian_CNMFe/BLA/
  - Reports every file patched and every file that needed no change

Usage (from the agent/ directory, after the data move):
    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe migrate_data_root.py

Add --dry-run to preview changes without writing:
    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe migrate_data_root.py --dry-run
"""

import argparse
from pathlib import Path

OLD_PREFIX = "D:/Julian_CNMFe/"
NEW_PREFIX = "D:/Julian_CNMFe/BLA/"

# After the move, all data lives under this root
BLA_ROOT = Path(r"D:\Julian_CNMFe\BLA")


def patch_file(path: Path, dry_run: bool) -> bool:
    """Returns True if the file was (or would be) patched."""
    text = path.read_text(encoding="latin-1")
    if OLD_PREFIX not in text:
        return False
    new_text = text.replace(OLD_PREFIX, NEW_PREFIX)
    if not dry_run:
        path.write_text(new_text, encoding="latin-1")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch run_final_review.m paths after data move.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files.")
    args = parser.parse_args()

    if not BLA_ROOT.exists():
        print(f"ERROR: BLA_ROOT not found: {BLA_ROOT}")
        print("Make sure you have completed step 3 (physical data move) before running this script.")
        return

    review_files = sorted(BLA_ROOT.rglob("run_final_review.m"))
    if not review_files:
        print(f"No run_final_review.m files found under {BLA_ROOT}")
        return

    patched = []
    skipped = []
    for f in review_files:
        if patch_file(f, dry_run=args.dry_run):
            patched.append(f)
        else:
            skipped.append(f)

    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action} {len(patched)} file(s):")
    for f in patched:
        rel = f.relative_to(BLA_ROOT)
        print(f"  {rel}")

    if skipped:
        print(f"\nAlready up-to-date ({len(skipped)} file(s)) — no change needed:")
        for f in skipped:
            rel = f.relative_to(BLA_ROOT)
            print(f"  {rel}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Remove --dry-run to apply.")
    else:
        print("\nDone. All run_final_review.m files now reference D:/Julian_CNMFe/BLA/")


if __name__ == "__main__":
    main()
