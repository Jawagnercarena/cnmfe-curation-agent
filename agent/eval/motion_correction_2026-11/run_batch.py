"""Run motion_metrics.py over a set of session tifs, a few at a time (read-only on the data)."""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew")
OUT = HERE / "out"

ap = argparse.ArgumentParser()
ap.add_argument("--match", nargs="+", required=True, help="substrings; a session needs any one")
ap.add_argument("--skip", nargs="*", default=[], help="substrings to leave out")
ap.add_argument("--jobs", type=int, default=3)
args = ap.parse_args()

tifs = []
for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
        continue
    if not any(m in d.name for m in args.match) or any(s in d.name for s in args.skip):
        continue
    tif = d / (d.name + ".tif")
    if tif.is_file() and not (OUT / (d.name + ".npz")).is_file():
        tifs.append(tif)
print(f"{len(tifs)} sessions to run", flush=True)


def run(tif: Path) -> tuple[str, int]:
    log = OUT / (tif.stem + ".log")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="ascii", errors="replace") as fh:
        rc = subprocess.run([sys.executable, str(HERE / "motion_metrics.py"), str(tif),
                             "--out", str(OUT)], stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"rc={rc} {tif.stem}", flush=True)
    return tif.stem, rc


with ThreadPoolExecutor(args.jobs) as ex:
    res = list(ex.map(run, tifs))
bad = [n for n, rc in res if rc != 0]
print(f"finished: {len(res) - len(bad)} ok, {len(bad)} failed {bad}", flush=True)
sys.exit(1 if bad else 0)
