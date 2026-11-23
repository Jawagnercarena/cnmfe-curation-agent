"""
run_seedtrace.py -- re-run CNMF-E on chosen sessions with every seed attempt and
every component-set change recorded (agent/eval/missed_somata_2026-11).

Question (user, 2026-11-04): GCaMP somata that are visible in the session mean
and pass both seeding gates still get no candidate -- were they never seeded, or
seeded and then merged/deleted?  Sessions: the 9/8/26 FOVA session of each of the
six red-guide animals (imaging sheet: FOVA for all six that day).

READS the session folder (movie .mat, agent_run.log).  WRITES only under
agent/eval/missed_somata_2026-11/seedtrace/<session>/ (seedtrace_ds.mat,
seedtrace.mat) and the log next to this file.  Never writes to the session
folder, D:\\retro_tdTomato_Alignment or the server.  No deletes.

MATLAB: same entry as production (cd repo; cnmfe_setup) plus this folder and
shadow/ first on the path; aborts unless the shadow greedyROI_endoscope.m wins.
One run at a time; waits while free RAM is below MIN_FREE_GB.

Usage:
  python run_seedtrace.py                       # the six sessions, serially
  python run_seedtrace.py --only <session>      # one session
  python run_seedtrace.py --movie X.mat --out DIR --params 9,36,0.7,6.6,10   # smoke test
"""
import argparse
import ctypes
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[1]
sys.path.insert(0, str(AGENT))
from local_config import DATA_PARENT  # noqa: E402
from run_cnmfe import _run_matlab, REPO_ROOT  # noqa: E402

TASK_DIR = Path(DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
SESSIONS = [
    "AVG5x-TSeries-090826-pnb99-463um-3z-000",
    "AVG5x-TSeries-090826-pnb102-497um-3z-000",
    "AVG5x-TSeries-090826-pnb104-425um-3z-000",
    "AVG5x-TSeries-090826-pnb105-385um-35z-000",
    "AVG5x-TSeries-090826-pnb106-209um-4z-000",
    "AVG5x-TSeries-090826-pnb109-412um-25z-000",
]
OUT_ROOT = HERE / "seedtrace"
LOG_FILE = HERE / "run_seedtrace.log"
MIN_FREE_GB = 110.0
TIMEOUT_H = 4.0

SCRIPT = """
cd('{repo}');
run('cnmfe_setup.m');
addpath('{shadow}', '-begin');
addpath('{here}', '-begin');
st_w = which('greedyROI_endoscope', '-all');
fprintf('[SEEDTRACE] greedyROI_endoscope resolution:\\n'); fprintf('[SEEDTRACE]   %s\\n', st_w{{:}});
if isempty(strfind(st_w{{1}}, 'missed_somata_2026-11')); error('seedtrace: shadow greedyROI_endoscope is not first on the path'); end;
fprintf('[SEEDTRACE] Sources2D -> %s\\n', which('Sources2D'));
fprintf('[SEEDTRACE] cnmfe_full -> %s\\n', which('cnmfe_full'));
fprintf('[SEEDTRACE] st_snap -> %s\\n', which('st_snap'));
nam = '{nam}'; gSig = {gSig}; gSiz = {gSiz}; min_corr = {min_corr}; min_pnr = {min_pnr}; bd = {bd}; st_out = '{out}';
seedtrace_run;
"""


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_ram_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def final_params(session_dir: Path) -> dict:
    txt = (session_dir / "agent_run.log").read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"Final parameters: \{'gSig': (\d+), 'gSiz': (\d+), 'min_corr': ([\d.]+), "
                      r"'min_pnr': ([\d.]+), 'bd': (\d+)\}", txt)
    if not hits:
        raise RuntimeError(f"no Final parameters line in {session_dir / 'agent_run.log'}")
    g, z, c, p, b = hits[-1]
    return dict(gSig=int(g), gSiz=int(z), min_corr=float(c), min_pnr=float(p), bd=int(b))


def run_one(movie: Path, out: Path, prm: dict) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    while free_ram_gb() < MIN_FREE_GB:
        log(f"  waiting: free RAM {free_ram_gb():.0f} GB < {MIN_FREE_GB:.0f} GB")
        time.sleep(300)
    fwd = lambda p: str(p).replace("\\", "/")
    script = SCRIPT.format(repo=fwd(REPO_ROOT), shadow=fwd(HERE / "shadow"), here=fwd(HERE),
                           nam=fwd(movie), out=fwd(out), **prm)
    # -batch joins the lines: a % outside a quoted string would comment out the rest
    assert all("%" not in re.sub(r"'[^']*'", "", ln) for ln in script.splitlines()), "% comment in -batch script"
    log(f"START {movie.name} params {prm} -> {out}")
    ok = _run_matlab(script, lambda s: log(s), timeout_hours=TIMEOUT_H)
    have = (out / "seedtrace.mat").exists()
    log(f"END {movie.name}: matlab_ok={ok} seedtrace.mat={'present' if have else 'MISSING'}"
        f" seedtrace_ds.mat={'present' if (out / 'seedtrace_ds.mat').exists() else 'MISSING'}")
    return ok and have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--movie")
    ap.add_argument("--out")
    ap.add_argument("--params", help="gSig,gSiz,min_corr,min_pnr,bd (smoke test only)")
    a = ap.parse_args()
    if a.movie:
        g, z, c, p, b = a.params.split(",")
        prm = dict(gSig=int(g), gSiz=int(z), min_corr=float(c), min_pnr=float(p), bd=int(b))
        sys.exit(0 if run_one(Path(a.movie), Path(a.out), prm) else 1)
    todo = [s for s in SESSIONS if (a.only is None or s == a.only)]
    log(f"seedtrace batch: {len(todo)} session(s)")
    for s in todo:
        sd = TASK_DIR / s
        out = OUT_ROOT / s
        if (out / "seedtrace.mat").exists():
            log(f"SKIP {s}: seedtrace.mat already present")
            continue
        try:
            run_one(sd / f"{s}.mat", out, final_params(sd))
        except Exception as e:  # keep going; the log says what failed
            log(f"FAIL {s}: {e!r}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
