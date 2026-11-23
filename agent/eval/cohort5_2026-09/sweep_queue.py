"""
sweep_queue.py -- run agent/sweep_gsig.py on a list of sessions, one at a time,
each as soon as the watcher has finished that session's reference run.

Meant to be launched fully detached (PowerShell Start-Process), so it survives
the launching shell.  One sweep arm at a time = one extra MATLAB process next to
the watcher's (the machine's cap is two heavy MATLAB jobs).

For each session in order:
  - wait until <session>/neuron.mat and <session>/candidate_features.npz exist
    (the watcher writes candidate_features.npz after the headless run);
  - skip the session if every requested arm already has neuron.mat and the
    comparison PDF exists;
  - otherwise run sweep_gsig.py --area <AREA> --session <dir> --gsig <arms>
    --no-wait, log to <log_dir>/sweep_<session>.log, and write
    <log_dir>/sweep_<session>.done with the exit code.

Usage:
    python sweep_queue.py --area vCA1 --gsig 7 11 --sessions <dir> [<dir> ...]
                          [--log-dir agent/logs] [--poll 120]
"""
import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent.parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import PYTHON_EXE, DATA_PARENT  # noqa: E402

SWEEP = AGENT_DIR / "sweep_gsig.py"


def log(fh, msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def reference_ready(sd: Path) -> bool:
    return (sd / "neuron.mat").exists() and (sd / "candidate_features.npz").exists()


def already_swept(area: str, sd: Path, arms: list[int]) -> bool:
    root = DATA_PARENT / area / ".sweep" / sd.name
    if not (root / "missed_by_reference.pdf").exists():
        return False
    return all((root / f"gSig{g:02d}" / "neuron.mat").exists() for g in arms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="vCA1")
    ap.add_argument("--gsig", type=int, nargs="+", default=[7, 11])
    ap.add_argument("--sessions", nargs="+", required=True, type=Path)
    ap.add_argument("--log-dir", type=Path, default=AGENT_DIR / "logs")
    ap.add_argument("--poll", type=int, default=120, help="seconds between readiness checks")
    a = ap.parse_args()
    a.log_dir.mkdir(parents=True, exist_ok=True)
    qlog = open(a.log_dir / "sweep_queue.log", "a", encoding="utf-8")
    log(qlog, f"[QUEUE] start: area {a.area}, arms {a.gsig}, {len(a.sessions)} sessions, python {PYTHON_EXE}")
    for sd in a.sessions:
        sd = sd.resolve()
        if not sd.is_dir():
            log(qlog, f"[QUEUE] {sd.name}: directory missing (not staged yet?) -- waiting for it")
        while not sd.is_dir() or not reference_ready(sd):
            time.sleep(a.poll)
        if already_swept(a.area, sd, a.gsig):
            log(qlog, f"[QUEUE] {sd.name}: already swept -- skipping")
            continue
        log(qlog, f"[QUEUE] {sd.name}: reference run present -- starting sweep")
        slog = a.log_dir / f"sweep_{sd.name}.log"
        cmd = [PYTHON_EXE, str(SWEEP), "--area", a.area, "--session", str(sd),
               "--gsig", *[str(g) for g in a.gsig], "--no-wait"]
        t0 = time.time()
        with open(slog, "a", encoding="utf-8") as out:
            rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(AGENT_DIR))
        (a.log_dir / f"sweep_{sd.name}.done").write_text(f"exit {rc}\n", encoding="ascii")
        log(qlog, f"[QUEUE] {sd.name}: sweep exit {rc} after {(time.time() - t0) / 60:.1f} min")
    log(qlog, "[QUEUE] all sessions processed")


if __name__ == "__main__":
    main()
