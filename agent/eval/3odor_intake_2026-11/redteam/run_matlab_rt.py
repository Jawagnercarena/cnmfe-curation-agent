"""Driver for the two MATLAB checks of the red team (one MATLAB at a time).

  c4   the real greedyROI_endoscope (production copy) on pure-noise movies
       -> cache/matlab/c4_T<T>_s<seed>.mat
  c5   the SEEDTRACE shadow copy, initialisation stage only, on real sessions at several
       min_corr -> cache/matlab/c5_<session>.mat
Reads session movies (.mat) only; writes only under redteam/cache/matlab and the log
redteam/matlab_rt.log.  The inline MATLAB handed to _run_matlab has no comments and is ASCII.
Usage: run_matlab_rt.py c4 | c5 [--only <substr> ...]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[2]
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(HERE))
from run_cnmfe import _run_matlab, REPO_ROOT  # noqa: E402
import rt_core as rc  # noqa: E402
import rt_labels as rl  # noqa: E402

OUT = rc.CACHE / "matlab"
LOG = HERE / "matlab_rt.log"
SHADOW = AGENT / "eval" / "missed_somata_2026-11" / "shadow"

C4_JOBS = [  # (T, seed, [min_corr ...], min_pnr, pick stage too)
    (2392, 1, [0.55, 0.575, 0.60], 3.0, True),
    (2392, 2, [0.55, 0.575, 0.60], 3.0, False),
    (6370, 1, [0.53, 0.55, 0.575], 3.0, True),
    (6370, 2, [0.53, 0.55, 0.575], 3.0, False),
    (8480, 1, [0.53, 0.55], 3.0, False),
]
C5_SESSIONS = ["090826-pnb102", "090826-pnb109", "090826-pnb99", "090826-pnb105"]
C5_MC = [0.60, 0.70, 0.80]


def log(msg):
    line = "%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fwd(p):
    return str(p).replace("\\", "/")


def run_c4():
    for T, seed, mcs, mp, pick in C4_JOBS:
        out = OUT / ("c4_T%d_s%d.mat" % (T, seed))
        if out.exists():
            log("c4 %s exists, skipped" % out.name)
            continue
        script = """
cd('{repo}');
run('cnmfe_setup.m');
addpath('{here}');
rt_T = {T}; rt_seed = {seed}; rt_mc = [{mcs}]; rt_mp = {mp}; rt_pick = {pick}; rt_out = '{out}';
c4_noise_greedy;
""".format(repo=fwd(REPO_ROOT), here=fwd(HERE), T=T, seed=seed, mcs=", ".join("%g" % m for m in mcs), mp=mp,
           pick="true" if pick else "false", out=fwd(out))
        log("c4 start T %d seed %d" % (T, seed))
        ok = _run_matlab(script, log, timeout_hours=2.0)
        log("c4 T %d seed %d: returned %s, output present %s" % (T, seed, ok, out.exists()))


def run_c5(only):
    for g, name, tif, sd, an in rc.sessions(only or C5_SESSIONS):
        if g != "6odor":
            continue
        out = OUT / ("c5_%s.mat" % name)
        if out.exists():
            log("c5 %s exists, skipped" % out.name)
            continue
        movie = sd / (sd.name + ".mat")
        if not movie.exists():
            log("c5 %s: movie .mat missing, skipped" % name)
            continue
        mc_run, mp_run = rl.run_params(sd)
        script = """
cd('{repo}');
run('cnmfe_setup.m');
addpath('{here}');
addpath('{shadow}', '-begin');
rt_nam = '{nam}'; rt_mc = [{mcs}]; rt_mp = {mp}; rt_bd = 10; rt_out = '{out}';
c5_seedtrace_init;
""".format(repo=fwd(REPO_ROOT), here=fwd(HERE), shadow=fwd(SHADOW), nam=fwd(movie),
           mcs=", ".join("%g" % m for m in C5_MC), mp=mp_run, out=fwd(out))
        log("c5 start %s (production run: min_corr %s, min_pnr %s)" % (name, mc_run, mp_run))
        ok = _run_matlab(script, log, timeout_hours=2.0)
        log("c5 %s: returned %s, output present %s" % (name, ok, out.exists()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["c4", "c5"])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "c4":
        run_c4()
    else:
        run_c5(a.only)


if __name__ == "__main__":
    main()
