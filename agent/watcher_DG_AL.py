"""
watcher_DG_AL.py
Runs watcher.py with DG AL configuration.

Usage:
    C:\ProgramData\anaconda3\envs\valence\python.exe watcher_DG_AL.py

Monitors D:\Julian_CNMFe\DG_AL\ for new .tif files and orchestrates the full
CNMFe pipeline for DG AL sessions.  Keep running in the background; it polls
every 60 seconds.

Logs are written to agent/logs/watcher_DG_AL.log and printed to the console.
"""
import logging
import sys
from pathlib import Path

import config_DG_AL
sys.modules["config"] = config_DG_AL

import watcher

# Redirect the auto-retrain subprocess to the DG_AL wrapper so that
# train_classifier.py picks up DG_AL DATA_ROOT and MODEL_DIR.  This is not
# optional: watcher._retrain_classifier runs the script as a SUBPROCESS, so the
# sys.modules injection above does not reach it.  Without this line the first
# DG_AL labels would retrain BLA's model and DG_AL would never get one.
watcher._TRAIN_SCRIPT = Path(__file__).parent / "train_classifier_DG_AL.py"

# Auto-reject nothing: DG AL is a cold start with no classifier, so the curator
# would otherwise fall back to an IsolationForest fitted on this area's own
# unreviewed candidates and drop neurons no human has seen.  A reviewer sees
# every candidate until there are enough reviewed sessions to measure a false
# auto-reject rate; raise this only from a measurement, never from a default.
import curator
curator.THRESHOLD_OVERRIDE = 0.0

# Use a separate log file so the BLA, vCA1 and DG_AL watchers don't interleave.
for _handler in list(watcher.logger.handlers):
    if isinstance(_handler, logging.FileHandler):
        _handler.close()
        watcher.logger.removeHandler(_handler)
_fh = logging.FileHandler(watcher.LOG_DIR / "watcher_DG_AL.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
watcher.logger.addHandler(_fh)

watcher.main()
