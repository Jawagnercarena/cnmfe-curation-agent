"""
watcher_vCA1.py
Runs watcher.py with vCA1 configuration.

Usage:
    C:\ProgramData\anaconda3\envs\valence\python.exe watcher_vCA1.py

Monitors D:\Julian_CNMFe\vCA1\ for new .tif files and orchestrates the full
CNMFe pipeline for vCA1 sessions.  Keep running in the background; it polls
every 60 seconds.

Logs are written to agent/logs/watcher_vCA1.log and printed to the console.
"""
import logging
import sys
from pathlib import Path

import config_vCA1
sys.modules["config"] = config_vCA1

import watcher

# Redirect the auto-retrain subprocess to the vCA1 wrapper so that
# train_classifier.py picks up vCA1 DATA_ROOT and MODEL_DIR.
watcher._TRAIN_SCRIPT = Path(__file__).parent / "train_classifier_vCA1.py"

# Use a separate log file so BLA and vCA1 watchers don't interleave.
for _handler in list(watcher.logger.handlers):
    if isinstance(_handler, logging.FileHandler):
        _handler.close()
        watcher.logger.removeHandler(_handler)
_fh = logging.FileHandler(watcher.LOG_DIR / "watcher_vCA1.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
watcher.logger.addHandler(_fh)

watcher.main()
