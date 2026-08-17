"""
train_classifier_DG_AL.py
Runs train_classifier.py with DG AL configuration.

All arguments are forwarded unchanged.  Examples:

    Full 3-way comparison + train:
        C:\ProgramData\anaconda3\envs\valence\python.exe train_classifier_DG_AL.py

    Prospective-only retrain (used by watcher_DG_AL auto-retrain):
        C:\ProgramData\anaconda3\envs\valence\python.exe train_classifier_DG_AL.py --prospective-only --model xgboost

NOTE: animal_params.json must contain DG_AL entries (see below -- on a cold
start extract_params.m has nothing to scan, so they are set by hand).
Model is saved to agent/model/DG_AL/classifier.joblib.
"""
import sys
from pathlib import Path

import config_DG_AL
sys.modules["config"] = config_DG_AL
from config import DATA_ROOT

# Count agent-reviewed sessions (have labels.mat AND ROIs_candidates.jpg,
# meaning they went through the full agent pipeline with human review).
_n_agent = sum(
    1 for td in DATA_ROOT.iterdir() if td.is_dir() and not td.name.startswith(".")
    for sd in td.iterdir()
    if sd.is_dir()
    and (sd / "ROIs_candidates.jpg").exists()
    and (sd / "labels.mat").exists()
) if DATA_ROOT.exists() else 0

# DG_AL reject_threshold policy (auto-injected unless --threshold is given).
#
# Without this injection train_classifier.py would stamp its per-model default
# (_THRESHOLD_BY_MODEL: xgboost 0.13) into the very first DG_AL model -- a
# number validated on BLA and never measured on DG AL.  The watcher retrains as
# soon as ONE reviewed session returns, so that would happen automatically and
# silently.
#
#   < _AGENT_THRESHOLD_SESSIONS  -> 0.0  (auto-reject nothing; a human sees every
#                                          candidate, matching the curator
#                                          THRESHOLD_OVERRIDE in watcher_DG_AL.py)
#   >= _AGENT_THRESHOLD_SESSIONS -> raise it, but only from a measured false
#                                    auto-reject rate.  Run diagnose_model.py
#                                    against this area first and record the
#                                    number here the way train_classifier_vCA1.py
#                                    documents its 0.05.
_COLD_START_THRESHOLD     = 0.0
_AGENT_THRESHOLD_SESSIONS = 10

if "--threshold" not in sys.argv:
    if _n_agent < _AGENT_THRESHOLD_SESSIONS:
        print(f"[DG_AL] {_n_agent} agent session(s) -> using threshold "
              f"{_COLD_START_THRESHOLD} (cold start, uncalibrated: nothing is "
              f"auto-rejected until a false-AR rate can be measured).")
        sys.argv.extend(["--threshold", str(_COLD_START_THRESHOLD)])
    else:
        print(f"[DG_AL] {_n_agent} agent sessions -> no threshold injected. "
              f"Measure a false auto-reject rate with diagnose_model.py and set "
              f"one explicitly with --threshold; do NOT accept the per-model "
              f"default, which is calibrated on BLA.")

import train_classifier
train_classifier.main()
