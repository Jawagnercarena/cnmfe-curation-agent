"""
train_classifier_vCA1.py
Runs train_classifier.py with vCA1 configuration.

All arguments are forwarded unchanged.  Examples:

    Full 3-way comparison + train:
        C:\ProgramData\anaconda3\envs\valence\python.exe train_classifier_vCA1.py

    Prospective-only retrain (used by watcher_vCA1 auto-retrain):
        C:\ProgramData\anaconda3\envs\valence\python.exe train_classifier_vCA1.py --prospective-only --model xgboost

    Dry-run eval:
        C:\ProgramData\anaconda3\envs\valence\python.exe train_classifier_vCA1.py --eval --dry-run --model auto

NOTE: animal_params.json must contain vCA1 entries (run extract_params.m first).
Model is saved to agent/model/vCA1/classifier.joblib.
"""
import sys
import json
from pathlib import Path

import config_vCA1
sys.modules["config"] = config_vCA1
from config import DATA_ROOT

# Count agent-reviewed sessions (have labels.mat AND ROIs_candidates.jpg,
# meaning they went through the full agent pipeline with human review).
# Dot-prefixed task dirs (.excluded/, .feature_expansion/, ...) are parked and
# skipped by every other scanner; skip them here too so a parked session cannot
# nudge the count that picks the threshold branch (matches train_classifier_DG_AL.py).
_n_agent = sum(
    1 for td in DATA_ROOT.iterdir() if td.is_dir() and not td.name.startswith(".")
    for sd in td.iterdir()
    if sd.is_dir()
    and (sd / "ROIs_candidates.jpg").exists()
    and (sd / "labels.mat").exists()
) if DATA_ROOT.exists() else 0

# vCA1 reject_threshold policy (auto-injected unless --threshold is given).
#
#   < 10 agent sessions  -> 0.04  (bootstrap-only, uncalibrated: 0.11 would
#                                   auto-reject too many borderline real neurons)
#   >= 10 agent sessions -> 0.04  (35-col contract, arm b0, deployed 2026-08-26:
#                                   8-seed OOF false-AR 0.48% mean / 0.96% worst
#                                   seed, 48.4% junk auto-caught; chosen over the
#                                   rule's 0.05 (0.66% / 0.96%, 51.6%) because
#                                   0.05's worst seed sits on the 1% ceiling and
#                                   flips with xgboost's thread count -- red team
#                                   2026-08-26, results/a08.json + a10.json).
#
# History: 0.05 was the 13-col value (validated 2026-06-23, kept 2026-08-24 as a
# deliberate 1.8%-false-AR posture).  Every threshold here is calibrated to
# animals already in the training set: for a held-out animal the same models run
# at ~4-5% false-AR (red team, attack #8), so a NEW animal/prep still gets the
# manual threshold-0 first pass before any auto-reject.
# The watcher's auto-retrain passes no --threshold, so this constant IS the
# deployed threshold; verify_vca1.py checks it against gate_decision.json.
_BOOTSTRAP_THRESHOLD      = 0.04
_VALIDATED_THRESHOLD      = 0.04
_AGENT_THRESHOLD_SESSIONS = 10

# Exact-token match only: `--threshold=0.07` (equals form) would slip past a
# substring test, get a second `--threshold` appended, and argparse would take
# the last one -- silently discarding the operator's value.
if not any(a == "--threshold" or a.startswith("--threshold=") for a in sys.argv):
    if _n_agent < _AGENT_THRESHOLD_SESSIONS:
        _t, _why = _BOOTSTRAP_THRESHOLD, "bootstrap-only, uncalibrated"
    else:
        _t, _why = _VALIDATED_THRESHOLD, "35-col arm b0, red-teamed 2026-08-26"
    print(f"[vCA1] {_n_agent} agent sessions -> using threshold {_t} ({_why}).")
    sys.argv.extend(["--threshold", str(_t)])

import train_classifier
train_classifier.main()
