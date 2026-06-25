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
_n_agent = sum(
    1 for td in DATA_ROOT.iterdir() if td.is_dir()
    for sd in td.iterdir()
    if sd.is_dir()
    and (sd / "ROIs_candidates.jpg").exists()
    and (sd / "labels.mat").exists()
)

# vCA1 reject_threshold policy (auto-injected unless --threshold is given).
#
#   < 10 agent sessions  -> 0.04  (bootstrap-only, uncalibrated: 0.11 would
#                                   auto-reject too many borderline real neurons)
#   >= 10 agent sessions -> 0.05  (VALIDATED 2026-06-23 via diagnose_model_vCA1.py
#                                   OOF Pareto sweep; agent-only AUC 0.881).
#
# Why 0.05 and not the per-model standard 0.11: vCA1's false-auto-reject rate at
# 0.11 is 3.6% (vs BLA's 0.8%) because these are low-yield "handful of cells"
# experiments where real-vs-garbage separation is weaker in the low-score region.
# 0.05 keeps false-AR at 1.8% (catches 27% garbage) to protect precious cells.
# Re-run diagnose_model_vCA1.py after ~10 more agent sessions to re-confirm.
_BOOTSTRAP_THRESHOLD      = 0.04
_VALIDATED_THRESHOLD      = 0.05
_AGENT_THRESHOLD_SESSIONS = 10

if "--threshold" not in sys.argv:
    if _n_agent < _AGENT_THRESHOLD_SESSIONS:
        _t, _why = _BOOTSTRAP_THRESHOLD, "bootstrap-only, uncalibrated"
    else:
        _t, _why = _VALIDATED_THRESHOLD, "validated OOF sweep 2026-06-23"
    print(f"[vCA1] {_n_agent} agent sessions -> using threshold {_t} ({_why}).")
    sys.argv.extend(["--threshold", str(_t)])

import train_classifier
train_classifier.main()
