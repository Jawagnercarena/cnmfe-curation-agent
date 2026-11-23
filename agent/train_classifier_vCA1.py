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
#   < 10 agent sessions  -> 0.02  (bootstrap-only, uncalibrated: 0.11 would
#                                   auto-reject too many borderline real neurons)
#   >= 10 agent sessions -> 0.02  (user decision 2026-11-20, after 19 of the 28
#                                   cohort-5 sessions came back labelled; 211-
#                                   session pool, 8 seeds, threads 8.  False-AR
#                                   mean / worst seed on the 12 threshold-0
#                                   pilots (410 kept cells, every candidate
#                                   shown): 0.21% / 0.49% with the session held
#                                   out, 0.34% / 0.49% with the animal held out
#                                   (0.03: 0.40% / 0.73% and 0.58% / 0.98%; 0.04
#                                   fails animal-held-out, 1.10% / 1.46%).  On
#                                   all 46 agent sessions reviewed without any
#                                   auto-reject (1389 kept): 0.24% / 0.36% and
#                                   0.31% / 0.36%.  Weakest animal pnb109: 0.71%
#                                   session, 1.30% animal held out.  Junk caught
#                                   28% vs 19% at 0.01.  Zero-weighting the
#                                   duplicate deletes (same day) moves the
#                                   pooled figure 0.19% -> 0.22%.  Dim sessions
#                                   are not represented in this test.
#                                   agent/eval/cohort5_2026-09/analysis_2026-11-20/threshold/)
#   (2026-10-28 to 11-20) -> 0.01  (user decision 2026-10-28 for the cohort-5
#                                   batch (pnb99/102/104/105/106/109): on the 12
#                                   threshold-0 pilot reviews (410 kept cells) the
#                                   within-animal OOF false-AR was 0.00% at 0.01 vs
#                                   0.49% mean / 0.73% worst seed at 0.02 (pnb109
#                                   alone 1.6%), junk caught 26% vs 37%; the
#                                   pre-pilot model that had never seen these
#                                   animals missed 3.2% at 0.02. Reassess after
#                                   the remaining 28 cohort-5 sessions return.
#                                   agent/eval/cohort5_2026-09/analysis_2026-10-28/)
#   (before that)        -> 0.02  (35-col contract, arm b0; user decision
#                                   2026-09-24 on the 161-session pool, 8 seeds,
#                                   threads pinned at 8, agent/eval/algo_bakeoff_2026-09/
#                                   results_0924: false-AR at 0.02 = 0.47% mean /
#                                   0.67% worst seed session-grouped, 0.44% / 0.67%
#                                   animal-grouped, 36% junk auto-caught; 0.03 =
#                                   0.82% / 1.15% session (fails the 1% worst-seed
#                                   cap) and 0.72% / 0.96% animal; the deployed
#                                   0.04 had drifted to 1.20% / 1.63%).
#
# History: 0.05 was the 13-col value (validated 2026-06-23, kept 2026-08-24 as a
# deliberate 1.8%-false-AR posture); 0.04 from 2026-08-26 (35-col deploy, red
# team results/a08.json + a10.json) to 2026-09-24.  Each returned batch adds dim
# real cells, so false-AR at a fixed T drifts up while AUC stays flat: re-derive
# after every ingest (rule: largest T with mean false-AR <= 0.85% and worst seed
# <= 1.0% under BOTH session and animal grouping).  Every threshold here is
# calibrated to animals already in the training set; a held-out animal of the
# same prep ran 1.0-1.3% at 0.02 and 1.4-1.8% at 0.03 (pnb97 held out), so a NEW
# animal/prep still gets a threshold-0 first pass (or a measured conservative T)
# before any auto-reject.
# The watcher's auto-retrain passes no --threshold, so this constant IS the
# deployed threshold; verify_vca1.py checks it against gate_decision.json.
_BOOTSTRAP_THRESHOLD      = 0.02
_VALIDATED_THRESHOLD      = 0.02
_AGENT_THRESHOLD_SESSIONS = 10

# Exact-token match only: `--threshold=0.07` (equals form) would slip past a
# substring test, get a second `--threshold` appended, and argparse would take
# the last one -- silently discarding the operator's value.
if not any(a == "--threshold" or a.startswith("--threshold=") for a in sys.argv):
    if _n_agent < _AGENT_THRESHOLD_SESSIONS:
        _t, _why = _BOOTSTRAP_THRESHOLD, "bootstrap-only, uncalibrated"
    else:
        _t, _why = _VALIDATED_THRESHOLD, "user decision 2026-11-20, 211-session pool"
    print(f"[vCA1] {_n_agent} agent sessions -> using threshold {_t} ({_why}).")
    sys.argv.extend(["--threshold", str(_t)])

import train_classifier
train_classifier.main()
