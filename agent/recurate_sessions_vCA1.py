"""
recurate_sessions_vCA1.py
Runs recurate_sessions.py with vCA1 configuration (DATA_ROOT + model/vCA1).

All arguments are forwarded unchanged.  Examples:

    Re-curate every pending vCA1 session with the calibrated threshold:
        C:\ProgramData\anaconda3\envs\valence\python.exe recurate_sessions_vCA1.py

    Re-curate specific sessions, sending every candidate to human review:
        C:\ProgramData\anaconda3\envs\valence\python.exe recurate_sessions_vCA1.py ^
            "D:\Julian_CNMFe\vCA1\6odorDualDiffRew\AVG5x-TSeries-061126-pnb97-561um-3z-000" ^
            --threshold 0

Without this wrapper, recurate_sessions.py imports config (BLA) and would both
scan the wrong DATA_ROOT and score with the BLA classifier.

When --threshold 0 is used the session's candidate_features.npz records an empty
auto_rejected array, so train_classifier's label reconstruction takes the
"sizes match" path and every candidate carries a human decision.
"""
import sys

import config_vCA1
sys.modules["config"] = config_vCA1

import recurate_sessions
recurate_sessions.main()
