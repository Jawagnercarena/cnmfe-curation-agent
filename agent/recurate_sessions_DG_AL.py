"""
recurate_sessions_DG_AL.py
Runs recurate_sessions.py with DG AL configuration (DATA_ROOT + model/DG_AL).

All arguments are forwarded unchanged.  Examples:

    Re-curate every pending DG_AL session:
        C:\ProgramData\anaconda3\envs\valence\python.exe recurate_sessions_DG_AL.py

    Re-curate specific sessions, sending every candidate to human review:
        C:\ProgramData\anaconda3\envs\valence\python.exe recurate_sessions_DG_AL.py ^
            "D:\Julian_CNMFe\DG_AL\odor_encoding\AVG4x-TSeries-040623-DG6D-356um-406um-2z-000A" ^
            --threshold 0

Without this wrapper, recurate_sessions.py imports config (BLA) and would both
scan the wrong DATA_ROOT and score with the BLA classifier.

With no session arguments, find_pending_sessions() scans only DG_AL's DATA_ROOT
and skips bootstrap sessions, so a bare run is safe here -- unlike the BLA
retro-tdTomato case, where only some sessions were the new prep and explicit
paths were required.

Note that watcher_DG_AL.py already pins curator.THRESHOLD_OVERRIDE = 0.0, so
sessions it prepares are not auto-rejecting anything in the first place; this
script is for re-scoring after the area's classifier is trained or retrained.
"""
import sys

import config_DG_AL
sys.modules["config"] = config_DG_AL

import recurate_sessions
recurate_sessions.main()
