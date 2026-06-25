"""
bootstrap_vCA1.py
Runs bootstrap_preagent.py with vCA1 configuration.

All arguments are forwarded unchanged.  Run two workers in separate
Anaconda Prompt windows to process sessions in parallel:

    Worker 0:
        cd C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent
        C:\ProgramData\anaconda3\envs\valence\python.exe bootstrap_vCA1.py --worker 0 --num-workers 2

    Worker 1 (separate window):
        cd C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent
        C:\ProgramData\anaconda3\envs\valence\python.exe bootstrap_vCA1.py --worker 1 --num-workers 2

Dry-run (see what would be processed without running):
    C:\ProgramData\anaconda3\envs\valence\python.exe bootstrap_vCA1.py --dry-run

NOTE: animal_params.json must contain entries for vCA1 animals before
running.  If not, bootstrap will fall back to default gSig=9/gSiz=36.
Run extract_params.m in MATLAB first (it now scans all areas automatically).
"""
import sys

# Inject vCA1 config before bootstrap_preagent is imported, so that all
# 'from config import ...' statements in that module resolve to vCA1 values.
import config_vCA1
sys.modules["config"] = config_vCA1

import bootstrap_preagent
bootstrap_preagent.main()
