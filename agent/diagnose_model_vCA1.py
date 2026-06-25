"""
diagnose_model_vCA1.py
Runs diagnose_model.py with vCA1 configuration.

Injects config_vCA1 as `config` before importing diagnose_model, so DATA_ROOT
and MODEL_DIR point at the vCA1 data root and vCA1 model.  Identical pattern to
train_classifier_vCA1.py / watcher_vCA1.py.

Run after vCA1 has >= 10 agent-reviewed sessions to re-derive the optimal
reject_threshold (the watcher auto-switches 0.04 -> 0.11 at that point, but the
0.11 jump is unvalidated until this OOF Pareto sweep confirms it).

    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe diagnose_model_vCA1.py
"""
import sys

import config_vCA1
sys.modules["config"] = config_vCA1

import diagnose_model
diagnose_model.main()
