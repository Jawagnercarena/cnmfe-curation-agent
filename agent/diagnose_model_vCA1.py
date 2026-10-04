"""
diagnose_model_vCA1.py
Runs diagnose_model.py with vCA1 configuration.

Injects config_vCA1 as `config` before importing diagnose_model, so DATA_ROOT
and MODEL_DIR point at the vCA1 data root and vCA1 model.  Identical pattern to
train_classifier_vCA1.py / watcher_vCA1.py.

Use it to re-derive vCA1's reject_threshold on the current pool.  The wrapper
train_classifier_vCA1.py injects the deployed threshold itself (0.04 on the
35-column contract since 2026-08-26; see that file), so this diagnostic only
informs the next decision.  Since 2026-08-26 diagnose_model honours
config_vCA1.AGENT_WEIGHT_OVERRIDE (5.0), so its absolute numbers match the
deployed weighting (they were computed at the sqrt recipe's 7.01x before).

    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe diagnose_model_vCA1.py
"""
import sys

import config_vCA1
sys.modules["config"] = config_vCA1

import diagnose_model
diagnose_model.main()
