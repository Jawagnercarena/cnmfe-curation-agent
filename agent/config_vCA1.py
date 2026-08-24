"""
config_vCA1.py -- pipeline configuration for vCA1 recordings.

Usage: scripts that need to run in vCA1 context import from this module
instead of config.  The bootstrap runner does this automatically.

When adding more areas, duplicate this file (e.g. config_DG.py) and
update AREA and DATA_ROOT.
"""
from pathlib import Path
from local_config import DATA_PARENT

AREA = "vCA1"

DATA_ROOT = DATA_PARENT / AREA

# Classifier for this area is stored separately from BLA.
MODEL_DIR = Path(__file__).parent / "model" / AREA

# Fixed agent up-weight (replaces the dynamic sqrt formula in
# train_classifier). Decided 2026-08-24 after the bootstrap pixel-order fix:
# the sqrt term resolves to 7.01x here and doubles false-AR at the deployed
# 0.05 threshold vs this value (3-seed sweep, AUC 0.886±0.004 vs 0.884±0.007;
# see agent/eval/bootstrap_matching_2026-08/c3_vca1_weight_sweep.log).
AGENT_WEIGHT_OVERRIDE = 5.0
