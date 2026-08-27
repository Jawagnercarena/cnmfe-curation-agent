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
# the sqrt term resolves to 7.01x here; the red team (2026-08-26, attack #9)
# found the "doubles false-AR" rationale was 3-seed noise but kept 5.0 as the
# best operating point by junk caught at matched false-AR
# (agent/eval/bootstrap_redteam_2026-08/results/a09.json).
AGENT_WEIGHT_OVERRIDE = 5.0

# Feature-contract version: 2 = the 35-column contract (13 base | 13 ranks |
# 8 v2b | v2_present), deployed for vCA1 2026-08-26 (arm b0, see
# agent/eval/vca1_v2_2026-08/VCA1_V2_LOG.md).  Shared code reads it with
# getattr(config, "FEATURE_VERSION", 1).
FEATURE_VERSION = 2

# Bootstrap rows under v2: "b0" = real v2b from the persisted candidate traces
# with ring_contrast forced to 0 and hiconf from the companion 13-col model
# (bootstrap_preagent honours this; BLA/DG_AL leave it unset = zero-fill).
BOOTSTRAP_V2B = "b0"
