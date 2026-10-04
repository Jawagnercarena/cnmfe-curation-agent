"""
config.py — pipeline configuration for the CNMFe agent.

This is the single source of truth for the data root directory.
When adding a new brain area, duplicate this file (e.g. config_vCA1.py)
or change DATA_ROOT here and retrain the classifier from scratch.
"""
from pathlib import Path
from local_config import DATA_PARENT

# Brain area this pipeline instance processes.
# Used for labelling in animal_params.json and model metadata.
AREA = "BLA"

# Root directory containing task-level subdirectories (2tones/, 3odor/, etc.)
# for this brain area, derived from the machine-local data parent.
DATA_ROOT = DATA_PARENT / AREA

# Directory where the trained classifier for this area is saved.
# Each brain area keeps its own model so areas don't overwrite each other.
MODEL_DIR = Path(__file__).parent / "model" / AREA

# Feature-contract version for this area.  2 = the 35-column expanded
# contract (13 base | 13 within-session percentile ranks | 8 v2b | v2_present)
# deployed 2026-08-20; see agent/eval/step4_2026-08/STEP4_LOG.md.  Areas
# whose config does not set this run the original 13-column contract
# (shared code reads it with getattr(config, "FEATURE_VERSION", 1)).
FEATURE_VERSION = 2
