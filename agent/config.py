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
