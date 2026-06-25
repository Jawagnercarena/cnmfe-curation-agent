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
