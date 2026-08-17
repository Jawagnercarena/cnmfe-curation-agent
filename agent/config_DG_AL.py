"""
config_DG_AL.py -- pipeline configuration for DG AL recordings.

Usage: scripts that need to run in DG AL context import from this module
instead of config.  The area wrappers do this automatically.

The area token is DG_AL, not "DG AL": it has to double as a Python module
name suffix (config_DG_AL.py), and module names cannot contain spaces.
"DG AL" remains the display name in docs and the paper.

When adding more areas, duplicate this file and update AREA and DATA_ROOT.
"""
from pathlib import Path
from local_config import DATA_PARENT

AREA = "DG_AL"

DATA_ROOT = DATA_PARENT / AREA

# Classifier for this area is stored separately from BLA and vCA1.
MODEL_DIR = Path(__file__).parent / "model" / AREA
