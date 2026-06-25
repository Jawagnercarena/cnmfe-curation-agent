"""
local_config.py — machine-local paths for the CNMFe agent.

Every machine-specific absolute path resolves here, from environment variables
(optionally loaded from an adjacent .env file) with fallbacks to this machine's
historical defaults — so existing behaviour is unchanged when nothing is set.

To override on a given machine, either set the environment variables below, or
copy .env.example -> .env and edit it. .env is gitignored.

Reviewer machines do NOT need this file (the final-review step is pure MATLAB).
It matters for the central agent/training machine and any future Python tools.
"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no third-party dependency).

    Real environment variables take precedence over .env entries.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent / ".env")


# Repo root: self-locating (parent of agent/). Override only for unusual layouts.
REPO_ROOT = Path(os.environ.get("CNMFE_REPO_ROOT",
                                str(Path(__file__).resolve().parent.parent)))

# Parent folder holding one subdirectory per brain area (BLA/, vCA1/, ...).
DATA_PARENT = Path(os.environ.get("CNMFE_DATA_PARENT", r"D:\Julian_CNMFe"))

# MATLAB executable for headless / subprocess calls.
# Default R2023b (the lab's working version); set CNMFE_MATLAB_EXE to override.
MATLAB_EXE = os.environ.get("CNMFE_MATLAB_EXE",
                            r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe")

# Python interpreter (the 'valence' conda env) for subprocess calls.
PYTHON_EXE = os.environ.get("CNMFE_PYTHON_EXE",
                            r"C:\ProgramData\anaconda3\envs\valence\python.exe")

# Lab-server exchange root for distributing review bundles (see W2 scripts:
# push_review_bundle.py / ingest_returns.py). Empty until the server path is set.
EXCHANGE_ROOT = os.environ.get("CNMFE_EXCHANGE_ROOT", "")
