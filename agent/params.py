"""
params.py
Estimates CNMFe hyperparameters for a new session.

Called by watcher.py before launching the MATLAB headless run.
Returns: gSig, gSiz, min_corr, min_pnr, bd
"""

import json
import re
import numpy as np
import scipy.io as sio
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config" / "animal_params.json"

# Safe fallback values (permissive / larger, per user preference)
DEFAULT_GSIZ = 36
DEFAULT_GSIG = 9
DEFAULT_MIN_CORR = 0.75
DEFAULT_MIN_PNR = 8.0
DEFAULT_BD = 20


def load_animal_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def parse_animal_id(session_name: str) -> str | None:
    """
    Parse animal ID from session folder name.
    Format: AVG5x-TSeries-DATE-ANIMALID-depthum-planesz-run
    """
    parts = session_name.split("-")
    if len(parts) >= 4:
        return parts[3]  # e.g. 'bla8', 'bla13', 'pnb88'
    return None


def get_gsig_gsiz(session_name: str, log) -> tuple[int, int]:
    """
    Look up gSig and gSiz for this session's animal from animal_params.json.
    Falls back to defaults if animal not found or if config flags inconsistency.
    """
    animal_id = parse_animal_id(session_name)
    if animal_id is None:
        log(f"  [PARAMS] Cannot parse animal ID from '{session_name}'. Using defaults.")
        return DEFAULT_GSIG, DEFAULT_GSIZ

    config = load_animal_config()
    if animal_id not in config:
        log(f"  [PARAMS] Animal '{animal_id}' not in config. Using defaults "
            f"(gSig={DEFAULT_GSIG}, gSiz={DEFAULT_GSIZ}). "
            f"Add this animal to animal_params.json after processing.")
        return DEFAULT_GSIG, DEFAULT_GSIZ

    entry = config[animal_id]
    gSig = entry["gSig"]
    gSiz = entry["gSiz"]

    if entry.get("flag") == "inconsistent_check_manually":
        all_gsiz = entry.get("gSiz_all", [gSiz])
        all_gsiz = [all_gsiz] if isinstance(all_gsiz, (int, float)) else all_gsiz
        log(f"  [PARAMS] Animal '{animal_id}' had inconsistent gSig/gSiz across sessions "
            f"(gSiz values seen: {all_gsiz}). Using chosen value gSig={gSig}, gSiz={gSiz}. "
            f"Edit config/animal_params.json to override.")
    else:
        log(f"  [PARAMS] Animal '{animal_id}': gSig={gSig}, gSiz={gSiz} "
            f"(consistent across {entry['n_sessions']} sessions).")

    return gSig, gSiz


def estimate_min_corr_min_pnr(session_dir: Path, log,
                               bootstrap_mode: bool = False) -> tuple[float, float]:
    """
    Estimate min_corr and min_pnr from the saved Cn and pnr images.

    Strategy (permissive bias):
    - min_corr: 85th percentile of nonzero Cn pixels (lower than expert would use,
      keeping more candidates for the classifier to handle)
    - min_pnr: 85th percentile of PNR pixels above a noise floor

    bootstrap_mode=True uses a more permissive 50th-percentile estimate with a
    lower clamp ceiling.  Pre-agent sessions were originally run with min_corr=0.3
    (very permissive) and expert review as the filter.  The bootstrap needs to seed
    neurons in lower-correlation regions to recover those same curated neurons.
    """
    cn_file  = session_dir / "Cn.mat"
    pnr_file = session_dir / "pnr.mat"

    if not cn_file.exists() or not pnr_file.exists():
        log(f"  [PARAMS] Cn.mat or pnr.mat not found — using defaults "
            f"(min_corr={DEFAULT_MIN_CORR}, min_pnr={DEFAULT_MIN_PNR}).")
        return DEFAULT_MIN_CORR, DEFAULT_MIN_PNR

    cn_data  = sio.loadmat(str(cn_file))
    pnr_data = sio.loadmat(str(pnr_file))

    Cn  = cn_data["Cn"].flatten()
    pnr = pnr_data["pnr"].flatten()

    # Only use pixels with meaningful signal
    cn_nonzero  = Cn[Cn > 0.1]
    pnr_nonzero = pnr[pnr > 1.0]

    if cn_nonzero.size == 0 or pnr_nonzero.size == 0:
        log("  [PARAMS] Cn/pnr arrays appear empty — using defaults.")
        return DEFAULT_MIN_CORR, DEFAULT_MIN_PNR

    if bootstrap_mode:
        # Pre-agent sessions used min_corr=0.3 with human expert filtering.
        # Use 65th percentile with a moderate ceiling so the bootstrap seeds
        # neurons in lower-correlation regions that the expert originally kept,
        # while keeping candidate counts close to the ~400 expert max.
        # Floor 0.55: avoids seeding in pure noise.
        # Ceiling 0.82: prevents over-restriction that misses curated neurons.
        min_corr = float(np.percentile(cn_nonzero, 65))
        min_pnr  = float(np.percentile(pnr_nonzero, 65))
        min_corr = float(np.clip(min_corr, 0.55, 0.82))
        min_pnr  = float(np.clip(min_pnr,  5.0,  13.0))
    else:
        # 85th percentile of the distribution (permissive: keeps more seeds)
        # Expert typically uses values near the 90th-95th percentile visually
        min_corr = float(np.percentile(cn_nonzero, 85))
        min_pnr  = float(np.percentile(pnr_nonzero, 85))
        min_corr = float(np.clip(min_corr, 0.6, 0.92))
        min_pnr  = float(np.clip(min_pnr, 5.0, 15.0))

    # Round to 2 decimal places for readability
    min_corr = round(min_corr, 2)
    min_pnr  = round(min_pnr, 1)

    log(f"  [PARAMS] Estimated from Cn/pnr images: min_corr={min_corr}, min_pnr={min_pnr}.")
    return min_corr, min_pnr


def estimate_bd(session_dir: Path, log) -> int:
    """
    Estimate boundary width (bd) from the correlation image.

    Motion correction artifacts appear as high-correlation lines near edges.
    We look for the outermost band where edge-pixel correlation is noticeably
    higher than the image interior, indicating motion blur.

    Falls back to DEFAULT_BD if Cn is not available yet.
    """
    cn_file = session_dir / "Cn.mat"
    if not cn_file.exists():
        log(f"  [PARAMS] Cn.mat not found for bd estimation — using default bd={DEFAULT_BD}.")
        return DEFAULT_BD

    cn_data = sio.loadmat(str(cn_file))
    Cn = cn_data["Cn"]
    h, w = Cn.shape

    # Interior mean (avoid edges entirely for reference)
    interior_strip = 40
    interior_mean = float(np.mean(Cn[interior_strip:-interior_strip,
                                     interior_strip:-interior_strip]))

    # Walk inward from each edge and find where the row/column mean drops
    # to within 20% above the interior mean
    threshold = interior_mean * 1.20

    def find_border(strip_means):
        for i, val in enumerate(strip_means):
            if val <= threshold:
                return max(i - 1, 0)
        return DEFAULT_BD

    top_means    = [float(np.mean(Cn[i, :])) for i in range(min(50, h // 4))]
    bottom_means = [float(np.mean(Cn[h-1-i, :])) for i in range(min(50, h // 4))]
    left_means   = [float(np.mean(Cn[:, i])) for i in range(min(50, w // 4))]
    right_means  = [float(np.mean(Cn[:, w-1-i])) for i in range(min(50, w // 4))]

    bd_candidates = [
        find_border(top_means),
        find_border(bottom_means),
        find_border(left_means),
        find_border(right_means),
    ]

    # Use the maximum (most conservative) — ensures all motion artifact bands are excluded
    bd = max(bd_candidates)

    # If the edge analysis returns a very small value it means there was no
    # detectable high-correlation border — the data doesn't show evidence of a
    # large motion artifact zone near the edges.  Use bd=10 as a minimal
    # safety margin rather than DEFAULT_BD (which is sized for data that
    # visibly has a motion border).
    if bd < 10:
        log(f"  [PARAMS] Edge analysis returned bd={bd} (no clear motion border detected). "
            f"Falling back to bd=10 (no visible border means less boundary to exclude).")
        return 10

    # Clamp to sane range [10, 40]
    bd = int(np.clip(bd, 10, 40))

    log(f"  [PARAMS] Estimated bd={bd} from Cn edge analysis "
        f"(candidates per edge: {bd_candidates}).")
    return bd


def estimate_all_params(session_name: str, session_dir: Path, log,
                        image_dir: Path = None,
                        bootstrap_mode: bool = False) -> dict:
    """
    Main entry point. Returns all five hyperparameters as a dict.

    Note: Cn.mat and pnr.mat must already exist in session_dir (or image_dir).
    The headless script saves them after the correlation step, so the agent
    runs a preliminary MATLAB call first (see run_cnmfe.py).

    image_dir: if provided, Cn.mat and pnr.mat are loaded from here instead of
    session_dir.  Used by bootstrap_preagent.py when the pre-pass is run into
    a temporary directory.
    """
    if image_dir is None:
        image_dir = session_dir
    gSig, gSiz = get_gsig_gsiz(session_name, log)
    min_corr, min_pnr = estimate_min_corr_min_pnr(image_dir, log,
                                                   bootstrap_mode=bootstrap_mode)
    bd = estimate_bd(image_dir, log)

    # Apply any per-session overrides from animal_params.json.
    # Add a "session_overrides" dict under the animal entry to hardcode params
    # for specific sessions (e.g. over-seeding cases needing a higher min_corr).
    animal_id = parse_animal_id(session_name)
    if animal_id:
        config = load_animal_config()
        overrides = config.get(animal_id, {}).get("session_overrides", {}).get(session_name, {})
        for key, val in overrides.items():
            if key == "min_corr":
                log(f"  [PARAMS] Session override min_corr={val} (estimated {min_corr}).")
                min_corr = val
            elif key == "min_pnr":
                log(f"  [PARAMS] Session override min_pnr={val} (estimated {min_pnr}).")
                min_pnr = val
            elif key == "gSig":
                log(f"  [PARAMS] Session override gSig={val} (estimated {gSig}).")
                gSig = val
            elif key == "gSiz":
                log(f"  [PARAMS] Session override gSiz={val} (estimated {gSiz}).")
                gSiz = val

    params = {
        "gSig": gSig,
        "gSiz": gSiz,
        "min_corr": min_corr,
        "min_pnr": min_pnr,
        "bd": bd,
    }
    log(f"  [PARAMS] Final parameters: {params}")
    return params
