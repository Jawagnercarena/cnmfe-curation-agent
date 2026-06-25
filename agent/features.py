"""
features.py
Extracts per-neuron features from CNMFe outputs for classification.

Called by curator.py after a headless run completes.
Features are divided into three groups:
  - Spatial:  footprint shape, size, compactness
  - Temporal: trace quality, transient statistics
  - Motion:   correlation of trace with global background (motion proxy)
"""

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
from pathlib import Path


# ---- Loaders ----

def load_spatial(session_dir: Path):
    """
    Load spatial footprints.
    Returns (N, H, W) array — one 2D footprint per neuron.
    """
    sf_file = session_dir / "spatial_footprints.mat"
    if sf_file.exists():
        data = sio.loadmat(str(sf_file))
        return data["spatial_footprints"]   # (N, H, W)

    # Fall back to A.txt if spatial_footprints.mat is missing
    a_file = session_dir / "A.txt"
    A = np.loadtxt(str(a_file))             # (pixels, N)
    n_neurons = A.shape[1]
    n_pixels  = A.shape[0]
    side = int(np.sqrt(n_pixels))
    footprints = np.zeros((n_neurons, side, side))
    for i in range(n_neurons):
        footprints[i] = A[:, i].reshape(side, side)
    return footprints


def load_traces(session_dir: Path) -> np.ndarray:
    """Load C_raw.txt — (N, T) raw calcium traces."""
    return np.loadtxt(str(session_dir / "C_raw.txt"))   # (N, T)


def load_cn(session_dir: Path) -> np.ndarray | None:
    """Load Cn.mat — correlation image (d1, d2). Returns None if not found."""
    cn_file = session_dir / "Cn.mat"
    if not cn_file.exists():
        return None
    data = sio.loadmat(str(cn_file))
    return data["Cn"]


def load_background(session_dir: Path) -> np.ndarray | None:
    """
    Load Ybg_mean.mat — mean background signal per frame saved by CNMFe_Biane_headless.m.
    Returns (T,) array, or None if not found (motion features will be skipped).

    Note: older sessions saved the full Ybg matrix (Ybg.mat) which exceeded
    MATLAB's v5 2GB limit and produced an empty file. Those sessions return None.
    """
    ybg_file = session_dir / "Ybg_mean.mat"
    if not ybg_file.exists():
        return None
    data = sio.loadmat(str(ybg_file))
    return data["mean_Ybg"].flatten()   # (T,) — mean pixel intensity per frame


# ---- Spatial features ----

def spatial_features(footprint: np.ndarray) -> dict:
    """
    Compute shape features for one (H, W) spatial footprint.
    The footprint values are spatial weights (non-negative).
    """
    # Threshold to binary mask at 20% of max weight
    thresh = 0.20 * footprint.max()
    mask = footprint > thresh

    area = float(mask.sum())
    if area == 0:
        return {"area": 0, "circularity": 0, "eccentricity": 1,
                "compactness": 0, "max_weight": 0, "weight_spread": 0}

    # Circularity: 4π * area / perimeter²  (1 = perfect circle, <1 = irregular)
    from skimage.measure import label, regionprops
    labeled = label(mask)
    props = regionprops(labeled, intensity_image=footprint)
    if not props:
        return {"area": 0, "circularity": 0, "eccentricity": 1,
                "compactness": 0, "max_weight": 0, "weight_spread": 0}

    # Use the largest connected region
    main = max(props, key=lambda p: p.area)

    area          = float(main.area)
    perimeter     = max(float(main.perimeter), 1.0)
    circularity   = float(4 * np.pi * area / perimeter ** 2)
    eccentricity  = float(main.eccentricity)   # 0=circle, 1=line

    # Compactness: fraction of footprint weight in the main connected region
    total_weight  = float(footprint.sum())
    region_weight = float(footprint[main.slice].sum()) if total_weight > 0 else 0
    compactness   = region_weight / total_weight if total_weight > 0 else 0

    max_weight    = float(footprint.max())

    # Weight spread: std of nonzero weights (uniformity of the footprint)
    nonzero = footprint[mask]
    weight_spread = float(nonzero.std() / (nonzero.mean() + 1e-9))

    return {
        "area":          area,
        "circularity":   circularity,
        "eccentricity":  eccentricity,
        "compactness":   compactness,
        "max_weight":    max_weight,
        "weight_spread": weight_spread,
    }


# ---- Temporal features ----

def temporal_features(trace: np.ndarray) -> dict:
    """
    Compute quality features for one (T,) calcium trace.
    """
    T = len(trace)

    # Baseline: median of the lower 25% of frames
    baseline_level = float(np.percentile(trace, 25))
    baseline_std   = float(np.std(trace[trace < np.percentile(trace, 50)]))
    noise          = max(baseline_std, 1e-9)

    # Peak SNR
    peak_snr = float((trace.max() - baseline_level) / noise)

    # Transient detection: frames > baseline + 2.5 * noise
    transient_mask = trace > (baseline_level + 2.5 * noise)
    transient_freq = float(transient_mask.sum() / T)   # fraction of frames

    # Number of distinct events (connected runs above threshold)
    labeled, n_events = ndi.label(transient_mask)
    events_per_min = 0.0
    if T > 0:
        # Assume 3.75 Hz frame rate (from CNMFe params); minutes = T / (3.75 * 60)
        duration_min = T / (3.75 * 60)
        events_per_min = n_events / max(duration_min, 1e-3)

    # Baseline stability: std of the baseline region (low = stable = good)
    baseline_frames = trace[~transient_mask]
    baseline_stability = float(baseline_frames.std()) if len(baseline_frames) > 0 else float(trace.std())

    # Skewness of trace (real transients → positive skew)
    mean_t  = float(trace.mean())
    std_t   = float(trace.std())
    skewness = float(np.mean(((trace - mean_t) / (std_t + 1e-9)) ** 3))

    return {
        "peak_snr":           peak_snr,
        "transient_freq":     transient_freq,
        "events_per_min":     events_per_min,
        "baseline_stability": baseline_stability,
        "skewness":           skewness,
    }


# ---- Motion features ----

def motion_features(trace: np.ndarray, background_signal: np.ndarray | None) -> dict:
    """
    Compute correlation of this neuron's trace with the global background signal.
    High correlation → likely motion artifact.
    """
    if background_signal is None or len(background_signal) != len(trace):
        return {"motion_correlation": 0.0}

    # Pearson correlation
    if trace.std() < 1e-9 or background_signal.std() < 1e-9:
        return {"motion_correlation": 0.0}

    corr = float(np.corrcoef(trace, background_signal)[0, 1])
    return {"motion_correlation": corr}


# ---- Cn-footprint correlation ----

def cn_features(footprint: np.ndarray, Cn) -> dict:
    """
    Pearson correlation between spatial footprint weights and the Cn image
    over the footprint's active region.

    Real neurons detected where local pixel correlations are high will have
    footprints that align with the Cn structure. Noise components often have
    footprints that land in low-correlation regions or don't match Cn at all.
    High value → likely real; low/negative value → likely noise or artifact.
    """
    if Cn is None:
        return {"cn_correlation": 0.0}
    thresh = 0.20 * footprint.max()
    mask = footprint > thresh
    if mask.sum() < 5:
        return {"cn_correlation": 0.0}
    fp_vals = footprint[mask]
    cn_vals = Cn[mask]
    if fp_vals.std() < 1e-9 or cn_vals.std() < 1e-9:
        return {"cn_correlation": 0.0}
    return {"cn_correlation": float(np.corrcoef(fp_vals, cn_vals)[0, 1])}


# ---- Overlap features (between neuron pairs) ----

def pairwise_overlap(footprints: np.ndarray) -> np.ndarray:
    """
    Compute spatial overlap ratio between all neuron pairs.
    overlap[i,j] = intersection_area / min(area_i, area_j)
    Returns (N, N) matrix.
    """
    N = footprints.shape[0]
    thresh = 0.20   # binary mask threshold (fraction of each neuron's max)
    masks = np.array([fp > thresh * fp.max() for fp in footprints])  # (N, H, W)

    overlap = np.zeros((N, N))
    areas = masks.sum(axis=(1, 2)).astype(float)   # (N,)

    for i in range(N):
        for j in range(i + 1, N):
            intersection = float((masks[i] & masks[j]).sum())
            min_area = min(areas[i], areas[j])
            if min_area > 0:
                ratio = intersection / min_area
            else:
                ratio = 0.0
            overlap[i, j] = ratio
            overlap[j, i] = ratio

    return overlap


# ---- Main entry point ----

def extract_all(session_dir: Path, log) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Extract features for all neurons in a session.

    Returns:
        feature_matrix : (N, F) array
        feature_names  : list of F feature name strings
        overlap_matrix : (N, N) spatial overlap ratios
    """
    log(f"  [FEATURES] Loading data from {session_dir.name}...")

    footprints = load_spatial(session_dir)       # (N, H, W)
    traces     = load_traces(session_dir)         # (N, T)
    bg_signal  = load_background(session_dir)     # (T,) or None
    Cn         = load_cn(session_dir)             # (H, W) or None

    N = footprints.shape[0]
    assert traces.shape[0] == N, \
        f"Mismatch: {N} footprints but {traces.shape[0]} traces in {session_dir}"

    log(f"  [FEATURES] Computing features for {N} neurons...")

    rows = []
    for i in range(N):
        sf = spatial_features(footprints[i])
        tf = temporal_features(traces[i])
        mf = motion_features(traces[i], bg_signal)
        cf = cn_features(footprints[i], Cn)

        row = {**sf, **tf, **mf, **cf}
        rows.append(row)

    # Build matrix
    feature_names  = list(rows[0].keys())
    feature_matrix = np.array([[r[k] for k in feature_names] for r in rows])

    # Pairwise spatial overlap
    log(f"  [FEATURES] Computing pairwise spatial overlap...")
    overlap_matrix = pairwise_overlap(footprints)

    log(f"  [FEATURES] Done. Feature matrix: {feature_matrix.shape}  "
        f"({len(feature_names)} features × {N} neurons)")

    return feature_matrix, feature_names, overlap_matrix
