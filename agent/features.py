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


# ===========================================================================
# v2 (35-column) feature contract — BLA feature expansion Step 4.
#
# Everything below is ADDITIVE.  extract_all and every function above are the
# v1 contract and must stay byte-identical: areas at FEATURE_VERSION 1
# (vCA1, DG_AL) never enter this section.  The v2 contract is
#     13 base | 13 within-session percentile ranks | 8 v2b | v2_present
# assembled positionally in that order (the deployed joblib stores no names).
#
# The v2b computation is a port of the evaluated reference implementation
# (agent/eval/step2_2026-08/compute_v2_features.py + compute_v2b_features.py).
# Its numerical parity against the pinned evaluation values
# (D:\Julian_CNMFe\BLA\.feature_expansion\_pinned\step2_v2b_features.npz) is a
# deploy invariant checked by agent/eval/step4_2026-08/parity_check.py — do
# not change constants or operation order here without re-running that check.
#
# Neighbor high-confidence sets enter ONLY as the hiconf_mask argument; the
# caller chooses the score source (production: the companion 13-column
# first-pass model in the deployed joblib at >= HICONF_SCORE; historical
# backfill: the pinned grouped-OOF scores).
# ===========================================================================

from scipy.stats import rankdata

V2B_NAMES = ["ev_rate", "ev_snr", "ev_template_corr", "ev_asym",
             "ev_frac_plausible", "nb_corr_max", "nb_corr_any", "ring_contrast"]

NB_DIST      = 60.0   # px, neighbor search radius (centroid distance)
HICONF_SCORE = 0.5    # first-pass score for a "high-confidence" neighbor

# v2b transient detector: noise sigma from the differenced trace, detection on
# a 3-frame smoothed trace at baseline + 3.5 sigma, events must reach peak
# z >= 5 within 15 frames, decay capped at 60 frames, template needs >= 3
# qualified events.
_V2B_REFRACT   = 4
_V2B_K         = 3.5
_V2B_PEAK_Z    = 5.0
_V2B_PEAK_WIN  = 15
_V2B_DECAY_CAP = 60
_V2B_SNIP_PRE  = 2
_V2B_SNIP_POST = 25


def v2_feature_names(base_names: list[str]) -> list[str]:
    """The 35 column names, in contract order."""
    return (list(base_names)
            + [f"rank_{n}" for n in base_names]
            + list(V2B_NAMES)
            + ["v2_present"])


def compute_ranks(X: np.ndarray) -> np.ndarray:
    """Within-session percentile ranks per column (deploy: the full candidate
    set of one session).  Matches the Step 2 evaluation exactly."""
    return rankdata(X, axis=0, method="average") / len(X)


def event_features_b(x: np.ndarray) -> list[float]:
    """[ev_rate, ev_snr, ev_template_corr, ev_asym, ev_frac_plausible] for one
    trace under the v2b shape-qualified detector."""
    T = len(x)
    xs = np.convolve(x, np.ones(3) / 3, mode="same")
    base_ = float(np.median(xs))
    dmad = float(np.median(np.abs(np.diff(x))))
    sig = 1.4826 * dmad / np.sqrt(2)
    if sig <= 0:
        sig = float(x.std()) or 1.0

    thr = base_ + _V2B_K * sig
    above = xs > thr
    ons = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(ons):
        keep = [ons[0]]
        for o in ons[1:]:
            if o - keep[-1] >= _V2B_REFRACT:
                keep.append(o)
        ons = np.asarray(keep)

    # qualify events by peak height
    events = []
    for o in ons:
        pk_end = min(o + _V2B_PEAK_WIN, T)
        pk = o + int(np.argmax(xs[o:pk_end]))
        z = (xs[pk] - base_) / sig
        if z >= _V2B_PEAK_Z:
            events.append((o, pk, z))

    n = len(events)
    ev_rate = n / T * 1000.0
    if n == 0:
        return [ev_rate, 0.0, 0.0, 0.0, 0.0]

    snips, peaks_z, asyms, plaus = [], [], [], []
    for o, pk, z in events:
        peaks_z.append(z)
        rise = max(pk - o, 1)
        half = base_ + 0.5 * (xs[pk] - base_)
        dec_end = min(pk + _V2B_DECAY_CAP, T)
        below = np.where(xs[pk:dec_end] < half)[0]
        decay = int(below[0]) if len(below) else dec_end - pk
        decay = max(decay, 1)
        asyms.append(decay / rise)
        plaus.append(1.0 if (rise <= 6 and decay >= rise) else 0.0)
        s, e = o - _V2B_SNIP_PRE, o + _V2B_SNIP_POST + 1
        if s >= 0 and e <= T:
            snips.append(xs[s:e] - base_)

    tmpl_corr = 0.0
    if len(snips) >= 3:
        S = np.array(snips)
        tmpl = S.mean(axis=0)
        cs = [np.corrcoef(row, tmpl)[0, 1] for row in S
              if row.std() > 0 and tmpl.std() > 0]
        if cs:
            tmpl_corr = float(np.mean(cs))
    return [ev_rate, float(np.median(peaks_z)), tmpl_corr,
            float(np.median(asyms)), float(np.mean(plaus))]


def _v2b_centroid_and_mask(img: np.ndarray):
    """Weight centroid (y, x) and half-max mask of one (H, W) footprint."""
    w = img.sum()
    if w <= 0:
        return (np.nan, np.nan), img > 0
    d1, d2 = img.shape
    ys, xs = np.mgrid[0:d1, 0:d2]
    cent = (float((ys * img).sum() / w), float((xs * img).sum() / w))
    thr = 0.5 * img.max()
    return cent, img >= thr


def _v2b_ring_contrast(mask: np.ndarray, Cn) -> float:
    """(mean Cn in footprint) - (mean Cn in 3-8 px ring), in Cn sd units."""
    if Cn is None or mask.sum() == 0 or not np.isfinite(Cn).any():
        return 0.0
    inner = ndi.binary_dilation(mask, iterations=3)
    outer = ndi.binary_dilation(mask, iterations=8)
    ring = outer & ~inner
    if ring.sum() == 0:
        return 0.0
    sd = np.nanstd(Cn)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float((np.nanmean(Cn[mask]) - np.nanmean(Cn[ring])) / sd)


def compute_v2b_features(traces: np.ndarray, footprints: np.ndarray, Cn,
                         hiconf_mask: np.ndarray) -> np.ndarray:
    """
    The 8 v2b candidate-level features for one session's candidate set.

    traces      : (N, T) raw traces (candidate order)
    footprints  : (N, H, W) spatial footprints, same order
    Cn          : (H, W) correlation image, or None (ring_contrast -> 0)
    hiconf_mask : (N,) bool — which candidates count as high-confidence
                  neighbors for nb_corr_max (score source is the caller's)

    Returns (N, 8) in V2B_NAMES order.
    """
    n = traces.shape[0]
    cents = np.full((n, 2), np.nan)
    masks = []
    for i in range(n):
        c, m = _v2b_centroid_and_mask(footprints[i])
        cents[i] = c
        masks.append(m)

    # z-normalized traces for fast pairwise correlation
    Cz = traces - traces.mean(axis=1, keepdims=True)
    sd_ = Cz.std(axis=1, keepdims=True)
    sd_[sd_ == 0] = 1.0
    Cz = Cz / sd_
    corr = (Cz @ Cz.T) / traces.shape[1]

    dist = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    with np.errstate(invalid="ignore"):
        near = (dist <= NB_DIST) & ~np.eye(n, dtype=bool)
    hi = np.asarray(hiconf_mask, dtype=bool)

    X = np.zeros((n, len(V2B_NAMES)))
    for i in range(n):
        X[i, :5] = event_features_b(traces[i].astype(float))
        nb_any = near[i]
        nb_hi = near[i] & hi
        X[i, 5] = float(corr[i, nb_hi].max()) if nb_hi.any() else 0.0
        X[i, 6] = float(corr[i, nb_any].max()) if nb_any.any() else 0.0
        X[i, 7] = _v2b_ring_contrast(masks[i], Cn)
    return X


def assemble_v2_matrix(X13: np.ndarray, v2b: np.ndarray, flag) -> np.ndarray:
    """13 | ranks(13) | v2b(8) | v2_present — the 35-column contract.
    flag: scalar or (N,) — 1 where v2b holds real values, 0 where zero-filled."""
    flag_col = np.asarray(flag, dtype=float)
    if flag_col.ndim == 0:
        flag_col = np.full(len(X13), float(flag_col))
    return np.hstack([X13, compute_ranks(X13), v2b, flag_col.reshape(-1, 1)])


def assemble_v2_bootstrap(X13: np.ndarray) -> np.ndarray:
    """Bootstrap sessions have no recoverable candidate traces: ranks are real
    (pure functions of the 13), v2b columns are zeros, v2_present = 0."""
    zeros = np.zeros((len(X13), len(V2B_NAMES)))
    return assemble_v2_matrix(X13, zeros, 0.0)


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
