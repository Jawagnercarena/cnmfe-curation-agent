"""
Step 2: compute candidate-level v2 trace/footprint features from the
review_neuron.mat extractions in D:\\Julian_CNMFe\\BLA\\.feature_expansion\\.

Per REVIEWED candidate (8 features, all deployable at curation time):
  ev_rate            onsets per 1000 frames (detector = July's: median +
                     2.5*robust-sigma rising edges, refractory 4)
  ev_snr             median event peak height in robust sigmas
  ev_template_corr   mean corr of onset-aligned snippets vs the cell's own
                     mean snippet ("does it repeat a stereotyped transient")
  ev_asym            median decay/rise frame ratio (GCaMP: slow decay >> rise)
  ev_frac_plausible  fraction of events with rise<=5 frames and decay>=rise
  nb_corr_max        max trace corr with a HIGH-CONF neighbor (deployed-model
                     score >= 0.5, centroid distance <= 60 px)
  nb_corr_any        max trace corr with ANY neighbor within 60 px
  ring_contrast      (mean Cn in footprint) - (mean Cn in 3-8px ring), / Cn sd

High-conf neighbor sets use the DEPLOYED joblib's scores on the session's own
npz (exactly what the curator would have at deploy time; label-free).
Cells with <2 events get template/asym/plausible = 0 (ev_rate carries the
information that events were absent).

Output: step2_v2_features.npz {sess}__X (n_review x 8), {sess}__idx (full
candidate indices of the review rows). Read-only on session dirs.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util  # noqa: F401  (sys.path setup for agent imports)

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
from config import DATA_ROOT

EXT = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion")
K_ONSET, REFRACT = 2.5, 4
SNIP_PRE, SNIP_POST = 2, 18
NB_DIST = 60.0
HICONF = 0.5

V2_NAMES = ["ev_rate", "ev_snr", "ev_template_corr", "ev_asym",
            "ev_frac_plausible", "nb_corr_max", "nb_corr_any", "ring_contrast"]

_model = joblib.load(AGENT / "model" / "BLA" / "classifier.joblib")

# High-conf neighbor scores: prefer the grouped-OOF means (label-free offline
# analog of the live score a curator would have); deployed in-sample scores
# only for sessions outside the OOF eval pool. Using deployed scores for all
# sessions leaked neighbor LABELS into the offline eval (the deployed model
# was trained on these sessions).
_oof = np.load(SP / "baseline_oof.npz", allow_pickle=True)
_oof_mean = _oof["oof_seeds"].mean(axis=0)
_oof_by_sess = {}
for _i in range(len(_oof_mean)):
    _oof_by_sess.setdefault(str(_oof["session"][_i]), {})[
        int(_oof["idx_in_session"][_i])] = float(_oof_mean[_i])


def deployed_scores(sd):
    npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
    X = npz["feature_matrix"]
    Xs = _model["scaler"].transform(X)
    scores = _model["clf"].predict_proba(Xs)[:, 1]
    rel = f"{sd.parent.name}/{sd.name}"
    if rel in _oof_by_sess:
        m = _oof_by_sess[rel]
        for full_idx, v in m.items():
            scores[full_idx] = v
    return scores, npz


def detect_onsets(x, base, sig):
    thr = base + K_ONSET * sig
    above = x > thr
    ons = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(ons):
        keep = [ons[0]]
        for o in ons[1:]:
            if o - keep[-1] >= REFRACT:
                keep.append(o)
        ons = np.asarray(keep)
    return ons


def event_features(x):
    base = float(np.median(x))
    mad = float(np.median(np.abs(x - base)))
    sig = 1.4826 * mad if mad > 0 else float(x.std()) or 1.0
    T = len(x)
    ons = detect_onsets(x, base, sig)
    n = len(ons)
    ev_rate = n / T * 1000.0
    if n == 0:
        return [ev_rate, 0.0, 0.0, 0.0, 0.0]

    snips, peaks_z, asyms, plaus = [], [], [], []
    for o in ons:
        pk_end = min(o + 10, T)
        pk = o + int(np.argmax(x[o:pk_end]))
        peaks_z.append((x[pk] - base) / sig)
        rise = max(pk - o, 1)
        half = base + 0.5 * (x[pk] - base)
        dec_end = min(pk + 30, T)
        below = np.where(x[pk:dec_end] < half)[0]
        decay = int(below[0]) if len(below) else dec_end - pk
        decay = max(decay, 1)
        asyms.append(decay / rise)
        plaus.append(1.0 if (rise <= 5 and decay >= rise) else 0.0)
        s, e = o - SNIP_PRE, o + SNIP_POST + 1
        if s >= 0 and e <= T:
            snips.append(x[s:e] - base)

    tmpl_corr = 0.0
    if len(snips) >= 2:
        S = np.array(snips)
        tmpl = S.mean(axis=0)
        cs = []
        for row in S:
            if row.std() > 0 and tmpl.std() > 0:
                cs.append(np.corrcoef(row, tmpl)[0, 1])
        if cs:
            tmpl_corr = float(np.mean(cs))
    return [ev_rate, float(np.median(peaks_z)), tmpl_corr,
            float(np.median(asyms)), float(np.mean(plaus))]


def centroids_and_masks(A, d1, d2):
    A = A.tocsc()
    cents, masks = [], []
    for k in range(A.shape[1]):
        col = np.asarray(A[:, k].todense()).ravel()
        img = col.reshape((d1, d2), order="F")
        w = img.sum()
        if w <= 0:
            cents.append((np.nan, np.nan)); masks.append(img > 0); continue
        ys, xs = np.mgrid[0:d1, 0:d2]
        cents.append((float((ys * img).sum() / w), float((xs * img).sum() / w)))
        thr = 0.5 * img.max()
        masks.append(img >= thr)
    return np.array(cents), masks


def ring_contrast(mask, Cn):
    if mask.sum() == 0 or not np.isfinite(Cn).any():
        return 0.0
    inner = binary_dilation(mask, iterations=3)
    outer = binary_dilation(mask, iterations=8)
    ring = outer & ~inner
    if ring.sum() == 0:
        return 0.0
    sd = np.nanstd(Cn)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float((np.nanmean(Cn[mask]) - np.nanmean(Cn[ring])) / sd)


def main():
    sessions = [l.strip() for l in
                (SP / "step2_sessions.txt").read_text().splitlines() if l.strip()]
    out = {}
    n_done = 0
    for rel in sessions:
        mat_f = EXT / (rel.replace("/", "__") + ".mat")
        if not mat_f.exists():
            print(f"MISSING extraction: {rel}")
            continue
        m = sio.loadmat(str(mat_f))
        C = m["C_raw"]                     # n_review x T
        A = m["A"]                         # pixels x n_review (sparse)
        Cn = m["Cn"]
        d1, d2 = int(m["d1"][0][0]), int(m["d2"][0][0])
        sd = DATA_ROOT / rel
        scores, npz = deployed_scores(sd)
        auto_rej = set(int(i) for i in npz["auto_rejected"])
        n_cand = int(npz["n_candidates"][0])
        review_idx = np.array([i for i in range(n_cand) if i not in auto_rej])
        if len(review_idx) != C.shape[0]:
            print(f"COUNT MISMATCH {rel}: review {len(review_idx)} vs C {C.shape[0]} — skipped")
            continue

        n = C.shape[0]
        cents, masks = centroids_and_masks(A, d1, d2)

        # z-normalized traces for fast pairwise corr
        Cz = C - C.mean(axis=1, keepdims=True)
        sd_ = Cz.std(axis=1, keepdims=True)
        sd_[sd_ == 0] = 1.0
        Cz /= sd_
        corr = (Cz @ Cz.T) / C.shape[1]

        dist = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
        near = (dist <= NB_DIST) & ~np.eye(n, dtype=bool)
        hi = scores[review_idx] >= HICONF

        X = np.zeros((n, len(V2_NAMES)))
        for i in range(n):
            X[i, :5] = event_features(C[i].astype(float))
            nb_any = near[i]
            nb_hi = near[i] & hi
            X[i, 5] = float(corr[i, nb_hi].max()) if nb_hi.any() else 0.0
            X[i, 6] = float(corr[i, nb_any].max()) if nb_any.any() else 0.0
            X[i, 7] = ring_contrast(masks[i], Cn)

        out[rel.replace("/", "__") + "__X"] = X
        out[rel.replace("/", "__") + "__idx"] = review_idx
        n_done += 1
        print(f"ok {rel}: {n} reviewed cands")

    np.savez(SP / "step2_v2_features.npz",
             feature_names=np.array(V2_NAMES), **out)
    print(f"\nsaved step2_v2_features.npz for {n_done} sessions")


if __name__ == "__main__":
    main()
