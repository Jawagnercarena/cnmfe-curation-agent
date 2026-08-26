"""
Step 3b.3: the vCA1 v2b REFERENCE values -- the independent target that the
shipping features.compute_v2b_features must reproduce (parity_vca1.py phase 1).

The functions below are VENDORED VERBATIM from the evaluated Step 2 reference
implementation (../step2_2026-08/compute_v2_features.py centroids_and_masks:134,
ring_contrast:150, the per-session loop:188-209, and compute_v2b_features.py
event_features_b:24-82 with K=3.5 / PEAK_Z=5 / PEAK_WIN=15 / DECAY_CAP=60 /
SNIP_PRE=2 / SNIP_POST=25 / REFRACT=4).  They are copied rather than imported
because compute_v2_features.py loads the BLA joblib and a BLA baseline_oof.npz
at module scope, which would both fail here and re-point paths at BLA.

Copying is the point: parity means two independent code paths agreeing.  If the
shipping code and the reference were the same object the check would be vacuous.
Do not "simplify" anything in this file to match features.py.

Writes PIN/vca1_v2b_reference.npz: {key}__X (n_review, 8), {key}__idx, feature_names.
Read-only.
"""
import sys

import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation

import vca1_common as vc

# --- reference constants (compute_v2_features.py:43-46, compute_v2b_features.py:20-21) ---
REFRACT = 4
NB_DIST = 60.0
HICONF = 0.5
K, PEAK_Z, PEAK_WIN, DECAY_CAP = 3.5, 5.0, 15, 60
SNIP_PRE, SNIP_POST = 2, 25

V2_NAMES = ["ev_rate", "ev_snr", "ev_template_corr", "ev_asym",
            "ev_frac_plausible", "nb_corr_max", "nb_corr_any", "ring_contrast"]


def event_features_b(x):
    T = len(x)
    xs = np.convolve(x, np.ones(3) / 3, mode="same")
    base_ = float(np.median(xs))
    dmad = float(np.median(np.abs(np.diff(x))))
    sig = 1.4826 * dmad / np.sqrt(2)
    if sig <= 0:
        sig = float(x.std()) or 1.0

    thr = base_ + K * sig
    above = xs > thr
    ons = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(ons):
        keep = [ons[0]]
        for o in ons[1:]:
            if o - keep[-1] >= REFRACT:
                keep.append(o)
        ons = np.asarray(keep)

    events = []
    for o in ons:
        pk_end = min(o + PEAK_WIN, T)
        pk = o + int(np.argmax(xs[o:pk_end]))
        z = (xs[pk] - base_) / sig
        if z >= PEAK_Z:
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
        dec_end = min(pk + DECAY_CAP, T)
        below = np.where(xs[pk:dec_end] < half)[0]
        decay = int(below[0]) if len(below) else dec_end - pk
        decay = max(decay, 1)
        asyms.append(decay / rise)
        plaus.append(1.0 if (rise <= 6 and decay >= rise) else 0.0)
        s, e = o - SNIP_PRE, o + SNIP_POST + 1
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
    vc.configure()
    hic = np.load(vc.PIN / "hiconf_scores.npz", allow_pickle=True)
    rels = [r for r in (vc.SP / "vca1_extract_sessions.txt").read_text().splitlines() if r.strip()]
    print(f"reference v2b for {len(rels)} labeled agent sessions")

    out = {"feature_names": np.array(V2_NAMES)}
    for rel in rels:
        sd = vc.DATA_ROOT / rel
        k = vc.key(rel)
        m = sio.loadmat(str(vc.EXT / (k + ".mat")))
        C = m["C_raw"].astype(float)
        A = m["A"]
        Cn = m["Cn"]
        d1, d2 = int(m["d1"][0][0]), int(m["d2"][0][0])

        npz = np.load(sd / vc.V1, allow_pickle=True)
        n_cand = int(npz["n_candidates"][0])
        auto = set(int(i) for i in npz["auto_rejected"])
        review_idx = np.array([i for i in range(n_cand) if i not in auto])
        assert len(review_idx) == C.shape[0], f"{rel}: review set {len(review_idx)} != C rows {C.shape[0]}"

        scores = hic[rel + "__scores"]
        assert len(scores) == n_cand, f"{rel}: hiconf covers {len(scores)} of {n_cand}"

        n = C.shape[0]
        cents, masks = centroids_and_masks(A, d1, d2)

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
            X[i, :5] = event_features_b(C[i].astype(float))
            nb_any = near[i]
            nb_hi = near[i] & hi
            X[i, 5] = float(corr[i, nb_hi].max()) if nb_hi.any() else 0.0
            X[i, 6] = float(corr[i, nb_any].max()) if nb_any.any() else 0.0
            X[i, 7] = ring_contrast(masks[i], Cn)

        out[k + "__X"] = X
        out[k + "__idx"] = review_idx
        print(f"  ok {rel}: {n} reviewed candidates, {int(hi.sum())} hi-conf neighbours "
              f"(source {hic[rel + '__source']})", flush=True)

    np.savez(vc.PIN / "vca1_v2b_reference.npz", **out)
    print(f"\nwrote {vc.PIN / 'vca1_v2b_reference.npz'} ({len(rels)} sessions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
