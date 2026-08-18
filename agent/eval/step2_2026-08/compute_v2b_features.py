"""
v2b: same 8 features as compute_v2_features.py but with a transient detector
built for SHAPE statistics rather than coincidence counting:
  - noise sigma from the differenced trace (robust to transient contamination):
    sigma = 1.4826 * MAD(diff(x)) / sqrt(2)
  - detection on a 3-frame smoothed trace at baseline + 3.5 sigma
  - events must reach peak z >= 5 within [onset, onset+15] or are dropped
  - decay cap 60 frames; template needs >= 3 qualified events
Output: step2_v2b_features.npz (same layout as v2).
"""
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import compute_v2_features as base

K, PEAK_Z, PEAK_WIN, DECAY_CAP = 3.5, 5.0, 15, 60
SNIP_PRE, SNIP_POST = 2, 25


def event_features_b(x):
    T = len(x)
    xs = np.convolve(x, np.ones(3) / 3, mode="same")
    base_ = float(np.median(xs))
    dmad = float(np.median(np.abs(np.diff(x))))
    sig = 1.4826 * dmad / np.sqrt(2)
    if sig <= 0:
        sig = float(x.std()) or 1.0

    # inline detector with our K
    thr = base_ + K * sig
    above = xs > thr
    ons = np.where(above[1:] & ~above[:-1])[0] + 1
    if len(ons):
        keep = [ons[0]]
        for o in ons[1:]:
            if o - keep[-1] >= base.REFRACT:
                keep.append(o)
        ons = np.asarray(keep)

    # qualify events by peak height
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


def main():
    base.event_features = event_features_b  # swap the detector
    # redirect output file
    orig_savez = np.savez

    def savez_b(path, **kw):
        orig_savez(SP / "step2_v2b_features.npz", **kw)
    np.savez = savez_b
    try:
        base.main()
    finally:
        np.savez = orig_savez
    print("v2b saved as step2_v2b_features.npz")


if __name__ == "__main__":
    main()
