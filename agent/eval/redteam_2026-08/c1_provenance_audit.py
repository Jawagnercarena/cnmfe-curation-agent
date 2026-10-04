"""C1: provenance audit of review_neuron.mat for all 79 step2 sessions.
Read-only. Checks, per session:
  1. mtime(review_neuron.mat) < mtime(labels.mat)  (curation strictly before review)
  2. N_extract (C_raw rows in .feature_expansion mat) == n_candidates - n_auto_rejected
  3. len(labels in labels.mat) == N_extract  (labels cover exactly the review set)
  4. pinned v2 idx array == recomputed review_indices (order + content)
"""
import datetime as dt
from pathlib import Path

import numpy as np
import scipy.io as sio

BASE = Path(r"D:\Julian_CNMFe\BLA")
EXT = BASE / ".feature_expansion"
PIN = EXT / "_pinned"

sessions = [l.strip() for l in (PIN / "step2_sessions.txt").read_text().splitlines()
            if l.strip()]
v2 = np.load(PIN / "step2_v2_features.npz", allow_pickle=True)

print(f"{len(sessions)} sessions listed")
viol = []
rows = []
for rel in sessions:
    sd = BASE / rel
    rn_f, lab_f = sd / "review_neuron.mat", sd / "labels.mat"
    ext_f = EXT / (rel.replace("/", "__") + ".mat")
    npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
    n_cand = int(npz["n_candidates"][0])
    auto_rej = set(int(i) for i in npz["auto_rejected"].flatten())
    exp_n = n_cand - len(auto_rej)

    rn_mt = rn_f.stat().st_mtime if rn_f.exists() else None
    lab_mt = lab_f.stat().st_mtime if lab_f.exists() else None

    # N from the extraction (whosmat: header-only, no data load)
    n_ext = None
    for name, shape, _ in sio.whosmat(str(ext_f)):
        if name == "C_raw":
            n_ext = shape[0]

    lab = sio.loadmat(str(lab_f))
    n_lab = lab["labels"].size

    key = rel.replace("/", "__")
    idx = v2[key + "__idx"]
    review_indices = np.array([i for i in range(n_cand) if i not in auto_rej])
    idx_ok = (len(idx) == len(review_indices)) and (idx == review_indices).all()

    probs = []
    if rn_mt is None:
        probs.append("review_neuron MISSING")
    if lab_mt is None:
        probs.append("labels MISSING")
    if rn_mt and lab_mt and rn_mt >= lab_mt:
        probs.append(f"MTIME INVERSION rn={dt.datetime.fromtimestamp(rn_mt)} "
                     f">= lab={dt.datetime.fromtimestamp(lab_mt)}")
    if n_ext != exp_n:
        probs.append(f"N MISMATCH extract={n_ext} expected={exp_n}")
    if n_lab != exp_n:
        probs.append(f"LABELS-N MISMATCH labels={n_lab} expected={exp_n}")
    if not idx_ok:
        probs.append("PINNED IDX != recomputed review_indices")

    rows.append((rel, rn_mt, lab_mt, n_cand, len(auto_rej), n_ext, n_lab))
    if probs:
        viol.append((rel, probs))
        print(f"VIOLATION {rel}: {'; '.join(probs)}")

ok = len(sessions) - len(viol)
print(f"\n{ok}/{len(sessions)} sessions pass all checks")
if viol:
    print(f"{len(viol)} VIOLATIONS — C1 at risk, chase each above")
else:
    dts = [(l - r) / 3600 for _, r, l, *_ in rows]
    print(f"review_neuron -> labels gap: min {min(dts):.1f} h, "
          f"median {sorted(dts)[len(dts)//2]:.1f} h, max {max(dts):.1f} h")
    rn_dates = sorted(dt.datetime.fromtimestamp(r) for _, r, *_ in rows)
    print(f"review_neuron mtimes span {rn_dates[0]} .. {rn_dates[-1]}")
