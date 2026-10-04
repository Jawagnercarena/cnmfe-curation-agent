"""
Record the CURRENT deployed model's scores on every v1 candidate_features.npz
(labeled + pending) before the swap.  Used for:
  - the rollback rehearsal (a restored v1 state must reproduce these exactly);
  - the before/after report;
  - identifying the two bla21 autopsy cells for the Step 7 smoke test
    (2tones/093025-bla21: the two lowest-scoring REVIEWED REAL cells, expected
    ~0.06 / ~0.10 under the old model, PDF numbering "Neuron 22" / "Neuron 25").

Output: preswap_scores.npz  {rel__scores: (N,), rel__thr: float}  + a printed
smoke-cell table.  Read-only on sessions.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio

SP = Path(__file__).parent
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")
SMOKE_REL = "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"

model = joblib.load(AGENT / "model" / "BLA" / "classifier.joblib")
thr = float(model.get("reject_threshold", 0.10))
print(f"deployed model: {model.get('model_type')}  threshold={thr:.2f}  "
      f"n_sessions={model.get('n_sessions')}")

out = {}
n = 0
for td in sorted(DATA_ROOT.iterdir()):
    if not td.is_dir() or td.name.startswith("."):
        continue
    for sd in sorted(td.iterdir()):
        f = sd / "candidate_features.npz"
        if not sd.is_dir() or not f.exists():
            continue
        npz = np.load(f, allow_pickle=True)
        X = npz["feature_matrix"]
        if X.shape[1] != 13:
            print(f"  UNEXPECTED width {X.shape[1]}: {sd.name} — skipped")
            continue
        s = model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]
        out[f"{td.name}/{sd.name}__scores"] = s
        n += 1
np.savez(SP / "preswap_scores.npz", reject_threshold=np.array([thr]), **out)
print(f"recorded deployed scores for {n} sessions -> preswap_scores.npz")

# ---- smoke-test cell identification ----
sd = DATA_ROOT / SMOKE_REL
npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
scores = out[SMOKE_REL + "__scores"]
auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
n_cand = int(npz["n_candidates"][0])
review_idx = [i for i in range(n_cand) if i not in auto]
y = sio.loadmat(str(sd / "labels.mat"))["labels"].flatten()
assert len(y) == len(review_idx), "labels/review size mismatch"
reals = [(scores[ci], ci) for k, ci in enumerate(review_idx) if y[k] == 1]
reals.sort()
print(f"\n{SMOKE_REL}: {n_cand} candidates, {len(review_idx)} reviewed, "
      f"{int(y.sum())} reals")
print("lowest-scoring REAL cells under the deployed model (smoke-test set):")
for s, ci in reals[:4]:
    print(f"  candidate idx {ci} ('Neuron {ci + 1}'): old score {s:.3f}"
          f"{'   <- below deployed threshold' if s < thr else ''}")
