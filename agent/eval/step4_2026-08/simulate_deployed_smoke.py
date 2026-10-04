"""
Pre-freeze simulation of the literal Step 7.6 smoke test: train the exact
model the freeze-window retrain would produce (35-col final + 13-col
companion, full arm-b corpus from the parallel v2 files, deployed weights,
xgboost random_state=42) and report the deployed in-sample scores of the
bla21 autopsy cells, next to their deploy-realistic OOF analogs.

Read-only on sessions; models stay in memory.
"""
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(SP))

import train_classifier as tc
from threshold_sweep_v2 import load_v2_records, SMOKE_REL, SMOKE_IDX

records, n_ag, n_bs = load_v2_records()
agent_weight = float(max(np.sqrt(n_bs / n_ag), tc.MIN_AGENT_WEIGHT))
X_all = np.vstack([r["X"] for r in records])
y_all = np.concatenate([r["y"] for r in records]).astype(float)
w_all = np.concatenate([r["w"] * (agent_weight if not r["is_bootstrap"] else 1.0)
                        for r in records])
print(f"corpus: {len(records)} sessions, {len(y_all)} rows, "
      f"agent_weight {agent_weight:.2f}")

scaler35, clf35 = tc.train_model(X_all, y_all, sample_weight=w_all,
                                 model_type="xgboost")
scaler13, clf13 = tc.train_model(X_all[:, :13], y_all, sample_weight=w_all,
                                 model_type="xgboost")

smoke = next(r for r in records if r["name"] == SMOKE_REL)
s35 = clf35.predict_proba(scaler35.transform(smoke["X"]))[:, 1]
s13 = clf13.predict_proba(scaler13.transform(smoke["X"][:, :13]))[:, 1]
print(f"\n{SMOKE_REL} — deployed IN-SAMPLE scores under the retrain-identical "
      f"models:")
for ci in SMOKE_IDX:
    print(f"  'Neuron {ci+1}': 35-col final {s35[ci]:.3f}   "
          f"13-col companion {s13[ci]:.3f}")
print("\n(reference: deploy-realistic OOF under rankv2b_35 = 0.065 / 0.052; "
      "old-model OOF = 0.064 / 0.101; old deployed threshold 0.12, "
      "proposed new threshold 0.06)")
