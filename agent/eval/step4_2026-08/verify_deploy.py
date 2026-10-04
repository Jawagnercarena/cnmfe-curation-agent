"""
Step 7 post-retrain verification (inside the freeze, before the watcher
restarts).  Checks, in order:

  1. joblib contract: model_type=xgboost, scaler width 35, companion
     first-pass present at width 13, reject_threshold == the Step 5 chosen T,
     feature_version 2, n_features 35, n_sessions / n_excluded_ambiguous
     match the pinned pool.
  2. corpus: every swapped candidate_features.npz byte-identical to what the
     Step 5 sweep evaluated (sha256 vs swap_report.json) — this transfers the
     Step 5 threshold table and reference eval to the live corpus without
     re-running them inside the freeze; first 13 columns + auto_rejected +
     n_candidates bit-identical to the _v1_backup of every session.
  3. deployed smoke: the bla21 autopsy cells under the NEW deployed joblib on
     the live npz (expected ~0.79/0.61 in-sample per the pre-freeze
     simulation; their deploy-realistic OOF analogs 0.065/0.052 are reported
     alongside and are the honest number).
  4. no-retrain-loop: joblib mtime newer than every labels.mat.

Exit 0 = all pass.  Read-only.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

SP = Path(__file__).parent
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")
BK = Path(r"D:\Julian_CNMFe\BLA\.feature_expansion\_v1_backup")
sys.path.insert(0, str(SP))
from swap_v2 import sha256, V1, SMOKE_REL  # noqa: E402

SMOKE_IDX = [21, 24]
ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def passed(msg):
    print(f"  ok    {msg}")


# ---- 1. joblib ----
res5 = json.loads((SP / "step5_results.json").read_text())
T = res5["chosen_T"]
m = joblib.load(AGENT / "model" / "BLA" / "classifier.joblib")
print("1. joblib contract")
(passed if m.get("model_type") == "xgboost" else fail)(
    f"model_type = {m.get('model_type')}")
(passed if m["scaler"].n_features_in_ == 35 else fail)(
    f"scaler width = {m['scaler'].n_features_in_}")
fp_ok = ("first_pass_scaler" in m and "first_pass_clf" in m
         and m["first_pass_scaler"].n_features_in_ == 13)
(passed if fp_ok else fail)("companion first-pass present at width 13")
(passed if abs(m.get("reject_threshold", -1) - T) < 1e-9 else fail)(
    f"reject_threshold = {m.get('reject_threshold')} (chosen T = {T})")
(passed if m.get("feature_version") == 2 else fail)(
    f"feature_version = {m.get('feature_version')}")
(passed if m.get("n_features") == 35 else fail)(f"n_features = {m.get('n_features')}")
man4 = json.loads((SP / "step4_pool_manifest.json").read_text())
(passed if m.get("n_sessions") == len(man4) else fail)(
    f"n_sessions = {m.get('n_sessions')} (manifest {len(man4)})")
(passed if m.get("n_excluded_ambiguous") == 1123 else fail)(
    f"n_excluded_ambiguous = {m.get('n_excluded_ambiguous')} (pinned 1123)")

# ---- 2. corpus ----
print("2. corpus identity")
swap_rep = json.loads((SP / "swap_report.json").read_text())
bkman = json.loads((BK / "backup_manifest.json").read_text())
n_sha = n_first13 = 0
for rel, e in sorted(swap_rep.items()):
    sd = DATA_ROOT / rel
    if sha256(sd / V1) != e["sha256"]:
        fail(f"{rel}: live npz != swept bytes")
        continue
    n_sha += 1
    live = np.load(sd / V1, allow_pickle=True)
    back = np.load(BK / bkman["sessions"][rel]["backup"], allow_pickle=True)
    if (np.array_equal(live["feature_matrix"][:, :13], back["feature_matrix"])
            and np.array_equal(live["auto_rejected"], back["auto_rejected"])
            and live["n_candidates"][0] == back["n_candidates"][0]):
        n_first13 += 1
    else:
        fail(f"{rel}: first-13/auto_rejected/n_candidates not identical to v1")
(passed if n_sha == len(swap_rep) else fail)(
    f"sha256 identical to swept corpus: {n_sha}/{len(swap_rep)}")
(passed if n_first13 == len(swap_rep) else fail)(
    f"first-13 + auto_rejected + n_candidates identical to v1 backup: "
    f"{n_first13}/{len(swap_rep)}")

# ---- 3. deployed smoke ----
print("3. deployed smoke (bla21 autopsy cells)")
npz = np.load(DATA_ROOT / SMOKE_REL / V1, allow_pickle=True)
X = npz["feature_matrix"]
s = m["clf"].predict_proba(m["scaler"].transform(X))[:, 1]
for ci in SMOKE_IDX:
    above = s[ci] > T
    (passed if above else fail)(
        f"'Neuron {ci+1}' deployed score {s[ci]:.3f} vs T={T:.2f} "
        f"(in-sample; deploy-realistic OOF analog "
        f"{res5['smoke'][str(ci)]['v2b']:.3f})")

# ---- 4. retrain loop ----
print("4. watcher retrain trigger")
jl_mtime = (AGENT / "model" / "BLA" / "classifier.joblib").stat().st_mtime
newest = max((sd / "labels.mat").stat().st_mtime
             for td in DATA_ROOT.iterdir() if td.is_dir()
             and not td.name.startswith(".")
             for sd in td.iterdir() if sd.is_dir()
             and (sd / "labels.mat").exists())
(passed if jl_mtime > newest else fail)(
    f"joblib newer than every labels.mat ({jl_mtime - newest:+.0f} s)")

print(f"\nVERIFY: {'ALL PASS' if ok else 'FAILURES — do not restart the watcher'}")
sys.exit(0 if ok else 1)
