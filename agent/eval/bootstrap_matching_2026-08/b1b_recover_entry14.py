"""
b1b_recover_entry14.py — recover the failed pilot entry (bla7-778um permissive)
from its preserved _bootstrap/ output, without re-running the 145-min CNMFe.

The run failed only at step 4: the >2GB spatial_footprints.mat save stub.
With the A.txt fallback now in bootstrap_preagent._load_candidates, this just
re-runs footprint extraction (15 s of MATLAB) + matching + save into the same
shadow dir. Session dir untouched (snapshot-verified like the pilot).
"""

import json
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENT_DIR))

import bootstrap_preagent as bp
from b1_run_pilot import snapshot, DATA, DIAG_ROOT

rel = "BLA/3odor/AVG5x-TSeries-042125-bla7-778um-27z-000"
session_dir = DATA / Path(rel)
shadow = DIAG_ROOT / "BLA" / "3odor" / session_dir.name / "permissive"
bootstrap_dir = shadow / "_bootstrap"
assert bootstrap_dir.is_dir(), f"missing {bootstrap_dir}"

before = snapshot(session_dir)
t0 = time.time()

res = bp._extract_final_footprints(session_dir, work_dir=shadow)
assert res is not None, "footprint extraction failed"
A_final, gSig, gSiz = res
print(f"extracted {A_final.shape[1]} curated neurons (gSig={gSig}, gSiz={gSiz})")

# run_params for the JSON: reuse what the pilot logged for this entry
run_params = {"gSig": gSig, "gSiz": gSiz, "min_corr": 0.30, "min_pnr": 4.0}
ok = bp._match_and_save(session_dir, bootstrap_dir, A_final,
                        out_dir=shadow, keep_candidates=True,
                        run_params=run_params)
after = snapshot(session_dir)
changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
print(f"match_and_save ok={ok} in {(time.time() - t0) / 60:.1f} min; "
      f"session dir untouched: {not changed}")
if changed:
    print("!! changes:", sorted(changed))

if ok:
    with open(shadow / "bootstrap_match_stats.json") as f:
        s = json.load(f)
    print(f"n_candidates={s['n_candidates']} n_curated={s['n_curated']} "
          f"n_matched={s['n_matched']}")
    print("recovery_by_threshold:", s["recovery_by_threshold"])
sys.exit(0 if ok else 1)
