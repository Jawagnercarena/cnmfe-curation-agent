"""
Step 3f: post-retrain verification, run inside the freeze before the watcher
restarts.  Port of ../step4_2026-08/verify_deploy.py for vCA1, with the BLA
literals replaced by computed values.

  1. joblib contract     model_type, widths, companion first-pass model,
                         reject_threshold == the decided T, feature_version,
                         n_sessions == the pinned manifest, and
                         n_excluded_ambiguous == the mask COMPUTED from the pool
                         (verify_deploy.py:70 hard-codes BLA's 1123).
  2. source flips        config_vCA1.FEATURE_VERSION == 2, BOOTSTRAP_V2B == the
                         decided arm, and -- the one that silently undoes a
                         deploy -- train_classifier_vCA1._VALIDATED_THRESHOLD ==
                         T.  The watcher's auto-retrain passes no --threshold
                         (watcher.py:521-522), so if that constant still reads
                         0.05 the first reviewer return quietly redeploys at the
                         old threshold.
  3. corpus identity     every live npz sha256 == what the gate evaluated, and
                         first-13 / auto_rejected / n_candidates identical to the
                         v1 backup.
  4. no retrain loop     joblib newer than every labels.mat.

  python verify_vca1.py                 full check (deploy day)
  python verify_vca1.py --rehearsal     check the rehearsal joblib instead, and
                                        skip the corpus/flip checks -- proves this
                                        script's own logic before the freeze
Exit 0 = all pass.  Read-only.
"""
import argparse
import json
import re
import sys

import joblib
import numpy as np

import vca1_common as vc

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  FAIL  {msg}")


def check(cond, msg):
    (print(f"  ok    {msg}") if cond else fail(msg))
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args()
    vc.configure()

    dec = json.loads((vc.SP / "gate_decision.json").read_text())
    T, arm = dec["chosen_T"], dec["arm"]
    W = dec.get("agent_weight") or vc.AGENT_WEIGHT
    manifest = json.loads((vc.SP / "vca1_pool_manifest.json").read_text())
    model_path = (vc.REHEARSAL / "model" / "classifier.joblib" if args.rehearsal
                  else vc.JOBLIB_LIVE)
    print(f"verifying {model_path}")
    print(f"decision: arm {arm}, weight {W:g}, T {T}\n")

    # ---- 1. joblib contract ----
    print("1. joblib contract")
    m = joblib.load(str(model_path))
    check(m.get("model_type") == "xgboost", f"model_type = {m.get('model_type')}")
    check(m["scaler"].n_features_in_ == 35, f"scaler width = {m['scaler'].n_features_in_}")
    check("first_pass_scaler" in m and "first_pass_clf" in m
          and m["first_pass_scaler"].n_features_in_ == 13,
          "companion first-pass model present at width 13")
    check(abs(m.get("reject_threshold", -1) - T) < 1e-9,
          f"reject_threshold = {m.get('reject_threshold')} (decided {T})")
    check(m.get("feature_version") == 2, f"feature_version = {m.get('feature_version')}")
    check(m.get("n_features") == 35, f"n_features = {m.get('n_features')}")
    check(abs(float(m.get("agent_weight", -1)) - W) < 1e-9,
          f"agent_weight = {m.get('agent_weight')} (decided {W:g})")
    check(m.get("n_sessions") == len(manifest),
          f"n_sessions = {m.get('n_sessions')} (pinned manifest {len(manifest)})")

    records = vc.load_pool(v2=False) if args.rehearsal else vc.load_pool(v2=False)
    computed_mask = sum(int((r["w"] == 0).sum()) for r in records if r["is_bootstrap"])
    check(m.get("n_excluded_ambiguous") == computed_mask,
          f"n_excluded_ambiguous = {m.get('n_excluded_ambiguous')} "
          f"(computed from the pool: {computed_mask})")

    # ---- 2. source flips ----
    print("\n2. source flips")
    if args.rehearsal:
        print("  --    skipped (--rehearsal: the flips are part of the deploy commit)")
    else:
        import config_vCA1
        check(getattr(config_vCA1, "FEATURE_VERSION", 1) == 2,
              f"config_vCA1.FEATURE_VERSION = {getattr(config_vCA1, 'FEATURE_VERSION', 1)}")
        check(getattr(config_vCA1, "BOOTSTRAP_V2B", None) == arm,
              f"config_vCA1.BOOTSTRAP_V2B = {getattr(config_vCA1, 'BOOTSTRAP_V2B', None)!r} "
              f"(decided {arm!r})")
        src = (vc.AGENT / "train_classifier_vCA1.py").read_text(encoding="utf-8")
        mt = re.search(r"_VALIDATED_THRESHOLD\s*=\s*([0-9.]+)", src)
        check(mt is not None and abs(float(mt.group(1)) - T) < 1e-9,
              f"train_classifier_vCA1._VALIDATED_THRESHOLD = "
              f"{mt.group(1) if mt else 'NOT FOUND'} (must be {T}, else the first "
              f"watcher auto-retrain silently reverts)")

    # ---- 3. corpus identity ----
    print("\n3. corpus identity")
    swap_rep = vc.SP / "swap_report.json"
    if args.rehearsal or not swap_rep.exists():
        print("  --    skipped (no swap_report.json: the swap has not run)")
    else:
        rep = json.loads(swap_rep.read_text())
        bkman = json.loads((vc.BK / "backup_manifest.json").read_text())
        from swap_vca1 import sha256
        n_sha = n_id = 0
        for rel, e in sorted(rep.items()):
            sd = vc.DATA_ROOT / rel
            if sha256(sd / vc.V1) != e["sha256"]:
                fail(f"{rel}: live npz != the bytes the gate evaluated")
                continue
            n_sha += 1
            live = np.load(sd / vc.V1, allow_pickle=True)
            back = np.load(vc.BK / bkman["sessions"][rel]["backup"], allow_pickle=True)
            if (np.array_equal(live["feature_matrix"][:, :13], back["feature_matrix"])
                    and np.array_equal(live["auto_rejected"], back["auto_rejected"])
                    and live["n_candidates"][0] == back["n_candidates"][0]):
                n_id += 1
            else:
                fail(f"{rel}: first-13 / auto_rejected / n_candidates differ from the v1 backup")
        check(n_sha == len(rep), f"sha256 matches the evaluated corpus: {n_sha}/{len(rep)}")
        check(n_id == len(rep), f"first-13 + auto_rejected + n_candidates preserved: "
                                f"{n_id}/{len(rep)}")

    # ---- 4. retrain loop ----
    print("\n4. watcher retrain trigger")
    newest = max((sd / "labels.mat").stat().st_mtime
                 for sd in vc.classify_sessions()[0] + vc.classify_sessions()[1])
    jl = model_path.stat().st_mtime
    check(jl > newest, f"joblib newer than every labels.mat ({jl - newest:+.0f} s)")

    print(f"\nVERIFY: {'ALL PASS' if ok else 'FAILURES -- do not restart the watcher'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
