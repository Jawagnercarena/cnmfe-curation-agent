"""
Operating-point comparison, computed PER SEED from the saved gate OOF vectors.

Why this exists: gate_vca1.matched_junk computed its metric on the seed-MEAN OOF
vector.  Averaging 8 OOF vectors smooths away the individual seed dips, so the
b13 reference false-AR at T=0.05 evaluated to 0.00% on the mean vector while the
honest per-seed mean is 0.84%.  Both sides then printed 0.00% and the ship
criterion's "false-AR not worse" check was vacuous.  gate_vca1.py is fixed for
future runs; this script re-derives the numbers from the stored OOF fixtures so
no refitting is needed.

Two calibration-fair comparisons, both per seed then averaged:
  * false-AR at MATCHED junk-caught  (hold junk fixed, compare safety)
  * junk-caught at MATCHED false-AR  (hold safety fixed, compare yield)

The second is the one that matters operationally: at a false-AR the lab already
accepts, how much junk does the model take off the reviewers' plate?
"""
import json
import sys

import numpy as np

import vca1_common as vc

GRID = np.arange(0.001, 0.5005, 0.001)


def per_seed(oof_ref, oof_new, y, t_ref):
    pos, neg = y == 1, y == 0
    far_ref, junk_ref, far_new, junk_new = [], [], [], []
    for r, n in zip(oof_ref, oof_new):
        jr = float((r[neg] < t_ref).sum() / neg.sum())
        fr = float((r[pos] < t_ref).sum() / pos.sum())
        junk_ref.append(jr * 100); far_ref.append(fr * 100)
        # new model at matched junk
        t = next((t for t in GRID if (n[neg] < t).sum() / neg.sum() >= jr), None)
        far_new.append(float((n[pos] < t).sum() / pos.sum() * 100) if t else np.nan)
        # new model at matched false-AR (largest T not exceeding the reference FAR)
        tj = None
        for t in GRID:
            if (n[pos] < t).sum() / pos.sum() <= fr:
                tj = t
            else:
                break
        junk_new.append(float((n[neg] < tj).sum() / neg.sum() * 100) if tj else np.nan)
    return (np.mean(far_ref), np.mean(junk_ref),
            np.nanmean(far_new), np.nanmean(junk_new))


def main():
    vc.configure()
    T_REF = vc.DEPLOYED_T
    out = {"t_ref": T_REF, "arms": {}}
    print(f"reference: the deployed 13-column model at its deployed T={T_REF}\n")
    print(f"{'arm':<5} {'b13 FAR':>8} {'b13 junk':>9} | {'v35 FAR':>8} {'v35 junk':>9} | "
          f"{'safety gain':>11} {'yield gain':>10}")
    for arm in (vc.ARM_A, vc.ARM_B0, vc.ARM_B1):
        z = np.load(vc.SP / f"gate_{arm}_w{vc.AGENT_WEIGHT:g}_oof.npz", allow_pickle=True)
        y = z["y"]
        fr, jr, fn, jn = per_seed(z["oof_b13"], z["oof_v35"], y, T_REF)
        out["arms"][arm] = {"far_b13": fr, "junk_b13": jr,
                            "far_v35_at_matched_junk": fn,
                            "junk_v35_at_matched_far": jn,
                            "safety_gain_pp": fr - fn, "yield_gain_pp": jn - jr}
        print(f"{arm:<5} {fr:>7.2f}% {jr:>8.1f}% | {fn:>7.2f}% {jn:>8.1f}% | "
              f"{fr - fn:>+10.2f}pp {jn - jr:>+9.1f}pp")
    print("\n  safety gain = false-AR reduction at the SAME junk-caught")
    print("  yield  gain = extra junk caught at the SAME false-AR")
    (vc.SP / "operating_point_vca1.json").write_text(json.dumps(out, indent=1))
    print("\nwrote operating_point_vca1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
