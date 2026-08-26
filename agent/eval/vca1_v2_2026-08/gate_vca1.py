"""
Step 3d: the 8-seed gate for one (arm, agent-weight) combination.

Runs b13 (the first 13 columns) and rankv2b_35 (all 35) through the identical
OOF harness so every delta is paired per seed.  The arm only changes BOOTSTRAP
columns 26-34, which b13 never sees, so b13 must reproduce the pin regardless of
arm -- that is asserted, and it is what makes the arms comparable to each other.

  python gate_vca1.py --arm a          --agent-weight 5.0
  python gate_vca1.py --arm b0         --agent-weight 5.0
  python gate_vca1.py --arm b1         --agent-weight 5.0

Reports, all 8-seed:
  1. AUC full and reviewed, paired delta (mean, min, all-seeds-positive).
  2. b13 consistency against PIN/baseline_oof.npz (allclose 1e-6 + per-seed AUC
     to 1e-4 -- NOT bitwise: xgboost runs multithreaded, and Step 4 only ever
     claimed AUC reproduction to 4 decimals).
  3. false-AR at matched junk-caught (the calibration-fair operating comparison).
  4. threshold table + the Step 4 rule's chosen T (STOP allowed).
  5. per-prep: pnb88 vs pnb97 reviewed AUC, from the same OOF vectors.
  6. 2-animal leave-one-animal-out, informational (2 animals is not a gate).

Writes gate_{arm}_w{W}.json (committed) + gate_{arm}_w{W}_oof.npz (fixture).
Read-only on session dirs.
"""
import argparse
import json
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import vca1_common as vc

VARIANTS = {"b13": slice(0, 13), "rankv2b_35": slice(0, 35)}


def matched_junk_far(oof_ref, oof_new, y, t_ref):
    """
    false-AR of the new model at the threshold where it catches as much junk as
    the reference does at t_ref.  Calibration-fair: the two models' score scales
    differ, so comparing FAR at a shared threshold would be meaningless.
    """
    pos, neg = y == 1, y == 0
    j_ref = float((oof_ref[neg] < t_ref).sum() / neg.sum())
    grid = np.arange(0.001, 0.501, 0.001)
    best = None
    for t in grid:
        if float((oof_new[neg] < t).sum() / neg.sum()) >= j_ref:
            best = t
            break
    if best is None:
        return None, j_ref, None
    return (float((oof_new[pos] < best).sum() / pos.sum() * 100), j_ref, float(best))


def loao(records, bs, animal, X_slice, weight, seeds):
    """Train on the other animal's agent sessions + all bootstrap; test on this one."""
    import diagnose_model as dm
    test = [r for r in records if r["animal"] == animal]
    train_ag = [r for r in records if r["animal"] != animal]
    if not test or not train_ag:
        return None
    Xte = np.vstack([r["X"] for r in test])[:, X_slice]
    yte = np.concatenate([r["y"] for r in test])
    if len(np.unique(yte)) < 2:
        return None
    Xtr = np.vstack([r["X"] for r in train_ag] + [r["X"] for r in bs])[:, X_slice]
    ytr = np.concatenate([r["y"] for r in train_ag] + [r["y"] for r in bs])
    wtr = np.concatenate([np.ones(sum(len(r["y"]) for r in train_ag)) * weight]
                         + [r["w"] for r in bs])
    keep = wtr > 0
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtr[keep])
    Xtes = sc.transform(Xte)
    aucs = []
    for seed in seeds:
        clf = dm.make_clf("xgb", dm.compute_spw(ytr[keep], wtr[keep]))
        clf.set_params(random_state=seed)     # the factory hard-codes 42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtrs, ytr[keep], sample_weight=wtr[keep])
            aucs.append(float(roc_auc_score(yte, clf.predict_proba(Xtes)[:, 1])))
    return {"n_sessions": len(test), "n_rows": int(len(yte)),
            "n_pos": int(yte.sum()), "auc": aucs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=[vc.ARM_A, vc.ARM_B0, vc.ARM_B1], required=True)
    ap.add_argument("--agent-weight", type=float, default=None)
    ap.add_argument("--skip-loao", action="store_true")
    args = ap.parse_args()
    vc.configure()
    W = vc.AGENT_WEIGHT if args.agent_weight is None else args.agent_weight
    tag = f"{args.arm}_w{W:g}"
    print(f"=== gate: arm {args.arm}, agent weight {W:g}, {len(vc.SEEDS)} seeds ===")

    records = vc.load_pool(v2=True, arm=args.arm, require_width=35)
    cv, rest, bs = vc.split_pool(records)
    X_ag, y_ag, g_ag = vc.stack(cv, keys=("X", "y"))
    rev = np.concatenate([r["reviewed"] for r in cv])
    animals = np.concatenate([[r["animal"]] * len(r["y"]) for r in cv])
    X_bs, y_bs, _ = vc.stack(bs, keys=("X", "y"))
    w_bs = np.concatenate([r["w"] for r in bs])
    pos, neg = y_ag == 1, y_ag == 0
    print(f"pool: {len(cv)} CV agent sessions, {len(bs)} bootstrap; "
          f"OOF {len(y_ag)} rows / {pos.sum()} real / {neg.sum()} junk "
          f"({(neg & rev).sum()} reviewed junk)")

    flag_bs = X_bs[:, 34]
    print(f"bootstrap v2_present: {int(flag_bs.sum())}/{len(flag_bs)} rows flagged; "
          f"ring_contrast nonzero on {int((X_bs[:, 33] != 0).sum())} rows")

    oof = {v: np.full((len(vc.SEEDS), len(y_ag)), np.nan) for v in VARIANTS}
    for si, seed in enumerate(vc.SEEDS):
        for v, sl in VARIANTS.items():
            oof[v][si] = vc.run_oof_fixed(X_ag[:, sl], y_ag, g_ag,
                                          X_bs[:, sl], y_bs, w_bs, seed,
                                          agent_weight=W)
        print(f"  seed {seed}: b13 {roc_auc_score(y_ag, oof['b13'][si]):.4f}  "
              f"v35 {roc_auc_score(y_ag, oof['rankv2b_35'][si]):.4f}", flush=True)

    res = {"arm": args.arm, "agent_weight": W, "seeds": vc.SEEDS,
           "n_cv_sessions": len(cv), "n_bootstrap": len(bs),
           "n_pool": int(len(y_ag)), "n_real": int(pos.sum()),
           "n_junk": int(neg.sum()), "n_junk_reviewed": int((neg & rev).sum()),
           "bootstrap_flagged_rows": int(flag_bs.sum()),
           "bootstrap_ring_nonzero_rows": int((X_bs[:, 33] != 0).sum())}

    # ---- 1. reference eval ----
    for scope, mask in (("full", np.ones(len(y_ag), bool)), ("reviewed", rev)):
        a13 = [roc_auc_score(y_ag[mask], o[mask]) for o in oof["b13"]]
        a35 = [roc_auc_score(y_ag[mask], o[mask]) for o in oof["rankv2b_35"]]
        d = [b - a for a, b in zip(a13, a35)]
        res[f"auc_{scope}"] = {"b13": a13, "rankv2b_35": a35}
        res[f"delta_{scope}"] = d
        print(f"\nAUC [{scope:>8}]  b13 {np.mean(a13):.4f}+/-{np.std(a13):.4f}   "
              f"v35 {np.mean(a35):.4f}+/-{np.std(a35):.4f}   "
              f"paired {np.mean(d):+.4f} (min {min(d):+.4f}, all>0 {all(x > 0 for x in d)})")

    # ---- 2. b13 consistency with the pin ----
    pin = np.load(vc.PIN / "baseline_oof.npz", allow_pickle=True)
    if abs(W - vc.AGENT_WEIGHT) < 1e-9 and pin["oof_seeds"].shape == oof["b13"].shape:
        md = float(np.max(np.abs(pin["oof_seeds"] - oof["b13"])))
        pin_auc = [roc_auc_score(y_ag, o) for o in pin["oof_seeds"]]
        own_auc = [roc_auc_score(y_ag, o) for o in oof["b13"]]
        dauc = float(np.max(np.abs(np.array(pin_auc) - np.array(own_auc))))
        res["b13_vs_pin"] = {"max_abs_score_diff": md, "max_abs_auc_diff": dauc}
        okc = md < 1e-6 and dauc < 1e-4
        print(f"\nb13 vs pin: max|score diff| {md:.2e}, max|AUC diff| {dauc:.2e} "
              f"-> {'consistent' if okc else 'DIFFERS (investigate)'}")
    else:
        res["b13_vs_pin"] = None
        print(f"\nb13 vs pin: skipped (weight {W:g} != pinned {vc.AGENT_WEIGHT:g})")

    # ---- 3. false-AR at matched junk ----
    m13 = oof["b13"].mean(axis=0)
    m35 = oof["rankv2b_35"].mean(axis=0)
    far_ref = float((m13[pos] < vc.DEPLOYED_T).sum() / pos.sum() * 100)
    far_new, j_ref, t_new = matched_junk_far(m13, m35, y_ag, vc.DEPLOYED_T)
    res["matched_junk"] = {"t_ref": vc.DEPLOYED_T, "junk_ref": j_ref,
                           "far_b13": far_ref, "far_v35": far_new, "t_v35": t_new}
    print(f"\nAt matched junk-caught ({100 * j_ref:.1f}%, b13 @ T={vc.DEPLOYED_T}): "
          f"false-AR b13 {far_ref:.2f}% -> v35 "
          f"{'n/a' if far_new is None else f'{far_new:.2f}%'} (v35 T={t_new})")

    # ---- 4. threshold table + the Step 4 rule ----
    print(f"\n{'T':>5}  {'false-AR %':>22}  {'junk full %':>12}  {'junk rev %':>11}")
    table = {}
    for t in vc.THRESHOLDS:
        far = [float((o[pos] < t).sum() / pos.sum() * 100) for o in oof["rankv2b_35"]]
        jf = [float((o[neg] < t).sum() / neg.sum() * 100) for o in oof["rankv2b_35"]]
        jr = [float((o[neg & rev] < t).sum() / (neg & rev).sum() * 100) for o in oof["rankv2b_35"]]
        table[f"{t:.2f}"] = {"far": far, "junk_full": jf, "junk_reviewed": jr}
        print(f"{t:5.2f}  {np.mean(far):6.2f} +/- {np.std(far):4.2f} (max {max(far):5.2f})"
              f"  {np.mean(jf):7.1f}  {np.mean(jr):10.1f}")
    res["threshold_table"] = table

    chosen = None
    for t in sorted(vc.THRESHOLDS):
        if t > 0.10:
            continue
        far = table[f"{t:.2f}"]["far"]
        if np.mean(far) <= 0.85 and max(far) <= 1.0:
            chosen = t
    res["chosen_T"] = chosen
    if chosen is None:
        res["gate_pass"] = False
        print("\nCHOSEN T: none in [0.03, 0.10] meets mean<=0.85% and worst-seed<=1.0% -> STOP")
    else:
        row = table[f"{chosen:.2f}"]
        gate = np.mean(row["far"]) <= 1.0 and np.mean(row["junk_full"]) >= 30.0
        res["gate_pass"] = bool(gate)
        print(f"\nCHOSEN T = {chosen:.2f}: false-AR {np.mean(row['far']):.2f}% "
              f"(max {max(row['far']):.2f}), junk full {np.mean(row['junk_full']):.1f}%, "
              f"reviewed {np.mean(row['junk_reviewed']):.1f}% -> "
              f"gate(far<=1%, junk>=30%): {'PASS' if gate else 'FAIL'}")

    # ---- 5. per-prep ----
    res["per_prep"] = {}
    print()
    for prep in sorted(set(animals)):
        m = rev & (animals == prep)
        if len(np.unique(y_ag[m])) < 2:
            continue
        a13 = [roc_auc_score(y_ag[m], o[m]) for o in oof["b13"]]
        a35 = [roc_auc_score(y_ag[m], o[m]) for o in oof["rankv2b_35"]]
        d = [b - a for a, b in zip(a13, a35)]
        res["per_prep"][prep] = {"n_sessions": len({r["name"] for r in cv if r["animal"] == prep}),
                                 "n_rows": int(m.sum()), "n_pos": int(y_ag[m].sum()),
                                 "b13": a13, "rankv2b_35": a35, "delta": d}
        print(f"per-prep [{prep}] {int(m.sum()):>4} reviewed rows, {int(y_ag[m].sum()):>3} real: "
              f"b13 {np.mean(a13):.4f} -> v35 {np.mean(a35):.4f}  "
              f"({np.mean(d):+.4f}, min {min(d):+.4f})")
    print("  (2022-23 animals are bootstrap-only, hence train-only: no agent test folds exist "
          "for them, so a pnb-vs-2022/23 split is not measurable.)")

    # ---- 6. LOAO, informational ----
    if not args.skip_loao:
        res["loao"] = {}
        agent_all = cv + rest
        for animal in sorted({r["animal"] for r in agent_all}):
            out = {}
            for v, sl in VARIANTS.items():
                out[v] = loao(agent_all, bs, animal, sl, W, vc.SEEDS[:4])
            if out["b13"] is None:
                continue
            res["loao"][animal] = out
            d = np.mean(out["rankv2b_35"]["auc"]) - np.mean(out["b13"]["auc"])
            print(f"LOAO [{animal}] {out['b13']['n_sessions']} sessions, "
                  f"{out['b13']['n_pos']} real: b13 {np.mean(out['b13']['auc']):.4f} -> "
                  f"v35 {np.mean(out['rankv2b_35']['auc']):.4f} ({d:+.4f})   [informational]")

    (vc.SP / f"gate_{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez(vc.SP / f"gate_{tag}_oof.npz",
             y=y_ag, groups=g_ag, reviewed=rev, animals=animals,
             names=np.array([r["name"] for r in cv]),
             oof_b13=oof["b13"], oof_v35=oof["rankv2b_35"])
    print(f"\nsaved gate_{tag}.json + gate_{tag}_oof.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
