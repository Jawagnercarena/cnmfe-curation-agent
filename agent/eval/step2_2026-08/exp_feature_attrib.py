"""
C2b: attribution on the existing 13 — (a) gain importances of a deployed-style
model, (b) native-XGB SHAP (pred_contribs) globally and on the false-AR set,
(c) leave-one-feature-out ablation (3-seed scan; 8-seed confirm within 0.002).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util
import harness
import diagnose_model as dm

FEATURE_NAMES = ["area", "circularity", "eccentricity", "compactness",
                 "max_weight", "weight_spread", "peak_snr", "transient_freq",
                 "events_per_min", "baseline_stability", "skewness",
                 "motion_correlation", "cn_correlation"]
T = 0.12


def main():
    manifest_util.assert_unchanged()
    d = np.load(SP / "baseline_oof.npz", allow_pickle=True)
    base_oof, y_base = d["oof_seeds"], d["y"]
    mean_oof = base_oof.mean(axis=0)

    records, agent_weight = dm.load_all_records()
    ag = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
    bs = [r for r in records if r["is_bootstrap"]]
    X_ag = np.vstack([r["X"] for r in ag])
    y_ag = np.concatenate([r["y"] for r in ag])
    X_bs = np.vstack([r["X"] for r in bs])
    y_bs = np.concatenate([r["y"] for r in bs])
    w_bs = np.concatenate([r["w"] for r in bs])
    assert (y_ag == y_base).all()

    # (a)+(b): one deployed-style fit on the full pool (in-sample for SHAP —
    # fine for "which features drag the false-AR set down", not for skill).
    X_all = np.vstack([X_ag, X_bs])
    y_all = np.concatenate([y_ag, y_bs])
    w_all = np.concatenate([np.ones(len(y_ag)) * agent_weight, w_bs])
    sc = StandardScaler()
    X_s = sc.fit_transform(X_all)
    clf = dm.make_clf("xgb", dm.compute_spw(y_all, w_all))
    clf.fit(X_s, y_all, sample_weight=w_all)

    booster = clf.get_booster()
    gain = booster.get_score(importance_type="gain")
    gain = {FEATURE_NAMES[int(k[1:])]: v for k, v in gain.items()}
    order = sorted(gain, key=gain.get, reverse=True)
    print("== gain importance (deployed-style fit) ==")
    for k in order:
        print(f"  {k:<20} {gain[k]:8.1f}")

    import xgboost as xgb
    dm_ag = xgb.DMatrix(sc.transform(X_ag))
    contribs = booster.predict(dm_ag, pred_contribs=True)[:, :-1]  # drop bias
    pos = y_ag == 1
    far = pos & (mean_oof < T)
    glob = np.abs(contribs).mean(axis=0)
    far_mean = contribs[far].mean(axis=0)
    kept_mean = contribs[pos & ~far].mean(axis=0)
    print("\n== SHAP (pred_contribs): false-AR set vs kept reals ==")
    print(f"{'feature':<20} {'glob|c|':>8} {'farC':>8} {'keptC':>8} {'drag':>8}")
    drag = {}
    for k in range(13):
        dr = far_mean[k] - kept_mean[k]
        drag[FEATURE_NAMES[k]] = float(dr)
        print(f"{FEATURE_NAMES[k]:<20} {glob[k]:>8.3f} {far_mean[k]:>8.3f} "
              f"{kept_mean[k]:>8.3f} {dr:>8.3f}")

    # (c) LOFO ablation
    base3 = harness.run_variant(records, lambda X: X, seeds=[42, 1, 7])
    from sklearn.metrics import roc_auc_score
    base3_auc = np.mean([roc_auc_score(base3[1], base3[0][s]) for s in range(3)])
    print(f"\n== LOFO ablation (3-seed scan; baseline3 {base3_auc:.4f}) ==")
    lofo = {}
    for k in range(13):
        keep = [j for j in range(13) if j != k]
        oof, y = harness.run_variant(records, lambda X, keep=keep: X[:, keep],
                                     seeds=[42, 1, 7])
        auc = np.mean([roc_auc_score(y, oof[s]) for s in range(3)])
        lofo[FEATURE_NAMES[k]] = float(auc - base3_auc)
        print(f"  drop {FEATURE_NAMES[k]:<20} AUC {auc:.4f} (d={auc-base3_auc:+.4f})")

    out = {"gain": gain, "shap_drag_far_minus_kept": drag,
           "lofo_delta_3seed": lofo, "base3_auc": float(base3_auc)}
    (SP / "exp_feature_attrib.json").write_text(json.dumps(out, indent=1))
    print("saved exp_feature_attrib.json")


if __name__ == "__main__":
    main()
