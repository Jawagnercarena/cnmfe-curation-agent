"""
C2a: per-session normalization variants of the existing 13 features.
Every variant is computable from candidate_features.npz alone (agent AND
bootstrap) and deployable at curation time (the curator scores the full
candidate set at once). 8-seed OOF vs corrected baseline B, paired per seed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util
import harness
import diagnose_model as dm

# session-context aggregate columns: peak_snr, events_per_min, baseline_stability
AGG_COLS = [6, 8, 9]


def t_identity(X):
    return X


def t_rank_aug(X):
    pct = rankdata(X, axis=0, method="average") / len(X)
    return np.hstack([X, pct])


def t_rank_replace(X):
    return rankdata(X, axis=0, method="average") / len(X)


def t_z_aug(X):
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    return np.hstack([X, (X - med) / iqr])


def t_sess_agg(X):
    med = np.median(X[:, AGG_COLS], axis=0)
    iqr = np.subtract(*np.percentile(X[:, AGG_COLS], [75, 25], axis=0))
    ctx = np.tile(np.concatenate([med, iqr]), (len(X), 1))
    return np.hstack([X, ctx])


VARIANTS = {
    "rank_aug_26": t_rank_aug,
    "rank_replace_13": t_rank_replace,
    "robustz_aug_26": t_z_aug,
    "sess_agg_19": t_sess_agg,
}


def main():
    manifest_util.assert_unchanged()
    d = np.load(SP / "baseline_oof.npz", allow_pickle=True)
    base_oof, y_base = d["oof_seeds"], d["y"]

    records, _ = dm.load_all_records()
    base = harness.summarize(base_oof, y_base, base_oof)
    print(f"baseline B: AUC {base['auc_mean']:.4f}+/-{base['auc_sd']:.4f}  "
          f"far@mj {base['far_at_matched_junk']:.2f}%  "
          f"gc@mf {base['gc_at_matched_far']:.1f}%  band {base['band_005_012']:.1f}")

    results = {"baseline": base}
    for name, fn in VARIANTS.items():
        oof, y = harness.run_variant(records, fn)
        assert (y == y_base).all()
        res = harness.summarize(oof, y, base_oof)
        res["delta_auc"] = res["auc_mean"] - base["auc_mean"]
        results[name] = res
        print(f"{name:<16} AUC {res['auc_mean']:.4f}+/-{res['auc_sd']:.4f} "
              f"(d={res['delta_auc']:+.4f})  "
              f"far@matched-junk {res['far_at_matched_junk']:.2f}%+/-{res['far_at_matched_junk_sd']:.2f}  "
              f"junk@matched-far {res['gc_at_matched_far']:.1f}%+/-{res['gc_at_matched_far_sd']:.1f}  "
              f"band {res['band_005_012']:.1f}")

    (SP / "exp_rank_norm.json").write_text(json.dumps(results, indent=1))
    print("saved exp_rank_norm.json")


if __name__ == "__main__":
    main()
