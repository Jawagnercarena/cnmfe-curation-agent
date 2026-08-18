"""
Step 2 evaluation. Pool = REVIEWED candidates only for agent sessions (v2
features exist only for the review set; every real is reviewed), bootstrap
sessions unchanged. Ranks are computed over the FULL candidate set (what the
curator would see at deploy time), then subset to reviewed rows.

Variants (mixed arm = agent + bootstrap [v2 zero-filled + present flag];
agent-only arm = no bootstrap):
  b13        the deployed 13 (reference on this pool)
  rank26     13 + within-session percentile ranks
  v2_22      13 + 8 v2 + v2_present
  rankv2_35  26 + 8 v2 + v2_present

Then promotion gates on the leaders: leave-one-animal-out and early-era
holdout (labels mtime < 2026-04-15), deterministic single fits.
"""
import datetime as dt
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util
import harness
import diagnose_model as dm

ANIMAL_RE = re.compile(r"-(bla\d+)-")
EARLY_CUTOFF = dt.datetime(2026, 4, 15).timestamp()


def build_records():
    manifest_util.assert_unchanged()
    records, _ = dm.load_all_records()
    v2 = np.load(SP / "step2_v2_features.npz", allow_pickle=True)
    v2b = np.load(SP / "step2_v2b_features.npz", allow_pickle=True)
    n_v2 = len(v2["feature_names"])

    out = {k: [] for k in ("b13", "rank26", "v2_22", "rankv2_35",
                           "v2b_22", "rankv2b_35")}
    n_agent_with_v2 = 0
    for r in records:
        X13 = r["X"]
        ranks = rankdata(X13, axis=0, method="average") / len(X13)
        X26 = np.hstack([X13, ranks])
        key = r["name"].replace("/", "__")
        if not r["is_bootstrap"]:
            if key + "__X" not in v2:
                continue  # extraction missing; count reported by compute step
            Xv2, Xv2b = v2[key + "__X"], v2b[key + "__X"]
            ridx = v2[key + "__idx"]
            assert (ridx == v2b[key + "__idx"]).all()
            flag = np.ones((len(ridx), 1))
            variants = {
                "b13": X13[ridx],
                "rank26": X26[ridx],
                "v2_22": np.hstack([X13[ridx], Xv2, flag]),
                "rankv2_35": np.hstack([X26[ridx], Xv2, flag]),
                "v2b_22": np.hstack([X13[ridx], Xv2b, flag]),
                "rankv2b_35": np.hstack([X26[ridx], Xv2b, flag]),
            }
            y, w = r["y"][ridx], r["w"][ridx]
            n_agent_with_v2 += 1
        else:
            z = np.zeros((len(X13), n_v2))
            f0 = np.zeros((len(X13), 1))
            variants = {
                "b13": X13,
                "rank26": X26,
                "v2_22": np.hstack([X13, z, f0]),
                "rankv2_35": np.hstack([X26, z, f0]),
                "v2b_22": np.hstack([X13, z, f0]),
                "rankv2b_35": np.hstack([X26, z, f0]),
            }
            y, w = r["y"], r["w"]
        for k, Xv in variants.items():
            out[k].append({**r, "X": Xv, "y": y, "w": w})
    print(f"records built: {n_agent_with_v2} agent sessions with v2, "
          f"{sum(1 for r in records if r['is_bootstrap'])} bootstrap")
    return out


def run_arm(recs_by_variant, agent_only, base_key="b13"):
    results = {}
    base_oof = None
    for name, recs in recs_by_variant.items():
        use = [r for r in recs if not (agent_only and r["is_bootstrap"])]
        oof, y = harness.run_variant(use, lambda X: X)
        if name == base_key:
            base_oof = oof
        res = harness.summarize(oof, y, base_oof if base_oof is not None else oof)
        results[name] = res
        tag = "agent-only" if agent_only else "mixed"
        print(f"[{tag}] {name:<10} AUC {res['auc_mean']:.4f}+/-{res['auc_sd']:.4f}  "
              f"far@mj {res['far_at_matched_junk']:.2f}%  "
              f"junk@mf {res['gc_at_matched_far']:.1f}%")
    for name in results:
        results[name]["delta_auc"] = (results[name]["auc_mean"]
                                      - results[base_key]["auc_mean"])
    return results


def holdout_eval(recs, test_names):
    """Deterministic single fit: train on everything except the named agent
    sessions, test pooled on them. Returns AUC and matched-junk false-AR."""
    train = [r for r in recs if r["name"] not in test_names]
    test = [r for r in recs if r["name"] in test_names]
    X_tr = np.vstack([r["X"] for r in train])
    y_tr = np.concatenate([r["y"] for r in train])
    n_ag = sum(len(r["y"]) for r in train if not r["is_bootstrap"])
    n_bs = sum(len(r["y"]) for r in train if r["is_bootstrap"])
    ag_w = float(max(np.sqrt(n_bs / n_ag), dm.MIN_AGENT_WEIGHT)) if n_ag and n_bs else dm.MIN_AGENT_WEIGHT
    w_tr = np.concatenate([np.ones(len(r["y"])) * ag_w if not r["is_bootstrap"]
                           else r["w"] for r in train])
    X_te = np.vstack([r["X"] for r in test])
    y_te = np.concatenate([r["y"] for r in test])
    sc = StandardScaler()
    X_trs = sc.fit_transform(X_tr)
    X_tes = sc.transform(X_te)
    clf = dm.make_clf("xgb", dm.compute_spw(y_tr, w_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_trs, y_tr, sample_weight=w_tr)
        s = clf.predict_proba(X_tes)[:, 1]
    return s, y_te


def main():
    recs_by_variant = build_records()

    print("\n=== 8-seed OOF, mixed arm (agent + bootstrap) ===")
    mixed = run_arm(recs_by_variant, agent_only=False)
    print("\n=== 8-seed OOF, agent-only arm ===")
    agent_only = run_arm(recs_by_variant, agent_only=True)

    # Promotion gates on all variants, mixed arm
    b13_recs = recs_by_variant["b13"]
    animals = sorted({ANIMAL_RE.search(r["name"]).group(1)
                      for r in b13_recs if not r["is_bootstrap"]})
    manifest = {s["rel"]: s for s in json.loads((SP / "pool_manifest.json").read_text())}
    early_names = {r["name"] for r in b13_recs if not r["is_bootstrap"]
                   and manifest[r["name"]]["labels_mtime"] < EARLY_CUTOFF}

    gates = {}
    print(f"\n=== Gates: leave-one-animal-out + early-era holdout "
          f"({len(early_names)} early sessions) ===")
    header = "variant     " + "".join(f"{a:>9}" for a in animals) + "   early-era"
    print(header)
    for name, recs in recs_by_variant.items():
        row = {}
        cells = []
        for a in animals:
            test = {r["name"] for r in recs if not r["is_bootstrap"]
                    and ANIMAL_RE.search(r["name"]).group(1) == a}
            s, y = holdout_eval(recs, test)
            auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
            row[a] = auc
            cells.append(f"{auc:>9.3f}")
        s, y = holdout_eval(recs, early_names)
        row["early_era"] = roc_auc_score(y, s)
        gates[name] = row
        print(f"{name:<12}" + "".join(cells) + f"   {row['early_era']:>9.3f}")

    json.dump({"mixed": mixed, "agent_only": agent_only, "gates": gates},
              open(SP / "step2_eval.json", "w"), indent=1)
    print("\nsaved step2_eval.json")


if __name__ == "__main__":
    main()
