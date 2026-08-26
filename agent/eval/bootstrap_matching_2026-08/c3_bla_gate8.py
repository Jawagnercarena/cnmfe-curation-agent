"""
c3_bla_gate8.py — BLA gates after the bootstrap pixel-order fix.

Faithful adaptation of step4_2026-08/threshold_sweep_v2.py (Step 5, the run
that produced the pinned post-Step-4 baseline step5_results.json): identical
seeds, CV structure, per-fold agent weight (deployed sqrt/4.0 recipe), model
factory and metrics — but reading the LIVE candidate_features.npz (35-col for
both agent and re-run bootstrap sessions) and, at the end, printing per-seed
PAIRED deltas against the pinned baseline (same seeds => paired).
Bootstrap masking = trainer's current helper (ambiguous + duplicates).

Reproduces agent/eval/threshold_robustness.py's 8-seed OOF methodology
(same seeds, same StratifiedGroupKFold structure, same per-fold agent
weight recomputation, weights identical to the deployed trainer) on the
35-column corpus, and in the same OOF runs:

  - threshold table T = 0.03..0.10 (+0.12 reference row): false-AR
    (% of reviewed REAL cells below T), junk-caught on the full pool and on
    the reviewed stratum, mean +/- sd over seeds;
  - reference eval: paired per-seed AUC b13 vs rankv2b_35 on the reviewed
    subset (Step 2 gate: delta >= +0.015, all seeds positive) and on the
    full pool;
  - OOF smoke check: the bla21 autopsy cells (2tones/093025, candidate idx
    21/24) — pinned-baseline OOF 0.064/0.101 — must score far above the
    chosen T under the new representation.

Chosen-T rule (red-team C9): largest T in [0.03, 0.10] with mean false-AR
<= 0.85% and worst-seed false-AR <= 1.0%; gate junk-caught >= 30%.

Outputs: step5_results.json (committed) + step5_oof.npz (uncommitted,
per-seed OOF vectors for reuse in Step 7 verification).  Read-only on
sessions.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SP = Path(__file__).parent
PINNED = SP.parent / "step4_2026-08" / "step5_results.json"
AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))

import diagnose_model as dm          # weighting/clf factories (deployed logic)
import train_classifier as tc

DATA_ROOT = Path(r"D:\Julian_CNMFe\BLA")
V2 = "candidate_features.npz"   # LIVE files (post-swap, post-re-run)
SEEDS = [42, 1, 7, 13, 100, 2024, 31337, 9]
THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]
MIN_POS = 5
SMOKE_REL = "2tones/AVG5x-TSeries-093025-bla21-313um-38z-000"
SMOKE_IDX = [21, 24]


def load_v2_records():
    """diagnose_model.load_all_records, but reading the parallel v2 files and
    attaching the reviewed-row mask."""
    records = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            f2, lab = sd / V2, sd / "labels.mat"
            if not f2.exists() or not lab.exists():
                continue
            npz = np.load(f2, allow_pickle=True)
            X = npz["feature_matrix"].astype(float)
            assert X.shape[1] == 35, f"{sd.name}: v2 width {X.shape[1]}"
            y_rev = sio.loadmat(str(lab))["labels"].flatten().astype(float)
            auto = npz["auto_rejected"].flatten().astype(int)
            reviewed = np.ones(len(X), dtype=bool)
            reviewed[auto] = False
            if len(y_rev) == len(X):
                y = y_rev
            else:
                assert len(y_rev) == int(reviewed.sum()), f"{sd.name}: sizes"
                y = np.zeros(len(X))
                y[reviewed] = y_rev
            records.append({
                "name": f"{td.name}/{sd.name}", "X": X,
                "y": (y == 1).astype(int), "reviewed": reviewed,
                "is_bootstrap": tc._is_bootstrap_session(sd),
                "session_dir": sd,
            })
    n_bs = sum(len(r["y"]) for r in records if r["is_bootstrap"])
    n_ag = sum(len(r["y"]) for r in records if not r["is_bootstrap"])
    for r in records:
        w = np.ones(len(r["y"]))
        if r["is_bootstrap"]:
            rec = dm._get_bootstrap_recovery(r["session_dir"])
            if rec is not None and rec < dm.BAD_SESSION_RECOVERY_THRESHOLD:
                w *= dm.BAD_SESSION_WEIGHT
            w[dm._get_bootstrap_ambiguous_mask(r["session_dir"], len(w))] = 0.0
        r["w"] = w
    return records, n_ag, n_bs


def run_oof(X_ag, y_ag, g_ag, X_bs, y_bs, w_bs, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y_ag), np.nan)
    for tr_idx, te_idx in cv.split(X_ag, y_ag, g_ag):
        n_ag_tr = len(tr_idx)
        ag_w = float(max(np.sqrt(len(y_bs) / n_ag_tr), dm.MIN_AGENT_WEIGHT))
        X_trC = np.vstack([X_ag[tr_idx], X_bs])
        y_trC = np.concatenate([y_ag[tr_idx], y_bs])
        w_trC = np.concatenate([np.ones(n_ag_tr) * ag_w, w_bs])
        sc = StandardScaler()
        X_trCs = sc.fit_transform(X_trC)
        X_teCs = sc.transform(X_ag[te_idx])
        clf = dm.make_clf("xgb", dm.compute_spw(y_trC, w_trC))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_trCs, y_trC, sample_weight=w_trC)
            oof[te_idx] = clf.predict_proba(X_teCs)[:, 1]
    return oof


def compare_to_pinned(res):
    """Per-seed paired deltas vs the pinned post-Step-4 baseline (Step 5)."""
    pin = json.load(open(PINNED))
    assert pin["seeds"] == res["seeds"], "seed lists differ - not paired"
    print("\n=== PAIRED vs pinned post-Step-4 baseline (rankv2b_35, same seeds) ===")
    print(f"pool rows: pinned {pin['n_pool']} vs now {res['n_pool']} "
          f"(real {pin['n_real']} vs {res['n_real']})")
    for scope in ("full", "reviewed"):
        a = np.array(pin[f"auc_{scope}"]["rankv2b_35"])
        b = np.array(res[f"auc_{scope}"]["rankv2b_35"])
        d = b - a
        print(f"G1 AUC [{scope}]: pinned {a.mean():.4f}+/-{a.std():.4f} -> now "
              f"{b.mean():.4f}+/-{b.std():.4f}  paired delta {d.mean():+.4f} "
              f"(min {d.min():+.4f}, max {d.max():+.4f}, all>=0: {(d >= 0).all()})")
    for t in ("0.06", "0.08"):
        if t in pin["threshold_table"] and t in res["threshold_table"]:
            pa, na = pin["threshold_table"][t], res["threshold_table"][t]
            print(f"G2 T={t}: false-AR pinned {np.mean(pa['far']):.2f}% -> now "
                  f"{np.mean(na['far']):.2f}% (worst seed {max(na['far']):.2f}%) | "
                  f"junk full {np.mean(pa['junk_full']):.1f}% -> {np.mean(na['junk_full']):.1f}% | "
                  f"junk reviewed {np.mean(pa['junk_reviewed']):.1f}% -> {np.mean(na['junk_reviewed']):.1f}%")


def main():
    records, n_ag, n_bs = load_v2_records()
    ag = [r for r in records if not r["is_bootstrap"]]
    bs = [r for r in records if r["is_bootstrap"]]
    ag_use = [r for r in ag if r["y"].sum() >= MIN_POS]
    print(f"pool: {len(ag)} agent ({len(ag_use)} with >= {MIN_POS} positives) "
          f"+ {len(bs)} bootstrap;  {n_ag} agent rows, {n_bs} bootstrap rows")

    X_ag = np.vstack([r["X"] for r in ag_use])
    y_ag = np.concatenate([r["y"] for r in ag_use])
    g_ag = np.concatenate([[i] * len(r["y"]) for i, r in enumerate(ag_use)])
    rev_ag = np.concatenate([r["reviewed"] for r in ag_use])
    X_bs = np.vstack([r["X"] for r in bs])
    y_bs = np.concatenate([r["y"] for r in bs])
    w_bs = np.concatenate([r["w"] for r in bs])

    # smoke-cell row positions in the pooled agent arrays
    smoke_rows = []
    off = 0
    for r in ag_use:
        if r["name"] == SMOKE_REL:
            smoke_rows = [off + i for i in SMOKE_IDX]
        off += len(r["y"])
    assert smoke_rows, "smoke session missing from OOF pool"

    pos, neg = y_ag == 1, y_ag == 0
    print(f"OOF pool: {len(y_ag)} rows, {pos.sum()} real, {neg.sum()} junk "
          f"({(neg & rev_ag).sum()} reviewed junk)")

    variants = {"b13": (X_ag[:, :13], X_bs[:, :13]),
                "rankv2b_35": (X_ag, X_bs)}
    oof_seeds = {v: [] for v in variants}
    for seed in SEEDS:
        for v, (Xa, Xb) in variants.items():
            oof_seeds[v].append(run_oof(Xa, y_ag, g_ag, Xb, y_bs, w_bs, seed))
        print(f"  seed {seed} done", flush=True)

    res = {"seeds": SEEDS, "n_pool": int(len(y_ag)), "n_real": int(pos.sum()),
           "n_junk": int(neg.sum()), "n_junk_reviewed": int((neg & rev_ag).sum())}

    # ---- reference eval ----
    for scope, mask in (("full", np.ones(len(y_ag), bool)), ("reviewed", rev_ag)):
        aucs = {v: [roc_auc_score(y_ag[mask], o[mask]) for o in oof_seeds[v]]
                for v in variants}
        d = [b - a for a, b in zip(aucs["b13"], aucs["rankv2b_35"])]
        res[f"auc_{scope}"] = {v: [float(x) for x in aucs[v]] for v in variants}
        res[f"delta_{scope}"] = [float(x) for x in d]
        print(f"\nAUC [{scope}]: b13 {np.mean(aucs['b13']):.4f}+/-"
              f"{np.std(aucs['b13']):.4f}  rankv2b_35 "
              f"{np.mean(aucs['rankv2b_35']):.4f}+/-"
              f"{np.std(aucs['rankv2b_35']):.4f}  paired delta "
              f"{np.mean(d):+.4f} (min {min(d):+.4f}, all>0: {all(x > 0 for x in d)})")

    # ---- threshold table (rankv2b_35) ----
    print(f"\n{'T':>5}  {'false-AR %':>18}  {'junk full %':>14}  {'junk reviewed %':>16}")
    table = {}
    for t in THRESHOLDS:
        far = [float((o[pos] < t).sum() / pos.sum() * 100)
               for o in oof_seeds["rankv2b_35"]]
        jf = [float((o[neg] < t).sum() / neg.sum() * 100)
              for o in oof_seeds["rankv2b_35"]]
        jr = [float((o[neg & rev_ag] < t).sum() / (neg & rev_ag).sum() * 100)
              for o in oof_seeds["rankv2b_35"]]
        table[f"{t:.2f}"] = {"far": far, "junk_full": jf, "junk_reviewed": jr}
        print(f"{t:5.2f}  {np.mean(far):6.2f} +/- {np.std(far):4.2f} "
              f"(max {max(far):4.2f})  {np.mean(jf):6.1f} +/- {np.std(jf):3.1f}"
              f"  {np.mean(jr):7.1f} +/- {np.std(jr):3.1f}")
    res["threshold_table"] = table

    # chosen T: largest with mean far <= 0.85 and worst-seed far <= 1.0
    chosen = None
    for t in sorted(THRESHOLDS):
        if t > 0.10:
            continue
        far = table[f"{t:.2f}"]["far"]
        if np.mean(far) <= 0.85 and max(far) <= 1.0:
            chosen = t
    res["chosen_T"] = chosen
    if chosen is not None:
        row = table[f"{chosen:.2f}"]
        gate = np.mean(row["far"]) <= 1.0 and np.mean(row["junk_full"]) >= 30.0
        res["gate_pass"] = bool(gate)
        print(f"\nCHOSEN T = {chosen:.2f}: false-AR "
              f"{np.mean(row['far']):.2f}%, junk full "
              f"{np.mean(row['junk_full']):.1f}%, junk reviewed "
              f"{np.mean(row['junk_reviewed']):.1f}%  -> gate "
              f"(far<=1%, junk>=30%): {'PASS' if gate else 'FAIL'}")
    else:
        res["gate_pass"] = False
        print("\nNO threshold in [0.03, 0.10] meets the false-AR rule — STOP.")

    # ---- OOF smoke check ----
    v2b_mean = np.mean(oof_seeds["rankv2b_35"], axis=0)
    b13_mean = np.mean(oof_seeds["b13"], axis=0)
    res["smoke"] = {}
    print()
    for ci, row in zip(SMOKE_IDX, smoke_rows):
        res["smoke"][str(ci)] = {"b13": float(b13_mean[row]),
                                 "v2b": float(v2b_mean[row])}
        print(f"smoke bla21 'Neuron {ci+1}': b13 OOF {b13_mean[row]:.3f} "
              f"(pinned 0.064/0.101) -> rankv2b_35 OOF {v2b_mean[row]:.3f}")

    json.dump(res, open(SP / "c3_bla_gate8_results.json", "w"), indent=1)
    np.savez(SP / "c3_bla_gate8_oof.npz",
             y=y_ag, groups=g_ag, reviewed=rev_ag,
             names=np.array([r["name"] for r in ag_use]),
             oof_b13=np.array(oof_seeds["b13"]),
             oof_v2b=np.array(oof_seeds["rankv2b_35"]),
             smoke_rows=np.array(smoke_rows))
    print("\nsaved c3_bla_gate8_results.json + c3_bla_gate8_oof.npz")
    compare_to_pinned(res)


if __name__ == "__main__":
    main()
