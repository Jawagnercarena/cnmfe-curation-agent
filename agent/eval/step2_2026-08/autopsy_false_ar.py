"""
C1: error autopsy. From the corrected seed-averaged OOF scores, list every
real-labeled candidate with mean OOF < 0.12 (the false-AR set), with feature
profiles, within-session percentile ranks, seed churn, and per-session
contrast rows (that session's lowest-scoring kept reals >= 0.12).
Writes autopsy_false_ar.csv + a summary to stdout.
"""
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util

AGENT = Path(r"c:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent")
sys.path.insert(0, str(AGENT))
import diagnose_model as dm
from config import DATA_ROOT

T = 0.12
RECENT_CUTOFF = dt.datetime(2026, 7, 30).timestamp()
ANIMAL_RE = re.compile(r"-(bla\d+)-")

FEATURE_NAMES = ["area", "circularity", "eccentricity", "compactness",
                 "max_weight", "weight_spread", "peak_snr", "transient_freq",
                 "events_per_min", "baseline_stability", "skewness",
                 "motion_correlation", "cn_correlation"]


def main():
    manifest_util.assert_unchanged()
    d = np.load(SP / "baseline_oof.npz", allow_pickle=True)
    oof_seeds, names = d["oof_seeds"], d["session"]
    idx_in_sess, y = d["idx_in_session"], d["y"]
    mean_oof = oof_seeds.mean(axis=0)

    records, _ = dm.load_all_records()
    ag_recs = [r for r in records if not r["is_bootstrap"] and r["y"].sum() >= 5]
    rebuilt = np.concatenate([[r["name"]] * len(r["y"]) for r in ag_recs])
    assert (rebuilt == names).all(), "record order mismatch vs baseline_oof"
    X = np.vstack([r["X"] for r in ag_recs])

    # Within-session percentile ranks of each feature (rank among ALL candidates)
    from scipy.stats import rankdata
    pct = np.zeros_like(X)
    for r in ag_recs:
        m = names == r["name"]
        pct[m] = rankdata(X[m], axis=0, method="average") / m.sum() * 100

    # Session metadata: mtime, auto_rejected, motion_delete (review-set indexed)
    meta = {}
    for r in ag_recs:
        sd = r["session_dir"]
        npz = np.load(sd / "candidate_features.npz", allow_pickle=True)
        auto_rej = set(int(i) for i in npz["auto_rejected"])
        n = int(npz["n_candidates"][0])
        review_indices = [i for i in range(n) if i not in auto_rej]
        rev_pos = {full: k for k, full in enumerate(review_indices)}
        lab = sio.loadmat(sd / "labels.mat")
        md = np.asarray(lab.get("motion_delete", np.zeros(0))).ravel()
        meta[r["name"]] = {
            "mtime": (sd / "labels.mat").stat().st_mtime,
            "rev_pos": rev_pos, "motion_delete": md,
        }

    pos = y == 1
    far_mask = pos & (mean_oof < T)
    far_idx = np.where(far_mask)[0]
    churn = (oof_seeds < T).mean(axis=0)  # fraction of seeds below T

    rows = []
    def add_row(i, kind):
        sess = str(names[i])
        m = meta[sess]
        full_idx = int(idx_in_sess[i])
        rp = m["rev_pos"].get(full_idx, -1)
        is_motion = (0 <= rp < len(m["motion_delete"])
                     and m["motion_delete"][rp] > 0)
        animal = ANIMAL_RE.search(sess)
        row = {
            "kind": kind, "session": sess,
            "animal": animal.group(1) if animal else "?",
            "task": sess.split("/")[0],
            "reviewed": dt.datetime.fromtimestamp(m["mtime"]).strftime("%Y-%m-%d"),
            "era": "recent" if m["mtime"] > RECENT_CUTOFF else "established",
            "cand_idx0": full_idx, "pdf_label": f"Neuron {full_idx + 1}",
            "oof_mean": round(float(mean_oof[i]), 4),
            "oof_sd": round(float(oof_seeds[:, i].std()), 4),
            "seed_frac_below_012": round(float(churn[i]), 3),
            "motion_delete": int(is_motion),
        }
        for k, fn in enumerate(FEATURE_NAMES):
            row[fn] = round(float(X[i, k]), 4)
            row[f"{fn}_pct"] = round(float(pct[i, k]), 1)
        rows.append(row)

    for i in far_idx:
        add_row(i, "false_AR")
    # Contrast rows: per affected session, the 3 lowest-scoring kept reals >= T
    for sess in sorted({str(names[i]) for i in far_idx}):
        m = (names == sess) & pos & (mean_oof >= T)
        cand = np.where(m)[0]
        for i in cand[np.argsort(mean_oof[cand])][:3]:
            add_row(i, "contrast")

    out = SP / "autopsy_false_ar.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    far_rows = [r for r in rows if r["kind"] == "false_AR"]
    print(f"false-AR set (mean OOF < {T}): {len(far_rows)} of {pos.sum()} reals "
          f"({len(far_rows)/pos.sum():.2%}) across "
          f"{len({r['session'] for r in far_rows})} sessions")
    from collections import Counter
    print("by animal:", dict(Counter(r["animal"] for r in far_rows)))
    print("by era:   ", dict(Counter(r["era"] for r in far_rows)))
    print("motion_delete-tagged:", sum(r["motion_delete"] for r in far_rows))
    stable = sum(1 for r in far_rows if r["seed_frac_below_012"] >= 0.75)
    print(f"stable across seeds (>=6/8 below 0.12): {stable}; "
          f"churny (<6/8): {len(far_rows) - stable}")
    # Recent-vs-established false-AR rates
    for era in ("recent", "established"):
        era_pos = [i for i in np.where(pos)[0]
                   if (meta[str(names[i])]["mtime"] > RECENT_CUTOFF) == (era == "recent")]
        n_far = sum(1 for i in era_pos if mean_oof[i] < T)
        print(f"{era}: {n_far}/{len(era_pos)} reals below 0.12 "
              f"({n_far/len(era_pos):.2%})" if era_pos else f"{era}: no reals")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
