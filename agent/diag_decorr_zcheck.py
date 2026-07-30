"""
diag_decorr_zcheck.py -- cheap z-hypothesis probe (no movie reload).

The QC flag (mot_mean+mot_p90, lateral LK) is worse-than-chance on 2/8 sessions
(bla12-660um, bla37-216um). Hypothesis: those sessions are z-dominated, and the
already-computed structure-change scalar `decorr_active` (col 6 of motion_vec.mat)
should separate their (m) cells even though displacement does not.

For each session, report per-feature single-feature AUC (Mann-Whitney) for
Q2 (motion vs other-deletes, the decision) and Q1 (motion vs keeps), for the
displacement feature (mot_mean) vs the structure-change feature (decorr_active).
If decorr_active >> 0.5 exactly where mot_mean <= 0.5, the z-story holds.

Read-only. python diag_decorr_zcheck.py
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score

AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))
from config import DATA_ROOT

SESSIONS = [
    ("6odorDualDiffRew", "AVG5x-TSeries-061226-bla37-213um-37z-000"),
    ("Block_Valence",    "AVG5x-TSeries-070226-bla37-262um-37z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-061126-bla37-277um-35z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-060426-bla37-275um-35z-000"),
    ("6odorDualDiffRew", "AVG5x-TSeries-052026-bla36-669um-29z-000"),
    ("2tones",           "AVG5x-TSeries-101525-bla12-660um-23z-000"),   # QC FAILS
    ("6odorDualDiffRew", "AVG5x-TSeries-052826-bla37-216um-37z-000"),   # QC FAILS
    ("2tones",           "AVG5x-TSeries-101525-bla16-278um-36z-000"),   # n=2, noise
]
FAILING = {"AVG5x-TSeries-101525-bla12-660um-23z-000",
           "AVG5x-TSeries-052826-bla37-216um-37z-000"}


def one_auc(feat, pos, neg):
    """AUC of a single feature separating pos(1) from neg(0); NaNs dropped."""
    v = np.concatenate([feat[pos], feat[neg]])
    lab = np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
    ok = ~np.isnan(v)
    v, lab = v[ok], lab[ok]
    if lab.sum() < 1 or (lab == 0).sum() < 1 or len(np.unique(lab)) < 2:
        return None, int(lab.sum()), int((lab == 0).sum())
    return roc_auc_score(lab, v), int(lab.sum()), int((lab == 0).sum())


def main():
    print("=" * 92)
    print("Z-HYPOTHESIS PROBE: structure-change (decorr_active) vs displacement (mot_mean)")
    print("Single-feature AUC per session. >0.5 = higher-in-motion = separates.")
    print("=" * 92)
    hdr = f"{'session':<40}{'nMot':>5}  " \
          f"{'Q2 mot_mean':>12}{'Q2 decorr':>11}   {'Q1 mot_mean':>12}{'Q1 decorr':>11}"
    print(hdr)
    print("-" * 92)
    for task, nm in SESSIONS:
        sd = DATA_ROOT / task / nm
        mv = sio.loadmat(str(sd / "motion_vec.mat"))
        feats = mv["feats"].astype(float)
        names = [str(x[0]) for x in mv["feature_names"][0]]
        i_mot = names.index("mot_mean")
        i_dec = names.index("decorr_active")

        lab = sio.loadmat(str(sd / "labels.mat"))
        y = lab["labels"].flatten().astype(int)
        ym = lab["motion_delete"].flatten().astype(int)

        pos = ym == 1
        neg_q2 = (y == 0) & (ym == 0)
        neg_q1 = y == 1

        a_mot_q2, nm2, _ = one_auc(feats[:, i_mot], pos, neg_q2)
        a_dec_q2, _, _ = one_auc(feats[:, i_dec], pos, neg_q2)
        a_mot_q1, _, _ = one_auc(feats[:, i_mot], pos, neg_q1)
        a_dec_q1, _, _ = one_auc(feats[:, i_dec], pos, neg_q1)

        def f(x):
            return f"{x:.3f}" if x is not None else "  n/a"
        mark = "  <-- QC FAILS" if nm in FAILING else ""
        print(f"{nm:<40}{int(pos.sum()):>5}  "
              f"{f(a_mot_q2):>12}{f(a_dec_q2):>11}   "
              f"{f(a_mot_q1):>12}{f(a_dec_q1):>11}{mark}")

    print("-" * 92)
    print("Read: on the two QC-FAILS rows, is Q2 decorr materially > 0.5 while Q2 mot_mean <= 0.5?")
    print("If yes -> those sessions are structure-change(z)-driven; onset-locked s(t) is the fix.")


if __name__ == "__main__":
    main()
