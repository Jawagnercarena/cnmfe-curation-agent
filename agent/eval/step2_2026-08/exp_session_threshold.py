"""
C2c: deploy-side lever — label-free per-session threshold rules, evaluated on
the corrected baseline OOF scores (pure post-processing, no refits).
The question: can a session-relative rule fix the false-AR drift (recent
sessions' dim reals rejected at fixed 0.12) without giving up junk-caught?
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
import manifest_util

T = 0.12
RECENT_CUTOFF = dt.datetime(2026, 7, 30).timestamp()


def eval_rule(reject_fn, oof_seeds, y, names, sess_list, recent):
    """reject_fn(scores) -> bool mask over one session's candidates."""
    pos, neg = y == 1, y == 0
    far, gc, worst, far_rec, far_est = [], [], [], [], []
    for s in range(oof_seeds.shape[0]):
        oof = oof_seeds[s]
        rej = np.zeros(len(y), dtype=bool)
        for sess in sess_list:
            m = names == sess
            rej[m] = reject_fn(oof[m])
        far.append(rej[pos].mean() * 100)
        gc.append(rej[neg].mean() * 100)
        w = 0.0
        for sess in sess_list:
            m = (names == sess) & pos
            if m.sum() >= 10:
                w = max(w, rej[m].mean() * 100)
        worst.append(w)
        far_rec.append(rej[pos & recent].mean() * 100 if (pos & recent).any() else 0)
        far_est.append(rej[pos & ~recent].mean() * 100)
    return {
        "far": float(np.mean(far)), "far_sd": float(np.std(far)),
        "junk": float(np.mean(gc)), "junk_sd": float(np.std(gc)),
        "worst_session_far": float(np.mean(worst)),
        "far_recent": float(np.mean(far_rec)),
        "far_established": float(np.mean(far_est)),
    }


def main():
    manifest_util.assert_unchanged()
    d = np.load(SP / "baseline_oof.npz", allow_pickle=True)
    oof_seeds, y, names = d["oof_seeds"], d["y"], d["session"]
    sess_list = sorted(set(names.tolist()))

    manifest = {s["rel"]: s for s in json.loads(
        (SP / "pool_manifest.json").read_text())}
    recent = np.array([manifest[str(n)]["labels_mtime"] > RECENT_CUTOFF
                       for n in names])
    n_rec = int((recent & (y == 1)).sum())
    print(f"pool: {len(y)} cands, {int((y==1).sum())} reals "
          f"({n_rec} in recent sessions, cutoff 2026-07-30)")

    rules = {"fixed_0.12": lambda s: s < T}
    for q in (5, 10, 15, 20):
        rules[f"min(0.12,P{q})"] = (
            lambda s, q=q: s < min(T, np.percentile(s, q)))
    for r in (10, 20, 30):
        def bottom_r(s, r=r):
            k = max(1, int(len(s) * r / 100))
            cut = np.partition(s, k - 1)[k - 1]
            return (s <= cut) & (s < T)
        rules[f"bottom{r}%_cap0.12"] = bottom_r
    # Two-sided: junk-rich sessions may ride above 0.12 (up to hi); clean
    # sessions drop below it (down to lo). The mirror of the shrink-only rules.
    for q, lo, hi in ((20, 0.05, 0.20), (30, 0.05, 0.20), (30, 0.08, 0.25)):
        rules[f"clip(P{q},{lo},{hi})"] = (
            lambda s, q=q, lo=lo, hi=hi:
            s < float(np.clip(np.percentile(s, q), lo, hi)))
    for z0 in (-1.5, -2.0, -2.5):
        def zrule(s, z0=z0):
            eps = 1e-6
            lg = np.log(np.clip(s, eps, 1 - eps) / (1 - np.clip(s, eps, 1 - eps)))
            med = np.median(lg)
            mad = np.median(np.abs(lg - med)) * 1.4826
            if mad == 0:
                return s < T
            return ((lg - med) / mad < z0) & (s < T)
        rules[f"robustz<{z0}_cap0.12"] = zrule

    results = {}
    print(f"\n{'rule':<22} {'FAR%':>6} {'junk%':>6} {'worstFAR':>8} "
          f"{'FAR-rec':>8} {'FAR-est':>8}")
    for name, fn in rules.items():
        res = eval_rule(fn, oof_seeds, y, names, sess_list, recent)
        results[name] = res
        print(f"{name:<22} {res['far']:>6.2f} {res['junk']:>6.1f} "
              f"{res['worst_session_far']:>8.1f} {res['far_recent']:>8.2f} "
              f"{res['far_established']:>8.2f}")

    (SP / "exp_session_threshold.json").write_text(json.dumps(results, indent=1))
    print("\nsaved exp_session_threshold.json")


if __name__ == "__main__":
    main()
