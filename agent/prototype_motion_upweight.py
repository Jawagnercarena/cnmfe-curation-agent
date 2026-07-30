"""
prototype_motion_upweight.py -- DRY-RUN experiment (writes nothing, changes no
deployed code). Question: if we upweight the motion-delete rows in the EXISTING
binary keep/delete classifier, does it auto-reject more motion artifacts, and at
what cost to real cells?

Mechanism: the model trains on the binary keep/delete `labels`, so motion cells
are just ordinary deletes today. Here we multiply the sample_weight of rows with
motion_delete==1 by a factor w_motion (on top of the normal agent up-weight) so
the training loss pays extra to push them below the reject_threshold. Everything
else -- features, labels, npz, bootstrap weighting -- is identical to the
deployed pipeline. No feature-contract change; this is the "seamless" lever.

Note on the metric floor: every motion-tagged cell reached human review, which
means it already scored ABOVE the production reject_threshold (auto-rejected
candidates are never shown, so never tagged). So baseline motion-catch starts
near zero by construction -- the question is whether the upweight can move a
meaningful fraction below threshold WITHOUT dragging real cells down with them.

Honesty caveat: grouped-by-session OOF lets the same animal appear in train and
test, and motion tags are 75% bla37, so motion-catch here is an OPTIMISTIC upper
bound (same cohort-leak the separability eval quantified at ~0.17 AUC). The
false-AR cost, measured on all agent real cells, is not inflated the same way.

Uses the active config (BLA). Read-only.

Usage:
    python prototype_motion_upweight.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))

import train_classifier as tc
from config import AREA, DATA_ROOT

# Mirror train_classifier.main (MIN_AGENT_WEIGHT is a local there, not importable).
MIN_AGENT_WEIGHT = 4.0
DEPLOYED_THRESHOLD = 0.14           # BLA reject_threshold
THRESHOLDS = [0.10, 0.14, 0.20]
W_MOTION_SWEEP = [1, 2, 3, 5, 10, 20]


def log(m=""):
    print(m, flush=True)


def reconstruct_motion(sd: Path, N: int) -> np.ndarray:
    """motion_delete reconstructed onto the full candidate set (auto-rejected=0),
    same index logic as train_classifier.load_prospective_session."""
    ld = sio.loadmat(str(sd / "labels.mat"))
    ym = np.zeros(N, dtype=float)
    if "motion_delete" not in ld:
        return ym
    md = ld["motion_delete"].flatten().astype(float)
    if N == len(md):
        return md
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
    ridx = [i for i in range(N) if i not in auto]
    if len(ridx) == len(md):
        for k, idx in enumerate(ridx):
            ym[idx] = md[k]
    return ym


def load_records():
    recs = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not sd.is_dir():
                continue
            if not ((sd / "candidate_features.npz").exists()
                    and (sd / "labels.mat").exists()):
                continue
            r = tc.load_prospective_session(sd)
            if r is None:
                continue
            X, y = r
            y = (y == 1).astype(int)
            is_bs = (sd / "bootstrap_match_stats.json").exists()
            ym = np.zeros(len(y)) if is_bs else reconstruct_motion(sd, len(X))
            recs.append({"X": X.astype(float), "y": y, "ym": ym.astype(int),
                         "is_bs": is_bs, "sd": sd})
    return recs


def base_weights(recs):
    """Deployed weighting: agent up-weight (floored), bad-session 0.4x, ambiguous=0."""
    n_bs = sum(len(r["y"]) for r in recs if r["is_bs"])
    n_ag = sum(len(r["y"]) for r in recs if not r["is_bs"])
    aw = (float(max(np.sqrt(n_bs / n_ag), MIN_AGENT_WEIGHT))
          if n_ag and n_bs else MIN_AGENT_WEIGHT)
    for gi, r in enumerate(recs):
        w = np.ones(len(r["y"]), dtype=float)
        if not r["is_bs"]:
            w *= aw
        else:
            rec = tc._get_bootstrap_recovery(r["sd"])
            if rec is not None and rec < tc.BAD_SESSION_RECOVERY_THRESHOLD:
                w *= tc.BAD_SESSION_WEIGHT
            amb = tc._get_bootstrap_ambiguous_mask(r["sd"], len(r["y"]))
            w[amb] = 0.0
        r["w_base"] = w
        r["grp"] = gi
    return aw


def oof(recs, w_motion):
    X    = np.vstack([r["X"] for r in recs])
    y    = np.concatenate([r["y"] for r in recs])
    ym   = np.concatenate([r["ym"] for r in recs])
    isbs = np.concatenate([[r["is_bs"]] * len(r["y"]) for r in recs])
    g    = np.concatenate([[r["grp"]] * len(r["y"]) for r in recs])
    w    = np.concatenate([r["w_base"] for r in recs]).copy()
    w[ym == 1] *= w_motion                      # THE upweight (motion rows only)

    scores = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in cv.split(X, y, g):
            wt = w[tr]; m = wt > 0; ytr = y[tr]
            ep = float(wt[(ytr == 1) & m].sum())
            en = float(wt[(ytr == 0) & m].sum())
            spw = en / ep if ep > 0 else 1.0
            sc = StandardScaler().fit(X[tr])
            clf = tc._make_clf("xgboost", scale_pos_weight=spw)
            clf.fit(sc.transform(X[tr]), ytr, sample_weight=wt)
            scores[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return scores, y, ym, isbs


def frac_below(scores, mask, thr):
    return float(np.mean(scores[mask] < thr)) if mask.sum() else float("nan")


def main():
    log("=" * 78)
    log(f"MOTION-UPWEIGHT PROTOTYPE (dry-run)   area: {AREA}")
    log("=" * 78)

    recs = load_records()
    base_weights(recs)
    n_ag = sum(1 for r in recs if not r["is_bs"])
    n_bs = sum(1 for r in recs if r["is_bs"])
    n_motion = int(sum(r["ym"].sum() for r in recs))
    log(f"\nSessions: {len(recs)} ({n_ag} agent, {n_bs} bootstrap)   "
        f"motion-tagged cells: {n_motion}")

    # Pre-compute masks/scores per w_motion; report at the deployed threshold.
    results = {}
    for wm in W_MOTION_SWEEP:
        scores, y, ym, isbs = oof(recs, wm)
        ag = ~isbs
        real   = ag & (y == 1)
        motion = ag & (ym == 1)
        otherd = ag & (y == 0) & (ym == 0)
        auc = roc_auc_score(y, scores)
        results[wm] = dict(scores=scores, y=y, ym=ym, isbs=isbs,
                           real=real, motion=motion, otherd=otherd, auc=auc)

    med0 = np.median(results[1]["scores"][results[1]["motion"]])
    log(f"Baseline (w_motion=1) median score of motion cells: {med0:.3f}  "
        f"(deployed threshold {DEPLOYED_THRESHOLD}) -- above threshold => they "
        f"currently reach review.")

    log("\n" + "-" * 78)
    log(f"Trade-off at the deployed threshold ({DEPLOYED_THRESHOLD}) "
        f"-- grouped OOF (optimistic on motion; see caveat)")
    log("-" * 78)
    log(f"  {'w_motion':>8}  {'keep/del AUC':>12}  {'motion caught':>13}  "
        f"{'real-cell false-AR':>18}  {'garbage caught':>14}")
    for wm in W_MOTION_SWEEP:
        R = results[wm]
        mc = frac_below(R["scores"], R["motion"], DEPLOYED_THRESHOLD)
        fa = frac_below(R["scores"], R["real"],   DEPLOYED_THRESHOLD)
        gc = frac_below(R["scores"], R["otherd"], DEPLOYED_THRESHOLD)
        tag = "  <- baseline (current model)" if wm == 1 else ""
        log(f"  {wm:>8}  {R['auc']:>12.3f}  {mc:>12.1%}  {fa:>17.1%}  "
            f"{gc:>13.1%}{tag}")

    log("\n" + "-" * 78)
    log("Threshold view for a moderate upweight vs baseline")
    log("-" * 78)
    for wm in (1, 3, 5):
        R = results[wm]
        log(f"\n  w_motion = {wm}   (keep/del AUC {R['auc']:.3f}, "
            f"median motion score {np.median(R['scores'][R['motion']]):.3f})")
        log(f"    {'thr':>5}  {'motion caught':>13}  {'real false-AR':>13}  "
            f"{'garbage caught':>14}")
        for thr in THRESHOLDS:
            mc = frac_below(R["scores"], R["motion"], thr)
            fa = frac_below(R["scores"], R["real"],   thr)
            gc = frac_below(R["scores"], R["otherd"], thr)
            log(f"    {thr:>5.2f}  {mc:>12.1%}  {fa:>12.1%}  {gc:>13.1%}")

    log("\n" + "=" * 78)
    log("How to read this")
    log("=" * 78)
    log("  'motion caught'  = fraction of motion-tagged cells the model would now")
    log("                     auto-reject (score < threshold). Higher = better.")
    log("  'real false-AR'  = fraction of genuine cells wrongly auto-rejected. This")
    log("                     is the cost; the deployed model holds it under ~1%.")
    log("  If motion-caught rises steeply while false-AR stays sub-1%, the upweight")
    log("  is a cheap win on today's features. If motion-caught barely moves, or")
    log("  false-AR climbs with it, the 13 features lack the motion signal and the")
    log("  upweight cannot manufacture it -- that argues for the trace/spatial")
    log("  features instead. Remember motion-caught here is cohort-optimistic.")


if __name__ == "__main__":
    main()
