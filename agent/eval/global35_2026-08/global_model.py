"""
global_model.py -- is a pooled BLA + vCA1 model worth it, now that both areas
run the 35-column contract?

Background: the cross-area question was last answered on the SHARED FIRST 13
COLUMNS (red team attack #18, agent/eval/bootstrap_redteam_2026-08/results/
a18.json): null for BLA and vCA1 (|delta| <= 0.004 AUC, both source weights).
Since 2026-08-20/26 both areas carry 35 columns, 13 of which are WITHIN-SESSION
RANKS -- which normalise away area- and session-level scale, the very thing that
made raw-feature pooling fail.  So the answer may have changed.  This measures it.

Design (mirrors a18's Cond A / Cond B, with two deliberate differences noted below)

  Cond A   target area alone, exactly the deployed recipe.
  Cond B   Cond A + every row of the OTHER area appended to training.
             w1.0      other area's native weights (agent 1.0, bootstrap its own)
             w0.3      the same, x0.3
             matched   other area's AGENT rows lifted to the target's own per-fold
                       agent weight (so "pooling" is not silently a 4-7x downweight)
             zscore    w1.0, but each area's features standardised within its own
                       area first (target mu/sd from TRAINING rows only), which is
                       the standard fix for cross-domain pooling
  Widths   35 (the question) and 13 (control: must reproduce a18's null)
  Grouping session (reproduces the deployed pins) and animal (the honest read;
           the test animals' own bootstrap sessions are dropped from training,
           as in a08/a18)
  Joint    one model over BOTH areas' CV sessions in shared folds, scored per
           area from the same fit -- the literal "one global model", plus whether
           a single threshold can serve both.

Differences from a18, both on purpose:
  - fit_predict(drop_zero_weight=False), the deployed harness convention.  a18
    used True to reproduce c5_global_model.py.  Consequence: Cond A here is
    bit-identical to the DEPLOYED out-of-fold scores, which is asserted.
  - Cond A is computed once per (target, width, grouping) rather than once per
    source weight; it does not depend on the source weight.

8 seeds.  Read-only: no session dir, npz, joblib or label file is written.
Outputs global35.json + oof fixtures in this directory.

Usage:  python global_model.py [--probe] [--skip-13col]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
RT = HERE.parents[0] / "bootstrap_redteam_2026-08"
sys.path.insert(0, str(RT))
import rt_lib as rt  # noqa: E402

PARETO_OOF = HERE.parents[0] / "pareto_2026-08" / "bla_deployed_oof.npz"
VCA1_OOF = HERE.parents[0] / "vca1_v2_2026-08" / "gate_b0_w5_oof.npz"
OUT = HERE / "global35.json"

AREAS = ("BLA", "vCA1")
# rt_lib.EXPECTED_WIDTH predates the 2026-08-26 vCA1 deploy and still says 13;
# both areas are 35 columns live (verified on disk).  rt_lib is the red team's
# pinned evaluator, so the width is overridden here rather than edited there.
LIVE_WIDTH = {"BLA": 35, "vCA1": 35}
T_DEPLOY = {"BLA": 0.04, "vCA1": 0.04}
B_CONDS_FULL = ("w1.0", "w0.3", "matched", "zscore")
B_CONDS_LITE = ("w1.0", "w0.3")


# ---------------------------------------------------------------------------
# pooling helpers
# ---------------------------------------------------------------------------

def zscore_blocks(X_target_tr, X_target_te, X_other):
    """Standardise each area within itself.  Target statistics come from the
    training rows only; the other area is standardised by its own statistics."""
    def stats(A):
        mu = A.mean(0)
        sd = A.std(0)
        sd[sd < 1e-12] = 1.0
        return mu, sd
    mt, st = stats(X_target_tr)
    mo, so = stats(X_other)
    return ((X_target_tr - mt) / st, (X_target_te - mt) / st, (X_other - mo) / so)


def other_rows(pools, target, width):
    """Every row of the non-target area: X, y, native weight, and an agent mask."""
    Xs, ys, ws, ags = [], [], [], []
    for area, recs in pools.items():
        if area == target:
            continue
        for r in recs:
            Xs.append(r["X"][:, :width])
            ys.append(r["y"])
            ws.append(r["w"])
            ags.append(np.full(len(r["y"]), not r["is_bootstrap"]))
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(ws),
            np.concatenate(ags).astype(bool))


# ---------------------------------------------------------------------------
# Cond A / Cond B for one target area
# ---------------------------------------------------------------------------

def evaluate(target, pools, width, grouping, b_conds, seeds, counter):
    recs = pools[target]
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g_sess, rev, animals, names = rt.stack_agent(cv)
    X_ag = X_ag[:, :width]
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs, width=LIVE_WIDTH[target])
    X_bs = X_bs[:, :width]
    bs_animals = (np.concatenate([[r["animal"]] * len(r["y"]) for r in bs])
                  if bs else np.zeros(0, dtype="<U8"))
    X_ot, y_ot, w_ot, ag_ot = other_rows(pools, target, width)
    override = rt.AGENT_WEIGHT_OVERRIDE.get(target)
    T = T_DEPLOY[target]

    if grouping == "session":
        groups = g_sess
    else:
        ids = {a: i for i, a in enumerate(sorted(set(animals)))}
        groups = np.array([ids[a] for a in animals])
    n_groups = len(np.unique(groups))

    conds = ["A"] + list(b_conds)
    oof = {c: np.full((len(seeds), len(y)), np.nan) for c in conds}

    # With fewer than 3 groups the fold split is fixed, so varying the CV seed
    # changes nothing; vary the xgboost seed instead so the 8 replicates are
    # genuinely independent (the a08/a18 convention for 2-animal LOAO).
    for si, seed in enumerate(seeds):
        xseed = 42 if n_groups >= 3 else seed
        splits = list(StratifiedGroupKFold(n_splits=min(5, n_groups), shuffle=True,
                                           random_state=seed).split(X_ag, y, groups))
        for tr, te in splits:
            if len(np.unique(y[te])) < 2:
                continue
            if grouping == "session" or len(bs_animals) == 0:
                keep = np.ones(len(y_bs), bool)
            else:
                keep = ~np.isin(bs_animals, list(set(animals[te])))
            agw = rt.fold_agent_weight(int(keep.sum()), len(tr), override)
            baseX = [X_ag[tr], X_bs[keep]]
            baseY = [y[tr], y_bs[keep]]
            baseW = [np.full(len(tr), agw), w_bs[keep]]

            for cond in conds:
                if cond == "A":
                    Xt = np.vstack(baseX); yt = np.concatenate(baseY); wt = np.concatenate(baseW)
                    Xte = X_ag[te]
                elif cond == "zscore":
                    a_tr, a_te, o = zscore_blocks(np.vstack(baseX), X_ag[te], X_ot)
                    Xt = np.vstack([a_tr, o]); Xte = a_te
                    yt = np.concatenate(baseY + [y_ot])
                    wt = np.concatenate(baseW + [w_ot])
                else:
                    w = np.array(w_ot, dtype=float)
                    if cond == "matched":
                        w[ag_ot] = agw
                    else:
                        w = w * float(cond[1:])
                    Xt = np.vstack(baseX + [X_ot]); Xte = X_ag[te]
                    yt = np.concatenate(baseY + [y_ot])
                    wt = np.concatenate(baseW + [w])
                oof[cond][si, te] = rt.fit_predict(Xt, yt, wt, Xte, xgb_seed=xseed)
                counter[0] += 1

    res = {"target": target, "width": width, "grouping": grouping,
           "n_cv_sessions": len(cv), "n_rows": int(len(y)), "n_real": int(y.sum()),
           "n_groups": int(n_groups), "n_other_rows": int(len(y_ot)),
           "n_other_real": int(y_ot.sum()), "T": T, "conds": {}}
    aucA = [rt.auc(y, o) for o in oof["A"]]
    for cond in conds:
        o = oof[cond]
        aucs = [rt.auc(y, s) for s in o]
        aucs_rev = [rt.auc(y, s, rev) for s in o]
        far = [rt.far_junk(s, y, T)[0] for s in o]
        junk = [rt.far_junk(s, y, T)[1] for s in o]
        tab = rt.threshold_table(o, y, rev)
        cT, gate, det = rt.rule_T(tab)
        entry = {"auc_full": rt.summarize(aucs), "auc_reviewed": rt.summarize(aucs_rev),
                 "far_at_T": rt.summarize(far), "junk_at_T": rt.summarize(junk),
                 "rule_T": cT}
        if cond != "A":
            entry["delta_vs_A"] = rt.paired_delta(aucA, aucs)
            mop = rt.matched_operating_point(oof["A"], o, y, T)
            entry["junk_at_matched_far"] = mop["junk_new_at_matched_far_mean"]
            entry["junk_A_ref"] = mop["junk_ref_mean"]
            entry["far_at_matched_junk"] = mop["far_new_at_matched_junk_mean"]
        res["conds"][cond] = entry
        d = entry.get("delta_vs_A")
        rt.log(f"  [{target:5s} {width}col {grouping:7s} {cond:8s}] AUC {entry['auc_full']['mean']:.4f}"
               + (f"  delta {d['mean']:+.4f} (min {d['min']:+.4f}, {d['n_positive']}/{d['n']})"
                  f"  junk@matchedFAR {entry['junk_A_ref']:.1f}->{entry['junk_at_matched_far']:.1f}%" if d else "")
               + f"  FAR@{T} {entry['far_at_T']['mean']:.2f}%  junk {entry['junk_at_T']['mean']:.1f}%")
    return res, oof, y, rev, names


# ---------------------------------------------------------------------------
# one joint model over both areas
# ---------------------------------------------------------------------------

def joint(pools, grouping, seeds, counter, width=35):
    """A single model trained on both areas' sessions in shared folds."""
    agX, agY, agRev, agArea, agGroup, agAnimal = [], [], [], [], [], []
    bsX, bsY, bsW, bsArea, bsAnimal = [], [], [], [], []
    gid = 0
    agw_area = []
    for area in AREAS:
        cv, rest, bs = rt.split_pool(pools[area])
        for r in cv:
            n = len(r["y"])
            agX.append(r["X"][:, :width]); agY.append(r["y"]); agRev.append(r["reviewed"])
            agArea.append([area] * n); agGroup.append([gid] * n)
            agAnimal.append([r["animal"] or "?"] * n)
            gid += 1
        for r in bs:
            n = len(r["y"])
            bsX.append(r["X"][:, :width]); bsY.append(r["y"]); bsW.append(r["w"])
            bsArea.append([area] * n); bsAnimal.append([r["animal"] or "?"] * n)
    X_ag = np.vstack(agX); y = np.concatenate(agY); rev = np.concatenate(agRev)
    area_ag = np.concatenate(agArea); g_sess = np.concatenate(agGroup)
    animals = np.concatenate(agAnimal)
    X_bs = np.vstack(bsX); y_bs = np.concatenate(bsY); w_bs = np.concatenate(bsW)
    area_bs = np.concatenate(bsArea); bs_animals = np.concatenate(bsAnimal)

    if grouping == "session":
        groups = g_sess
    else:
        ids = {a: i for i, a in enumerate(sorted(set(animals)))}
        groups = np.array([ids[a] for a in animals])
    n_groups = len(np.unique(groups))

    oof = np.full((len(seeds), len(y)), np.nan)
    for si, seed in enumerate(seeds):
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True,
                                           random_state=seed).split(X_ag, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            if grouping == "session":
                keep = np.ones(len(y_bs), bool)
            else:
                keep = ~np.isin(bs_animals, list(set(animals[te])))
            # each area keeps its own deployed agent-weight recipe
            w_tr = np.empty(len(tr))
            for area in AREAS:
                m = area_ag[tr] == area
                n_bs_a = int((keep & (area_bs == area)).sum())
                n_ag_a = int(m.sum())
                w_tr[m] = rt.fold_agent_weight(n_bs_a, n_ag_a, rt.AGENT_WEIGHT_OVERRIDE.get(area))
            Xt = np.vstack([X_ag[tr], X_bs[keep]])
            yt = np.concatenate([y[tr], y_bs[keep]])
            wt = np.concatenate([w_tr, w_bs[keep]])
            oof[si, te] = rt.fit_predict(Xt, yt, wt, X_ag[te], xgb_seed=42)
            counter[0] += 1

    res = {"grouping": grouping, "n_sessions": int(len(np.unique(g_sess))),
           "n_rows": int(len(y)), "n_real": int(y.sum()), "n_groups": int(n_groups),
           "per_area": {}, "shared_threshold": {}}
    for area in AREAS:
        m = area_ag == area
        T = T_DEPLOY[area]
        res["per_area"][area] = {
            "n_rows": int(m.sum()), "n_real": int(y[m].sum()),
            "auc_full": rt.summarize([rt.auc(y[m], s[m]) for s in oof]),
            "auc_reviewed": rt.summarize([rt.auc(y[m], s[m], rev[m]) for s in oof]),
            "far_at_deployed_T": rt.summarize([rt.far_junk(s[m], y[m], T)[0] for s in oof]),
            "junk_at_deployed_T": rt.summarize([rt.far_junk(s[m], y[m], T)[1] for s in oof]),
            "rule_T": rt.rule_T(rt.threshold_table(oof[:, m], y[m], rev[m]))[0],
        }
    for t in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        row = {}
        for area in AREAS:
            m = area_ag == area
            f = [rt.far_junk(s[m], y[m], t) for s in oof]
            row[area] = {"far": float(np.mean([x[0] for x in f])),
                         "far_worst": float(np.max([x[0] for x in f])),
                         "junk": float(np.mean([x[1] for x in f]))}
        res["shared_threshold"][f"{t:.3f}"] = row
    rt.log(f"  [joint {grouping:7s}] " + " | ".join(
        f"{a}: AUC {res['per_area'][a]['auc_full']['mean']:.4f} "
        f"FAR@{T_DEPLOY[a]} {res['per_area'][a]['far_at_deployed_T']['mean']:.2f}% "
        f"junk {res['per_area'][a]['junk_at_deployed_T']['mean']:.1f}%" for a in AREAS))
    return res, oof, y, rev, area_ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="time a single pooled fit and exit")
    ap.add_argument("--skip-13col", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rt.log("loading pools ...")
    pools = {a: rt.load_pool(a, width=LIVE_WIDTH[a]) for a in AREAS}
    for a in AREAS:
        cv, rest, bs = rt.split_pool(pools[a])
        X, y, *_ = rt.stack_agent(cv)
        Xb, yb, wb = rt.stack_bootstrap(bs, width=35)
        rt.log(f"  {a}: {len(pools[a])} labeled ({len(cv)} CV / {len(bs)} bootstrap), "
               f"{len(y)} CV rows / {int(y.sum())} real, {len(yb)} bootstrap rows "
               f"({int((wb == 0).sum())} masked)")

    if args.probe:
        counter = [0]
        t = time.time()
        evaluate("BLA", pools, 35, "session", ("w1.0",), [42], counter)
        dt = time.time() - t
        rt.log(f"probe: {counter[0]} fits in {dt:.1f}s -> {dt / counter[0]:.2f}s per fit")
        rt.log(f"full run is ~960 fits -> ~{960 * dt / counter[0] / 60:.0f} min")
        return

    out = {"seeds": rt.SEEDS, "deployed_T": T_DEPLOY,
           "note": "Cond A = deployed recipe; Cond B appends the other area. Read-only.",
           "aux": {}, "joint": {}, "selfcheck": {}}
    counter = [0]
    store = {}

    grid = [(35, "session", B_CONDS_FULL), (35, "animal", B_CONDS_LITE)]
    if not args.skip_13col:
        grid.append((13, "session", B_CONDS_LITE))
    for width, grouping, bconds in grid:
        for target in AREAS:
            rt.log(f"--- {target} {width}col {grouping} ---")
            res, oof, y, rev, names = evaluate(target, pools, width, grouping,
                                               bconds, rt.SEEDS, counter)
            key = f"{target}_{width}col_{grouping}"
            out["aux"][key] = res
            store[f"{key}__y"] = y
            for c, o in oof.items():
                store[f"{key}__{c}"] = o
            if width == 35 and grouping == "session":
                chk = {"max_abs_score_diff": float(np.max(np.abs(
                    (np.load(PARETO_OOF)["oof_session"] if target == "BLA"
                     else np.load(VCA1_OOF)["oof_v35"]) - oof["A"])))}
                chk["pass"] = chk["max_abs_score_diff"] < 1e-9
                out["selfcheck"][target] = chk
                rt.log(f"  self-check vs DEPLOYED oof: "
                       f"{'PASS' if chk['pass'] else 'FAIL'} "
                       f"(max |diff| {chk['max_abs_score_diff']:.2e})")

    for grouping in ("session", "animal"):
        rt.log(f"--- joint model, {grouping} ---")
        res, oof, y, rev, area_ag = joint(pools, grouping, rt.SEEDS, counter)
        out["joint"][grouping] = res
        store[f"joint_{grouping}__oof"] = oof
        store[f"joint_{grouping}__y"] = y
        store[f"joint_{grouping}__area"] = area_ag

    out["n_fits"] = counter[0]
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, default=rt._jsonable))
    np.savez_compressed(HERE / "global35_oof.npz", **store)
    rt.log(f"\n{counter[0]} fits in {out['runtime_s'] / 60:.1f} min")
    rt.log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
