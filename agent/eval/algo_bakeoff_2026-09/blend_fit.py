"""
blend_fit.py -- README round-2 amendment: a 3-parameter logistic combiner of
logit(control OOF) and logit(CNN OOF), fitted per outer fold on the training
folds' agent rows and applied to the test fold (nested by construction), versus
the fixed 50/50 probability blend.

Rows without a CNN score (auto-rejected candidates, no raw data) keep the
control score, as in stack_eval.  Folds are the control's (StratifiedGroupKFold
on the same y / groups / seed).

Usage: python blend_fit.py --area BLA --grouping session,animal [--cnn <npz>] [--name r1]
Outputs results/r2/blendfit_<name>_<AREA>_<grouping>.{npz,json}, appends to results/r2/blendfit_<name>.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import stack_eval as SE  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
R2 = RESULTS / "r2"


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fitted_blend(ref, cnn, y, groups, n_splits, seed):
    """Per-seed OOF of the logistic combiner; returns (oof, per-fold coefficients)."""
    out = ref.copy()
    has = ~np.isnan(cnn)
    Z = np.stack([logit(ref), logit(np.where(has, cnn, 0.5))], axis=1)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    coefs = []
    for tr, te in cv.split(np.zeros((len(y), 1)), y, groups):
        trh = tr[has[tr]]
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(Z[trh], y[trh])
        teh = te[has[te]]
        if len(teh):
            out[teh] = lr.predict_proba(Z[teh])[:, 1]
        coefs.append([float(lr.coef_[0, 0]), float(lr.coef_[0, 1]), float(lr.intercept_[0])])
    return out, coefs


def run_cell(area, grouping, cnn_path, name):
    ctrl = np.load(RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
    cnn = np.load(cnn_path, allow_pickle=True)
    assert np.array_equal(ctrl["y"], cnn["y"]), "rows differ"
    y, rev = ctrl["y"].astype(int), ctrl["reviewed"].astype(bool)
    groups = ctrl["groups_cv"].astype(int)
    n_splits = min(5, len(np.unique(groups)))
    seeds = [int(s) for s in cnn["seeds"]]
    ci = [[int(t) for t in ctrl["seeds"]].index(s) for s in seeds]
    ref, new = ctrl["oof"][ci], cnn["oof"].astype(float)
    T = B.T_DEPLOY[area]
    fixed = np.where(np.isnan(new), ref, 0.5 * (ref + np.nan_to_num(new)))
    fit = np.zeros_like(ref)
    coefs = []
    t0 = time.time()
    for si, s in enumerate(seeds):
        fit[si], c = fitted_blend(ref[si], new[si], y, groups, n_splits, s)
        coefs.append(c)
    blocks = {
        "fixed_vs_control": SE.block(y, rev, ref, fixed, T, "fixed 50/50 blend vs control"),
        "fitted_vs_control": SE.block(y, rev, ref, fit, T, "fitted logistic combiner vs control"),
        "fitted_vs_fixed": SE.block(y, rev, fixed, fit, T, "fitted combiner vs fixed blend"),
    }
    c_arr = np.array(coefs)                      # (seeds, folds, 3)
    w_cnn = c_arr[:, :, 1] / (c_arr[:, :, 0] + c_arr[:, :, 1])
    out = {"area": area, "grouping": grouping, "cnn": str(cnn_path), "name": name, "seeds": seeds, "T": T,
           "blocks": blocks, "coef_mean": c_arr.mean(axis=(0, 1)).tolist(),
           "cnn_share_of_logit_weight": {"mean": float(w_cnn.mean()), "min": float(w_cnn.min()), "max": float(w_cnn.max())},
           "runtime_s": time.time() - t0, "written_at": dt.datetime.now().isoformat(timespec="seconds")}
    R2.mkdir(exist_ok=True)
    stem = f"blendfit_{name}_{area}_{grouping}"
    np.savez_compressed(R2 / f"{stem}.npz", oof_fitted=fit, oof_fixed=fixed, y=y, reviewed=rev, seeds=np.array(seeds),
                        coefs=c_arr)
    (R2 / f"{stem}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    lines = [f"## {area} {grouping}-grouped, CNN = {cnn_path.name}, {len(seeds)} seeds ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
             "| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [SE.fmt(blocks[k]) for k in ("fixed_vs_control", "fitted_vs_control", "fitted_vs_fixed")]
    lines += ["", f"combiner coefficients (mean over seeds x folds): control-logit {out['coef_mean'][0]:.3f}, CNN-logit "
              f"{out['coef_mean'][1]:.3f}, intercept {out['coef_mean'][2]:.3f}; CNN share of logit weight "
              f"{out['cnn_share_of_logit_weight']['mean']:.2f} [{out['cnn_share_of_logit_weight']['min']:.2f}, "
              f"{out['cnn_share_of_logit_weight']['max']:.2f}]", ""]
    with (R2 / f"blendfit_{name}.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA")
    ap.add_argument("--grouping", default="session,animal")
    ap.add_argument("--cnn", default="")
    ap.add_argument("--name", default="r1")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    for grouping in [g.strip() for g in a.grouping.split(",") if g.strip()]:
        p = Path(a.cnn) if a.cnn else RESULTS / f"{a.area}_{grouping}_cnn_t8.npz"
        if not p.exists():
            log(f"missing {p}")
            continue
        run_cell(a.area, grouping, p, a.name)


if __name__ == "__main__":
    main()
