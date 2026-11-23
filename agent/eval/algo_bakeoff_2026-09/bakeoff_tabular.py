"""
bakeoff_tabular.py -- Stage 1 of the 2026-09 algorithm bake-off: on the SAME
35-column contract, rows, weights, folds and seeds as the deployed model, does
any other tabular learner beat the deployed (never tuned) XGBoost?

Built on the red team's independent evaluator rt_lib (loading, labels,
weights, CV structure, metrics) so every number is on the deployed protocol.
rt_lib is NOT edited; its per-fold fit is re-implemented here for the other
learners (run_oof_generic mirrors rt.run_oof: agent-train-fold rows first,
then bootstrap rows in sorted-walk order, StandardScaler on the training
block, weight-0 rows kept in the xgboost/lightgbm fit call).  The control
contender xgb_deployed calls rt.make_xgb / rt.run_oof directly.

Tuned contenders use NESTED selection: per outer fold, one inner
session-grouped split (80/20) of the outer-train agent rows; every sampled
configuration is fit with early stopping on the inner-val rows and scored by
inner-val AUC; the best is refit on the whole outer-train block with its
early-stopped size and scored on the outer test fold.  The outer test fold
never informs the choice.

Read-only with respect to the corpus.  Outputs: results/*.npz (OOF scores),
results/*.json (summaries), results/report_*.md (--report).

See README.md for the pre-registered decision rule and the contender list.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path


def _early_threads() -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 8


THREADS = _early_threads()
if THREADS > 0:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
N_JOBS = THREADS if THREADS > 0 else -1

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
RT = HERE.parents[0] / "bootstrap_redteam_2026-08"
sys.path.insert(0, str(RT))
import rt_lib as rt  # noqa: E402

RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
PARETO_OOF = HERE.parents[0] / "pareto_2026-08" / "bla_deployed_oof.npz"
# rt_lib.EXPECTED_WIDTH / DEPLOYED_T predate the 2026-08-26 vCA1 deploy; both
# areas are 35 columns at T=0.04 live (verified on disk 2026-09-06).
LIVE_WIDTH = {"BLA": 35, "vCA1": 35, "DG_AL": 13}
T_DEPLOY = {"BLA": 0.04, "vCA1": 0.04, "DG_AL": 0.0}
TAG = f"t{THREADS}"

# ---------------------------------------------------------------------------
# Contenders
# ---------------------------------------------------------------------------
# agent/train_classifier.py:429-440 (identical to rt.make_xgb)
DEPLOYED_XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, gamma=0.0)
XGB_SPACE = dict(max_depth=[3, 4, 5, 6, 8], learning_rate=[0.02, 0.03, 0.05, 0.08, 0.10],
                 min_child_weight=[1, 5, 20], subsample=[0.6, 0.8, 1.0],
                 colsample_bytree=[0.5, 0.8, 1.0], reg_lambda=[1.0, 5.0, 20.0],
                 gamma=[0.0, 0.5, 2.0])
# agent/train_classifier.py:444-454 verbatim.  NOTE: LightGBM ignores `subsample`
# unless subsample_freq > 0 (default 0), so the production mirror does not bag.
DEPLOYED_LGBM = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                     colsample_bytree=0.8)
LGBM_SPACE = dict(max_depth=[3, 4, 5, 6, 8], learning_rate=[0.02, 0.03, 0.05, 0.08, 0.10],
                  min_child_samples=[5, 20, 50], subsample=[0.6, 0.8, 1.0], subsample_freq=[1],
                  colsample_bytree=[0.5, 0.8, 1.0], reg_lambda=[0.0, 1.0, 5.0, 20.0])
CAT_DEFAULT = dict(iterations=1000, learning_rate=0.05, depth=6)
CAT_SPACE = dict(depth=[4, 6, 8], learning_rate=[0.03, 0.05, 0.10], l2_leaf_reg=[1, 3, 10],
                 random_strength=[1, 5], bagging_temperature=[0, 1])
HISTGB_CFG = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0)
ET_CFG = dict(n_estimators=500, min_samples_leaf=5, max_features="sqrt")
MLP_CFG = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=200,
               early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, batch_size=256)

CONTENDERS = {
    "xgb_deployed": dict(family="xgb", cfg=DEPLOYED_XGB),
    "xgb_tuned": dict(family="xgb", space=XGB_SPACE, n_configs=30),
    "lgbm_deployed": dict(family="lgbm", cfg=DEPLOYED_LGBM),
    "lgbm_tuned": dict(family="lgbm", space=LGBM_SPACE, n_configs=30),
    "cat_default": dict(family="cat", cfg=CAT_DEFAULT),
    "cat_tuned": dict(family="cat", space=CAT_SPACE, n_configs=10),
    "histgb": dict(family="histgb", cfg=HISTGB_CFG),
    "extratrees": dict(family="et", cfg=ET_CFG),
    "mlp": dict(family="mlp", cfg=MLP_CFG),
}
N_KEY = {"xgb": "n_estimators", "lgbm": "n_estimators", "cat": "iterations"}
REPORT_ORDER = ["xgb_deployed", "xgb_tuned", "xgb_winner", "lgbm_deployed", "lgbm_tuned",
                "lgbm_winner", "cat_default", "cat_tuned", "cat_winner", "histgb", "extratrees",
                "mlp", "ens_mean"]
ENS_COMPONENTS = ["xgb_deployed", "lgbm_deployed", "cat_default"]


def log(msg: str = ""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def versions() -> dict:
    out = {"python": sys.executable, "cpu_count": os.cpu_count(), "threads": THREADS,
           "n_jobs": N_JOBS, "platform": platform.platform(),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    for m in ("numpy", "scipy", "sklearn", "xgboost", "lightgbm", "catboost"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


# ---------------------------------------------------------------------------
# Pool (with a per-area snapshot so the rows cannot move under a contender)
# ---------------------------------------------------------------------------

def load_area_pool(area: str) -> list[dict]:
    width = LIVE_WIDTH[area]
    snap_p = RESULTS / f"pool_{area}.json"
    dirs = rt.labeled_sessions(area)
    current = {rt.rel(sd): rt._sha256(sd / "labels.mat") for sd in dirs}
    if snap_p.exists():
        snap = json.loads(snap_p.read_text())
        want = {s["name"]: s["labels_sha256"] for s in snap["sessions"]}
        missing = sorted(set(want) - set(current))
        changed = sorted(k for k in want if k in current and current[k] != want[k])
        if missing or changed:
            raise RuntimeError(f"{area}: pool moved since {snap_p.name}: "
                               f"missing={missing} changed={changed}")
        extra = sorted(set(current) - set(want))
        if extra:
            log(f"{area}: {len(extra)} session(s) ingested after the snapshot are EXCLUDED: {extra}")
        recs = rt.load_pool(area, width=width, exclude=set(extra))
        log(f"{area}: pool matches snapshot {snap_p.name} ({len(recs)} labeled sessions)")
        return recs
    recs = rt.load_pool(area, width=width)
    RESULTS.mkdir(exist_ok=True)
    snap = {"area": area, "created": dt.datetime.now().isoformat(timespec="seconds"), "width": width,
            "sessions": [{"name": r["name"], "is_bootstrap": bool(r["is_bootstrap"]),
                          "n_cand": int(r["n_cand"]), "n_pos": int(r["y"].sum()),
                          "animal": r["animal"], "labels_sha256": current[r["name"]]}
                         for r in recs]}
    snap_p.write_text(json.dumps(snap, indent=1))
    log(f"{area}: wrote pool snapshot {snap_p.name} ({len(recs)} labeled sessions)")
    try:  # how far the corpus has moved since the red team's pin, for the record
        old = rt.load_pin()
        old = {"areas": {area: old["areas"][area]}}
        d = rt.diff_pin(old, rt.compute_pin((area,)))[area]
        (RESULTS / f"pin_diff_{area}.json").write_text(json.dumps(d, indent=1))
        log(f"{area}: vs red-team pin_manifest: new={len(d['new'])} gone={len(d['gone'])} "
            f"changed={len(d['changed'])} joblib_changed={d['joblib_changed']}")
    except Exception as e:  # pragma: no cover
        log(f"{area}: pin diff unavailable: {e}")
    return recs


def prepare(area: str, recs: list[dict], grouping: str) -> dict:
    cv, rest, bs = rt.split_pool(recs)
    X_ag, y, g_sess, rev, animals, names = rt.stack_agent(cv)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs, width=LIVE_WIDTH[area])
    bs_animals = (np.concatenate([[r["animal"] or "?"] * len(r["y"]) for r in bs])
                  if bs else np.zeros(0, dtype="<U8"))
    if grouping == "session":
        groups, mask = g_sess, None
    elif grouping == "animal":
        ids = {a: i for i, a in enumerate(sorted(set(animals)))}
        groups = np.array([ids[a] for a in animals])

        def mask(te, _bsa=bs_animals, _an=animals):
            # the test animals' own bootstrap sessions leave the training block
            return ~np.isin(_bsa, np.unique(_an[te]))
    else:
        raise ValueError(grouping)
    n_groups = int(len(np.unique(groups)))
    n_splits = min(5, n_groups)
    return dict(cv=cv, rest=rest, bs=bs, X_ag=X_ag, y=y, g_sess=g_sess, rev=rev, animals=animals,
                names=names, X_bs=X_bs, y_bs=y_bs, w_bs=w_bs, groups=groups, mask=mask,
                n_groups=n_groups, n_splits=n_splits, fixed_split=(n_splits == n_groups),
                override=rt.AGENT_WEIGHT_OVERRIDE.get(area))


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

def _num_leaves(depth: int) -> int:
    return min(2 ** depth - 1, 63)


def build(family: str, cfg: dict, spw: float, seed: int, early_stopping: bool = False):
    if family == "xgb":
        kw = dict(cfg)
        if early_stopping:
            kw.update(n_estimators=2000, early_stopping_rounds=50)
        return XGBClassifier(**kw, scale_pos_weight=spw, eval_metric="auc", verbosity=0,
                             random_state=seed, n_jobs=N_JOBS)
    if family == "lgbm":
        from lightgbm import LGBMClassifier
        kw = dict(cfg)
        if early_stopping:
            kw.update(n_estimators=2000)
        return LGBMClassifier(**kw, scale_pos_weight=spw, random_state=seed, n_jobs=N_JOBS,
                              verbose=-1)
    if family == "cat":
        from catboost import CatBoostClassifier
        kw = dict(cfg)
        if early_stopping:
            kw.update(iterations=2000, od_type="Iter", od_wait=50, eval_metric="AUC")
        return CatBoostClassifier(**kw, scale_pos_weight=spw, random_seed=seed,
                                  thread_count=(N_JOBS if N_JOBS > 0 else -1), verbose=False,
                                  allow_writing_files=False)
    if family == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**cfg, random_state=seed, early_stopping=False)
    if family == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**cfg, n_jobs=N_JOBS, random_state=seed)
    if family == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**cfg, random_state=seed)
    raise ValueError(family)


def fit_predict_generic(family, cfg, X_tr, y_tr, w_tr, X_te, seed, eval_block=None):
    """One fit on the harness convention (scaler on the whole training block,
    spw = weighted neg/pos over w>0).  eval_block=(Xv, yv, wv) turns on early
    stopping for xgb/lgbm/cat and returns the early-stopped size."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    spw = rt.spw_of(y_tr, w_tr)
    best_iter = None
    if family == "mlp":
        Xtr, Xte = np.clip(Xtr, -5, 5), np.clip(Xte, -5, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family in ("xgb", "lgbm", "cat"):
            clf = build(family, cfg, spw, seed, early_stopping=eval_block is not None)
            if family == "cat":
                keep = w_tr > 0
                Xf, yf, wf = Xtr[keep], y_tr[keep], w_tr[keep]
            else:
                Xf, yf, wf = Xtr, y_tr, w_tr
            if eval_block is None:
                clf.fit(Xf, yf, sample_weight=wf)
            else:
                Xv = sc.transform(eval_block[0])
                yv, wv = eval_block[1], eval_block[2]
                if family == "xgb":
                    clf.fit(Xf, yf, sample_weight=wf, eval_set=[(Xv, yv)],
                            sample_weight_eval_set=[wv], verbose=False)
                    best_iter = int(clf.best_iteration) + 1
                elif family == "lgbm":
                    import lightgbm as lgb
                    clf.fit(Xf, yf, sample_weight=wf, eval_set=[(Xv, yv)],
                            eval_sample_weight=[wv], eval_metric="auc",
                            callbacks=[lgb.early_stopping(50, verbose=False)])
                    best_iter = int(clf.best_iteration_)
                else:
                    from catboost import Pool
                    clf.fit(Xf, yf, sample_weight=wf, eval_set=Pool(Xv, yv, weight=wv))
                    best_iter = int(clf.get_best_iteration()) + 1
            return clf.predict_proba(Xte)[:, 1], best_iter
        # sklearn learners: scale_pos_weight semantics via the positive sample weights
        w_eff = w_tr * np.where(y_tr == 1, spw, 1.0)
        keep = w_eff > 0
        clf = build(family, cfg, spw, seed)
        clf.fit(Xtr[keep], y_tr[keep], sample_weight=w_eff[keep])
        return clf.predict_proba(Xte)[:, 1], None


def sample_configs(family: str, space: dict, n: int) -> list[dict]:
    rng = random.Random(0)
    base = {"xgb": DEPLOYED_XGB, "lgbm": DEPLOYED_LGBM, "cat": CAT_DEFAULT}[family]
    configs = [dict(base)]
    seen = {json.dumps(base, sort_keys=True)}
    keys = sorted(space)
    tries = 0
    while len(configs) < n and tries < 10000:
        tries += 1
        c = {k: rng.choice(space[k]) for k in keys}
        if family == "lgbm":
            c["num_leaves"] = _num_leaves(c["max_depth"])
        j = json.dumps(c, sort_keys=True)
        if j in seen:
            continue
        seen.add(j)
        configs.append(c)
    return configs


# ---------------------------------------------------------------------------
# OOF harness (mirrors rt.run_oof)
# ---------------------------------------------------------------------------

def _train_block(P, tr, Xb, yb, wb, agw):
    X_ag, y = P["X_ag"], P["y"]
    X_tr = np.vstack([X_ag[tr], Xb]) if len(yb) else X_ag[tr]
    y_tr = np.concatenate([y[tr], yb])
    w_tr = np.concatenate([np.full(len(tr), agw), wb])
    return X_tr, y_tr, w_tr


def run_oof_generic(family, cfg, P, seed, configs=None):
    cv = StratifiedGroupKFold(n_splits=P["n_splits"], shuffle=True, random_state=seed)
    model_seed = seed if P["fixed_split"] else 42
    X_ag, y, g_sess = P["X_ag"], P["y"], P["g_sess"]
    oof = np.full(len(y), np.nan)
    trials, chosen = [], []
    for fi, (tr, te) in enumerate(cv.split(X_ag, y, P["groups"])):
        if P["mask"] is not None:
            m = P["mask"](te)
            Xb, yb, wb = P["X_bs"][m], P["y_bs"][m], P["w_bs"][m]
        else:
            Xb, yb, wb = P["X_bs"], P["y_bs"], P["w_bs"]
        agw = rt.fold_agent_weight(len(yb), len(tr), P["override"])
        if configs is None:
            X_tr, y_tr, w_tr = _train_block(P, tr, Xb, yb, wb, agw)
            oof[te], _ = fit_predict_generic(family, cfg, X_tr, y_tr, w_tr, X_ag[te], model_seed)
            continue
        # nested selection on one inner session-grouped split of the outer-train rows
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000 + seed)
        itr, ival = next(inner.split(X_ag[tr], y[tr], g_sess[tr]))
        itr, ival = tr[itr], tr[ival]
        X_in, y_in, w_in = _train_block(P, itr, Xb, yb, wb, agw)
        Xv, yv, wv = X_ag[ival], y[ival], np.ones(len(ival))
        best = None
        for ci, c in enumerate(configs):
            t0 = time.time()
            s, nb = fit_predict_generic(family, c, X_in, y_in, w_in, Xv, model_seed,
                                        eval_block=(Xv, yv, wv))
            a = rt.auc(yv, s)
            trials.append({"seed": seed, "fold": fi, "cfg": ci, "n_best": nb, "inner_auc": a,
                           "fit_s": round(time.time() - t0, 1)})
            if best is None or (np.isfinite(a) and a > best[1]):
                best = (ci, a, nb)
        ci, a, nb = best
        c = dict(configs[ci])
        c[N_KEY[family]] = nb
        X_tr, y_tr, w_tr = _train_block(P, tr, Xb, yb, wb, agw)
        oof[te], _ = fit_predict_generic(family, c, X_tr, y_tr, w_tr, X_ag[te], model_seed)
        chosen.append({"seed": seed, "fold": fi, "cfg": ci, "n_best": nb, "inner_auc": a})
        log(f"    seed {seed} fold {fi}: cfg {ci} (inner AUC {a:.4f}, {N_KEY[family]}={nb})")
    return oof, trials, chosen


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def result_stem(area, grouping, name) -> str:
    return f"{area}_{grouping}_{name}_{TAG}"


def save(area, grouping, name, oof, P, seeds, meta: dict):
    RESULTS.mkdir(exist_ok=True)
    stem = result_stem(area, grouping, name)
    y, rev = P["y"], P["rev"]
    np.savez_compressed(RESULTS / f"{stem}.npz", oof=oof, y=y, reviewed=rev, animals=P["animals"],
                        groups=P["g_sess"], groups_cv=P["groups"], names=P["names"],
                        seeds=np.array(seeds))
    T = T_DEPLOY[area]
    tab = rt.threshold_table(oof, y, rev)
    chosen, gate, det = rt.rule_T(tab)
    summ = {
        "area": area, "grouping": grouping, "contender": name, "seeds": list(seeds),
        "n_sessions": int(len(P["names"])), "n_rows": int(len(y)), "n_real": int(y.sum()),
        "n_reviewed": int(rev.sum()), "n_bootstrap_rows": int(len(P["y_bs"])),
        "n_groups": P["n_groups"], "n_splits": P["n_splits"], "fixed_split": bool(P["fixed_split"]),
        "agent_weight_override": P["override"],
        "auc_full": [rt.auc(y, o) for o in oof], "auc_reviewed": [rt.auc(y, o, rev) for o in oof],
        "far_at_T": [rt.far_junk(o, y, T)[0] for o in oof],
        "junk_at_T": [rt.far_junk(o, y, T)[1] for o in oof],
        "junk_reviewed_at_T": [rt.far_junk(o, y, T, rev)[2] for o in oof],
        "T_deploy": T, "rule_T": chosen, "rule_gate": gate, "rule_detail": det,
        "threshold_table": tab, "env": versions(),
        "written_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    summ.update(meta)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summ, indent=1, default=rt._jsonable))
    log(f"  {name:14s} AUC full {np.mean(summ['auc_full']):.4f}  reviewed {np.mean(summ['auc_reviewed']):.4f}"
        f" | FAR@{T} {np.mean(summ['far_at_T']):.2f}% (worst {max(summ['far_at_T']):.2f})"
        f" junk {np.mean(summ['junk_at_T']):.1f}% | rule_T {chosen} | {meta.get('runtime_s', 0):.0f}s")


def run_control(area, grouping, P, seeds):
    t0 = time.time()
    oofs = []
    for s in seeds:
        oofs.append(rt.run_oof(P["X_ag"], P["y"], P["groups"], P["X_bs"], P["y_bs"], P["w_bs"], s,
                               agent_weight=P["override"], n_splits=P["n_splits"],
                               xgb_seed=(s if P["fixed_split"] else 42), bs_train_mask=P["mask"],
                               n_jobs=N_JOBS))
        log(f"    seed {s} done ({time.time() - t0:.0f}s)")
    save(area, grouping, "xgb_deployed", np.array(oofs), P, seeds,
         {"family": "xgb", "cfg": DEPLOYED_XGB, "runtime_s": time.time() - t0,
          "note": "rt.run_oof / rt.make_xgb verbatim"})


def run_fixed(area, grouping, name, family, cfg, P, seeds, extra=None):
    t0 = time.time()
    oofs = []
    for s in seeds:
        oofs.append(run_oof_generic(family, cfg, P, s)[0])
        log(f"    seed {s} done ({time.time() - t0:.0f}s)")
    meta = {"family": family, "cfg": cfg, "runtime_s": time.time() - t0}
    if extra:
        meta.update(extra)
    save(area, grouping, name, np.array(oofs), P, seeds, meta)


def run_tuned(area, grouping, name, spec, P, seeds):
    t0 = time.time()
    configs = sample_configs(spec["family"], spec["space"], spec["n_configs"])
    log(f"  {name}: {len(configs)} configs (cfg 0 = deployed shape), nested on an inner 80/20 session split")
    oofs, trials, chosen = [], [], []
    for s in seeds:
        o, t, c = run_oof_generic(spec["family"], None, P, s, configs=configs)
        oofs.append(o)
        trials += t
        chosen += c
    save(area, grouping, name, np.array(oofs), P, seeds,
         {"family": spec["family"], "configs": configs, "trials": trials, "chosen": chosen,
          "runtime_s": time.time() - t0})


def winner_config(area, grouping, tuned_name):
    j = json.loads((RESULTS / f"{result_stem(area, grouping, tuned_name)}.json").read_text())
    configs, chosen = j["configs"], j["chosen"]
    by = {}
    for c in chosen:
        by.setdefault(c["cfg"], []).append(c)
    best = max(by, key=lambda k: (len(by[k]), float(np.mean([c["inner_auc"] for c in by[k]]))))
    n = int(round(float(np.median([c["n_best"] for c in by[best]]))))
    cfg = dict(configs[best])
    cfg[N_KEY[j["family"]]] = n
    info = {"winner_of": tuned_name, "cfg_index": best, "times_chosen": len(by[best]),
            "of_folds": len(chosen),
            "mean_inner_auc": float(np.mean([c["inner_auc"] for c in by[best]]))}
    return j["family"], cfg, info


# ---------------------------------------------------------------------------
# Reproduction of the pinned deployed OOF (pareto_2026-08)
# ---------------------------------------------------------------------------

def repro(area, recs):
    ref = np.load(PARETO_OOF, allow_pickle=True)
    names_ref = [str(n) for n in ref["names"]]
    cv, rest, bs = rt.split_pool(recs)
    cv_ref = [r for r in cv if r["name"] in set(names_ref)]
    names_got = [r["name"] for r in cv_ref]
    if names_got != names_ref:
        log(f"REPRO: session list differs from the reference: "
            f"missing={sorted(set(names_ref) - set(names_got))} order_ok={sorted(names_got) == sorted(names_ref)}")
    X_ag, y, g, rev, animals, names = rt.stack_agent(cv_ref)
    X_bs, y_bs, w_bs = rt.stack_bootstrap(bs, width=LIVE_WIDTH[area])
    same_y = len(y) == len(ref["y"]) and bool(np.array_equal(y, ref["y"]))
    log(f"REPRO: {len(cv_ref)} sessions, {len(y)} rows (ref {len(ref['y'])}), labels identical: {same_y}; "
        f"{len(y_bs)} bootstrap rows; n_jobs={N_JOBS}")
    t0 = time.time()
    oof = rt.run_oof_seeds(X_ag, y, g, X_bs, y_bs, w_bs, agent_weight=None, n_jobs=N_JOBS)
    out = {"area": area, "threads": THREADS, "n_jobs": N_JOBS, "rows": int(len(y)),
           "labels_identical": same_y, "runtime_s": round(time.time() - t0, 1), "env": versions()}
    if same_y:
        r = ref["oof_session"]
        out["max_abs_score_diff_per_seed"] = [float(np.max(np.abs(a - b))) for a, b in zip(oof, r)]
        fns = {"auc_full": lambda o: rt.auc(y, o), "auc_reviewed": lambda o: rt.auc(y, o, rev),
               "far_0.04": lambda o: rt.far_junk(o, y, 0.04)[0],
               "junk_0.04": lambda o: rt.far_junk(o, y, 0.04)[1],
               "far_0.03": lambda o: rt.far_junk(o, y, 0.03)[0],
               "junk_0.03": lambda o: rt.far_junk(o, y, 0.03)[1]}
        stats = {}
        for k, f in fns.items():
            a = float(np.mean([f(o) for o in oof]))
            b = float(np.mean([f(o) for o in r]))
            stats[k] = {"recomputed": a, "reference": b, "abs_diff": abs(a - b)}
        out["stats"] = stats
        out["pass_5e-3"] = all(v["abs_diff"] < 5e-3 for v in stats.values())
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / f"repro_{area}_{TAG}.npz", oof=oof, y=y, reviewed=rev, names=names,
                        seeds=np.array(rt.SEEDS))
    (RESULTS / f"repro_{area}_{TAG}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    for k, v in out.get("stats", {}).items():
        log(f"REPRO: {k:13s} recomputed {v['recomputed']:.5f}  reference {v['reference']:.5f}  "
            f"abs diff {v['abs_diff']:.2e}")
    log(f"REPRO: max |score diff| per seed = {out.get('max_abs_score_diff_per_seed')}")
    log(f"REPRO: pass(<5e-3 on all six) = {out.get('pass_5e-3')}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def load_res(area, grouping, name):
    p = RESULTS / f"{result_stem(area, grouping, name)}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    j = json.loads(p.with_suffix(".json").read_text())
    return {"oof": z["oof"], "y": z["y"], "rev": z["reviewed"], "seeds": [int(s) for s in z["seeds"]],
            "json": j}


def align(ref, new):
    common = [s for s in new["seeds"] if s in ref["seeds"]]
    ri = [ref["seeds"].index(s) for s in common]
    ni = [new["seeds"].index(s) for s in common]
    return ref["oof"][ri], new["oof"][ni], common


def ensemble(area, grouping, ref):
    comps = [(n, load_res(area, grouping, n)) for n in ENS_COMPONENTS]
    comps = [(n, r) for n, r in comps if r is not None]
    if len(comps) < 2:
        return None
    seeds = set(ref["seeds"])
    for _, r in comps:
        seeds &= set(r["seeds"])
    seeds = [s for s in ref["seeds"] if s in seeds]
    oof = np.mean([r["oof"][[r["seeds"].index(s) for s in seeds]] for _, r in comps], axis=0)
    return {"oof": oof, "y": ref["y"], "rev": ref["rev"], "seeds": seeds,
            "json": {"family": "ensemble", "components": [n for n, _ in comps], "runtime_s": 0}}


def report(area, groupings, contenders):
    lines = [f"# {area} bake-off report ({TAG}, {dt.datetime.now():%Y-%m-%d %H:%M})", ""]
    out = {"area": area, "tag": TAG, "groupings": {}}
    for grouping in groupings:
        ref = load_res(area, grouping, "xgb_deployed")
        if ref is None:
            log(f"{area}/{grouping}: no pinned control result; skipping")
            continue
        y, rev = ref["y"], ref["rev"]
        T = T_DEPLOY[area]
        j = ref["json"]
        lines += [f"## {grouping}-grouped: {j['n_sessions']} CV sessions, {j['n_rows']} rows, "
                  f"{j['n_real']} real, {j['n_bootstrap_rows']} bootstrap rows, "
                  f"{j['n_splits']} folds over {j['n_groups']} groups, T = {T}", "",
                  "| contender | seeds | AUC full | AUC reviewed | dAUC rev vs control (min; n+/n) | rule T | "
                  "FAR@T mean / worst | junk@T | FAR at matched junk | junk at matched FAR (dpp) | min |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = {}
        for name in contenders:
            res = ensemble(area, grouping, ref) if name == "ens_mean" else load_res(area, grouping, name)
            if res is None:
                continue
            if not np.array_equal(res["y"], y):
                log(f"{area}/{grouping}/{name}: rows differ from the control -- skipped")
                continue
            ro, no, common = align(ref, res)
            auc_f = [rt.auc(y, o) for o in no]
            auc_r = [rt.auc(y, o, rev) for o in no]
            ref_r = [rt.auc(y, o, rev) for o in ro]
            d = rt.paired_delta(ref_r, auc_r)
            tab = rt.threshold_table(no, y, rev)
            cT, gate, det = rt.rule_T(tab)
            far = [rt.far_junk(o, y, T)[0] for o in no]
            junk = [rt.far_junk(o, y, T)[1] for o in no]
            mop = rt.matched_operating_point(ro, no, y, T)
            djunk = mop["junk_new_at_matched_far_mean"] - mop["junk_ref_mean"]
            rows[name] = {
                "seeds": common, "auc_full": auc_f, "auc_reviewed": auc_r,
                "d_auc_reviewed": d, "rule_T": cT, "rule_gate": gate, "rule_detail": det,
                "far_at_T": far, "junk_at_T": junk,
                "far_at_matched_junk": mop["far_new_at_matched_junk_mean"],
                "junk_at_matched_far": mop["junk_new_at_matched_far_mean"],
                "junk_ref": mop["junk_ref_mean"], "d_junk_pp": djunk,
                "runtime_min": res["json"].get("runtime_s", 0) / 60,
                "pass_auc": bool(d["mean"] >= 0.003 and d["all_positive"]),
                "pass_junk": bool(djunk >= 2.0),
            }
            sd = np.std(auc_f, ddof=1) if len(auc_f) > 1 else 0.0
            lines.append(
                f"| {name} | {len(common)} | {np.mean(auc_f):.4f} +- {sd:.4f} "
                f"| {np.mean(auc_r):.4f} | {d['mean']:+.4f} ({d['min']:+.4f}; {d['n_positive']}/{d['n']}) "
                f"| {cT} | {np.mean(far):.2f} / {max(far):.2f} | {np.mean(junk):.1f} "
                f"| {mop['far_new_at_matched_junk_mean']:.2f} | {mop['junk_new_at_matched_far_mean']:.1f} ({djunk:+.1f}) "
                f"| {rows[name]['runtime_min']:.0f} |")
        out["groupings"][grouping] = rows
        lines.append("")
    # decision across groupings
    lines += ["## Decision rule (pre-registered in README.md)", ""]
    verdicts = {}
    names = set()
    for g in out["groupings"].values():
        names |= set(g)
    for name in sorted(names - {"xgb_deployed"}):
        per = {g: out["groupings"][g].get(name) for g in out["groupings"]}
        ok = all(r is not None and r["pass_auc"] and r["pass_junk"] for r in per.values()) \
            and len(per) == len(groupings)
        verdicts[name] = ok
        detail = "; ".join(
            (f"{g}: dAUC {r['d_auc_reviewed']['mean']:+.4f} ({r['d_auc_reviewed']['n_positive']}/{r['d_auc_reviewed']['n']}), "
             f"junk@matchedFAR {r['d_junk_pp']:+.1f} pp") if r else f"{g}: missing"
            for g, r in per.items())
        lines.append(f"- **{name}**: {'ADOPTABLE' if ok else 'null'} -- {detail}")
    out["verdicts"] = verdicts
    lines.append("")
    md = RESULTS / f"report_{area}_{TAG}.md"
    md.write_text("\n".join(lines))
    (RESULTS / f"report_{area}_{TAG}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    print("\n".join(lines))
    log(f"wrote {md}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA", choices=list(LIVE_WIDTH))
    ap.add_argument("--grouping", default="session", help="comma list of session,animal")
    ap.add_argument("--contenders", default="", help=f"comma list of {sorted(CONTENDERS)}")
    ap.add_argument("--winner-of", default="", help="comma list of *_tuned results to re-run as fixed winners")
    ap.add_argument("--seeds", type=int, default=8, help="first N of rt_lib.SEEDS")
    ap.add_argument("--threads", type=int, default=8, help="0 = unpinned (n_jobs=-1)")
    ap.add_argument("--repro", action="store_true", help="reproduce pareto_2026-08/bla_deployed_oof.npz")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    groupings = [g.strip() for g in a.grouping.split(",") if g.strip()]
    contenders = [c.strip() for c in a.contenders.split(",") if c.strip()]
    winners = [w.strip() for w in a.winner_of.split(",") if w.strip()]
    if a.report:
        report(a.area, groupings, contenders or REPORT_ORDER)
        return
    log(f"env: {versions()}")
    recs = load_area_pool(a.area)
    if a.repro:
        repro(a.area, recs)
        return
    seeds = rt.SEEDS[:a.seeds]
    # "name" or "name:N" -- N overrides --seeds for that contender (e.g. cat_tuned:3)
    plan = []
    for c in contenders:
        name, _, n = c.partition(":")
        plan.append((name, rt.SEEDS[:int(n)] if n else seeds))
    for grouping in groupings:
        P = prepare(a.area, recs, grouping)
        log(f"{a.area}/{grouping}: {len(P['cv'])} CV sessions ({len(P['rest'])} agent sessions <5 reals dropped), "
            f"{len(P['y'])} rows, {int(P['y'].sum())} real, {len(P['y_bs'])} bootstrap rows "
            f"({int((P['w_bs'] == 0).sum())} weight-0); {P['n_groups']} groups -> {P['n_splits']} folds"
            f"{' (fixed split: model seed varies)' if P['fixed_split'] else ''}; "
            f"agent weight override {P['override']}; seeds {seeds}")
        for name, c_seeds in plan:
            spec = CONTENDERS[name]
            log(f"  == {name} (seeds {c_seeds}) ==")
            if name == "xgb_deployed":
                run_control(a.area, grouping, P, c_seeds)
            elif "space" in spec:
                run_tuned(a.area, grouping, name, spec, P, c_seeds)
            else:
                run_fixed(a.area, grouping, name, spec["family"], spec["cfg"], P, c_seeds)
        for tuned in winners:
            fam, cfg, info = winner_config(a.area, grouping, tuned)
            log(f"  == {fam}_winner (from {tuned}: cfg {info['cfg_index']} chosen "
                f"{info['times_chosen']}/{info['of_folds']} folds) == {cfg}")
            run_fixed(a.area, grouping, f"{fam}_winner", fam, cfg, P, seeds, info)


if __name__ == "__main__":
    main()
