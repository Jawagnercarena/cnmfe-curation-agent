"""
train_cnn_pooled.py -- Round 2 follow-up (2026-09-08): is a POOLED BLA + vCA1
CNN better than the per-area CNN for each area?

Mirror of agent/eval/global35_2026-08/global_model.py Cond B for the raw-data
model: for each target cell (area x grouping) the folds, rows, seeds and the
target area's weights are exactly those of train_cnn2.py (so the OOF pairs 1:1
with the per-area run at the same variant and seeds), and EVERY row of the
other area's corpus (agent + bootstrap, at that area's own native weights:
agent weight = its override or max(sqrt(n_bs/n_ag), 4.0), bootstrap weights as
stored) is appended to the training block, multiplied by --other-weight.  The
target fold is scored by the pooled model.  Nothing of the other area is ever
in a test fold.

Usage:
  python train_cnn_pooled.py --variant fixed12+nbtrace --cells BLA:session,BLA:animal,vCA1:session,vCA1:animal --seeds 3
Outputs results/r2/pooled<W>_<variant>_<AREA>_<grouping>.{npz,json} (W = other-weight, e.g. pooled1.0_...)
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
import torch
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import train_cnn as T  # noqa: E402
import train_cnn2 as T2  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
R2 = RESULTS / "r2"
OTHER = {"BLA": "vCA1", "vCA1": "BLA"}


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_both(target, root, tag, nb):
    """Concatenate target + other corpora into one GpuCorpus2-ready set; target rows first."""
    parts = []
    for area in (target, OTHER[target]):
        X, tr, off, meta, sessions, p = T2.load_corpus(area, root, tag)
        if nb and "nb_row" not in meta:
            raise SystemExit(f"{p.name} has no nb_row")
        parts.append(dict(area=area, X=X, tr=tr, off=off, meta=meta, sessions=sessions, path=p))
    X = np.concatenate([q["X"] for q in parts])
    tr = np.concatenate([q["tr"] for q in parts])
    offs, base = [np.array([0], dtype=np.int64)], 0
    for q in parts:
        offs.append(q["off"][1:] + base)
        base += q["off"][-1]
    off = np.concatenate(offs)
    n0 = len(parts[0]["meta"]["y"])
    meta = {"T_new": np.concatenate([q["meta"]["T_new"].astype(np.int64) for q in parts])}
    if nb:
        nb0 = parts[0]["meta"]["nb_row"].astype(np.int64)
        nb1 = parts[1]["meta"]["nb_row"].astype(np.int64)
        meta["nb_row"] = np.concatenate([nb0, np.where(nb1 >= 0, nb1 + n0, -1)])
    return X, tr, off, meta, parts, n0


def other_block(q, n0, other_weight):
    """All rows of the other area: corpus row indices (offset), y, native weight x other_weight."""
    m = q["meta"]
    y = m["y"].astype(int)
    w = m["w"].astype(float)
    is_bs = m["is_bootstrap"].astype(bool)
    n_bs, n_ag = int(is_bs.sum()), int((~is_bs).sum())
    agw = rt.fold_agent_weight(n_bs, n_ag, rt.AGENT_WEIGHT_OVERRIDE.get(q["area"]))
    w = np.where(is_bs, w, agw) * other_weight
    keep = w > 0
    rows = np.arange(len(y))[keep] + n0
    return rows, y[keep], w[keep], agw, int(keep.sum())


def run(area, grouping, seeds, variant, cfg, a):
    ctrl_p = RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz"
    ctrl = np.load(ctrl_p, allow_pickle=True)
    y, rev = ctrl["y"].astype(int), ctrl["reviewed"].astype(bool)
    groups = ctrl["groups_cv"].astype(int)
    g_sess = ctrl["groups"].astype(int)
    animals = np.asarray(ctrl["animals"]).astype(str)
    X, trd, off, meta, parts, n0 = load_both(area, a.out_root, cfg["corpus"], bool(cfg["nb"]))
    tm = parts[0]["meta"]
    h5row = T.align_rows(ctrl, parts[0]["sessions"], tm)          # target rows are first in the combined corpus
    T_new = meta["T_new"]
    bs_all = np.where(tm["is_bootstrap"].astype(bool))[0]
    bs_w = tm["w"][bs_all].astype(float)
    bs_y = tm["y"][bs_all].astype(int)
    bs_animal = np.array([rt.animal_of(area, Path(parts[0]["sessions"][int(s)]).name) or "?" for s in tm["sess_idx"][bs_all]])
    o_rows, o_y, o_w, o_agw, o_n = other_block(parts[1], n0, a.other_weight)
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    fixed_split = n_splits == n_groups
    override = rt.AGENT_WEIGHT_OVERRIDE.get(area)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = T2.GpuCorpus2(X, trd, off, meta, dev, nb=bool(cfg["nb"]))
    stem = f"{'smoke_' if a.smoke else ''}pooled{a.other_weight:g}_{variant}_{area}_{grouping}"
    log(f"{stem}: target {len(y)} rows ({int(y.sum())} real), raw for {int((h5row >= 0).sum())}; target bootstrap {len(bs_all)}; "
        f"OTHER area {parts[1]['area']}: {o_n} rows at native weights x {a.other_weight:g} (its agw {o_agw:.2f}, "
        f"{int(o_y.sum())} real); {n_groups} groups -> {n_splits} folds; corpus rows {len(X)}; cfg {cfg}")
    oof = np.full((len(seeds), len(y)), np.nan)
    folds_log = []
    t0 = time.time()
    for si, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fi, (tr, te) in enumerate(cv.split(np.zeros((len(y), 1)), y, groups)):
            if a.smoke and fi > 0:
                break
            keep_bs = ~np.isin(bs_animal, np.unique(animals[te])) if grouping == "animal" else np.ones(len(bs_all), dtype=bool)
            agw = rt.fold_agent_weight(int(keep_bs.sum()), len(tr), override)
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1000 + seed)
            itr, ival = next(inner.split(np.zeros((len(tr), 1)), y[tr], g_sess[tr]))
            itr, ival = tr[itr], tr[ival]
            if cfg["fixed_epochs"]:
                itr, ival = tr, np.zeros(0, dtype=int)
            ag_rows, va_rows = h5row[itr], h5row[ival]
            ag_ok, va_ok = ag_rows >= 0, va_rows >= 0
            bsel = keep_bs & (bs_w > 0)
            tr_rows = np.concatenate([ag_rows[ag_ok], bs_all[bsel], o_rows])
            tr_y = np.concatenate([y[itr][ag_ok], bs_y[bsel], o_y])
            tr_w = np.concatenate([np.full(int(ag_ok.sum()), agw), bs_w[bsel], o_w])
            spw = rt.spw_of(tr_y, tr_w)
            tr_w = tr_w * np.where(tr_y == 1, spw, 1.0)
            model_seed = seed * 100 + fi if fixed_split else 42 * 100 + fi + seed
            te_rows = h5row[te]
            te_ok = te_rows >= 0
            preds, infos = [], []
            for m in range(int(cfg["ens"])):
                model, info = T2.train_fold(corpus, T_new, tr_rows, tr_y, tr_w, va_rows[va_ok], y[ival][va_ok],
                                            model_seed + 7919 * m, cfg)
                preds.append(T2.predict(model, corpus, te_rows[te_ok], T_new))
                infos.append(info)
                del model
            p = np.mean(preds, axis=0)
            oof[si, te[te_ok]] = p
            fold_auc = rt.auc(y[te][te_ok], p)
            folds_log.append({"seed": seed, "fold": fi, "n_train": int(len(tr_rows)), "n_other": int(len(o_rows)),
                              "agw": agw, "spw": spw, "n_test_scored": int(te_ok.sum()), "test_auc": fold_auc,
                              "best_epoch": [i["best_epoch"] for i in infos], "epochs_run": [i["epochs_run"] for i in infos],
                              "curve": [i["curve"] for i in infos]})
            log(f"  seed {seed} fold {fi}: train {len(tr_rows)} (target agw {agw:.2f}, spw {spw:.2f}) | test AUC {fold_auc:.4f} "
                f"({int(te_ok.sum())} rows) | {time.time() - t0:.0f}s")
        has = ~np.isnan(oof[si])
        if has.any():
            log(f"  seed {seed}: OOF AUC on rows with raw data: full {rt.auc(y[has], oof[si][has]):.4f} "
                f"reviewed {rt.auc(y[has], oof[si][has], rev[has]):.4f}")
    R2.mkdir(exist_ok=True)
    np.savez_compressed(R2 / f"{stem}.npz", oof=oof, y=y, reviewed=rev, has_raw=(h5row >= 0), groups=g_sess,
                        groups_cv=groups, names=ctrl["names"], seeds=np.array(seeds))
    (R2 / f"{stem}.json").write_text(json.dumps({
        "variant": variant, "cfg": cfg, "other_area": parts[1]["area"], "other_weight": a.other_weight, "other_rows": o_n,
        "other_agw": o_agw, "area": area, "grouping": grouping, "seeds": list(seeds), "control": ctrl_p.name,
        "corpora": [str(q["path"]) for q in parts], "smoke": bool(a.smoke), "runtime_s": time.time() - t0, "folds": folds_log,
        "written_at": dt.datetime.now().isoformat(timespec="seconds")}, indent=1, default=rt._jsonable))
    log(f"wrote r2/{stem}.npz/.json ({time.time() - t0:.0f}s)")
    del corpus
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="fixed12+nbtrace")
    ap.add_argument("--cells", default="BLA:session,BLA:animal,vCA1:session,vCA1:animal")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--other-weight", type=float, default=1.0)
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    cfg = T2.parse_variant(a.variant)
    if a.smoke:
        cfg["fixed_epochs"] = 2 if cfg["fixed_epochs"] else 0
        cfg["epochs"] = 2
    seeds = rt.SEEDS[:1] if a.smoke else rt.SEEDS[:a.seeds]
    for area, grouping in [tuple(c.split(":")) for c in a.cells.split(",") if c.strip()]:
        run(area, grouping, seeds, a.variant, cfg, a)


if __name__ == "__main__":
    main()
