"""
stack_leak.py -- the pre-registered measurement of the non-nested stack's
optimism (README "Round 2"), for ONE seed on ONE cell.

Non-nested stack (stack_xgb.py): the level-1 XGBoost of outer fold f trains on
rows whose CNN score came from a CNN that HAD seen fold f (the round-1 OOF of
fold g was produced by the model trained on the other four folds, f included).

Nested version measured here: CNNs are trained with PAIRS of outer folds held
out (10 models for 5 folds).  For outer fold f, every training row in fold g
gets its CNN score from model (f, g), which never saw fold f; the test rows of
fold f keep the round-1 fold-f model's OOF score (trained on the four training
folds -- what a deployment would do).  The stacked XGBoost is then fitted and
evaluated exactly as in stack_xgb.py.

Reported: reviewed AUC of the non-nested stack minus the nested stack at the
same seed (positive = optimism), plus both against the pinned control at the
matched operating point.

Usage: python stack_leak.py --area BLA --grouping session --seed 42 [--variant base]
Outputs results/r2/leak_<variant>_<AREA>_<grouping>_s<seed>.{npz,json}
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
import stack_eval as SE  # noqa: E402
import stack_xgb as SX  # noqa: E402
import train_cnn as T  # noqa: E402
import train_cnn2 as T2  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
R2 = RESULTS / "r2"


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA")
    ap.add_argument("--grouping", default="session")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="base", help="CNN configuration (train_cnn2 variant syntax)")
    ap.add_argument("--cnn", default="", help="level-0 OOF npz for the test rows (default: round-1 cnn_t8)")
    ap.add_argument("--stack", default="", help="non-nested stack npz to compare with (default: r2/stack_r1_*)")
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    area, grouping, seed = a.area, a.grouping, a.seed
    cfg = T2.parse_variant(a.variant)

    # ---- harness pool (tabular side) and the level-0 OOF used for the test rows
    P = B.prepare(area, B.load_area_pool(area), grouping)
    cnn_p = Path(a.cnn) if a.cnn else RESULTS / f"{area}_{grouping}_cnn_t8.npz"
    cnn = np.load(cnn_p, allow_pickle=True)
    si = [int(s) for s in cnn["seeds"]].index(seed)
    ctrl = np.load(RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
    ci = [int(s) for s in ctrl["seeds"]].index(seed)
    y, rev, g_sess = P["y"].astype(int), P["rev"].astype(bool), P["g_sess"]
    if not (np.array_equal(cnn["y"].astype(int), y) and np.array_equal(ctrl["y"].astype(int), y)):
        raise SystemExit("rows differ between the harness pool and the saved OOFs")
    groups = P["groups"]
    animals = np.asarray(ctrl["animals"]).astype(str)

    # ---- raw corpus (CNN side), aligned to the same rows
    X, trd, off, meta, sessions, h5p = T2.load_corpus(area, a.out_root, cfg["corpus"])
    h5row = T.align_rows(ctrl, sessions, meta)
    T_new = meta["T_new"].astype(np.int64)
    bs_all = np.where(meta["is_bootstrap"].astype(bool))[0]
    bs_w = meta["w"][bs_all].astype(float)
    bs_y = meta["y"][bs_all].astype(int)
    bs_animal = np.array([rt.animal_of(area, Path(sessions[int(s)]).name) or "?" for s in meta["sess_idx"][bs_all]])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = T2.GpuCorpus2(X, trd, off, meta, dev, nb=bool(cfg["nb"]))
    override = rt.AGENT_WEIGHT_OVERRIDE.get(area)

    # ---- outer folds of this seed (identical to rt.run_oof / train_cnn)
    cv = StratifiedGroupKFold(n_splits=P["n_splits"], shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros((len(y), 1)), y, groups))
    K = len(folds)
    fold_of = np.full(len(y), -1)
    for fi, (_, te) in enumerate(folds):
        fold_of[te] = fi
    log(f"{area}/{grouping} seed {seed}: {len(y)} rows, {K} folds, corpus {h5p.name}, cfg {cfg}")

    # ---- pairwise-held-out CNNs: nested[f][rows of g] = score from model (f, g)
    nested = np.full((K, len(y)), np.nan)
    t0 = time.time()
    pair_log = []
    for f in range(K):
        for g in range(f + 1, K):
            held = (fold_of == f) | (fold_of == g)
            tr = np.where(~held)[0]
            if grouping == "animal":
                keep_bs = ~np.isin(bs_animal, np.unique(animals[held]))
            else:
                keep_bs = np.ones(len(bs_all), dtype=bool)
            agw = rt.fold_agent_weight(int(keep_bs.sum()), len(tr), override)
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1000 + seed)
            itr, ival = next(inner.split(np.zeros((len(tr), 1)), y[tr], g_sess[tr]))
            itr, ival = tr[itr], tr[ival]
            if cfg["fixed_epochs"]:
                itr, ival = tr, np.zeros(0, dtype=int)
            ag_rows, va_rows = h5row[itr], h5row[ival]
            ag_ok, va_ok = ag_rows >= 0, va_rows >= 0
            bsel = keep_bs & (bs_w > 0)
            bval = np.zeros(len(bs_all), dtype=bool)
            if cfg["val_bs_frac"] > 0:
                rng = np.random.default_rng(3000 + 100 * seed + 10 * f + g)
                sess_ids = np.unique(meta["sess_idx"][bs_all][bsel])
                hold = rng.choice(sess_ids, max(1, int(round(len(sess_ids) * cfg["val_bs_frac"]))), replace=False)
                bval = bsel & np.isin(meta["sess_idx"][bs_all], hold)
                bsel = bsel & ~bval
            tr_rows = np.concatenate([ag_rows[ag_ok], bs_all[bsel]])
            tr_y = np.concatenate([y[itr][ag_ok], bs_y[bsel]])
            tr_w = np.concatenate([np.full(int(ag_ok.sum()), agw), bs_w[bsel]])
            spw = rt.spw_of(tr_y, tr_w)
            tr_w = tr_w * np.where(tr_y == 1, spw, 1.0)
            val_rows = np.concatenate([va_rows[va_ok], bs_all[bval]])
            val_y = np.concatenate([y[ival][va_ok], bs_y[bval]])
            preds = []
            for m in range(int(cfg["ens"])):
                model, info = T2.train_fold(corpus, T_new, tr_rows, tr_y, tr_w, val_rows, val_y,
                                            seed * 100 + 10 * f + g + 7919 * m, cfg)
                held_rows = np.where(held)[0]
                hr = h5row[held_rows]
                ok = hr >= 0
                p = np.full(len(held_rows), np.nan)
                p[ok] = T2.predict(model, corpus, hr[ok], T_new)
                preds.append(p)
                del model
            p = np.nanmean(np.stack(preds), axis=0)
            held_rows = np.where(held)[0]
            in_g = fold_of[held_rows] == g
            nested[f, held_rows[in_g]] = p[in_g]        # fold-g rows scored without fold f
            nested[g, held_rows[~in_g]] = p[~in_g]      # fold-f rows scored without fold g
            auc_pair = rt.auc(y[held_rows][~np.isnan(p)], p[~np.isnan(p)])
            pair_log.append({"f": f, "g": g, "n_train": int(len(tr_rows)), "agw": agw, "spw": spw,
                             "best_epoch": info["best_epoch"], "best_val_auc": info["best_val_auc"], "heldout_auc": auc_pair})
            log(f"  pair ({f},{g}): train {len(tr_rows)} | best ep {info['best_epoch']} val {info['best_val_auc']:.4f} "
                f"| held-out AUC {auc_pair:.4f} | {time.time() - t0:.0f}s")

    # ---- nested stack per outer fold
    oof_r1 = cnn["oof"][si].astype(float)                 # level-0 scores for the test rows
    n_bs = len(P["y_bs"])
    oof_nested = np.full(len(y), np.nan)
    model_seed = seed if P["fixed_split"] else 42
    for f, (tr, te) in enumerate(folds):
        sc_tr = nested[f]                                  # NaN on fold f itself (and rows without raw)
        rk_tr = SX.session_rank(sc_tr, g_sess)
        sc_te = np.where(fold_of == f, oof_r1, np.nan)
        rk_te = SX.session_rank(sc_te, g_sess)
        X_ag2 = np.hstack([P["X_ag"], np.where(fold_of == f, sc_te, sc_tr)[:, None],
                           np.where(fold_of == f, rk_te, rk_tr)[:, None]])
        if P["mask"] is not None:
            m = P["mask"](te)
            Xb, yb, wb = P["X_bs"][m], P["y_bs"][m], P["w_bs"][m]
        else:
            Xb, yb, wb = P["X_bs"], P["y_bs"], P["w_bs"]
        Xb2 = np.hstack([Xb, np.full((len(yb), 2), np.nan)]) if len(yb) else Xb
        agw = rt.fold_agent_weight(len(yb), len(tr), P["override"])
        X_tr = np.vstack([X_ag2[tr], Xb2]) if len(yb) else X_ag2[tr]
        y_tr = np.concatenate([y[tr], yb])
        w_tr = np.concatenate([np.full(len(tr), agw), wb])
        oof_nested[te], _ = B.fit_predict_generic("xgb", B.DEPLOYED_XGB, X_tr, y_tr, w_tr, X_ag2[te], model_seed)
        log(f"  outer fold {f}: nested stack fold AUC {rt.auc(y[te], oof_nested[te]):.4f} "
            f"(train rows with a nested CNN score: {int(np.isfinite(sc_tr[tr]).sum())}/{len(tr)})")

    # ---- compare with the non-nested stack at the same seed
    stack_p = Path(a.stack) if a.stack else R2 / f"stack_r1_{area}_{grouping}.npz"
    st = np.load(stack_p, allow_pickle=True)
    sj = [int(s) for s in st["seeds"]].index(seed)
    oof_nn = st["oof"][sj].astype(float)
    ref = ctrl["oof"][ci][None, :]
    Tdep = B.T_DEPLOY[area]
    b_nn = SE.block(y, rev, ref, oof_nn[None, :], Tdep, "non-nested stack vs control")
    b_ne = SE.block(y, rev, ref, oof_nested[None, :], Tdep, "nested stack vs control")
    auc_nn, auc_ne = rt.auc(y, oof_nn, rev), rt.auc(y, oof_nested, rev)
    out = {"area": area, "grouping": grouping, "seed": seed, "variant": a.variant, "cfg": cfg,
           "cnn": str(cnn_p), "stack_nonnested": str(stack_p), "pairs": pair_log,
           "auc_reviewed_nonnested": auc_nn, "auc_reviewed_nested": auc_ne, "optimism": auc_nn - auc_ne,
           "auc_full_nonnested": rt.auc(y, oof_nn), "auc_full_nested": rt.auc(y, oof_nested),
           "block_nonnested": b_nn, "block_nested": b_ne,
           "runtime_s": time.time() - t0, "written_at": dt.datetime.now().isoformat(timespec="seconds")}
    stem = f"leak_{a.variant}_{area}_{grouping}_s{seed}"
    R2.mkdir(exist_ok=True)
    np.savez_compressed(R2 / f"{stem}.npz", oof_nested=oof_nested, oof_nonnested=oof_nn, nested_level0=nested,
                        y=y, reviewed=rev, seed=seed)
    (R2 / f"{stem}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    log(f"reviewed AUC: non-nested stack {auc_nn:.4f} | nested stack {auc_ne:.4f} | optimism {auc_nn - auc_ne:+.4f}; "
        f"vs control: non-nested {b_nn['d_auc_reviewed']['mean']:+.4f} / junk {b_nn['d_junk_pp']:+.1f} pp, "
        f"nested {b_ne['d_auc_reviewed']['mean']:+.4f} / junk {b_ne['d_junk_pp']:+.1f} pp; wrote r2/{stem}")


if __name__ == "__main__":
    main()
