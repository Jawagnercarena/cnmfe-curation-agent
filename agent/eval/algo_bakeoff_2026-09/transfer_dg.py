"""
transfer_dg.py -- Stage 2 step 5: how well does a raw-data CNN trained on
BLA + vCA1 ONLY score DG_AL candidates it has never seen an example of?

This is the data-starved-area question the pooled tabular prior answered with
+0.079 AUC under 2-animal LOAO (agent/eval/global35_2026-08, red team #18).
Footprint patches are gSig-scaled and traces are rate-normalised, so the raw
representation is the one place a cross-area model could transfer without the
score-scale collapse that sank raw-feature pooling.

Design
  train   every BLA and vCA1 row of the raw corpora (agent + bootstrap), with
          each area's own weighting rule (BLA agent weight max(sqrt(n_bs/n_ag),
          4.0), vCA1 override 5.0; bootstrap weights as stored); early stopping
          on a held-out sixth of the training SESSIONS (grouped); 3 seeds
  score   all DG_AL rows (17 sessions; DG has no auto-rejected candidates)
  report  DG_AL AUC of the transfer CNN (full pool + per animal), against the
          DG_AL own-model control (results/DG_AL_{session,animal}_xgb_deployed_t8.npz:
          0.900 session-grouped / 0.864 2-animal LOAO), and the rank-blend of
          transfer CNN + own-model OOF (the "ranking prior" use)

Usage: python transfer_dg.py [--seeds 3] [--epochs 25]
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
from scipy.stats import rankdata
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import train_cnn as T  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def area_rows(area, root):
    X, trd, off, meta, sessions, p = T.load_corpus(area, root)
    y = meta["y"].astype(int)
    w = meta["w"].astype(float)
    is_bs = meta["is_bootstrap"].astype(bool)
    n_bs, n_ag = int(is_bs.sum()), int((~is_bs).sum())
    agw = rt.fold_agent_weight(n_bs, n_ag, rt.AGENT_WEIGHT_OVERRIDE.get(area))
    w = np.where(is_bs, w, agw)
    return dict(X=X, tr=trd, off=off, meta=meta, sessions=sessions, y=y, w=w, sess=meta["sess_idx"].astype(int),
                agw=agw, path=p)


def concat_corpora(parts):
    """Stack several areas' corpora into one GpuCorpus-compatible set (offsets re-based)."""
    X = np.concatenate([p["X"] for p in parts])
    tr = np.concatenate([p["tr"] for p in parts])
    offs, base = [np.array([0], dtype=np.int64)], 0
    for p in parts:
        offs.append(p["off"][1:] + base)
        base += p["off"][-1]
    off = np.concatenate(offs)
    T_new = np.concatenate([p["meta"]["T_new"].astype(np.int64) for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    w = np.concatenate([p["w"] for p in parts])
    sess = np.concatenate([p["sess"] + 100000 * i for i, p in enumerate(parts)])
    return X, tr, off, T_new, y, w, sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = [area_rows(ar, a.out_root) for ar in ("BLA", "vCA1")]
    X, tr, off, T_new, y, w, sess = concat_corpora(parts)
    dg = area_rows("DG_AL", a.out_root)
    log(f"train rows {len(y)} ({int(y.sum())} real; BLA agw {parts[0]['agw']:.2f}, vCA1 agw {parts[1]['agw']:.2f}); "
        f"DG_AL rows {len(dg['y'])} ({int(dg['y'].sum())} real) from {len(dg['sessions'])} sessions; device {dev}")
    # one GPU corpus holding train + DG rows (DG appended after the training rows)
    Xa = np.concatenate([X, dg["X"]])
    tra = np.concatenate([tr, dg["tr"]])
    offa = np.concatenate([off, dg["off"][1:] + off[-1]])
    Ta = np.concatenate([T_new, dg["meta"]["T_new"].astype(np.int64)])
    corpus = T.GpuCorpus(Xa, tra, offa, {"T_new": Ta}, dev)
    n_tr = len(y)
    dg_rows = np.arange(n_tr, n_tr + len(dg["y"]))
    keep = w > 0
    seeds = rt.SEEDS[:a.seeds]
    scores = np.zeros((len(seeds), len(dg["y"])))
    fold_log = []
    t0 = time.time()
    for si, seed in enumerate(seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=1 / 6, random_state=seed)
        itr, ival = next(gss.split(np.zeros((n_tr, 1)), y, sess))
        itr = itr[keep[itr]]
        ival = ival[keep[ival]]
        spw = rt.spw_of(y[itr], w[itr])
        tw = w[itr] * np.where(y[itr] == 1, spw, 1.0)
        model, info = T.train_fold(corpus, Ta, itr, y[itr], tw, ival, y[ival], seed, a.epochs, a.patience, a.batch, a.lr)
        p = T.predict(model, corpus, dg_rows, Ta)
        scores[si] = p
        auc_dg = rt.auc(dg["y"], p)
        fold_log.append({"seed": seed, "n_train": int(len(itr)), "n_val": int(len(ival)), "spw": spw,
                         "best_epoch": info["best_epoch"], "best_val_auc": info["best_val_auc"], "dg_auc": auc_dg})
        log(f"  seed {seed}: train {len(itr)} val {len(ival)} | best epoch {info['best_epoch']} val AUC "
            f"{info['best_val_auc']:.4f} | DG_AL AUC {auc_dg:.4f} | {time.time() - t0:.0f}s")
    # comparison with the DG_AL own-model control (session- and animal-grouped OOF), rows aligned by (session, cand)
    out = {"seeds": seeds, "n_dg_rows": int(len(dg["y"])), "folds": fold_log,
           "transfer_auc_mean": float(np.mean([rt.auc(dg["y"], s) for s in scores])),
           "transfer_auc_per_seed": [rt.auc(dg["y"], s) for s in scores]}
    dg_key = {(dg["sessions"][int(s)], int(c)): i for i, (s, c) in enumerate(zip(dg["meta"]["sess_idx"], dg["meta"]["cand_idx"]))}
    dg_animal = np.array([rt.animal_of("DG_AL", Path(dg["sessions"][int(s)]).name) or "?" for s in dg["meta"]["sess_idx"]])
    out["transfer_auc_per_animal"] = {an: float(np.mean([rt.auc(dg["y"][dg_animal == an], s[dg_animal == an]) for s in scores]))
                                      for an in sorted(set(dg_animal))}
    for grouping in ("session", "animal"):
        cp = RESULTS / f"DG_AL_{grouping}_xgb_deployed_t8.npz"
        if not cp.exists():
            continue
        c = np.load(cp, allow_pickle=True)
        h5row = T.align_rows(c, dg["sessions"], dg["meta"])
        ok = h5row >= 0
        yc = c["y"].astype(int)
        own = c["oof"]
        ctrl_auc = float(np.mean([rt.auc(yc[ok], o[ok]) for o in own]))
        tr_auc = float(np.mean([rt.auc(yc[ok], s[h5row[ok]]) for s in scores]))
        blend = []
        for o in own:
            r1 = rankdata(o[ok]) / ok.sum()
            r2 = rankdata(scores.mean(axis=0)[h5row[ok]]) / ok.sum()
            blend.append(rt.auc(yc[ok], 0.5 * (r1 + r2)))
        out[f"vs_own_{grouping}"] = {"rows": int(ok.sum()), "own_auc": ctrl_auc, "transfer_auc": tr_auc,
                                     "rank_blend_auc": float(np.mean(blend)),
                                     "d_blend_vs_own": float(np.mean(blend) - ctrl_auc)}
        log(f"  vs own-model ({grouping}-grouped OOF, {int(ok.sum())} rows): own {ctrl_auc:.4f} | transfer CNN {tr_auc:.4f} "
            f"| rank blend {np.mean(blend):.4f} ({np.mean(blend) - ctrl_auc:+.4f})")
    np.savez_compressed(RESULTS / "DG_AL_transfer_cnn.npz", scores=scores, y=dg["y"], sess_idx=dg["meta"]["sess_idx"],
                        cand_idx=dg["meta"]["cand_idx"], sessions=np.array(dg["sessions"]), seeds=np.array(seeds))
    out["written_at"] = dt.datetime.now().isoformat(timespec="seconds")
    out["runtime_s"] = time.time() - t0
    (RESULTS / "DG_AL_transfer_cnn.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    log(f"transfer AUC on DG_AL (all 17 sessions, {len(dg['y'])} rows): {out['transfer_auc_mean']:.4f} "
        f"per animal {out['transfer_auc_per_animal']}; wrote DG_AL_transfer_cnn.npz/.json")


if __name__ == "__main__":
    main()
