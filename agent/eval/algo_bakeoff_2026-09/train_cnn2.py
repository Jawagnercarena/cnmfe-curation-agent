"""
train_cnn2.py -- Round 2 of the Stage 2 CNN probe: training-regime and input
variants, one change per variant, on the SAME rows / folds / weights / seeds as
train_cnn.py so every OOF pairs 1:1 with the round-1 CNN and the deployed
control.  Pre-registered in README.md ("Round 2").

Variants (combine with '+'; later entries override earlier keys):
  base      round-1 configuration re-run (pairing / nondeterminism check only)
  valfix    inner early-stopping split also holds out 1/8 of the fold's
            bootstrap SESSIONS (they leave that fold's training block); cap 40
  fixed12   no early stopping: 12 epochs, cosine LR decay, every session trains
  ens3      3 members per fold (init seeds differ, data identical), mean prob
  nbtrace   2-channel trace: own + max-correlated neighbour within 60 px
            (corpus v2; zeros when the candidate has no neighbour)
  patch48   48x48 footprint patches (corpus v2_p48)

Outputs results/r2/<variant>_<AREA>_<grouping>.npz (oof: seeds x rows, NaN
where a row has no raw data; y, reviewed, has_raw, groups, groups_cv, names,
seeds) and a .json with the per-fold curves.

Usage:
  python train_cnn2.py --variant valfix,fixed12,ens3 --cells BLA:session,vCA1:animal --seeds 3
  python train_cnn2.py --variant valfix+nbtrace+ens3 --cells BLA:session,BLA:animal,vCA1:session,vCA1:animal --seeds 8
  python train_cnn2.py --variant valfix --cells BLA:session --smoke        # 1 seed, fold 0, 2 epochs
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import train_cnn as T  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
R2 = RESULTS / "r2"
CROP = T.CROP

VARIANTS = {
    "base": {},
    "valfix": dict(val_bs_frac=0.125, epochs=40),
    "fixed12": dict(fixed_epochs=12),
    "ens3": dict(ens=3),
    "nbtrace": dict(corpus="v2", nb=True),
    "patch48": dict(corpus="v2_p48"),
}
DEFAULT_CFG = dict(val_bs_frac=0.0, epochs=25, fixed_epochs=0, ens=1, corpus="", nb=False,
                   patience=5, batch=256, lr=1e-3)


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse_variant(name: str) -> dict:
    cfg = dict(DEFAULT_CFG)
    for part in name.split("+"):
        if part not in VARIANTS:
            raise SystemExit(f"unknown variant '{part}' (known: {sorted(VARIANTS)})")
        cfg.update(VARIANTS[part])
    return cfg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_corpus(area, root, tag):
    tag = os.environ.get(f"BAKEOFF_CORPUS_TAG_{area}", tag) if tag == "v2" else tag   # refresh runs point "v2" at a newer corpus
    suffix = f"_{tag}" if tag else ""
    p = Path(root) / area / ".feature_expansion" / "_raw_corpus" / f"{area}_raw_corpus{suffix}.h5"
    with h5py.File(p, "r") as h:
        X = h["images"][:]
        tr = h["trace_data"][:]
        off = h["trace_offsets"][:].astype(np.int64)
        keys = ["sess_idx", "cand_idx", "y", "reviewed", "is_bootstrap", "w", "T_new"]
        if "nb_row" in h:
            keys.append("nb_row")
        meta = {k: h[k][:] for k in keys}
        sessions = [s.decode() if isinstance(s, bytes) else str(s) for s in h["sessions"][:]]
    return X, tr, off, meta, sessions, p


class GpuCorpus2(T.GpuCorpus):
    """Round-1 corpus on the GPU plus the optional neighbour-trace channel."""

    def __init__(self, X, tr, off, meta, device, nb=False):
        super().__init__(X, tr, off, meta, device)
        self.nb = nb
        if nb:
            self.nb_row = torch.from_numpy(meta["nb_row"].astype(np.int64)).to(device)

    def trace_input(self, idx, starts):
        own = self.traces(idx, starts)                                   # (B,1,CROP)
        if not self.nb:
            return own
        nbr = self.nb_row[idx]
        has = (nbr >= 0)
        other = self.traces(nbr.clamp(min=0), starts)                    # same session -> same T and start
        other = other * has[:, None, None].to(other.dtype)
        return torch.cat([own, other], dim=1)                             # (B,2,CROP)

    def train_batch(self, idx):
        img = self.images(idx)
        k = int(torch.randint(0, 4, (1,)))
        img = torch.rot90(img, k, dims=(2, 3))
        if torch.rand(1).item() < 0.5:
            img = img.flip(3)
        Tn = self.T[idx]
        starts = (torch.rand(len(idx), device=self.dev) * Tn.float()).long()
        return img, self.trace_input(idx, starts)


# ---------------------------------------------------------------------------
# model (round-1 architecture; only the trace stem's input width can change)
# ---------------------------------------------------------------------------

class TraceNet2(nn.Module):
    def __init__(self, in_ch=1, c=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(c, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c), nn.ReLU(),
                          nn.Conv1d(c, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c))
            for d in (1, 2, 4, 8, 16)])

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = Fnn.relu(h + b(h))
        return torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)


class Net2(nn.Module):
    def __init__(self, trace_ch=1):
        super().__init__()
        self.img = T.ImgNet()
        self.tr = TraceNet2(trace_ch)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, img, trace):
        return self.head(torch.cat([self.img(img), self.tr(trace)], dim=1)).squeeze(1)


@torch.no_grad()
def predict(model, corpus, rows, T_new, bs=512):
    """Mean logit over evenly spaced CROP windows -> probability, per row."""
    model.eval()
    if len(rows) == 0:
        return np.zeros(0)
    pairs_r, pairs_s = [], []
    for r in rows:
        Tn = int(T_new[r])
        n = max(1, math.ceil(Tn / CROP))
        starts = np.linspace(0, max(Tn - CROP, 0), n).astype(int) if Tn > CROP else np.zeros(1, dtype=int)
        pairs_r += [r] * len(starts)
        pairs_s += list(starts)
    pairs_r = np.asarray(pairs_r)
    pairs_s = np.asarray(pairs_s)
    logits = np.zeros(len(pairs_r))
    for i in range(0, len(pairs_r), bs):
        idx = torch.as_tensor(pairs_r[i:i + bs], device=corpus.dev)
        st = torch.as_tensor(pairs_s[i:i + bs], device=corpus.dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=corpus.dev.type == "cuda"):
            out = model(corpus.images(idx), corpus.trace_input(idx, st))
        logits[i:i + bs] = out.float().cpu().numpy()
    sums = np.zeros(len(rows))
    cnts = np.zeros(len(rows))
    pos = {r: i for i, r in enumerate(rows)}
    for r, l in zip(pairs_r, logits):
        sums[pos[r]] += l
        cnts[pos[r]] += 1
    return 1.0 / (1.0 + np.exp(-(sums / cnts)))


def train_fold(corpus, T_new, tr_rows, tr_y, tr_w, val_rows, val_y, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = corpus.dev
    model = Net2(2 if corpus.nb else 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    fixed = int(cfg["fixed_epochs"]) > 0
    n_ep = int(cfg["fixed_epochs"]) if fixed else int(cfg["epochs"])
    if fixed:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    rows_t = torch.as_tensor(tr_rows, device=dev)
    y_t = torch.as_tensor(tr_y, dtype=torch.float32, device=dev)
    w_t = torch.as_tensor(tr_w, dtype=torch.float32, device=dev)
    n = len(tr_rows)
    bs = int(cfg["batch"])
    best = (-1.0, None, -1)
    hist = []
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            img, trace = corpus.train_batch(rows_t[sel])
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(img, trace)
            loss = (Fnn.binary_cross_entropy_with_logits(logit.float(), y_t[sel], reduction="none") * w_t[sel]).sum() / w_t[sel].sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            cnt += 1
        if fixed:
            sched.step()
            hist.append({"epoch": ep, "loss": tot / max(cnt, 1), "val_auc": float("nan")})
            continue
        auc_val = rt.auc(val_y, predict(model, corpus, val_rows, T_new))
        sched.step(auc_val)
        hist.append({"epoch": ep, "loss": tot / max(cnt, 1), "val_auc": auc_val})
        if auc_val > best[0]:
            best = (auc_val, copy.deepcopy(model.state_dict()), ep)
        elif ep - best[2] >= int(cfg["patience"]):
            break
    if fixed:
        return model, {"best_epoch": n_ep - 1, "best_val_auc": float("nan"), "epochs_run": n_ep, "curve": hist}
    model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "best_val_auc": best[0], "epochs_run": len(hist), "curve": hist}


# ---------------------------------------------------------------------------

def run(area, grouping, seeds, variant, cfg, a):
    ctrl_p = RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz"
    ctrl = np.load(ctrl_p, allow_pickle=True)
    y, rev = ctrl["y"].astype(int), ctrl["reviewed"].astype(bool)
    groups = ctrl["groups_cv"].astype(int)
    g_sess = ctrl["groups"].astype(int)
    animals = np.asarray(ctrl["animals"]).astype(str)
    X, trd, off, meta, sessions, h5p = load_corpus(area, a.out_root, cfg["corpus"])
    if cfg["nb"] and "nb_row" not in meta:
        raise SystemExit(f"{h5p.name} has no nb_row; build it with prep_raw_corpus2.py")
    h5row = T.align_rows(ctrl, sessions, meta)
    T_new = meta["T_new"].astype(np.int64)
    bs_all = np.where(meta["is_bootstrap"].astype(bool))[0]
    bs_w = meta["w"][bs_all].astype(float)
    bs_y = meta["y"][bs_all].astype(int)
    bs_sess = meta["sess_idx"][bs_all].astype(int)
    bs_animal = np.array([rt.animal_of(area, Path(sessions[int(s)]).name) or "?" for s in bs_sess])
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    fixed_split = n_splits == n_groups
    override = rt.AGENT_WEIGHT_OVERRIDE.get(area)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    corpus = GpuCorpus2(X, trd, off, meta, dev, nb=bool(cfg["nb"]))
    members = list(a.members) if a.members else list(range(int(cfg["ens"])))
    stem = (f"{'smoke_' if a.smoke else ''}{variant}"
            f"{'_m' + '-'.join(map(str, members)) if a.members else ''}_{area}_{grouping}")
    log(f"{stem}: control {ctrl_p.name}: {len(y)} rows, {int(y.sum())} real; raw for {int((h5row >= 0).sum())} rows; "
        f"bootstrap rows {len(bs_all)} ({int((bs_w == 0).sum())} weight-0); {n_groups} groups -> {n_splits} folds; "
        f"corpus {h5p.name} images {tuple(X.shape[1:])}; device {dev}; cfg {cfg}")
    oof = np.full((len(seeds), len(y)), np.nan)
    oof_m = np.full((len(seeds), len(members), len(y)), np.nan)      # per-member scores (member m = init seed + 7919*m)
    folds_log = []
    t0 = time.time()
    for si, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fi, (tr, te) in enumerate(cv.split(np.zeros((len(y), 1)), y, groups)):
            if a.smoke and fi > 0:
                break
            if grouping == "animal":
                keep_bs = ~np.isin(bs_animal, np.unique(animals[te]))
            else:
                keep_bs = np.ones(len(bs_all), dtype=bool)
            agw = rt.fold_agent_weight(int(keep_bs.sum()), len(tr), override)
            # inner split of the agent train rows by session (round-1 seeds), for early stopping
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1000 + seed)
            itr, ival = next(inner.split(np.zeros((len(tr), 1)), y[tr], g_sess[tr]))
            itr, ival = tr[itr], tr[ival]
            if cfg["fixed_epochs"]:
                itr, ival = tr, np.zeros(0, dtype=int)             # every session trains
            ag_rows = h5row[itr]
            ag_ok = ag_rows >= 0
            va_rows = h5row[ival]
            va_ok = va_rows >= 0
            bsel = keep_bs & (bs_w > 0)
            bval = np.zeros(len(bs_all), dtype=bool)
            if cfg["val_bs_frac"] > 0:
                rng = np.random.default_rng(2000 + 10 * seed + fi)
                sess_ids = np.unique(bs_sess[bsel])
                n_hold = max(1, int(round(len(sess_ids) * cfg["val_bs_frac"])))
                hold = rng.choice(sess_ids, n_hold, replace=False)
                bval = bsel & np.isin(bs_sess, hold)
                bsel = bsel & ~bval
            tr_rows = np.concatenate([ag_rows[ag_ok], bs_all[bsel]])
            tr_y = np.concatenate([y[itr][ag_ok], bs_y[bsel]])
            tr_w = np.concatenate([np.full(int(ag_ok.sum()), agw), bs_w[bsel]])
            spw = rt.spw_of(tr_y, tr_w)
            tr_w = tr_w * np.where(tr_y == 1, spw, 1.0)
            val_rows = np.concatenate([va_rows[va_ok], bs_all[bval]])
            val_y = np.concatenate([y[ival][va_ok], bs_y[bval]])
            model_seed = seed * 100 + fi if fixed_split else 42 * 100 + fi + seed
            te_rows = h5row[te]
            te_ok = te_rows >= 0
            preds, infos = [], []
            for mi, m in enumerate(members):
                model, info = train_fold(corpus, T_new, tr_rows, tr_y, tr_w, val_rows, val_y,
                                         model_seed + 7919 * m, cfg)
                pm = predict(model, corpus, te_rows[te_ok], T_new)
                preds.append(pm)
                oof_m[si, mi, te[te_ok]] = pm
                infos.append(info)
                del model
            p = np.mean(preds, axis=0)
            oof[si, te[te_ok]] = p
            fold_auc = rt.auc(y[te][te_ok], p)
            member_auc = [rt.auc(y[te][te_ok], q) for q in preds]
            folds_log.append({"seed": seed, "fold": fi, "n_train": int(len(tr_rows)), "agw": agw, "spw": spw,
                              "n_val_agent": int(va_ok.sum()), "n_val_bs": int(bval.sum()),
                              "n_test_scored": int(te_ok.sum()), "test_auc": fold_auc, "member_auc": member_auc,
                              "best_epoch": [i["best_epoch"] for i in infos],
                              "best_val_auc": [i["best_val_auc"] for i in infos],
                              "epochs_run": [i["epochs_run"] for i in infos],
                              "curve": [i["curve"] for i in infos]})
            log(f"  seed {seed} fold {fi}: train {len(tr_rows)} (agw {agw:.2f}, spw {spw:.2f}) val {len(val_rows)} "
                f"({int(bval.sum())} bs) | best ep {[i['best_epoch'] for i in infos]} val AUC "
                f"{[round(i['best_val_auc'], 4) for i in infos]} | test AUC {fold_auc:.4f} "
                f"(members {[round(x, 4) for x in member_auc]}; {int(te_ok.sum())} rows) | {time.time() - t0:.0f}s")
        has = ~np.isnan(oof[si])
        if has.any():
            log(f"  seed {seed}: OOF AUC on rows with raw data: full {rt.auc(y[has], oof[si][has]):.4f} "
                f"reviewed {rt.auc(y[has], oof[si][has], rev[has]):.4f}")
    R2.mkdir(exist_ok=True)
    np.savez_compressed(R2 / f"{stem}.npz", oof=oof, oof_members=oof_m, members=np.array(members), y=y, reviewed=rev,
                        has_raw=(h5row >= 0), groups=g_sess, groups_cv=groups, names=ctrl["names"], seeds=np.array(seeds))
    (R2 / f"{stem}.json").write_text(json.dumps({
        "variant": variant, "cfg": cfg, "members": members, "area": area, "grouping": grouping, "seeds": list(seeds),
        "control": ctrl_p.name, "corpus": str(h5p), "n_rows": int(len(y)), "n_rows_with_raw": int((h5row >= 0).sum()),
        "smoke": bool(a.smoke), "device": str(dev), "torch": torch.__version__,
        "runtime_s": time.time() - t0, "folds": folds_log,
        "written_at": dt.datetime.now().isoformat(timespec="seconds")}, indent=1, default=rt._jsonable))
    log(f"wrote r2/{stem}.npz/.json ({time.time() - t0:.0f}s)")
    del corpus
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="comma-separated; combine with '+'")
    ap.add_argument("--cells", default="BLA:session,vCA1:animal")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--threads", type=int, default=8)   # consumed by bakeoff_tabular's early parse
    ap.add_argument("--smoke", action="store_true", help="1 seed, fold 0 only, 2 epochs")
    ap.add_argument("--members", default="", help="train only these ensemble member indices (e.g. 1,2); "
                                                  "output is tagged _m1-2 and can be averaged with member 0's run")
    a = ap.parse_args()
    a.members = [int(x) for x in a.members.split(",") if x.strip()]
    torch.set_num_threads(max(1, a.threads))
    seeds = rt.SEEDS[:1] if a.smoke else rt.SEEDS[:a.seeds]
    cells = [tuple(c.split(":")) for c in a.cells.split(",") if c.strip()]
    for variant in [v.strip() for v in a.variant.split(",") if v.strip()]:
        cfg = parse_variant(variant)
        if a.smoke:
            cfg["epochs"] = 2
            if cfg["fixed_epochs"]:
                cfg["fixed_epochs"] = 2
        for area, grouping in cells:
            run(area, grouping, seeds, variant, cfg, a)


if __name__ == "__main__":
    main()
