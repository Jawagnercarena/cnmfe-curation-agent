"""
train_cnn.py -- Stage 2: the raw-data CNN probe.

A small two-branch network -- 2-D CNN on the 3-channel 32x32 footprint patch,
1-D dilated CNN on the raw trace -- trained on the SAME rows, weights, folds
and seeds as the tabular bake-off, so its out-of-fold scores pair 1:1 with the
deployed XGBoost's (results/<AREA>_<grouping>_xgb_deployed_t8.npz supplies the
fold structure: y, session groups, animal groups, seeds).

Protocol mirror of rt.run_oof / bakeoff_tabular:
  * outer folds  StratifiedGroupKFold(n_splits, shuffle, random_state=seed) on
                 the agent CV pool rows (identical to the tabular folds because
                 the split depends only on y and groups)
  * training     agent-train-fold rows that have raw data + every bootstrap
                 row with weight > 0 (test animals' bootstrap sessions dropped
                 under animal grouping); agent weight per fold = the tabular
                 rule (override or max(sqrt(n_bs / n_ag_tr), 4.0)); positives
                 additionally weighted by spw = weighted neg/pos (xgboost's
                 scale_pos_weight semantics)
  * early stop   one inner session-grouped split (~1/6 of the train sessions)
                 monitored by AUC; best epoch restored
  * inputs       patch channels (own, Cn z-score clipped to [-5, 10], others);
                 trace = asinh(robust-z), random 2048-frame crops in training
                 (tiled when shorter), evenly spaced crops averaged at eval
  * output       results/<AREA>_<grouping>_cnn_t8.npz  oof (seeds x rows, NaN
                 where a row has no raw data = auto-rejected candidates), plus
                 a json of per-fold curves

GPU: everything lives on the device (BLA ~1 GB); no DataLoader.  Determinism:
seeded, but cuDNN kernels are not bit-reproducible; report seeds, not bits.

Usage: python train_cnn.py --area BLA --grouping session,animal [--seeds 8] [--epochs 25]
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

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
CROP = 2048


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_corpus(area, root):
    p = Path(root) / area / ".feature_expansion" / "_raw_corpus" / f"{area}_raw_corpus.h5"
    with h5py.File(p, "r") as h:
        X = h["images"][:]
        tr = h["trace_data"][:]
        off = h["trace_offsets"][:].astype(np.int64)
        meta = {k: h[k][:] for k in ("sess_idx", "cand_idx", "y", "reviewed", "is_bootstrap", "w", "T_new")}
        sessions = [s.decode() if isinstance(s, bytes) else str(s) for s in h["sessions"][:]]
    return X, tr, off, meta, sessions, p


def align_rows(control, sessions, meta):
    """control (tabular) row -> h5 row, or -1 when the candidate has no raw data."""
    names = [str(n) for n in control["names"]]
    g = control["groups"].astype(int)
    first, cand = {}, np.zeros(len(g), dtype=int)
    for r, s in enumerate(g):
        if s not in first:
            first[s] = r
        cand[r] = r - first[s]
    key = {(sessions[int(si)], int(ci)): r for r, (si, ci) in enumerate(zip(meta["sess_idx"], meta["cand_idx"]))}
    return np.array([key.get((names[s], int(c)), -1) for s, c in zip(g, cand)], dtype=int)


class GpuCorpus:
    def __init__(self, X, tr, off, meta, device):
        self.dev = device
        self.X = torch.from_numpy(X).to(device)                       # (N,3,32,32) float16
        self.tr = torch.from_numpy(tr).to(device)                     # flat float16
        self.off = torch.from_numpy(off).to(device)                   # (N+1,)
        self.T = torch.from_numpy(meta["T_new"].astype(np.int64)).to(device)
        self.ar = torch.arange(CROP, device=device)

    def images(self, idx):
        img = self.X[idx].float()
        img[:, 1].clamp_(-5.0, 10.0)
        return img

    def traces(self, idx, starts):
        """idx, starts: (B,) tensors -> (B,1,CROP) with modulo tiling."""
        T = self.T[idx]
        pos = (starts[:, None] + self.ar[None, :]) % T[:, None]
        flat = self.off[idx][:, None] + pos
        return torch.asinh(self.tr[flat].float()).unsqueeze(1)

    def train_batch(self, idx):
        img = self.images(idx)
        k = int(torch.randint(0, 4, (1,)))
        img = torch.rot90(img, k, dims=(2, 3))
        if torch.rand(1).item() < 0.5:
            img = img.flip(3)
        T = self.T[idx]
        starts = (torch.rand(len(idx), device=self.dev) * T.float()).long()
        return img, self.traces(idx, starts)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class ImgNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class TraceNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, c, 7, padding=3), nn.BatchNorm1d(c), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(c, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c), nn.ReLU(),
                          nn.Conv1d(c, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c))
            for d in (1, 2, 4, 8, 16)])

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = Fnn.relu(h + b(h))
        return torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = ImgNet()
        self.tr = TraceNet()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, img, trace):
        return self.head(torch.cat([self.img(img), self.tr(trace)], dim=1)).squeeze(1)


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, corpus, rows, T_new, bs=512):
    """Mean logit over evenly spaced CROP windows -> probability, per row."""
    model.eval()
    pairs_r, pairs_s = [], []
    for r in rows:
        T = int(T_new[r])
        n = max(1, math.ceil(T / CROP))
        starts = np.linspace(0, max(T - CROP, 0), n).astype(int) if T > CROP else np.zeros(1, dtype=int)
        pairs_r += [r] * len(starts)
        pairs_s += list(starts)
    pairs_r = np.asarray(pairs_r)
    pairs_s = np.asarray(pairs_s)
    logits = np.zeros(len(pairs_r))
    for i in range(0, len(pairs_r), bs):
        idx = torch.as_tensor(pairs_r[i:i + bs], device=corpus.dev)
        st = torch.as_tensor(pairs_s[i:i + bs], device=corpus.dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=corpus.dev.type == "cuda"):
            out = model(corpus.images(idx), corpus.traces(idx, st))
        logits[i:i + bs] = out.float().cpu().numpy()
    sums = np.zeros(len(rows))
    cnts = np.zeros(len(rows))
    pos = {r: i for i, r in enumerate(rows)}
    for r, l in zip(pairs_r, logits):
        sums[pos[r]] += l
        cnts[pos[r]] += 1
    return 1.0 / (1.0 + np.exp(-(sums / cnts)))


def train_fold(corpus, T_new, tr_rows, tr_y, tr_w, val_rows, val_y, seed, epochs, patience, bs, lr):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = corpus.dev
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    rows_t = torch.as_tensor(tr_rows, device=dev)
    y_t = torch.as_tensor(tr_y, dtype=torch.float32, device=dev)
    w_t = torch.as_tensor(tr_w, dtype=torch.float32, device=dev)
    n = len(tr_rows)
    best = (-1.0, None, -1)
    hist = []
    for ep in range(epochs):
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
        auc_val = rt.auc(val_y, predict(model, corpus, val_rows, T_new))
        sched.step(auc_val)
        hist.append({"epoch": ep, "loss": tot / max(cnt, 1), "val_auc": auc_val})
        if auc_val > best[0]:
            best = (auc_val, copy.deepcopy(model.state_dict()), ep)
        elif ep - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "best_val_auc": best[0], "epochs_run": len(hist), "curve": hist}


# ---------------------------------------------------------------------------

def run(area, grouping, seeds, a):
    ctrl_p = RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz"
    ctrl = np.load(ctrl_p, allow_pickle=True)
    y, rev = ctrl["y"].astype(int), ctrl["reviewed"].astype(bool)
    groups = ctrl["groups_cv"].astype(int)
    g_sess = ctrl["groups"].astype(int)
    animals = np.asarray(ctrl["animals"]).astype(str)
    X, trd, off, meta, sessions, h5p = load_corpus(area, a.out_root)
    h5row = align_rows(ctrl, sessions, meta)
    T_new = meta["T_new"].astype(np.int64)
    bs_all = np.where(meta["is_bootstrap"].astype(bool))[0]
    bs_w = meta["w"][bs_all].astype(float)
    bs_y = meta["y"][bs_all].astype(int)
    bs_animal = np.array([rt.animal_of(area, Path(sessions[int(s)]).name) or "?" for s in meta["sess_idx"][bs_all]])
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    fixed = n_splits == n_groups
    override = rt.AGENT_WEIGHT_OVERRIDE.get(area)
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    corpus = GpuCorpus(X, trd, off, meta, dev)
    log(f"{area}/{grouping}: control {ctrl_p.name}: {len(y)} rows, {int(y.sum())} real; raw for "
        f"{int((h5row >= 0).sum())} rows ({int((h5row < 0).sum())} without); bootstrap rows {len(bs_all)} "
        f"({int((bs_w == 0).sum())} weight-0); {n_groups} groups -> {n_splits} folds; device {dev}; corpus {h5p.name}")
    oof = np.full((len(seeds), len(y)), np.nan)
    folds_log = []
    t0 = time.time()
    for si, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fi, (tr, te) in enumerate(cv.split(np.zeros((len(y), 1)), y, groups)):
            if grouping == "animal":
                keep_bs = ~np.isin(bs_animal, np.unique(animals[te]))
            else:
                keep_bs = np.ones(len(bs_all), dtype=bool)
            agw = rt.fold_agent_weight(int(keep_bs.sum()), len(tr), override)
            # inner split of the agent train rows by session, for early stopping
            inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=1000 + seed)
            itr, ival = next(inner.split(np.zeros((len(tr), 1)), y[tr], g_sess[tr]))
            itr, ival = tr[itr], tr[ival]
            ag_rows = h5row[itr]
            ag_ok = ag_rows >= 0
            va_rows = h5row[ival]
            va_ok = va_rows >= 0
            bsel = keep_bs & (bs_w > 0)
            tr_rows = np.concatenate([ag_rows[ag_ok], bs_all[bsel]])
            tr_y = np.concatenate([y[itr][ag_ok], bs_y[bsel]])
            tr_w = np.concatenate([np.full(int(ag_ok.sum()), agw), bs_w[bsel]])
            spw = rt.spw_of(tr_y, tr_w)
            tr_w = tr_w * np.where(tr_y == 1, spw, 1.0)
            model_seed = seed * 100 + fi if fixed else 42 * 100 + fi + seed
            model, info = train_fold(corpus, T_new, tr_rows, tr_y, tr_w, va_rows[va_ok], y[ival][va_ok],
                                     model_seed, a.epochs, a.patience, a.batch, a.lr)
            te_rows = h5row[te]
            te_ok = te_rows >= 0
            p = predict(model, corpus, te_rows[te_ok], T_new)
            oof[si, te[te_ok]] = p
            fold_auc = rt.auc(y[te][te_ok], p)
            folds_log.append({"seed": seed, "fold": fi, "n_train": int(len(tr_rows)), "agw": agw, "spw": spw,
                              "n_val": int(va_ok.sum()), "n_test_scored": int(te_ok.sum()),
                              "test_auc": fold_auc, **{k: v for k, v in info.items() if k != "curve"},
                              "curve": info["curve"]})
            log(f"  seed {seed} fold {fi}: train {len(tr_rows)} (agw {agw:.2f}, spw {spw:.2f}) | best epoch "
                f"{info['best_epoch']} val AUC {info['best_val_auc']:.4f} | test AUC {fold_auc:.4f} "
                f"({int(te_ok.sum())} rows) | {time.time() - t0:.0f}s")
        has = ~np.isnan(oof[si])
        log(f"  seed {seed}: OOF AUC on rows with raw data: full {rt.auc(y[has], oof[si][has]):.4f} "
            f"reviewed {rt.auc(y[has], oof[si][has], rev[has]):.4f}")
    stem = f"{area}_{grouping}_cnn_t8"
    np.savez_compressed(RESULTS / f"{stem}.npz", oof=oof, y=y, reviewed=rev, has_raw=(h5row >= 0),
                        groups=g_sess, groups_cv=groups, names=ctrl["names"], seeds=np.array(seeds))
    (RESULTS / f"{stem}.json").write_text(json.dumps({
        "area": area, "grouping": grouping, "seeds": list(seeds), "control": ctrl_p.name, "corpus": str(h5p),
        "n_rows": int(len(y)), "n_rows_with_raw": int((h5row >= 0).sum()), "epochs": a.epochs,
        "patience": a.patience, "batch": a.batch, "lr": a.lr, "device": str(dev), "torch": torch.__version__,
        "runtime_s": time.time() - t0, "folds": folds_log,
        "written_at": dt.datetime.now().isoformat(timespec="seconds")}, indent=1, default=rt._jsonable))
    log(f"wrote {stem}.npz/.json ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA")
    ap.add_argument("--grouping", default="session,animal")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=str(rt.DATA_PARENT))
    ap.add_argument("--threads", type=int, default=8)   # consumed by bakeoff_tabular's early parse
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    seeds = rt.SEEDS[:a.seeds]
    for grouping in [g.strip() for g in a.grouping.split(",") if g.strip()]:
        run(a.area, grouping, seeds, a)


if __name__ == "__main__":
    main()
