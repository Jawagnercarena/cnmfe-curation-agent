"""Read-only quality look at the 13 cohort-5 sessions ingested 2026-11-20.
Scores every candidate with the model that CURATED them (snapshot of
2026-10-28, 177 sessions, T 0.01), which never saw these labels.
Writes nothing outside stdout."""
import sys, re, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import joblib
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "agent"))
import local_config as lc

DATA = Path(lc.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
PAT = re.compile(r"pnb(99|102|104|105|106|109)-")
CUR = joblib.load(str(REPO / "agent/model/vCA1/classifier_cohort5_T001_2026-10-28.joblib"))
CUR["clf"].set_params(n_jobs=1)
print("curation model: n_sessions", CUR.get("n_sessions"), "T", CUR.get("reject_threshold"),
      "n_features", CUR.get("n_features"))


def score(m, X):
    return m["clf"].predict_proba(m["scaler"].transform(
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)))[:, 1]


def load(sd):
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = np.asarray(npz["feature_matrix"], dtype=float)
    n = len(X)
    auto = sorted(set(npz["auto_rejected"].ravel().astype(int).tolist()))
    out = {"X": X, "n": n, "auto": auto}
    lp = sd / "labels.mat"
    if lp.exists():
        d = sio.loadmat(str(lp))
        lab = d["labels"].ravel().astype(int)
        mot = d["motion_delete"].ravel().astype(int) if "motion_delete" in d else np.zeros(len(lab), int)
        rows = np.array([i for i in range(n) if i not in set(auto)], dtype=int) if len(lab) != n else np.arange(n)
        assert len(rows) == len(lab), (sd.name, len(rows), len(lab))
        out.update(lab=lab, mot=mot, rows=rows)
    return out


sess = sorted(p for p in DATA.iterdir() if p.is_dir() and PAT.search(p.name))
new, pilots, out = [], [], []
for sd in sess:
    d = load(sd)
    d["name"] = sd.name.replace("AVG5x-TSeries-", "")
    d["sd"] = sd
    d["animal"] = re.search(r"(pnb\d+)", sd.name).group(1)
    if "lab" not in d:
        out.append(d)
    elif d["name"].startswith("08"):
        pilots.append(d)
    else:
        new.append(d)
print("new labelled %d, pilots %d, still out %d" % (len(new), len(pilots), len(out)))

# ---- A. does the snapshot reproduce the curation? ----
print("\n== A. curation reproduced by the 10-28 snapshot (score < 0.01 == stored auto_rejected) ==")
for d in new + out:
    s = score(CUR, d["X"])
    d["s"] = s
    mine = set(np.where(s < 0.01)[0].tolist())
    st = set(d["auto"])
    d["repro"] = (mine == st)
    if mine != st:
        print("  %-36s stored %3d  recomputed %3d  only-stored %d  only-recomputed %d" % (
            d["name"], len(st), len(mine), len(st - mine), len(mine - st)))
print("  exact match in %d of %d sessions (13 labelled + 15 outstanding)" % (
    sum(d["repro"] for d in new + out), len(new + out)))

# ---- B. per-session table ----
print("\n== B. per session (scores = curating model, out of sample for these labels) ==")
h = "%-34s %4s %4s %4s %4s %5s %5s | kept by score: %5s %5s %5s %5s %5s | %4s %4s %4s"
print(h % ("session", "N", "auto", "rev", "keep", "rate", "AUC", "<.02", "<.05", "<.2", "<.5", ">=.5", "d>.9", "mot", "m>=.5"))
bands = [0.01, 0.02, 0.05, 0.2, 0.5, 1.01]
P = {"k": [], "s": [], "mot": [], "animal": [], "sess": []}
for d in new:
    s = d["s"][d["rows"]]
    lab, mot = d["lab"], d["mot"]
    kb = [int(((lab == 1) & (s >= lo) & (s < hi)).sum()) for lo, hi in zip(bands[:-1], bands[1:])]
    auc = roc_auc_score(lab, s)
    dh = int(((lab == 0) & (mot == 0) & (s > 0.9)).sum())
    print(h % (d["name"][:34], d["n"], len(d["auto"]), len(lab), int(lab.sum()), "%.2f" % lab.mean(), "%.3f" % auc,
               kb[0], kb[1], kb[2], kb[3], kb[4], dh, int(mot.sum()), int((s >= 0.5).sum())))
    P["k"].append(lab); P["s"].append(s); P["mot"].append(mot)
    P["animal"] += [d["animal"]] * len(lab); P["sess"] += [d["name"]] * len(lab)
k = np.concatenate(P["k"]); s = np.concatenate(P["s"]); mot = np.concatenate(P["mot"])
an = np.array(P["animal"])
print("pooled: reviewed %d, kept %d (%.1f%%), AUC %.3f" % (len(k), k.sum(), 100 * k.mean(), roc_auc_score(k, s)))
for a in sorted(set(an)):
    m = an == a
    print("   %-7s reviewed %4d kept %3d  AUC %.3f" % (a, m.sum(), k[m].sum(), roc_auc_score(k[m], s[m])))

# ---- C. threshold bands, pooled over the 13 ----
print("\n== C. what a higher auto-reject threshold would have done on these 13 (scores from the curating model) ==")
print("   (candidates below 0.01 were never shown, so losses below 0.01 are invisible here)")
nk, nd = int(k.sum()), int((k == 0).sum())
nauto = sum(len(d["auto"]) for d in new)
ntot = sum(d["n"] for d in new)
print("   at 0.01 (as run): auto-rejected %d of %d candidates (%.1f%%)" % (nauto, ntot, 100 * nauto / ntot))
for T in (0.02, 0.03, 0.04, 0.05, 0.10):
    lk = int(((k == 1) & (s < T)).sum()); ld = int(((k == 0) & (s < T)).sum())
    per = {}
    for a in sorted(set(an)):
        m = an == a
        per[a] = "%d/%d" % (int(((k == 1) & (s < T) & m).sum()), int(k[m].sum()))
    print("   T %.2f: kept cells lost %3d of %d (%.2f%%) | extra junk removed %4d of %d reviewed deletes (%.1f%%) | review load %d -> %d | lost by animal %s" % (
        T, lk, nk, 100 * lk / nk, ld, nd, 100 * ld / nd, len(k), len(k) - lk - ld, per))

# ---- D. kept rate by score decile-ish band (where does the review time go?) ----
print("\n== D. kept rate by score band, 13 sessions pooled ==")
for lo, hi in [(0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
    m = (s >= lo) & (s < hi)
    print("   %.2f-%.2f: %4d reviewed (%4.1f%% of load), kept %3d (%.1f%%)" % (lo, min(hi, 1.0), m.sum(), 100 * m.mean(), k[m].sum(), 100 * k[m].mean() if m.any() else 0))

# ---- E. per animal: kept counts across days, pilots + new ----
print("\n== E. kept cells per session by animal (pilots 08xx at threshold 0, new 09xx at 0.01) ==")
by = {}
for d in pilots + new:
    by.setdefault(d["animal"], []).append((d["name"], d["n"], len(d["lab"]), int(d["lab"].sum())))
for a in sorted(by):
    print("  " + a)
    for nm, n, r, kk in sorted(by[a], key=lambda t: (t[0][4:6], t[0][0:4])):
        print("     %-34s N %3d  reviewed %3d  kept %3d" % (nm, n, r, kk))

# ---- F. still out ----
print("\n== F. still with Taylor ==")
tot = 0
for d in out:
    r = d["n"] - len(d["auto"]); tot += r
    print("   %-34s N %3d  to review %3d" % (d["name"], d["n"], r))
print("   15 sessions, %d candidates to review" % tot)

# ---- G. 090526-pnb99: the 0.60 re-run vs the old 0.70 set ----
print("\n== G. 090526-pnb99 (re-run at min_corr 0.60): kept cells vs the old 0.70 candidate set ==")
d = [x for x in new if x["name"].startswith("090526-pnb99")][0]
sd = d["sd"]
Anew = sio.loadmat(str(sd / ".pre_ingest" / "spatial_footprints.mat"))["spatial_footprints"]
Aold = sio.loadmat(str(sd / ".prev_mincorr070_2026-11-02" / "spatial_footprints.mat"))["spatial_footprints"]
print("   footprints: new %s (candidates %d), old %s" % (Anew.shape, d["n"], Aold.shape))


def masks(A):
    mx = A.reshape(len(A), -1).max(1)
    return A > (0.2 * mx[:, None, None])


Mn, Mo = masks(Anew).reshape(len(Anew), -1), masks(Aold).reshape(len(Aold), -1)
inter = Mn.astype(np.float32) @ Mo.astype(np.float32).T
un = Mn.sum(1)[:, None] + Mo.sum(1)[None, :] - inter
J = inter / np.maximum(un, 1)
best = J.max(1)
full = np.zeros(d["n"], int) - 1
full[d["rows"]] = d["lab"]          # -1 = auto-rejected (not shown)
in_old = best >= 0.5
print("   new candidates matching an old one (IoU >= 0.5): %d of %d; old candidates matched: %d of %d" % (
    in_old.sum(), d["n"], (J.max(0) >= 0.5).sum(), len(Aold)))
for nm, m in (("present at 0.70", in_old), ("added by 0.60", ~in_old)):
    print("   %-16s: %2d candidates -> auto-rejected %2d, reviewed %2d, KEPT %2d" % (
        nm, m.sum(), ((full == -1) & m).sum(), ((full >= 0) & m).sum(), ((full == 1) & m).sum()))
sc = d["s"]
add_k = np.where((full == 1) & ~in_old)[0]
print("   scores of the kept cells added by 0.60:", np.round(np.sort(sc[add_k]), 3).tolist())
print("   best IoU with any old candidate for those:", np.round(best[add_k], 2).tolist())
