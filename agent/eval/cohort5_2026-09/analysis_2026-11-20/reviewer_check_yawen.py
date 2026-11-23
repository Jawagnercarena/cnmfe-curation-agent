"""Read-only check of a new reviewer (Yawen): the six pnb105 sessions ingested
2026-11-20 15:29, against Taylor's cohort-5 reviews.

  A  per-session table, every labelled cohort-5 session, by reviewer
  B  keep decisions at matched model score (pre-pilot model = out of sample
     for every cohort-5 animal)
  C  same field on different days: what happened in session B to the cells
     kept in session A (within Taylor, within Yawen, across the two)
  D  Yawen's training drills (progress.csv on the server, read only)
  E  final footprints that do not sit on a KEEP candidate, the six sessions

Writes nothing except an optional mask cache (--cache DIR, scratch only).
Usage: python reviewer_check_yawen.py [--cache DIR] [--skip-c]
"""
import sys, re, csv, argparse, itertools
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "agent"))
import local_config as lc

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=None)
ap.add_argument("--skip-c", action="store_true")
args = ap.parse_args()
CACHE = Path(args.cache) if args.cache else None

DATA = Path(lc.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
PAT = re.compile(r"pnb(99|102|104|105|106|109)-")
MODELS = {
    "pre": "classifier_prepilot_2026-09-24.joblib",
    "cur": "classifier_cohort5_T001_2026-10-28.joblib",
}
M = {}
for k, fn in MODELS.items():
    M[k] = joblib.load(str(REPO / "agent" / "model" / "vCA1" / fn))
    M[k]["clf"].set_params(n_jobs=1)
    print("model %s = %s: n_sessions %s, T %s, n_features %s" % (
        k, fn, M[k].get("n_sessions"), M[k].get("reject_threshold"), M[k].get("n_features")))


def score(m, X):
    return m["clf"].predict_proba(m["scaler"].transform(
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)))[:, 1]


def read_kv(p):
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                a, b = line.split(":", 1)
                out[a.strip()] = b.strip()
    return out


S = []
for sd in sorted(p for p in DATA.iterdir() if p.is_dir() and PAT.search(p.name)):
    if not (sd / "labels.mat").exists() or not (sd / "candidate_features.npz").exists():
        continue
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = np.asarray(npz["feature_matrix"], dtype=float)
    n = len(X)
    auto = sorted(set(npz["auto_rejected"].ravel().astype(int).tolist()))
    d = sio.loadmat(str(sd / "labels.mat"))
    lab = d["labels"].ravel().astype(int)
    mot = d["motion_delete"].ravel().astype(int) if "motion_delete" in d else np.zeros(len(lab), int)
    rows = np.arange(n) if len(lab) == n else np.array([i for i in range(n) if i not in set(auto)], dtype=int)
    assert len(rows) == len(lab), (sd.name, len(rows), len(lab))
    full = np.full(n, -1)
    full[rows] = lab
    fmot = np.zeros(n, int)
    fmot[rows] = mot
    name = sd.name.replace("AVG5x-TSeries-", "")
    S.append(dict(sd=sd, name=name, animal=re.search(r"(pnb\d+)", name).group(1), n=n, auto=auto,
                  lab=lab, mot=mot, rows=rows, full=full, fmot=fmot, X=X,
                  reviewer=read_kv(sd / "labels_provenance.txt").get("reviewer", "?"),
                  meta=read_kv(sd / "review_meta.txt"), pilot=name.startswith("08"),
                  s_pre=score(M["pre"], X), s_cur=score(M["cur"], X)))
print("labelled cohort-5 sessions: %d (%s)" % (len(S), ", ".join(
    "%s %d" % (r, sum(1 for x in S if x["reviewer"] == r)) for r in sorted({x["reviewer"] for x in S}))))

# ---------------------------------------------------------------- A
print("\n== A. per session. AUC over reviewed candidates; pre = pre-pilot model (161 sessions, no cohort-5 animal),")
print("      cur = the curating model of 10-28 (out of sample for 09xx sessions, IN sample for the 08xx pilots).")
print("      exp = sum of pre-pilot scores over reviewed candidates (keeps that model expected); k/exp = kept / exp.")
h = "%-30s %-7s %4s %4s %4s %4s %5s %6s %6s %6s %5s %5s %5s %4s %5s %5s"
print(h % ("session", "review", "N", "auto", "rev", "keep", "rate", "AUCpre", "AUCcur", "exp", "k/exp",
           "d>.9", "k<.05", "mot", "min", "min/c"))
for x in sorted(S, key=lambda x: (x["reviewer"], x["animal"], x["name"][4:6], x["name"][:4])):
    sp_, sc_ = x["s_pre"][x["rows"]], x["s_cur"][x["rows"]]
    lab, mot = x["lab"], x["mot"]
    exp = float(sp_.sum())
    dur = x["meta"].get("duration_min", "")
    try:
        durf = float(dur)
    except ValueError:
        durf = float("nan")
    x["exp"] = exp
    print(h % (x["name"][:30], x["reviewer"], x["n"], len(x["auto"]), len(lab), int(lab.sum()), "%.2f" % lab.mean(),
               "%.3f" % roc_auc_score(lab, sp_), "%.3f" % roc_auc_score(lab, sc_), "%.1f" % exp,
               "%.2f" % (lab.sum() / exp), int(((lab == 0) & (mot == 0) & (sc_ > 0.9)).sum()),
               int(((lab == 1) & (sc_ < 0.05)).sum()), int(mot.sum()),
               "%.0f" % durf if durf == durf else "-", "%.2f" % (durf / len(lab)) if durf == durf else "-"))
print("  (d>.9 = deleted, not motion-tagged, cur score > 0.9; k<.05 = kept with cur score < 0.05;")
print("   min = review_meta duration_min, min/c = minutes per reviewed candidate; a resumed review restarts the clock)")

GROUPS = [
    ("Taylor 09xx (13 sess, pnb99/102/109)", lambda x: x["reviewer"] == "Taylor" and not x["pilot"]),
    ("Taylor pilots, not pnb105 (thr 0)", lambda x: x["reviewer"] == "Taylor" and x["pilot"] and x["animal"] != "pnb105"),
    ("Taylor pilots pnb105 (thr 0)", lambda x: x["reviewer"] == "Taylor" and x["pilot"] and x["animal"] == "pnb105"),
    ("Yawen 09xx (pnb105)", lambda x: x["reviewer"] == "Yawen"),
]
print("\n  pooled by group (reviewed candidates):")
for gname, f in GROUPS:
    g = [x for x in S if f(x)]
    if not g:
        continue
    lab = np.concatenate([x["lab"] for x in g])
    s1 = np.concatenate([x["s_pre"][x["rows"]] for x in g])
    s2 = np.concatenate([x["s_cur"][x["rows"]] for x in g])
    print("   %-38s sessions %2d  reviewed %4d  kept %3d (%.1f%%)  AUC pre %.3f  cur %.3f  kept/expected %.2f" % (
        gname, len(g), len(lab), lab.sum(), 100 * lab.mean(), roc_auc_score(lab, s1), roc_auc_score(lab, s2),
        lab.sum() / s1.sum()))

# ---------------------------------------------------------------- B
print("\n== B. keep rate at matched model score (pre-pilot model, out of sample for every group) ==")
print("   population: reviewed candidates. The 09xx sets were cut at cur score 0.01 before review, the pilots were")
print("   not (threshold 0), so the lowest band is not comparable across groups; read 0.05 upward.")
BANDS = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
print("   %-38s " % "group" + " ".join("%14s" % ("%.2f-%.2f" % (lo, min(hi, 1.0))) for lo, hi in BANDS))
for gname, f in GROUPS:
    g = [x for x in S if f(x)]
    if not g:
        continue
    lab = np.concatenate([x["lab"] for x in g])
    s1 = np.concatenate([x["s_pre"][x["rows"]] for x in g])
    cells = []
    for lo, hi in BANDS:
        m = (s1 >= lo) & (s1 < hi)
        cells.append("%3d/%3d %3.0f%%" % (lab[m].sum(), m.sum(), 100 * lab[m].mean() if m.any() else 0))
    print("   %-38s " % gname + " ".join("%14s" % c for c in cells))
print("   per Yawen session:")
for x in [x for x in S if x["reviewer"] == "Yawen"]:
    s1 = x["s_pre"][x["rows"]]
    cells = []
    for lo, hi in BANDS:
        m = (s1 >= lo) & (s1 < hi)
        cells.append("%3d/%3d %3.0f%%" % (x["lab"][m].sum(), m.sum(), 100 * x["lab"][m].mean() if m.any() else 0))
    print("   %-38s " % x["name"][:38] + " ".join("%14s" % c for c in cells))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_offset(sessions, flag):
    lab = np.concatenate([x["lab"] for x in sessions])
    z = np.concatenate([logit(x["s_pre"][x["rows"]]) for x in sessions])
    r = np.concatenate([np.full(len(x["lab"]), 1.0 if flag(x) else 0.0) for x in sessions])
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([z, r]), lab)
    return float(lr.coef_[0][1]), float(lr.coef_[0][0])


def boot(sessions, flag, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = [x for x in sessions if flag(x)]
    b = [x for x in sessions if not flag(x)]
    out = []
    for _ in range(nb):
        samp = [a[i] for i in rng.integers(0, len(a), len(a))] + [b[i] for i in rng.integers(0, len(b), len(b))]
        try:
            out.append(fit_offset(samp, flag)[0])
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5]), len(out)


print("\n   logistic model keep ~ logit(pre-pilot score) + [reviewer is Yawen]; reviewed candidates with pre score >= 0.05.")
print("   coefficient > 0: Yawen keeps MORE than Taylor at the same score; < 0: fewer. CI = bootstrap over sessions")
print("   within reviewer (2000 draws). Reviewer and animal are confounded except in the pnb105-only row.")


def cut(x):
    y = dict(x)
    m = x["s_pre"][x["rows"]] >= 0.05
    y["lab"] = x["lab"][m]
    y["rows"] = x["rows"][m]
    return y


isY = lambda x: x["reviewer"] == "Yawen"
for nm, sel in (("Yawen 6 vs Taylor 09xx 13", lambda x: isY(x) or (x["reviewer"] == "Taylor" and not x["pilot"])),
                ("Yawen 6 vs all Taylor (13 + 12 pilots)", lambda x: x["reviewer"] in ("Yawen", "Taylor")),
                ("pnb105 only: Yawen 6 vs Taylor 2 pilots", lambda x: x["animal"] == "pnb105")):
    ss = [cut(x) for x in S if sel(x)]
    b, slope = fit_offset(ss, isY)
    (lo, hi), nb = boot(ss, isY)
    print("   %-42s offset %+.2f (odds x %.2f), 95%% CI %+.2f .. %+.2f; slope %.2f; sessions Y %d / T %d" % (
        nm, b, np.exp(b), lo, hi, slope, sum(1 for x in ss if isY(x)), sum(1 for x in ss if not isY(x))))

# ---------------------------------------------------------------- C
H = W = 512


def coords(x):
    """(cand, y, x) of every candidate's support (> 0.2 * its max), from .pre_ingest."""
    if CACHE is not None:
        cp = CACHE / (x["name"] + "_masks.npz")
        if cp.exists():
            z = np.load(str(cp))
            return z["c"], z["y"], z["x"]
    p = x["sd"] / ".pre_ingest" / "spatial_footprints.mat"
    if not p.exists():
        return None
    A = sio.loadmat(str(p))["spatial_footprints"]
    assert A.shape == (x["n"], H, W), (x["name"], A.shape, x["n"])
    mx = A.reshape(len(A), -1).max(1)
    c, y, xx = np.nonzero(A > (0.2 * mx[:, None, None]))
    c, y, xx = c.astype(np.int32), y.astype(np.int16), xx.astype(np.int16)
    if CACHE is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(CACHE / (x["name"] + "_masks.npz")), c=c, y=y, x=xx)
    return c, y, xx


def density(co):
    img = np.zeros((H, W), np.float32)
    np.add.at(img, (co[1].astype(int), co[2].astype(int)), 1.0)
    return gaussian_filter(np.minimum(img, 1.0), 1.5)


def xshift(a, b, maxs=150):
    """shift (dy, dx) to ADD to b's pixel coordinates so that b lands on a; peak height / runner-up."""
    a0, b0 = a - a.mean(), b - b.mean()
    cc = np.fft.ifft2(np.fft.fft2(a0) * np.conj(np.fft.fft2(b0))).real
    cc = np.fft.fftshift(cc)
    cy, cx = H // 2, W // 2
    win = cc[cy - maxs:cy + maxs + 1, cx - maxs:cx + maxs + 1].copy()
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    pk = win[iy, ix]
    w2 = win.copy()
    w2[max(0, iy - 10):iy + 11, max(0, ix - 10):ix + 11] = -np.inf
    return int(iy - maxs), int(ix - maxs), float(pk / max(w2.max(), 1e-9))


def overlap_r(a, b, dy, dx):
    ya0, ya1 = max(0, dy), min(H, H + dy)
    xa0, xa1 = max(0, dx), min(W, W + dx)
    pa = a[ya0:ya1, xa0:xa1]
    pb = b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    return float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])


def csr(co, n, dy=0, dx=0):
    c, y, xx = co[0], co[1].astype(int) + dy, co[2].astype(int) + dx
    ok = (y >= 0) & (y < H) & (xx >= 0) & (xx < W)
    return sp.csr_matrix((np.ones(ok.sum(), np.float32), (c[ok], y[ok] * W + xx[ok])), shape=(n, H * W))


def cn_hp(x):
    cn = sio.loadmat(str(x["sd"] / "Cn.mat"))["Cn"].astype(np.float32)
    return cn - gaussian_filter(cn, 10)


def centroids(co, n):
    cy = np.bincount(co[0], weights=co[1].astype(float), minlength=n)
    cx = np.bincount(co[0], weights=co[2].astype(float), minlength=n)
    k = np.maximum(np.bincount(co[0], minlength=n), 1)
    return cy / k, cx / k


def vote_shift(A, B, maxs=150):
    """Translation by voting: every (A candidate, B candidate) pair votes for the displacement between
    their centroids; votes are pooled over +/-2 px. Returns the shift to ADD to B coordinates, the votes
    at the peak, and the best count more than 10 px away from it (the chance level for this pair)."""
    ay, ax = A["cen"]
    by, bx = B["cen"]
    dy = np.rint(ay[:, None] - by[None, :]).astype(int).ravel()
    dx = np.rint(ax[:, None] - bx[None, :]).astype(int).ravel()
    ok = (np.abs(dy) <= maxs) & (np.abs(dx) <= maxs)
    hist = np.zeros((2 * maxs + 1, 2 * maxs + 1), np.float32)
    np.add.at(hist, (dy[ok] + maxs, dx[ok] + maxs), 1.0)
    from scipy.ndimage import uniform_filter
    sm = uniform_filter(hist, 5, mode="constant") * 25.0
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    pk = float(sm[iy, ix])
    s2 = sm.copy()
    s2[max(0, iy - 10):iy + 11, max(0, ix - 10):ix + 11] = 0
    return int(iy - maxs), int(ix - maxs), pk, float(s2.max())


def fate(A, B, dy, dx, thr):
    """Cells KEPT in A: outcome in B. (dy, dx) is added to B coordinates to land on A."""
    Ma = csr(A["co"], A["n"])
    Mb = csr(B["co"], B["n"], dy, dx)
    kept = np.where(A["full"] == 1)[0]
    cen_y = np.array([A["co"][1][A["co"][0] == k].mean() for k in kept]) - dy
    cen_x = np.array([A["co"][2][A["co"][0] == k].mean() for k in kept]) - dx
    infield = (cen_y >= 15) & (cen_y < H - 15) & (cen_x >= 15) & (cen_x < W - 15)
    kept = kept[infield]
    inter = (Ma[kept] @ Mb.T).toarray()
    un = np.asarray(Ma[kept].sum(1)) + np.asarray(Mb.sum(1)).T - inter
    J = inter / np.maximum(un, 1)
    out = {"kept": 0, "deleted": 0, "auto": 0, "none": 0}
    for i in range(len(kept)):
        m = np.where(J[i] >= thr)[0]
        lb = B["full"][m]
        if (lb == 1).any():
            out["kept"] += 1
        elif (lb == 0).any():
            out["deleted"] += 1
        elif len(m):
            out["auto"] += 1
        else:
            out["none"] += 1
    out["n"] = len(kept)
    out["outfield"] = int((~infield).sum())
    return out


def near(a, b, tol=3):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def local_votes(A, B, dy, dx, rad=4, maxs=190):
    """Refine a predicted shift: best vote count within +/-rad px of it, and the best count > 10 px away."""
    ay, ax = A["cen"]
    by, bx = B["cen"]
    vy = np.rint(ay[:, None] - by[None, :]).astype(int).ravel()
    vx = np.rint(ax[:, None] - bx[None, :]).astype(int).ravel()
    ok = (np.abs(vy) <= maxs) & (np.abs(vx) <= maxs)
    hist = np.zeros((2 * maxs + 1, 2 * maxs + 1), np.float32)
    np.add.at(hist, (vy[ok] + maxs, vx[ok] + maxs), 1.0)
    from scipy.ndimage import uniform_filter
    sm = uniform_filter(hist, 5, mode="constant") * 25.0
    cy, cx = dy + maxs, dx + maxs
    if not (rad <= cy < 2 * maxs + 1 - rad and rad <= cx < 2 * maxs + 1 - rad):
        return dy, dx, 0.0, float(sm.max())
    win = sm[cy - rad:cy + rad + 1, cx - rad:cx + rad + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    pk = float(win[iy, ix])
    s2 = sm.copy()
    s2[max(0, cy - 10):cy + 11, max(0, cx - 10):cx + 11] = 0
    return int(dy + iy - rad), int(dx + ix - rad), pk, float(s2.max())


def day_of(x):
    import datetime as _dt
    nm = x["name"]
    return _dt.date(2000 + int(nm[4:6]), int(nm[0:2]), int(nm[2:4]))


if not args.skip_c:
    print("\n== C. same field, different days: fate in session B of the cells KEPT in session A ==")
    print("   registration, three independent estimates of the shift between two sessions of one animal:")
    print("   (1) voting: every (A candidate, B candidate) pair votes for the displacement between the centroids,")
    print("   pooled over +/-2 px; (2) cross-correlation of the candidate-support density images; (3) cross-")
    print("   correlation of the high-passed Cn. A pair is a RELIABLE edge when at least two of the three agree")
    print("   within 3 px. Sessions linked by reliable edges form one field; every session gets a position by")
    print("   least squares over those edges, so every pair in the field gets a shift by composition, refined to")
    print("   the best vote count within +/-4 px ('votes'; 'next' = best count > 10 px away = chance level).")
    print("   A kept cell in A is matched to B candidates with support IoU >= 0.3 after the shift: 'kept' if any")
    print("   matching candidate was kept, 'deleted' if matched only to reviewed deletes, 'auto' if matched only to")
    print("   auto-rejected candidates, 'none' if no candidate. Cells whose centre falls within 15 px of B's edge")
    print("   are left out. A cell can be silent on another day, so the within-reviewer rows are the baseline.")
    for x in S:
        x["co"] = coords(x)
        if x["co"] is not None:
            x["den"] = density(x["co"])
            x["cnhp"] = cn_hp(x)
            x["cen"] = centroids(x["co"], x["n"])
    rows_c = []
    rows_all = []
    for an in sorted({x["animal"] for x in S}):
        ss = [x for x in S if x["animal"] == an and x.get("co") is not None]
        edges = []
        for i, j in itertools.combinations(range(len(ss)), 2):
            A, B = ss[i], ss[j]
            v = vote_shift(A, B)[:2]
            dn = xshift(A["den"], B["den"])[:2]
            cn = xshift(A["cnhp"], B["cnhp"])[:2]
            if near(v, dn) or near(v, cn):
                edges.append((i, j, v[0], v[1]))
            elif near(dn, cn):
                edges.append((i, j, int(round((dn[0] + cn[0]) / 2.0)), int(round((dn[1] + cn[1]) / 2.0))))
        comp = list(range(len(ss)))
        for i, j, _, _ in edges:
            a, b = comp[i], comp[j]
            if a != b:
                comp = [a if c == b else c for c in comp]
        print("\n   %s: %d sessions, %d reliable edges of %d pairs" % (an, len(ss), len(edges), len(ss) * (len(ss) - 1) // 2))
        for c in sorted(set(comp)):
            mem = [k for k in range(len(ss)) if comp[k] == c]
            if len(mem) < 2:
                print("     alone (no reliable link): %s" % ss[mem[0]]["name"][:28])
                continue
            ed = [e for e in edges if comp[e[0]] == c]
            pos = {}
            for dim in (2, 3):
                Amat = np.zeros((len(ed) + 1, len(mem)))
                bvec = np.zeros(len(ed) + 1)
                for r, e in enumerate(ed):
                    Amat[r, mem.index(e[0])] = 1.0
                    Amat[r, mem.index(e[1])] = -1.0
                    bvec[r] = e[dim]
                Amat[len(ed), 0] = 1.0
                sol = np.linalg.lstsq(Amat, bvec, rcond=None)[0]
                pos[dim] = sol
                res = np.abs(Amat[:len(ed)] @ sol - bvec[:len(ed)])
                pos[("res", dim)] = float(res.max()) if len(res) else 0.0
            print("     field of %d: %s | edges %d, worst edge residual %.1f px (y) %.1f px (x)" % (
                len(mem), ", ".join(ss[k]["name"][:13] for k in mem), len(ed), pos[("res", 2)], pos[("res", 3)]))
            for a, b in itertools.combinations(range(len(mem)), 2):
                A, B = ss[mem[a]], ss[mem[b]]
                py = int(round(pos[2][a] - pos[2][b]))
                px = int(round(pos[3][a] - pos[3][b]))
                dy, dx, pk, nx = local_votes(A, B, py, px)
                ok = pk >= 1.3 * max(nx, 1.0)
                print("       %-13s %-13s %s>%s gap %2dd  composed %4d,%4d  refined %4d,%4d  votes %4.0f  next %4.0f  %s" % (
                    A["name"][:13], B["name"][:13], A["reviewer"][:1], B["reviewer"][:1],
                    abs((day_of(B) - day_of(A)).days), py, px, dy, dx, pk, nx, "used" if ok else "NOT used (votes < 1.3 x next)"))
                if ok:
                    rows_c.append((A, B, dy, dx))
                rows_all.append((A, B, dy, dx))
    for thr in (0.3, 0.5):
        print("\n   fate table, IoU >= %.1f (both directions of every used pair)" % thr)
        if thr == 0.3:
            print("   %-13s -> %-13s %-5s %4s %5s %5s %5s %5s %5s | %s" % (
                "A (kept)", "B", "rev", "gap", "n", "kept", "del", "auto", "none", "kept/(kept+del)  kept/n"))
        agg = {}
        for A, B, dy, dx in rows_c:
            gap = abs((day_of(B) - day_of(A)).days)
            for P, Q, sy, sx in ((A, B, dy, dx), (B, A, -dy, -dx)):
                o = fate(P, Q, sy, sx, thr)
                key = "%s -> %s, %s, %s" % (P["reviewer"], Q["reviewer"],
                                            "pnb105" if P["animal"] == "pnb105" else "other animals",
                                            "1-5 days apart" if gap <= 5 else "9+ days apart")
                a = agg.setdefault(key, {"kept": 0, "deleted": 0, "auto": 0, "none": 0, "n": 0, "pairs": 0})
                for k in ("kept", "deleted", "auto", "none", "n"):
                    a[k] += o[k]
                a["pairs"] += 1
                rev = o["kept"] + o["deleted"]
                if thr == 0.3:
                    print("   %-13s -> %-13s %-5s %3dd %5d %5d %5d %5d %5d | %s  %.2f" % (
                        P["name"][:13], Q["name"][:13], "%s>%s" % (P["reviewer"][:1], Q["reviewer"][:1]), gap, o["n"],
                        o["kept"], o["deleted"], o["auto"], o["none"], "%.2f" % (o["kept"] / rev) if rev else "-",
                        o["kept"] / max(o["n"], 1)))
        print("   summary:")
        for key in sorted(agg):
            a = agg[key]
            rev = a["kept"] + a["deleted"]
            print("     %-46s directed pairs %2d  kept cells %4d -> kept %4d, deleted %4d, auto %3d, none %4d"
                  " | kept/(kept+del) %.3f | kept/n %.3f" % (
                      key, a["pairs"], a["n"], a["kept"], a["deleted"], a["auto"], a["none"],
                      a["kept"] / rev if rev else float("nan"), a["kept"] / a["n"] if a["n"] else float("nan")))
    print("\n   sensitivity, IoU >= 0.3: EVERY pair of a field at its composed shift, including the pairs set aside")
    print("   above for a weak vote peak (few shared cells, or a wrong shift; the two cannot be told apart):")
    agg = {}
    for A, B, dy, dx in rows_all:
        gap = abs((day_of(B) - day_of(A)).days)
        for P, Q, sy, sx in ((A, B, dy, dx), (B, A, -dy, -dx)):
            o = fate(P, Q, sy, sx, 0.3)
            key = "%s -> %s, %s, %s" % (P["reviewer"], Q["reviewer"],
                                        "pnb105" if P["animal"] == "pnb105" else "other animals",
                                        "1-5 days apart" if gap <= 5 else "9+ days apart")
            a = agg.setdefault(key, {"kept": 0, "deleted": 0, "auto": 0, "none": 0, "n": 0, "pairs": 0})
            for k in ("kept", "deleted", "auto", "none", "n"):
                a[k] += o[k]
            a["pairs"] += 1
    for key in sorted(agg):
        a = agg[key]
        rev = a["kept"] + a["deleted"]
        print("     %-46s directed pairs %2d  kept cells %4d -> kept %4d, deleted %4d, auto %3d, none %4d"
              " | kept/(kept+del) %.3f | kept/n %.3f" % (
                  key, a["pairs"], a["n"], a["kept"], a["deleted"], a["auto"], a["none"],
                  a["kept"] / rev if rev else float("nan"), a["kept"] / a["n"] if a["n"] else float("nan")))

# ---------------------------------------------------------------- D
print("\n== D. Yawen's training drills (progress.csv in the server training folders; read only) ==")
exch = getattr(lc, "EXCHANGE_ROOT", None)
tdir = Path(exch) / "training" / "Yawen" if exch else None
if tdir is None or not tdir.is_dir():
    print("   training folder not reachable; skipped")
else:
    rows = []
    for p in sorted(tdir.rglob("progress.csv")):
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            rows += list(csv.DictReader(fh))
    fin = sorted([r for r in rows if r["pass"] == "final"], key=lambda r: r["timestamp"])
    print("   attempts: %d (one 'final' row each); modes: %s" % (len(fin), sorted({r["mode"] for r in fin})))
    print("   %-16s %-5s %-22s %3s %6s %6s %6s %6s %-10s %-10s %7s" % (
        "finished", "area", "session", "stg", "shown", "visit", "scored", "agree", "false keep", "false del", "kappa"))
    for r in fin:
        print("   %-16s %-5s %-22s %3s %6s %6s %6s %6s %-10s %-10s %7s" % (
            r["timestamp"][:16], r["area"], r["session"].replace("AVG5x-TSeries-", "")[:22], r["stage"], r["n_items"],
            r["n_visited"], r["n_scored"], r["agreement"], "%s/%s" % (r["false_keep"], r["n_ref_delete_scored"]),
            "%s/%s" % (r["false_delete"], r["n_ref_keep_scored"]), r["kappa"]))
    print("   (coach mode scores the first decision, before the feedback; contested items are not scored)")

# ---------------------------------------------------------------- E
print("\n== E. the six sessions: final footprints that do not best-match a KEEP candidate (IoU of supports) ==")
for x in [x for x in S if x["reviewer"] == "Yawen"]:
    co = x.get("co") or coords(x)
    Mc = csr(co, x["n"])
    fin = sio.loadmat(str(x["sd"] / "spatial_footprints.mat"))["spatial_footprints"]
    mx = fin.reshape(len(fin), -1).max(1)
    c, y, xx = np.nonzero(fin > (0.2 * mx[:, None, None]))
    Mf = sp.csr_matrix((np.ones(len(c), np.float32), (c, y * W + xx)), shape=(len(fin), H * W))
    inter = (Mf @ Mc.T).toarray()
    J = inter / np.maximum(np.asarray(Mf.sum(1)) + np.asarray(Mc.sum(1)).T - inter, 1)
    keepc = np.where(x["full"] == 1)[0]
    for i in range(len(fin)):
        b = int(J[i].argmax())
        if J[i, b] >= 0.5 and x["full"][b] == 1:
            continue
        bk = keepc[int(J[i, keepc].argmax())] if len(keepc) else -1
        print("   %-28s final #%3d: best candidate %3d (IoU %.2f, %s, cur score %.2f) | best KEEP candidate %3d (IoU %.2f)" % (
            x["name"][:28], i + 1, b, J[i, b], {1: "KEEP", 0: "DELETE", -1: "AUTO-REJECTED"}[int(x["full"][b])],
            x["s_cur"][b], bk, J[i, bk] if bk >= 0 else float("nan")))

# ---------------------------------------------------------------- F
print("\n== F. deleted candidates the models liked: duplicates of a kept cell, or unique objects? ==")
print("   deleted, not motion-tagged, score >= 0.8. 'dup' = support IoU >= 0.5 with a candidate the same reviewer KEPT")
print("   in that session (a second copy of a kept cell); 'near' = IoU 0.2-0.5; 'unique' = IoU < 0.2 with every kept one.")
print("   Columns for both models: cur (10-28) and pre (pre-pilot).")
tot = {}
print("   %-30s %-7s | cur: %4s %4s %4s %6s | pre: %4s %4s %4s %6s | all high cur: kept/total" % (
    "session", "review", "dup", "near", "uniq", "n", "dup", "near", "uniq", "n"))
for x in sorted(S, key=lambda x: (x["reviewer"], x["animal"], x["name"][4:6], x["name"][:4])):
    co = x.get("co") or coords(x)
    if co is None:
        continue
    Mc = csr(co, x["n"])
    keepc = np.where(x["full"] == 1)[0]
    line = []
    for key in ("s_cur", "s_pre"):
        dl = np.where((x["full"] == 0) & (x["fmot"] == 0) & (x[key] >= 0.8))[0]
        if len(dl) and len(keepc):
            inter = (Mc[dl] @ Mc[keepc].T).toarray()
            J = inter / np.maximum(np.asarray(Mc[dl].sum(1)) + np.asarray(Mc[keepc].sum(1)).T - inter, 1)
            best = J.max(1)
        else:
            best = np.zeros(len(dl))
        d3 = (int((best >= 0.5).sum()), int(((best >= 0.2) & (best < 0.5)).sum()), int((best < 0.2).sum()))
        line.append(d3 + (len(dl),))
        g = (x["reviewer"], "pnb105" if x["animal"] == "pnb105" else "other", key)
        a = tot.setdefault(g, [0, 0, 0, 0, 0, 0])
        for i in range(3):
            a[i] += d3[i]
        a[3] += len(dl)
        hi = (x["full"] >= 0) & (x[key] >= 0.8)
        a[4] += int((x["full"][hi] == 1).sum())
        a[5] += int(hi.sum())
    hi = (x["full"] >= 0) & (x["s_cur"] >= 0.8)
    print("   %-30s %-7s |      %4d %4d %4d %6d |      %4d %4d %4d %6d | %d/%d" % (
        (x["name"][:30], x["reviewer"]) + line[0] + line[1] + (int((x["full"][hi] == 1).sum()), int(hi.sum()))))
print("   totals:")
for g in sorted(tot):
    a = tot[g]
    print("     %-7s %-6s %-5s deleted high-score %3d: dup %3d, near %3d, unique %3d | kept %d of %d high-score reviewed"
          " (%.0f%%); unique deletes = %.1f%% of high-score reviewed" % (
              g[0], g[1], g[2][2:], a[3], a[0], a[1], a[2], a[4], a[5], 100.0 * a[4] / max(a[5], 1),
              100.0 * a[2] / max(a[5], 1)))

# ---------------------------------------------------------------- G
print("\n== G. duplicates at every score, and section B again without them ==")
print("   duplicate = a reviewed DELETE (not motion-tagged) whose support has IoU >= 0.5 with a candidate KEPT in the")
print("   same session. These rows enter training as agent-weight negatives (train_classifier.py zero-weights")
print("   duplicates only for bootstrap sessions).")
for x in S:
    co = x.get("co") or coords(x)
    x["dup"] = np.zeros(x["n"], bool)
    if co is None:
        continue
    Mc = csr(co, x["n"])
    keepc = np.where(x["full"] == 1)[0]
    dl = np.where((x["full"] == 0) & (x["fmot"] == 0))[0]
    if len(dl) and len(keepc):
        inter = (Mc[dl] @ Mc[keepc].T).toarray()
        J = inter / np.maximum(np.asarray(Mc[dl].sum(1)) + np.asarray(Mc[keepc].sum(1)).T - inter, 1)
        x["dup"][dl[J.max(1) >= 0.5]] = True
print("   %-38s %8s %8s %8s %10s %14s" % ("group", "sessions", "deletes", "dup", "dup/delete", "dup per kept"))
for gname, f in GROUPS:
    g = [x for x in S if f(x)]
    if not g:
        continue
    nd = sum(int((x["full"] == 0).sum()) for x in g)
    ndup = sum(int(x["dup"].sum()) for x in g)
    nk = sum(int((x["full"] == 1).sum()) for x in g)
    print("   %-38s %8d %8d %8d %9.1f%% %14.2f" % (gname, len(g), nd, ndup, 100.0 * ndup / max(nd, 1), ndup / max(nk, 1)))
print("   per Yawen session: " + "; ".join("%s %d dup of %d deletes" % (
    x["name"][:13], int(x["dup"].sum()), int((x["full"] == 0).sum())) for x in S if x["reviewer"] == "Yawen"))
print("\n   keep rate by pre-pilot score band with the duplicates removed from the denominator:")
print("   %-38s " % "group" + " ".join("%14s" % ("%.2f-%.2f" % (lo, min(hi, 1.0))) for lo, hi in BANDS))
for gname, f in GROUPS:
    g = [x for x in S if f(x)]
    if not g:
        continue
    lab = np.concatenate([x["lab"] for x in g])
    s1 = np.concatenate([x["s_pre"][x["rows"]] for x in g])
    dp = np.concatenate([x["dup"][x["rows"]] for x in g])
    cells = []
    for lo, hi in BANDS:
        m = (s1 >= lo) & (s1 < hi) & ~dp
        cells.append("%3d/%3d %3.0f%%" % (lab[m].sum(), m.sum(), 100 * lab[m].mean() if m.any() else 0))
    print("   %-38s " % gname + " ".join("%14s" % c for c in cells))


def cut_nodup(x):
    y = dict(x)
    m = (x["s_pre"][x["rows"]] >= 0.05) & ~x["dup"][x["rows"]]
    y["lab"] = x["lab"][m]
    y["rows"] = x["rows"][m]
    return y


print("   logistic offset for Yawen as in B, duplicates removed:")
for nm, sel in (("Yawen 6 vs Taylor 09xx 13", lambda x: isY(x) or (x["reviewer"] == "Taylor" and not x["pilot"])),
                ("Yawen 6 vs all Taylor (13 + 12 pilots)", lambda x: x["reviewer"] in ("Yawen", "Taylor")),
                ("pnb105 only: Yawen 6 vs Taylor 2 pilots", lambda x: x["animal"] == "pnb105")):
    ss = [cut_nodup(x) for x in S if sel(x)]
    b, slope = fit_offset(ss, isY)
    (lo, hi), nb = boot(ss, isY)
    print("   %-42s offset %+.2f (odds x %.2f), 95%% CI %+.2f .. %+.2f; slope %.2f" % (nm, b, np.exp(b), lo, hi, slope))
