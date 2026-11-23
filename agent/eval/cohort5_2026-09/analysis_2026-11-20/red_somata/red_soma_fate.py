"""red_soma_fate.py -- what happened to the red (retro-tdTomato) somata on the
cohort-5 vCA1 sessions that now carry reviewer labels (2026-11-20).

READ-ONLY on DATA_PARENT session folders and on the retro project's results
(local_config.RETRO_RESULTS).  Writes only red_soma_fate.txt and
red_soma_fate_somata.csv next to this file.

Soma-to-candidate rule (the one agent/eval/missed_somata_2026-11/red_exempt_yield.py
used): a candidate is ON a red soma when its footprint mask (>= 0.3 x its own
max) covers >= 30% of the 7-px-radius disk around a red_blobs.csv centre with
bleed_flag False.  Somata whose disk crosses the image edge are not assessed
(as there).  Sensitivity variant: the red centre pixel lies inside the mask.

Fate of a red soma, in this order: KEPT if any candidate on it was kept by the
reviewer; DELETED if any candidate on it was reviewed (none kept); AUTO if its
candidates were all auto-rejected (never shown); NONE if no candidate is on it.

Groups: new19 = 09xx sessions with labels (13 Taylor, 6 Yawen), pilot12 = 08xx
sessions reviewed at threshold 0, out9 = 09xx sessions still out (no labels:
only shown / auto / none can be said).

Scores: classifier_cohort5_T001_2026-10-28.joblib (the curating model, out of
sample for the 09xx sessions); classifier_prepilot_2026-09-24.joblib for the
pilots (out of sample for them).  The live classifier.joblib is never loaded.
"""
import csv
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "agent"))
import local_config as lc

DATA = Path(lc.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
RETRO = Path(lc.RETRO_RESULTS)
OLD_TXT = REPO / "agent" / "eval" / "missed_somata_2026-11" / "red_exempt_yield.txt"
PAT = re.compile(r"pnb(99|102|104|105|106|109)-")
FOOT_LEVEL, DISK_R, COVER = 0.3, 7, 0.30
CEN_R = 6.0
DUP_LEVEL, DUP_IOU = 0.2, 0.5
H = W = 512
BANDS = [(0.01, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]


def load_model(fn):
    m = joblib.load(str(REPO / "agent" / "model" / "vCA1" / fn))
    m["clf"].set_params(n_jobs=1)
    return m


def score(m, X):
    assert X.shape[1] == m["n_features"], (X.shape, m["n_features"])
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


def red_centres(p):
    """(row, col, id, flagged) for every blob; flagged = bleed_flag True."""
    rows = []
    with open(p, encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(round(float(r["row"]))), int(round(float(r["col"]))), r["id"],
                         str(r["bleed_flag"]).strip() == "True"))
    return rows


def rate(k, n):
    return "%d/%d (%.0f%%)" % (k, n, 100.0 * k / n) if n else "%d/0" % k


def retro_dir(name):
    full = "AVG5x-TSeries-" + name
    hits = [d for d in RETRO.glob("pnb*/*/" + full) if d.is_dir()]
    return hits[0] if len(hits) == 1 else None


def load_session(sd):
    name = sd.name.replace("AVG5x-TSeries-", "")
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = np.asarray(npz["feature_matrix"], dtype=float)
    n = len(X)
    auto = sorted(set(npz["auto_rejected"].ravel().astype(int).tolist()))
    x = dict(sd=sd, name=name, animal=re.search(r"(pnb\d+)", name).group(1), n=n, auto=auto, X=X,
             pilot=name.startswith("08"), labelled=(sd / "labels.mat").exists(),
             reviewer=read_kv(sd / "labels_provenance.txt").get("reviewer", "-"))
    full = np.full(n, -1)          # -1 auto-rejected (never shown), 0 deleted, 1 kept, -2 shown but unlabelled
    fmot = np.zeros(n, int)
    rows = np.array([i for i in range(n) if i not in set(auto)], dtype=int)
    if x["labelled"]:
        d = sio.loadmat(str(sd / "labels.mat"))
        lab = d["labels"].ravel().astype(int)
        mot = d["motion_delete"].ravel().astype(int) if "motion_delete" in d else np.zeros(len(lab), int)
        if len(lab) == n:
            rows = np.arange(n)
        assert len(rows) == len(lab), (name, len(rows), len(lab))
        full[rows] = lab
        fmot[rows] = mot
        fp = sd / ".pre_ingest" / "spatial_footprints.mat"
    else:
        full[rows] = -2
        fp = sd / "spatial_footprints.mat"
    x.update(full=full, fmot=fmot, rows=rows, fp=fp)
    return x


def masks_of(x):
    A = sio.loadmat(str(x["fp"]))["spatial_footprints"]
    if A.ndim == 2:
        A = A[None]
    assert A.shape == (x["n"], H, W), (x["name"], A.shape, x["n"])
    mx = np.maximum(A.reshape(len(A), -1).max(1), 1e-12)
    m03 = A >= (FOOT_LEVEL * mx[:, None, None])
    c, y, xx = np.nonzero(A > (DUP_LEVEL * mx[:, None, None]))
    m02 = sp.csr_matrix((np.ones(len(c), np.float32), (c, y * W + xx)), shape=(x["n"], H * W))
    return m03, m02


def guide_check(x, rd, m03):
    """A 09xx red guide must describe the live candidate set. Returns (ok, text)."""
    g = rd / "red_guide"
    gj = g / "guide.json"
    cs = g / "candidates_shown.npz"
    if not gj.exists() or not cs.exists():
        return False, "no guide.json / candidates_shown.npz"
    gd = json.load(open(gj))
    z = np.load(str(cs), allow_pickle=True)
    if int(z["n_candidates"].ravel()[0]) != x["n"] or int(gd["n_candidates"]) != x["n"]:
        return False, "guide N %s != live N %d" % (gd.get("n_candidates"), x["n"])
    if sorted(set(z["auto_rejected"].ravel().astype(int).tolist())) != x["auto"]:
        return False, "guide auto-rejected set differs from the live one"
    ridx = z["review_idx"].ravel().astype(int)
    mk = np.unpackbits(z["masks_packed"], axis=1)[:, :H * W].reshape(len(ridx), H, W).astype(bool)
    diff = int((mk != m03[ridx]).sum())
    return True, "guide N/auto match; shown masks vs .pre_ingest masks differ in %d px (of %d set)" % (diff, int(mk.sum()))


def analyse(x, cur, pre):
    rd = retro_dir(x["name"])
    if rd is None:
        x["skip"] = "no retro results folder"
        return
    rb = (rd / "red_blobs.csv") if x["pilot"] else (rd / "red_guide" / "red_blobs.csv")
    if not rb.exists():
        x["skip"] = "no red_blobs.csv"
        return
    m03, m02 = masks_of(x)
    if not x["pilot"]:
        ok, txt = guide_check(x, rd, m03)
        x["guide"] = txt
        if not ok:
            x["skip"] = txt
            return
    else:
        x["guide"] = "pilot: red_blobs.csv of 10-28 (positions in the session-mean frame; no candidates_shown to compare)"
    x["s_cur"] = score(cur, x["X"])
    x["s_pre"] = score(pre, x["X"])
    # duplicates: reviewed, deleted, not motion-tagged, IoU >= 0.5 with a kept candidate
    dup = np.zeros(x["n"], bool)
    keepc = np.where(x["full"] == 1)[0]
    dl = np.where((x["full"] == 0) & (x["fmot"] == 0))[0]
    if len(keepc) and len(dl):
        inter = (m02[dl] @ m02[keepc].T).toarray()
        J = inter / np.maximum(np.asarray(m02[dl].sum(1)) + np.asarray(m02[keepc].sum(1)).T - inter, 1)
        dup[dl[J.max(1) >= DUP_IOU]] = True
    x["dup"] = dup
    yy, xx = np.mgrid[-DISK_R:DISK_R + 1, -DISK_R:DISK_R + 1]
    disk = (yy ** 2 + xx ** 2) <= DISK_R ** 2
    red = np.zeros(x["n"], bool)
    red_c = np.zeros(x["n"], bool)
    red_s = np.zeros(x["n"], bool)
    ci_, yi_, xi_ = np.nonzero(m03)
    kk_ = np.maximum(np.bincount(ci_, minlength=x["n"]), 1)
    ceny = np.bincount(ci_, weights=yi_.astype(float), minlength=x["n"]) / kk_
    cenx = np.bincount(ci_, weights=xi_.astype(float), minlength=x["n"]) / kk_
    somata = []
    blobs = red_centres(rb)
    x["n_blobs"] = len(blobs)
    x["n_bleed"] = sum(1 for b in blobs if b[3])
    x["n_edge"] = 0
    for r, c, bid, flagged in blobs:
        if flagged:
            continue
        r0, r1, c0, c1 = r - DISK_R, r + DISK_R + 1, c - DISK_R, c + DISK_R + 1
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
            x["n_edge"] += 1
            continue
        cov = (m03[:, r0:r1, c0:c1] & disk).sum((1, 2)) / float(disk.sum())
        on = np.where(cov >= COVER)[0]
        red[on] = True
        red_c |= m03[:, r, c]
        lb = x["full"][on]
        fate = fate_of(lb, len(on))
        rev = on[x["full"][on] >= 0]
        # sensitivity: the red centre pixel lies inside the candidate mask
        on_c = np.where(m03[:, r, c])[0]
        # stricter variant: the candidate is CENTRED on the soma (mask centroid within CEN_R px)
        on_s = np.where(np.hypot(ceny - r, cenx - c) <= CEN_R)[0]
        red_s[on_s] = True
        fate_s = fate_of(x["full"][on_s], len(on_s))
        for key, (rr, cc) in (("ctrlS_T", (c, r)), ("ctrlS_R", (H - 1 - r, W - 1 - c))):
            if rr - DISK_R < 0 or cc - DISK_R < 0 or rr + DISK_R + 1 > H or cc + DISK_R + 1 > W:
                continue
            oc = np.where(np.hypot(ceny - rr, cenx - cc) <= CEN_R)[0]
            x.setdefault(key, []).append(fate_of(x["full"][oc], len(oc)))
        fate_c = fate_of(x["full"][on_c], len(on_c))
        # control positions in the same field: transposed (col, row) and rotated 180 degrees
        for key, (rr, cc) in (("ctrl_T", (c, r)), ("ctrl_R", (H - 1 - r, W - 1 - c))):
            a0, a1, b0, b1 = rr - DISK_R, rr + DISK_R + 1, cc - DISK_R, cc + DISK_R + 1
            if a0 < 0 or b0 < 0 or a1 > H or b1 > W:
                continue
            cv = (m03[:, a0:a1, b0:b1] & disk).sum((1, 2)) / float(disk.sum())
            oc = np.where(cv >= COVER)[0]
            x.setdefault(key, []).append(fate_of(x["full"][oc], len(oc)))
        somata.append(dict(session=x["name"], animal=x["animal"], reviewer=x["reviewer"], id=bid, row=r, col=c,
                           fate=fate, fate_centre=fate_c, n_on_centre=len(on_c),
                           fate_centroid=fate_s, n_on_centroid=len(on_s),
                           max_cur_centroid=float(x["s_cur"][on_s].max()) if len(on_s) else float("nan"),
                           max_pre_centroid=float(x["s_pre"][on_s].max()) if len(on_s) else float("nan"),
                           n_on=len(on), n_on_auto=int((lb == -1).sum()),
                           on=";".join(str(int(i)) for i in on),
                           max_cur=float(x["s_cur"][on].max()) if len(on) else float("nan"),
                           max_pre=float(x["s_pre"][on].max()) if len(on) else float("nan"),
                           del_all_dup=bool(fate == "DELETED" and len(rev) and x["dup"][rev].all()),
                           del_any_dup=bool(fate == "DELETED" and x["dup"][on].any()),
                           del_any_motion=bool(fate == "DELETED" and (x["fmot"][on] == 1).any())))
    x["red"], x["red_c"], x["red_s"], x["somata"] = red, red_c, red_s, somata


def fate_of(lb, n_on):
    if (lb == 1).any():
        return "KEPT"
    if (lb == 0).any():
        return "DELETED"
    if (lb == -2).any():
        return "SHOWN"
    return "AUTO" if n_on else "NONE"


def fate_counts(som, key="fate"):
    out = {k: 0 for k in ("KEPT", "DELETED", "SHOWN", "AUTO", "NONE")}
    for s in som:
        out[s[key] if isinstance(s, dict) else s] += 1
    out["n"] = len(som)
    return out


def old_part_b():
    """session -> (exempt disk, exempt centre) from the 11-04 record, for a reproduction check."""
    out = {}
    if not OLD_TXT.exists():
        return out
    for line in OLD_TXT.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"AVG5x-TSeries-(\S+)\s+N\s+(\d+)\s+review\s+(\d+)\s+auto-rej\s+(\d+)\s+-> exempt \(disk\)\s+(\d+)\s+\(centre rule\s+(\d+)\)", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(4)), int(m.group(5)), int(m.group(6)))
    return out


def band_table(sess, key, title, bands=BANDS, drop_dup=False, rk="red"):
    print("   %s" % title)
    print("   %-10s " % "" + " ".join("%15s" % ("%.2f-%.2f" % (lo, min(hi, 1.0))) for lo, hi in bands))
    for nm, flag in (("red", True), ("not red", False)):
        cells = []
        for lo, hi in bands:
            k = n = 0
            for x in sess:
                m = (x["full"] >= 0) & (x[key] >= lo) & (x[key] < hi) & (x[rk] == flag)
                if drop_dup:
                    m &= ~x["dup"]
                k += int((x["full"][m] == 1).sum())
                n += int(m.sum())
            cells.append("%4d/%4d %3.0f%%" % (k, n, 100.0 * k / n if n else 0))
        print("   %-10s " % nm + " ".join("%15s" % c for c in cells))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def red_offset(sess, key, drop_dup, rk="red"):
    lab, z, r = [], [], []
    for x in sess:
        m = x["full"] >= 0
        if drop_dup:
            m = m & ~x["dup"]
        lab.append((x["full"][m] == 1).astype(int))
        z.append(logit(x[key][m]))
        r.append(x[rk][m].astype(float))
    lab, z, r = np.concatenate(lab), np.concatenate(z), np.concatenate(r)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([z, r]), lab)
    return float(lr.coef_[0][1])


def boot_offset(sess, key, drop_dup, nb=2000, seed=0, rk="red"):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nb):
        samp = [sess[i] for i in rng.integers(0, len(sess), len(sess))]
        try:
            out.append(red_offset(samp, key, drop_dup, rk))
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5])


def main():
    cur = load_model("classifier_cohort5_T001_2026-10-28.joblib")
    pre = load_model("classifier_prepilot_2026-09-24.joblib")
    print("models: curating 10-28 n_sessions %s T %s | prepilot n_sessions %s T %s" % (
        cur.get("n_sessions"), cur.get("reject_threshold"), pre.get("n_sessions"), pre.get("reject_threshold")))
    print("rule: candidate ON a red soma = mask (>= %.1f x max) covers >= %.0f%% of the %d-px-radius disk (%d px) at a"
          " red_blobs.csv centre with bleed_flag False; disks crossing the edge not assessed." % (
              FOOT_LEVEL, 100 * COVER, DISK_R, int(((np.mgrid[-DISK_R:DISK_R + 1, -DISK_R:DISK_R + 1] ** 2).sum(0) <= DISK_R ** 2).sum())))
    S = []
    for sd in sorted(p for p in DATA.iterdir() if p.is_dir() and PAT.search(p.name)):
        if not (sd / "candidate_features.npz").exists():
            continue
        x = load_session(sd)
        analyse(x, cur, pre)
        S.append(x)
    used = [x for x in S if "skip" not in x]
    new19 = [x for x in used if x["labelled"] and not x["pilot"]]
    pil = [x for x in used if x["labelled"] and x["pilot"]]
    out9 = [x for x in used if not x["labelled"]]
    print("sessions found %d: labelled 09xx %d, pilots %d, still out %d; skipped %d" % (
        len(S), len(new19), len(pil), len(out9), len(S) - len(used)))
    for x in S:
        if "skip" in x:
            print("  SKIPPED %-32s %s" % (x["name"], x["skip"]))

    # ------------------------------------------------------------ 0
    print("\n== 0. inputs per session ==")
    for x in used:
        check = ""
        if x["labelled"]:
            cur_auto = sorted(np.where(x["s_cur"] < 0.01)[0].tolist())
            if not x["pilot"]:
                check = "curating model reproduces auto-rejects: %s" % ("yes" if cur_auto == x["auto"] else "NO")
        print("  %-30s %-7s N %3d auto %3d | blobs %2d bleed-flagged %d edge %d assessed %2d | %s | %s" % (
            x["name"][:30], x["reviewer"], x["n"], len(x["auto"]), x["n_blobs"], x["n_bleed"], x["n_edge"],
            len(x["somata"]), x["guide"], check))

    # ------------------------------------------------------------ 1
    print("\n== 1. fate of the red somata (soma level) ==")
    print("   KEPT = a candidate on it was kept; DELETED = reviewed, none kept; AUTO = only auto-rejected candidates;")
    print("   NONE = no candidate. dup = DELETED somata whose reviewed candidates are ALL second copies of a kept cell")
    print("   (support IoU >= 0.5 with a kept candidate); mot = DELETED somata with a motion-tagged candidate.")
    h = "   %-30s %-7s %4s %5s %5s %5s %5s | %4s %4s | %8s %8s"
    print(h % ("session", "review", "red", "KEPT", "DEL", "AUTO", "NONE", "dup", "mot", "reviewed", "kept"))

    def line(nm, rev, som):
        f = fate_counts(som)
        nd = sum(1 for s in som if s["del_all_dup"])
        nm_ = sum(1 for s in som if s["del_any_motion"])
        print(h % (nm[:30], rev, f["n"], f["KEPT"], f["DELETED"], f["AUTO"], f["NONE"], nd, nm_,
                   "%.0f%%" % (100.0 * (f["KEPT"] + f["DELETED"]) / f["n"]) if f["n"] else "-",
                   "%.0f%%" % (100.0 * f["KEPT"] / f["n"]) if f["n"] else "-"))

    for x in sorted(new19, key=lambda x: (x["animal"], x["name"][4:6], x["name"][:4])):
        line(x["name"], x["reviewer"], x["somata"])
    print("   pooled by animal (labelled 09xx):")
    for an in sorted({x["animal"] for x in new19}):
        line(an, "", [s for x in new19 if x["animal"] == an for s in x["somata"]])
    line("ALL 19 labelled 09xx", "", [s for x in new19 for s in x["somata"]])
    line("  Taylor 13", "", [s for x in new19 if x["reviewer"] == "Taylor" for s in x["somata"]])
    line("  Yawen 6", "", [s for x in new19 if x["reviewer"] == "Yawen" for s in x["somata"]])
    print("   pilots (threshold 0: every candidate reviewed, so AUTO is empty by construction):")
    for x in sorted(pil, key=lambda x: (x["animal"], x["name"][:4])):
        line(x["name"], x["reviewer"], x["somata"])
    line("ALL 12 pilots", "", [s for x in pil for s in x["somata"]])
    for an in sorted({x["animal"] for x in pil}):
        line("  pilots " + an, "", [s for x in pil if x["animal"] == an for s in x["somata"]])

    # ------------------------------------------------------------ 1b
    print("\n== 1b. how specific is the matching rule? ==")
    print("   centre rule (red centre pixel inside the mask) and two control positions per soma in the same field")
    print("   (transposed = (col, row); rotated = 180 degrees). A control lands on a candidate by chance, so the")
    print("   control rows are the chance level of each column under the disk rule.")

    def frow(nm, f):
        n = max(f["n"], 1)
        print("   %-44s n %3d: KEPT %3d (%2.0f%%)  DELETED %3d (%2.0f%%)  AUTO %3d (%2.0f%%)  NONE %3d (%2.0f%%)" % (
            nm, f["n"], f["KEPT"], 100.0 * f["KEPT"] / n, f["DELETED"], 100.0 * f["DELETED"] / n,
            f["AUTO"], 100.0 * f["AUTO"] / n, f["NONE"], 100.0 * f["NONE"] / n))

    for nm, ss in (("19 labelled 09xx", new19), ("12 pilots", pil)):
        som = [s for x in ss for s in x["somata"]]
        frow(nm + ", red somata, disk rule", fate_counts(som))
        frow(nm + ", red somata, centre rule", fate_counts(som, "fate_centre"))
        frow(nm + ", control transposed, disk rule", fate_counts([f for x in ss for f in x.get("ctrl_T", [])]))
        frow(nm + ", control rotated 180, disk rule", fate_counts([f for x in ss for f in x.get("ctrl_R", [])]))

    # ------------------------------------------------------------ 4
    print("\n== 4. share of red somata reaching the reviewer; comparison with the 11-04 measurement ==")
    print("   11-04 (memory note, 28 Taylor sessions, 502 red somata; script not in the repo): 50% reach the reviewer,")
    print("   38% no candidate, 12% auto-rejected.")

    def reach(nm, som):
        f = fate_counts(som)
        shown = f["KEPT"] + f["DELETED"] + f["SHOWN"]
        print("   %-34s red somata %3d: reach reviewer %s, auto-rejected only %s, no candidate %s" % (
            nm, f["n"], rate(shown, f["n"]), rate(f["AUTO"], f["n"]), rate(f["NONE"], f["n"])))

    reach("19 labelled 09xx", [s for x in new19 for s in x["somata"]])
    reach("9 still out (unlabelled)", [s for x in out9 for s in x["somata"]])
    reach("all 28 of the batch", [s for x in new19 + out9 for s in x["somata"]])
    for an in sorted({x["animal"] for x in new19 + out9}):
        reach("  " + an + " (all its 09xx)", [s for x in new19 + out9 if x["animal"] == an for s in x["somata"]])
    reach("12 pilots (threshold 0)", [s for x in pil for s in x["somata"]])
    f = fate_counts([s for x in new19 for s in x["somata"]])
    print("   final set, 19 labelled: %s of the red somata have a KEPT candidate; of those shown to the reviewer %s." % (
        rate(f["KEPT"], f["n"]), rate(f["KEPT"], f["KEPT"] + f["DELETED"])))

    # ------------------------------------------------------------ 2
    print("\n== 2. the exemption: red-overlapping candidates that were auto-rejected at 0.01 ==")
    old = old_part_b()
    tot = dict(c=0, cc=0, s=0, oc=0, occ=0, rev=0, ar=0)
    print("   %-30s %4s %5s | cand exempt disk (11-04) centre (11-04) | somata AUTO-only | cur scores of exempt candidates" % (
        "session", "rev", "auto"))
    allsc = []
    for x in sorted(new19 + out9, key=lambda x: (x["animal"], x["name"][4:6], x["name"][:4])):
        ar = np.array(x["auto"], dtype=int)
        ex = ar[x["red"][ar]] if len(ar) else ar
        exc = ar[x["red_c"][ar]] if len(ar) else ar
        ns = sum(1 for s in x["somata"] if s["fate"] == "AUTO")
        o = old.get(x["name"])
        sc = np.sort(x["s_cur"][ex])
        if x["labelled"]:
            allsc += sc.tolist()
            tot["c"] += len(ex); tot["cc"] += len(exc); tot["s"] += ns
            tot["rev"] += x["n"] - len(ar); tot["ar"] += len(ar)
            if o:
                tot["oc"] += o[2]; tot["occ"] += o[3]
        print("   %-30s %4d %5d | %3d (%s) %3d (%s) | %3d | %s%s" % (
            x["name"][:30], x["n"] - len(ar), len(ar), len(ex), o[2] if o else "-", len(exc), o[3] if o else "-", ns,
            " ".join("%.4f" % v for v in sc), "" if x["labelled"] else "   [still out]"))
    allsc = np.array(allsc)
    print("   19 labelled: exempt candidates %d (11-04 record for the same sessions: %d), centre rule %d (%d);"
          " = +%.1f%% on %d reviewed; red somata with ONLY auto-rejected candidates: %d" % (
              tot["c"], tot["oc"], tot["cc"], tot["occ"], 100.0 * tot["c"] / max(tot["rev"], 1), tot["rev"], tot["s"]))
    if len(allsc):
        print("   curating-model score of the %d exempt candidates: median %.4f, IQR %.4f-%.4f, max %.4f;"
              " < 0.001: %d, 0.001-0.003: %d, 0.003-0.005: %d, 0.005-0.01: %d" % (
                  len(allsc), np.median(allsc), np.percentile(allsc, 25), np.percentile(allsc, 75), allsc.max(),
                  int((allsc < 0.001).sum()), int(((allsc >= 0.001) & (allsc < 0.003)).sum()),
                  int(((allsc >= 0.003) & (allsc < 0.005)).sum()), int((allsc >= 0.005).sum())))
    som_auto = [s for x in new19 for s in x["somata"] if s["fate"] == "AUTO"]
    if som_auto:
        mx = np.array([s["max_cur"] for s in som_auto])
        print("   the %d AUTO-only somata, best curating score among their candidates: median %.4f, range %.4f-%.4f;"
              " by animal: %s" % (len(som_auto), np.median(mx), mx.min(), mx.max(), ", ".join(
                  "%s %d" % (an, sum(1 for s in som_auto if s["animal"] == an)) for an in sorted({s["animal"] for s in som_auto}))))
    print("\n   proxy A (measured, 19 labelled sessions, curating scores): reviewer keep rate of candidates just ABOVE the")
    print("   threshold, red vs not red. Nothing below 0.01 was reviewed in these sessions.")
    band_table(new19, "s_cur", "all reviewed candidates:", bands=BANDS[:3])
    band_table(new19, "s_cur", "duplicates of a kept cell removed:", bands=BANDS[:3], drop_dup=True)
    for lo, hi in ((0.01, 0.05), (0.01, 0.02)):
        a = b = c = e = 0
        for x in new19:
            m = (x["full"] >= 0) & (x["s_cur"] >= lo) & (x["s_cur"] < hi)
            a += int((x["full"][m & x["red"]] == 1).sum()); b += int((m & x["red"]).sum())
            c += int((x["full"][m & ~x["red"]] == 1).sum()); e += int((m & ~x["red"]).sum())
        p = fisher_exact([[a, b - a], [c, e - c]])[1] if b and e else float("nan")
        print("   band %.2f-%.2f: red kept %s, not red kept %s, Fisher p %.3g" % (lo, hi, rate(a, b), rate(c, e), p))
    print("\n   proxy B (measured, 12 pilots at threshold 0, PREPILOT scores = out of sample; every candidate has a label):")
    for T in (0.01, 0.02):
        a = b = c = e = 0
        for x in pil:
            lo_ = x["s_pre"] < T
            a += int((x["full"][lo_ & x["red"]] == 1).sum()); b += int((lo_ & x["red"]).sum())
            c += int((x["full"][lo_ & ~x["red"]] == 1).sum()); e += int((lo_ & ~x["red"]).sum())
        p = fisher_exact([[a, b - a], [c, e - c]])[1] if b and e else float("nan")
        print("   prepilot score < %.2f: red-overlapping kept %s, not red kept %s, Fisher p %.3g" % (T, rate(a, b), rate(c, e), p))
    ps = [s for x in pil for s in x["somata"] if s["n_on"] > 0 and s["max_pre"] < 0.01]
    print("   soma level, pilots: red somata whose candidates ALL score < 0.01 (prepilot): %d; KEPT %d, DELETED %d" % (
        len(ps), sum(1 for s in ps if s["fate"] == "KEPT"), sum(1 for s in ps if s["fate"] == "DELETED")))
    ps2 = [s for x in pil for s in x["somata"] if s["n_on"] > 0]
    print("   soma level, pilots: red somata with any candidate %d; KEPT %d, DELETED %d" % (
        len(ps2), sum(1 for s in ps2 if s["fate"] == "KEPT"), sum(1 for s in ps2 if s["fate"] == "DELETED")))

    # ------------------------------------------------------------ 3
    print("\n== 3. at the same model score, is a red-overlapping candidate kept more often? (reviewed candidates) ==")
    nred = sum(int(x["red"][x["full"] >= 0].sum()) for x in new19)
    nall = sum(int((x["full"] >= 0).sum()) for x in new19)
    kred = sum(int((x["full"][x["red"]] == 1).sum()) for x in new19)
    kall = sum(int((x["full"] == 1).sum()) for x in new19)
    ndup = sum(int(x["dup"].sum()) for x in new19)
    ndup_red = sum(int((x["dup"] & x["red"]).sum()) for x in new19)
    print("   19 labelled: reviewed %d, kept %d; red-overlapping reviewed %d, kept %d (%.0f%% vs %.0f%% of the rest);"
          " duplicates of a kept cell %d, of which red-overlapping %d" % (
              nall, kall, nred, kred, 100.0 * kred / max(nred, 1), 100.0 * (kall - kred) / max(nall - nred, 1), ndup, ndup_red))
    for nm, ss in (("19 labelled 09xx", new19), ("Taylor 13", [x for x in new19 if x["reviewer"] == "Taylor"]),
                   ("Yawen 6 (pnb105)", [x for x in new19 if x["reviewer"] == "Yawen"])):
        band_table(ss, "s_cur", "%s, curating scores, all reviewed:" % nm)
        band_table(ss, "s_cur", "%s, duplicates of a kept cell removed:" % nm, drop_dup=True)
    for drop in (False, True):
        b = red_offset(new19, "s_cur", drop)
        lo, hi = boot_offset(new19, "s_cur", drop)
        print("   logistic keep ~ logit(curating score) + [red-overlapping], 19 sessions%s: red offset %+.2f (odds x %.2f),"
              " 95%% CI %+.2f .. %+.2f (bootstrap over sessions, 2000 draws)" % (
                  ", duplicates removed" if drop else "", b, np.exp(b), lo, hi))
    b = red_offset(pil, "s_pre", False)
    lo, hi = boot_offset(pil, "s_pre", False)
    print("   same on the 12 pilots with prepilot scores (all candidates): red offset %+.2f (odds x %.2f), 95%% CI %+.2f .. %+.2f" % (
        b, np.exp(b), lo, hi))
    print("   Yawen duplicate check: %d duplicate deletes on the six pnb105 sessions (reviewer_check_yawen.txt G: 282)" % (
        sum(int(x["dup"].sum()) for x in new19 if x["reviewer"] == "Yawen")))

    # ------------------------------------------------------------ 5
    print("\n== 5. stricter variant: a candidate CENTRED on the red soma (mask centroid within %.0f px of the red centre) ==" % CEN_R)
    print("   Added because 1b shows the disk rule is met by an arbitrary point of the field about as often as by a")
    print("   red soma. Same fate order; controls as in 1b.")
    for nm, ss in (("19 labelled 09xx", new19), ("12 pilots", pil)):
        som = [s for x in ss for s in x["somata"]]
        frow(nm + ", red somata, centroid rule", fate_counts(som, "fate_centroid"))
        frow(nm + ", control transposed", fate_counts([f for x in ss for f in x.get("ctrlS_T", [])]))
        frow(nm + ", control rotated 180", fate_counts([f for x in ss for f in x.get("ctrlS_R", [])]))
    print("   by animal, 19 labelled 09xx:")
    for an in sorted({x["animal"] for x in new19}):
        frow("  " + an, fate_counts([s for x in new19 if x["animal"] == an for s in x["somata"]], "fate_centroid"))
    frow("  Taylor 13", fate_counts([s for x in new19 if x["reviewer"] == "Taylor" for s in x["somata"]], "fate_centroid"))
    frow("  Yawen 6", fate_counts([s for x in new19 if x["reviewer"] == "Yawen" for s in x["somata"]], "fate_centroid"))
    frow("9 still out (SHOWN counted under KEPT/DELETED = 0)", fate_counts([s for x in out9 for s in x["somata"]], "fate_centroid"))
    f9 = fate_counts([s for x in out9 for s in x["somata"]], "fate_centroid")
    print("   9 still out: shown to the reviewer %s, auto-rejected only %s, none %s" % (
        rate(f9["SHOWN"], f9["n"]), rate(f9["AUTO"], f9["n"]), rate(f9["NONE"], f9["n"])))
    nex = sum(int(x["red_s"][np.array(x["auto"], dtype=int)].sum()) for x in new19 if len(x["auto"]))
    nrev = sum(x["n"] - len(x["auto"]) for x in new19)
    sa = [s for x in new19 for s in x["somata"] if s["fate_centroid"] == "AUTO"]
    print("   exemption under this rule, 19 labelled: auto-rejected candidates centred on a red soma %d (+%.1f%% on %d"
          " reviewed); red somata whose centred candidates were all auto-rejected: %d (best curating score: %s)" % (
              nex, 100.0 * nex / max(nrev, 1), nrev, len(sa),
              " ".join("%.4f" % v for v in sorted(s["max_cur_centroid"] for s in sa))))
    band_table(new19, "s_cur", "19 labelled, curating scores, candidates centred on a red soma vs the rest, all reviewed:", rk="red_s")
    band_table(new19, "s_cur", "same, duplicates of a kept cell removed:", drop_dup=True, rk="red_s")
    for drop in (False, True):
        b = red_offset(new19, "s_cur", drop, rk="red_s")
        lo, hi = boot_offset(new19, "s_cur", drop, rk="red_s")
        print("   logistic keep ~ logit(curating score) + [centred on a red soma]%s: offset %+.2f (odds x %.2f), 95%% CI %+.2f .. %+.2f" % (
            ", duplicates removed" if drop else "", b, np.exp(b), lo, hi))
    for T in (0.01, 0.02):
        a = b_ = c = e = 0
        for x in pil:
            lo_ = x["s_pre"] < T
            a += int((x["full"][lo_ & x["red_s"]] == 1).sum()); b_ += int((lo_ & x["red_s"]).sum())
            c += int((x["full"][lo_ & ~x["red_s"]] == 1).sum()); e += int((lo_ & ~x["red_s"]).sum())
        print("   pilots, prepilot score < %.2f: centred-on-red kept %s, the rest kept %s" % (T, rate(a, b_), rate(c, e)))
    ps = [s for x in pil for s in x["somata"] if s["n_on_centroid"] > 0 and s["max_pre_centroid"] < 0.01]
    print("   pilots, soma level: red somata whose centred candidates all score < 0.01 (prepilot): %d; KEPT %d, DELETED %d" % (
        len(ps), sum(1 for s in ps if s["fate_centroid"] == "KEPT"), sum(1 for s in ps if s["fate_centroid"] == "DELETED")))

    # per-soma record
    cols = ["session", "animal", "reviewer", "id", "row", "col", "fate", "fate_centre", "n_on_centre", "fate_centroid", "n_on_centroid", "max_cur_centroid", "max_pre_centroid", "n_on", "n_on_auto",
            "on", "max_cur", "max_pre",
            "del_all_dup", "del_any_dup", "del_any_motion"]
    with open(HERE / "red_soma_fate_somata.csv", "w", encoding="ascii", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group"] + cols)
        w.writeheader()
        for grp, ss in (("labelled_09xx", new19), ("pilot", pil), ("still_out", out9)):
            for x in ss:
                for s in x["somata"]:
                    row = {"group": grp}
                    row.update({k: s[k] for k in cols})
                    w.writerow(row)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()
    print(out)
    (HERE / "red_soma_fate.txt").write_text(out, encoding="ascii", errors="replace")
