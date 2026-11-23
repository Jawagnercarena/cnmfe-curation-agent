"""
red_exempt_yield.py -- what would "exempt red-overlapping candidates from
auto-reject" add, and are those candidates real?  (agent/eval/missed_somata_2026-11)

READ-ONLY on D:\\Julian_CNMFe and D:\\retro_tdTomato_Alignment.  Writes only
red_exempt_yield.txt next to this file.

Rule (same as the 2026-11-04 soma-coverage measurement): a candidate overlaps a
red soma when its footprint mask (>= 0.3 x its own max) covers >= 30% of the
7-px-radius disk around a red_blobs.csv centre with bleed_flag False.
Sensitivity variant: the red centre pixel lies inside the mask.

Part A (yield, measured): the 12 cohort-5 pilot sessions (082626/082726/082826,
reviewed by Taylor at threshold 0, so every candidate has a human label).  Scores
come from classifier_prepilot_2026-09-24.joblib, trained before those labels
existed (out-of-sample).  The deployed model's scores are shown for reference
only (it was trained on these labels: in-sample).
Part B (size, measured): the 28 sessions now at Taylor -- how many of their
currently auto-rejected candidates the rule would send to the reviewer.
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import joblib
import numpy as np
import scipy.io as sio
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RES = Path(r"D:\retro_tdTomato_Alignment\results")
V = Path(r"D:\Julian_CNMFe\vCA1")
FOOT_LEVEL, DISK_R, COVER = 0.3, 7, 0.30
T_DEPLOYED = 0.01


def load_mat_var(p: Path, name: str) -> np.ndarray:
    try:
        return sio.loadmat(p)[name]
    except NotImplementedError:
        with h5py.File(p, "r") as h:
            a = np.array(h[name])
        return a.T  # h5py reverses MATLAB axis order


def load_footprints(d: Path) -> np.ndarray:
    sf = load_mat_var(d / "spatial_footprints.mat", "spatial_footprints")
    if sf.ndim == 2:
        sf = sf[None]
    return sf.astype(np.float32)  # (N, d1, d2)


def red_centres(csv: Path) -> np.ndarray:
    rb = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8"))
    if rb.size == 0:
        return np.zeros((0, 2), int)
    ok = np.array([str(x).strip() != "True" for x in rb["bleed_flag"]])
    return np.stack([rb["row"][ok], rb["col"][ok]], 1).astype(int)


def overlap_flags(sf: np.ndarray, cen: np.ndarray):
    """Per candidate: (disk rule, centre-pixel rule)."""
    N, d1, d2 = sf.shape
    mx = sf.reshape(N, -1).max(1)
    masks = sf >= FOOT_LEVEL * np.maximum(mx, 1e-12)[:, None, None]
    yy, xx = np.mgrid[-DISK_R:DISK_R + 1, -DISK_R:DISK_R + 1]
    disk = (yy ** 2 + xx ** 2) <= DISK_R ** 2
    f_disk = np.zeros(N, bool); f_ctr = np.zeros(N, bool)
    for r, c in cen:
        r0, r1, c0, c1 = r - DISK_R, r + DISK_R + 1, c - DISK_R, c + DISK_R + 1
        if r0 < 0 or c0 < 0 or r1 > d1 or c1 > d2:
            continue
        cov = (masks[:, r0:r1, c0:c1] & disk).sum((1, 2)) / disk.sum()
        f_disk |= cov >= COVER
        f_ctr |= masks[:, r, c]
    return f_disk, f_ctr


def acorn_dir(session: str) -> Path:
    hits = [d for d in V.glob(f"*/{session}") if d.is_dir() and not d.parent.name.startswith(".")]
    if len(hits) != 1:
        raise RuntimeError(f"{session}: {len(hits)} ACORN folders")
    return hits[0]


def labels_of(d: Path) -> np.ndarray:
    return load_mat_var(d / "labels.mat", "labels").ravel()


def score(model: dict, X: np.ndarray) -> np.ndarray:
    assert X.shape[1] == model["n_features"], (X.shape, model["n_features"])
    return model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]


def rate(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else f"{k}/0"


def part_a():
    pre = joblib.load(REPO / "agent/model/vCA1/classifier_prepilot_2026-09-24.joblib")
    dep = joblib.load(REPO / "agent/model/vCA1/classifier.joblib")
    T_pre = float(pre["reject_threshold"])
    print(f"prepilot model: n_sessions {pre.get('n_sessions')}, n_features {pre['n_features']}, "
          f"reject_threshold {T_pre}; deployed: n_sessions {dep.get('n_sessions')}, T {dep['reject_threshold']}")
    pilots = sorted(p.parent for p in RES.glob("pnb*/0826*/*/red_blobs.csv")) + \
        sorted(p.parent for p in RES.glob("pnb*/0827*/*/red_blobs.csv")) + \
        sorted(p.parent for p in RES.glob("pnb*/0828*/*/red_blobs.csv"))
    rows = []
    print(f"\nPart A: {len(pilots)} pilot sessions")
    for g in pilots:
        s = g.name; d = acorn_dir(s)
        z = np.load(d / "candidate_features.npz")
        X = z["feature_matrix"]; N = int(z["n_candidates"][0])
        assert len(z["auto_rejected"]) == 0, f"{s}: has auto-rejects"
        L = labels_of(d); assert len(L) == N, (s, len(L), N)
        # after review spatial_footprints.mat holds only the kept cells; the
        # first ingest snapshotted the all-candidate headless file into .pre_ingest/
        sf = load_footprints(d / ".pre_ingest"); assert sf.shape[0] == N, (s, sf.shape, N)
        cen = red_centres(g / "red_blobs.csv")
        fd, fc = overlap_flags(sf, cen)
        sp, sd = score(pre, X), score(dep, X)
        print(f"  {s:46s} N {N:4d} kept {int((L == 1).sum()):3d} red somata {len(cen):3d} "
              f"red-overlap cand disk {int(fd.sum()):3d} ctr {int(fc.sum()):3d} labels {sorted(set(np.unique(L).tolist()))}")
        for i in range(N):
            rows.append(dict(s=s, keep=int(L[i] == 1), red=bool(fd[i]), red_c=bool(fc[i]),
                             sp=float(sp[i]), sd=float(sd[i])))
    keep = np.array([r["keep"] for r in rows]); red = np.array([r["red"] for r in rows])
    red_c = np.array([r["red_c"] for r in rows])
    sp = np.array([r["sp"] for r in rows]); sd = np.array([r["sd"] for r in rows])
    print(f"\npooled candidates {len(rows)}, kept {int(keep.sum())} ({100 * keep.mean():.1f}%), "
          f"red-overlapping (disk rule) {int(red.sum())}, kept among them {rate(int(keep[red].sum()), int(red.sum()))}")
    for name, s_, T in [("prepilot (out-of-sample)", sp, T_DEPLOYED), ("prepilot (out-of-sample)", sp, T_pre),
                        ("deployed (IN-sample, reference)", sd, T_DEPLOYED)]:
        lo = s_ < T
        a, b = int(keep[lo & red].sum()), int((lo & red).sum())
        c, e = int(keep[lo & ~red].sum()), int((lo & ~red).sum())
        p = fisher_exact([[a, b - a], [c, e - c]])[1] if b and e else float("nan")
        print(f"- {name}, T {T:.3f}: below T {int(lo.sum())} (kept {int(keep[lo].sum())}); "
              f"red-overlap below T {b}, kept {rate(a, b)}; non-red below T {e}, kept {rate(c, e)}; Fisher p {p:.3g}")
        lo_c = lo & red_c
        print(f"    centre rule: red-overlap below T {int(lo_c.sum())}, kept {rate(int(keep[lo_c].sum()), int(lo_c.sum()))}")
        print(f"    above T: red kept {rate(int(keep[~lo & red].sum()), int((~lo & red).sum()))}, "
              f"non-red kept {rate(int(keep[~lo & ~red].sum()), int((~lo & ~red).sum()))}")
    lo = sp < T_DEPLOYED
    print("  per session (prepilot, T 0.01): red-overlap below T -> kept")
    for s in sorted({r["s"] for r in rows}):
        m = np.array([r["s"] == s for r in rows])
        print(f"    {s:46s} {int((m & lo & red).sum()):3d} -> {int(keep[m & lo & red].sum()):3d} kept")


def part_b():
    print("\nPart B: the 28 sessions at Taylor (current npz auto_rejected)")
    tot_ar = tot_add = tot_add_c = tot_rev = 0
    for gj in sorted(RES.glob("*/*/*/red_guide/guide.json")):
        g = json.load(open(gj)); d = Path(g["acorn_dir"]); s = gj.parent.parent.name
        z = np.load(d / "candidate_features.npz"); N = int(z["n_candidates"][0])
        ar = np.asarray(z["auto_rejected"]).ravel().astype(int)
        sf = load_footprints(d); assert sf.shape[0] == N, (s, sf.shape, N)
        fd, fc = overlap_flags(sf, red_centres(gj.parent / "red_blobs.csv"))
        add, add_c = int(fd[ar].sum()), int(fc[ar].sum())
        tot_ar += len(ar); tot_add += add; tot_add_c += add_c; tot_rev += N - len(ar)
        print(f"  {s:46s} N {N:4d} review {N - len(ar):4d} auto-rej {len(ar):4d} "
              f"-> exempt (disk) {add:3d} (centre rule {add_c:3d})")
    print(f"  TOTAL: review {tot_rev}, auto-rejected {tot_ar}, would be exempted {tot_add} "
          f"(centre rule {tot_add_c}) = +{100 * tot_add / tot_rev:.1f}% review load")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        part_a()
        part_b()
    out = buf.getvalue()
    print(out)
    (HERE / "red_exempt_yield.txt").write_text(out, encoding="utf-8")
