"""Helpers for the 2026-11-20 seeding re-check (C8 / C9 / C10 of the 2026-11-11 red team) on
every labelled cohort-5 6odor session.  Read-only on the data and on the 11-11 record.

Reuses, by import only, agent/eval/3odor_intake_2026-11/redteam/rt_core.py and rt_labels.py
(seeding statistics, per-cell ceiling) and READS that folder's cache/<session>.npz (the
greedyROI seeding images cn / pnr on the full half-rate movie, 256 x 256).  Nothing is
written there: bytecode writing is switched off before the import and this folder has its
own cache/ for the per-session candidate arrays.

Difference from rt_labels.candidates(): that one refuses a session whose labels.mat does not
cover every candidate.  Here the full vector is rebuilt from candidate_features.npz
(auto_rejected, 0-based):  1 = kept, 0 = reviewed delete, -1 = auto-rejected (never shown).
"""
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REDTEAM = HERE.parents[2] / "3odor_intake_2026-11" / "redteam"
sys.path.insert(0, str(REDTEAM))
import rt_core as rc  # noqa: E402
import rt_labels as rl  # noqa: E402

CACHE = HERE / "cache"
H = W = 512
F_PREV = 0.444
PILOT_DATES = ("082626", "082726", "082826")
FOVA_105 = ("090726", "090826", "090926", "091026")


def floor_prev(T):
    Ts = np.array([500, 2400, 6400, 8445])
    return {500: 0.65, 2400: 0.575, 6400: 0.55, 8445: 0.55}[int(Ts[np.argmin(np.abs(Ts - T))])]


def read_kv(p):
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                a, b = line.split(":", 1)
                out[a.strip()] = b.strip()
    return out


def group_of(name, reviewer, mc):
    """Analysis group of a labelled session."""
    an = re.search(r"(pnb\d+)", name).group(1)
    d = name[:6]
    if d in PILOT_DATES:
        return "pilot 0.60 (082726-pnb105)" if mc < 0.65 else "pilots at 0.70 (11)"
    if an == "pnb105":
        if d in FOVA_105:
            return "Yawen pnb105 FOVA at 0.60 (4)"
        return "Yawen pnb105 %s at 0.60" % ("FOVC 090526" if d == "090526" else "FOVB 090626")
    if mc < 0.65:
        return "Taylor 090526-pnb99 dim FOVB at 0.60"
    return "Taylor 09xx at 0.70 (12)"


GROUP_ORDER = ["pilots at 0.70 (11)", "pilot 0.60 (082726-pnb105)", "Taylor 09xx at 0.70 (12)",
               "Taylor 090526-pnb99 dim FOVB at 0.60", "Yawen pnb105 FOVA at 0.60 (4)",
               "Yawen pnb105 FOVC 090526 at 0.60", "Yawen pnb105 FOVB 090626 at 0.60"]


def support_coords(a):
    """(cand, y, x) of every footprint's support (> 0.2 * its max) at 512 x 512."""
    mx = a.reshape(len(a), -1).max(1)
    c, y, x = np.nonzero(a > (0.2 * mx[:, None, None]))
    return c.astype(np.int32), y.astype(np.int16), x.astype(np.int16)


def csr(co, n):
    c, y, x = co
    return sp.csr_matrix((np.ones(len(c), np.float32), (c, y.astype(np.int64) * W + x.astype(np.int64))),
                         shape=(n, H * W))


def iou(Ma, Mb):
    inter = (Ma @ Mb.T).toarray()
    un = np.asarray(Ma.sum(1)) + np.asarray(Mb.sum(1)).T - inter
    return inter / np.maximum(un, 1)


def load_session(name, sd):
    """Everything per candidate that does not depend on thresholds; cached in CACHE."""
    cp = CACHE / (name + ".npz")
    if cp.exists():
        z = np.load(str(cp))
        core = np.unpackbits(z["core_bits"], axis=1)[:, :256 * 256].reshape(-1, 256, 256).astype(bool)
        return dict(n=int(z["n"]), full=z["full"], fmot=z["fmot"], core=core,
                    co=(z["c"], z["y"], z["x"]), n_final=int(z["n_final"]), join_keep=int(z["join_keep"]),
                    join_min_cos=float(z["join_min_cos"]))
    pre = rl.load_sf(sd / ".pre_ingest" / "spatial_footprints.mat")
    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    n = len(npz["feature_matrix"])
    if pre.shape != (n, H, W):
        raise RuntimeError("%s: footprints %s vs %d candidates" % (name, pre.shape, n))
    auto = sorted(set(npz["auto_rejected"].ravel().astype(int).tolist()))
    d = loadmat(str(sd / "labels.mat"))
    lab = np.asarray(d["labels"]).ravel().astype(int)
    mot = np.asarray(d["motion_delete"]).ravel().astype(int) if "motion_delete" in d else np.zeros(len(lab), int)
    rows = np.arange(n) if len(lab) == n else np.array([i for i in range(n) if i not in set(auto)], dtype=int)
    if len(rows) != len(lab):
        raise RuntimeError("%s: %d labels for %d reviewed candidates" % (name, len(lab), len(rows)))
    full = np.full(n, -1)
    full[rows] = lab
    fmot = np.zeros(n, int)
    fmot[rows] = mot
    # label join check, as rt_labels.candidates(): the reviewer's final footprints against the candidates
    fin = rl.load_sf(sd / "spatial_footprints.mat")
    P = pre.reshape(n, -1)
    F = fin.reshape(fin.shape[0], -1)
    S = (F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)) @ (P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)).T
    best = S.argmax(axis=1)
    join_keep = int((full[best] == 1).sum())
    A = pre.reshape(n, 256, 2, 256, 2).mean(axis=(2, 4))
    mx = A.reshape(n, -1).max(axis=1)
    core = A >= 0.5 * mx[:, None, None]
    core &= mx[:, None, None] > 0
    co = support_coords(pre)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cp), n=n, full=full, fmot=fmot, core_bits=np.packbits(core.reshape(n, -1), axis=1),
                        c=co[0], y=co[1], x=co[2], n_final=fin.shape[0], join_keep=join_keep,
                        join_min_cos=float(S.max(axis=1).min()))
    return dict(n=n, full=full, fmot=fmot, core=core, co=co, n_final=int(fin.shape[0]), join_keep=join_keep,
                join_min_cos=float(S.max(axis=1).min()))


def sessions():
    """Every labelled cohort-5 6odor session with its seeding images, ceilings and duplicate flags."""
    out = []
    for g, name, tif, sd, an in rc.sessions():
        if g != "6odor" or not (sd / "labels.mat").exists():
            continue
        f = rc.CACHE / (name + ".npz")
        if not f.exists():
            print("  NOT IN THE 11-11 IMAGE CACHE, skipped: %s" % name)
            continue
        z = np.load(f)
        cn, pnr, T = z["cn"].astype(float), z["pnr"].astype(float), int(z["T"])
        mc, mp = rl.run_params(sd)
        s = load_session(name, sd)
        s.update(name=name, sd=sd, animal=an, cn=cn, pnr=pnr, T=T, mc=mc, mp=mp,
                 reviewer=read_kv(sd / "labels_provenance.txt").get("reviewer", "?"))
        s["group"] = group_of(name, s["reviewer"], mc)
        s["pk50"] = rc.pk(cn, 50)
        s["floor"] = floor_prev(T)
        s["rule"] = round(s["floor"] + F_PREV * (s["pk50"] - s["floor"]), 2)
        s["ce"], s["pk"] = rl.cell_ceiling(s["core"], cn, pnr, mp)
        M = csr(s["co"], s["n"])
        J = iou(M, M)
        np.fill_diagonal(J, 0.0)
        s["J"] = J
        kept = s["full"] == 1
        s["ovk"] = J[:, kept].max(1) if kept.any() else np.zeros(s["n"])
        # the main session's definition (reviewer_check_yawen.py section G)
        s["dup"] = (s["full"] == 0) & (s["fmot"] == 0) & (s["ovk"] >= 0.5)
        out.append(s)
    return out
