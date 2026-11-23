"""
analyze_seedtrace.py -- why do visible GCaMP somata end up with no ACORN candidate?
(agent/eval/missed_somata_2026-11, 2026-11-05)

Inputs (all read-only):
  seedtrace/<session>/seedtrace.mat   from run_seedtrace.py (every seed attempt of
                                      the greedy init + the residual pick, and a
                                      snapshot of the component set after every
                                      merge / delete step)
  somata/<date-animal>/blobs.json     GCaMP mean-image somata (DoG), 2026-11-04
  somata/<date-animal>/reds.json      red somata (red_blobs.csv), 2026-11-04
                                      (scripts that made them: somata/scripts/)
  D:\\Julian_CNMFe\\vCA1\\6odorDualDiffRew\\<session>\\spatial_footprints.mat and
  candidate_features.npz             production output (reproduction check)
  D:\\retro_tdTomato_Alignment\\results\\...\\red_guide\\guide.json   (um per px)
Writes only seedtrace_analysis.txt and seedtrace_somata.csv next to this file.

Coverage rule (same as the 2026-11-04 measurement): a soma is covered by a
component when the component's footprint at >= 0.3 of its own max covers >= 30%
of the 7-px (full-res) disk at the soma; 3.5 px on the 2x-downsampled grid.
A seed attempt belongs to a soma when its pixel lies inside that ds disk.
"""
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
TASK_DIR = Path(r"D:\Julian_CNMFe\vCA1\6odorDualDiffRew")
RES = Path(r"D:\retro_tdTomato_Alignment\results")
R_FULL = 7.0
R_DS = R_FULL / 2
COVER_LVL, COVER_FRAC = 0.3, 0.3
OUTCOME = {1: "below min at its turn", 2: "weak signal", 3: "extract fail/NaN", 4: "min_pixel", 5: "initialised"}


# ---------------------------------------------------------------- loading
def scal(x):
    return np.asarray(x).ravel()[0]


def strv(x):
    a = np.asarray(x).ravel()
    return str(a[0]) if a.size else ""


def cell_list(x):
    a = np.asarray(x)
    if a.size == 0:
        return []
    if a.dtype == object:
        return [e for e in a.ravel()]
    return [a]


def as_struct(e):
    a = np.asarray(e)
    return a.ravel()[0] if a.dtype == object else e


def load_trace(p: Path):
    m = sio.loadmat(p, squeeze_me=False, struct_as_record=False)
    snaps = []
    for e in cell_list(m["ST_SNAPS"]):
        s = as_struct(e)
        K, npix = int(scal(s.K)), int(scal(s.npix))
        d = int(round(np.sqrt(npix)))
        assert d * d == npix
        if K > 0 and np.size(s.i):
            i = np.asarray(s.i).ravel().astype(np.int64) - 1
            j = np.asarray(s.j).ravel().astype(np.int64) - 1
            v = np.asarray(s.v).ravel().astype(np.float64)
            R = sp.csc_matrix((v, (i, j)), shape=(npix, K))
        else:
            R = sp.csc_matrix((npix, K))
        groups = [np.asarray(g).ravel().astype(int) - 1 for g in cell_list(s.merged)]
        newIDs = np.asarray(s.newIDs).ravel().astype(int) - 1 if np.size(s.newIDs) else np.zeros(0, int)
        snaps.append(dict(tag=strv(s.tag), K=K, d=d, R=R, groups=groups, newIDs=newIDs))
    seeds = {}
    for e in cell_list(m["ST_SEEDS"]):
        z = as_struct(e)
        seeds[strv(z.stage)] = dict(
            att=np.asarray(z.att, float).reshape(-1, 5), v1=np.asarray(z.v1, float),
            Cn0=np.asarray(z.Cn0, float), PNR0=np.asarray(z.PNR0, float), K=int(scal(z.K)), k=int(scal(z.k)),
            d1=int(scal(z.d1)), d2=int(scal(z.d2)), min_corr=float(scal(z.min_corr)), min_pnr=float(scal(z.min_pnr)),
            gSig=float(scal(z.gSig)), gSiz=float(scal(z.gSiz)), end=int(scal(z.end_reason)),
            center=np.asarray(z.center, float).reshape(-1, 2))
    return snaps, seeds, sp.csc_matrix(m["st_final_A"])


# ---------------------------------------------------------------- linking
def step_type(tag: str) -> str:
    """init, order, qm, mn, ut, us, tc, pick, full"""
    return re.sub(r"\d+$", "", tag.split("_")[-1])


def to_ds(R: sp.csc_matrix, d: int) -> sp.csc_matrix:
    """full-res (d x d, column-major) -> d/2 grid, summing each 2x2 block."""
    coo = R.tocoo()
    r, c = coo.row % d, coo.row // d
    h = d // 2
    return sp.csc_matrix((coo.data, ((r // 2) + (c // 2) * h, coo.col)), shape=(h * h, R.shape[1]))


def cos_sim(A: sp.csc_matrix, B: sp.csc_matrix) -> np.ndarray:
    na = np.sqrt(np.asarray(A.multiply(A).sum(0))).ravel()
    nb = np.sqrt(np.asarray(B.multiply(B).sum(0))).ravel()
    S = (A.T @ B).toarray()
    return S / np.maximum(na[:, None] * nb[None, :], 1e-12)


def monotone_align(S: np.ndarray):
    """prev i -> next j, order kept, K_prev - K_next prev components deleted;
    maximises the summed similarity of the matched pairs."""
    Kp, Kn = S.shape
    nd = Kp - Kn
    assert nd >= 0
    if nd == 0:
        return np.arange(Kp)
    NEG = -1e18
    f = np.full((Kp + 1, nd + 1), NEG)   # f[i, x]: first i prev aligned with x deletions
    f[0, 0] = 0.0
    back = np.zeros((Kp + 1, nd + 1), np.int8)
    for i in range(1, Kp + 1):
        for x in range(0, min(i, nd) + 1):
            j = i - x
            best, b = NEG, 0
            if x >= 1 and f[i - 1, x - 1] > NEG:            # delete prev i-1
                best, b = f[i - 1, x - 1], 1
            if j >= 1 and j <= Kn and f[i - 1, x] > NEG:    # match prev i-1 -> next j-1
                val = f[i - 1, x] + S[i - 1, j - 1]
                if val > best:
                    best, b = val, 2
            f[i, x] = best
            back[i, x] = b
    fwd = np.full(Kp, -1)
    i, x = Kp, nd
    while i > 0:
        if back[i, x] == 2:
            fwd[i - 1] = i - x - 1
        else:
            x -= 1
        i -= 1
    return fwd


def link(prev, nxt, notes):
    """Return (fwd, merged_into, method).  fwd[k_prev] = k_next or -1."""
    Kp, Kn, tag = prev["K"], nxt["K"], nxt["tag"]
    merged_into = {}
    if step_type(tag) in ("qm", "mn"):
        groups = nxt["groups"]
        if not groups:
            assert Kp == Kn, (tag, Kp, Kn)
            return np.arange(Kp), merged_into, "merge step, no merge"
        base_del = set()
        for g in groups:
            base_del.update(g[1:].tolist())
        f = (Kp - len(base_del)) - Kn          # groups whose deconvolution failed: all members deleted
        assert 0 <= f <= len(groups), (tag, Kp, Kn, len(base_del))
        sols = []
        for failed in combinations(range(len(groups)), f):
            D = set(base_del) | {int(groups[g][0]) for g in failed}
            surv = sorted(int(groups[g][0]) for g in range(len(groups)) if g not in failed)
            Ds = np.array(sorted(D))
            post = [p - int(np.searchsorted(Ds, p)) for p in surv]
            if Kp - len(D) == Kn and np.array_equal(np.array(post, int), np.sort(nxt["newIDs"])):
                sols.append((failed, D))
        assert len(sols) >= 1, f"{tag}: merge bookkeeping does not match K / newIDs"
        if len(sols) > 1:
            notes.append(f"{tag}: {len(sols)} failed-group assignments fit; first taken")
        failed, D = sols[0]
        Ds = np.array(sorted(D))
        fwd = np.full(Kp, -1)
        for k in range(Kp):
            if k not in D:
                fwd[k] = k - int(np.searchsorted(Ds, k))
        for gi, g in enumerate(groups):
            if gi in failed:
                notes.append(f"{tag}: merge group {g.tolist()} failed -> all deleted")
                continue
            for k in g[1:]:
                merged_into[int(k)] = int(fwd[g[0]])
        # verification: components outside any group are unchanged
        inside = set(np.concatenate(groups).tolist())
        keep = [k for k in range(Kp) if k not in inside]
        if keep:
            S = cos_sim(prev["R"][:, keep], nxt["R"][:, fwd[keep]])
            dmin = float(np.min(np.diag(S)))
            if dmin < 0.9999:
                notes.append(f"{tag}: untouched components changed (min cos {dmin:.4f})")
        surv = [int(g[0]) for gi, g in enumerate(groups) if gi not in failed]
        if surv:
            S = cos_sim(prev["R"][:, surv], nxt["R"][:, fwd[surv]])
            nxt.setdefault("surv_unchanged", []).extend((np.diag(S) >= 0.9999).tolist())
        return fwd, merged_into, f"merge bookkeeping ({len(groups)} groups, {f} failed)"
    A, B = prev["R"], nxt["R"]
    if prev["d"] != nxt["d"]:
        assert nxt["d"] == 2 * prev["d"], (prev["d"], nxt["d"])
        B = to_ds(B, nxt["d"])
    if tag == "order":
        S = cos_sim(A, B)
        r, c = linear_sum_assignment(-S)
        fwd = np.full(Kp, -1)
        fwd[r] = c
        if float(S[r, c].min()) < 0.9999:
            notes.append(f"order: permutation not exact (min cos {float(S[r, c].min()):.4f})")
        return fwd, merged_into, "permutation"
    if tag == "L1_pick":
        assert Kn >= Kp
        S = cos_sim(A, B[:, :Kp])
        if float(np.diag(S).min()) < 0.9999:
            notes.append(f"L1_pick: existing components changed (min cos {float(np.diag(S).min()):.4f})")
        return np.arange(Kp), merged_into, f"append {Kn - Kp}"
    S = cos_sim(A, B)
    fwd = monotone_align(S)
    m = fwd >= 0
    sims = S[np.where(m)[0], fwd[m]]
    if Kp > Kn:
        notes.append(f"{tag}: {Kp - Kn} deleted (aligned; matched cos min {sims.min():.3f}, median {np.median(sims):.3f})")
    return fwd, merged_into, f"monotone ({Kp - Kn} deleted)"


# ---------------------------------------------------------------- per soma
def disk_index(r, c, d, rad):
    yy, xx = np.mgrid[:d, :d]
    m = (yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2
    rr, cc = np.nonzero(m)
    return rr + cc * d, m


def cover_of(R: sp.csc_matrix, idx: np.ndarray) -> np.ndarray:
    sub = R[idx, :]
    return np.asarray((sub >= COVER_LVL).sum(0)).ravel() / len(idx)


def centroid(R: sp.csc_matrix, k: int, d: int):
    col = R[:, k].tocoo()
    w = col.data
    rr, cc = col.row % d, col.row // d
    return float((rr * w).sum() / w.sum()), float((cc * w).sum() / w.sum())


def follow(maps, t0, k):
    """Follow component k from snapshot t0 to the end."""
    for t in range(t0, len(maps)):
        fwd, merged_into, _ = maps[t]
        f = int(fwd[k])
        if f < 0:
            if k in merged_into:
                return dict(fate="merged", t=t + 1, into=merged_into[k])
            return dict(fate="deleted", t=t + 1)
        k = f
    return dict(fate="survived", t=len(maps), k=k)


def seeds_near(z, r_ds, c_ds, rad):
    att = z["att"]
    if att.size == 0:
        return att
    ind = att[:, 1].astype(np.int64) - 1
    ar, ac = ind % z["d1"], ind // z["d1"]
    return att[np.hypot(ar - r_ds, ac - c_ds) <= rad]


def analyse_session(sess, out_rows):
    tag = "-".join(re.search(r"-(\d{6})-(pnb\d+)-", sess).groups())
    animal, date = tag.split("-")[1], tag.split("-")[0]
    um = json.load(open(RES / animal / date / sess / "red_guide" / "guide.json"))["um_per_px"]
    snaps, seeds, Afin = load_trace(HERE / "seedtrace" / sess / "seedtrace.mat")
    print(f"\n=== {sess}  ({um:.3f} um/px full res)")

    # ---- reproduction
    SF = sio.loadmat(TASK_DIR / sess / "spatial_footprints.mat")["spatial_footprints"]
    if SF.ndim == 2:
        SF = SF[None]
    N = int(np.load(TASK_DIR / sess / "candidate_features.npz")["n_candidates"][0])
    K = Afin.shape[1]
    rep = f"re-run K {K} vs production spatial_footprints {SF.shape[0]} / npz N {N}"
    if K == SF.shape[0]:
        prod = np.stack([SF[k].ravel(order="F") for k in range(K)], 1)
        A = Afin.toarray()
        dmax = float(np.abs(A - prod).max())
        na, npr = np.linalg.norm(A, axis=0), np.linalg.norm(prod, axis=0)
        cs = (A * prod).sum(0) / np.maximum(na * npr, 1e-12)
        rep += f"; max |diff| {dmax:.3g} (max value {float(np.abs(prod).max()):.3g}); per-component cos min {cs.min():.6f}"
        exact = dmax == 0.0
    else:
        exact = False
    print("reproduction: " + rep + ("  -> EXACT" if exact else ""))

    # ---- seeding summary
    for st in ("init", "pick"):
        z = seeds[st]
        o = Counter(int(x) for x in z["att"][:, 3])
        print(f"seeding {st}: gSig {z['gSig']:g} gSiz {z['gSiz']:g} (ds grid {z['d1']}x{z['d2']}), min_corr {z['min_corr']}, "
              f"min_pnr {z['min_pnr']}; K cap {z['K']}, initialised {z['k']}, end reason {z['end']} (1 = no local max left, 2 = K cap); "
              f"attempts {len(z['att'])}: " + ", ".join(f"{OUTCOME[k]} {o[k]}" for k in sorted(o)))

    # ---- linking
    notes, maps = [], []
    for t in range(len(snaps) - 1):
        maps.append(link(snaps[t], snaps[t + 1], notes))
    ks = " -> ".join(f"{s['tag']}:{s['K']}" for s in snaps if s["tag"] in ("init", "L1_qm0", "L1_pick", "F_full", "Q_mn"))
    print(f"component count: {ks}")
    mn_s = [x for s in snaps if s["tag"].split("_")[-1].startswith("mn") for x in s.get("surv_unchanged", [])]
    qm_s = [x for s in snaps if s["tag"].split("_")[-1].startswith("qm") for x in s.get("surv_unchanged", [])]
    print(f"merge survivors with an UNCHANGED footprint: MergeNeighbors {sum(mn_s)}/{len(mn_s)}, quickMerge {sum(qm_s)}/{len(qm_s)}")
    for n in notes:
        print("  link note: " + n)
    # merged / deleted counts per stage family
    fam = Counter()
    for t, (fwd, mi, _) in enumerate(maps):
        step = step_type(snaps[t + 1]["tag"])
        fam[(step, "merged")] += len(mi)
        fam[(step, "deleted")] += int((fwd < 0).sum()) - len(mi)
    print("components removed, by step type: " + ", ".join(f"{s} {w} {n}" for (s, w), n in sorted(fam.items()) if n))

    # ---- birth bookkeeping: own components of init / pick seeds
    t_pick = next(i for i, s in enumerate(snaps) if s["tag"] == "L1_pick")
    Kpre_pick = snaps[t_pick - 1]["K"]
    assert snaps[t_pick]["K"] == Kpre_pick + seeds["pick"]["k"], "pick append count mismatch"
    assert snaps[0]["K"] == seeds["init"]["k"], "init count mismatch"
    zi = seeds["init"]
    min_v = zi["min_corr"] * zi["min_pnr"]
    halfmax = np.asarray((snaps[0]["R"] > 0.5).sum(1)).ravel() > 0      # ind_search blocks from initialised neighbours

    for s in snaps:
        s["Rr"] = s["R"].tocsr()
    # birth record of every component at every snapshot: ('init'|'pick', greedy index)
    birth = [[("init", k) for k in range(snaps[0]["K"])]]
    for t, (fwd, _, _) in enumerate(maps):
        nb = [None] * snaps[t + 1]["K"]
        for k, f in enumerate(fwd):
            if f >= 0:
                nb[f] = birth[t][k]
        if snaps[t + 1]["tag"] == "L1_pick":
            for j in range(Kpre_pick, snaps[t + 1]["K"]):
                nb[j] = ("pick", j - Kpre_pick)
        assert all(b is not None for b in nb), snaps[t + 1]["tag"]
        birth.append(nb)

    def seed_full(b):   # birth seed position, full-res (row, col)
        rc = seeds[b[0]]["center"][b[1]] - 1.0
        return 2 * rc[0] + 0.5, 2 * rc[1] + 0.5

    def cdist_um(t, k, r, c):
        sc = centroid(snaps[t]["R"], k, snaps[t]["d"])
        s_ = 512 / snaps[t]["d"]
        return float(np.hypot(sc[0] * s_ - r, sc[1] * s_ - c) * um)

    som = json.load(open(HERE / "somata" / tag / "blobs.json"))
    reds = json.load(open(HERE / "somata" / tag / "reds.json"))
    items = [("gcamp", x) for x in som] + [("red", x) for x in reds]
    # final-snapshot cover of every GCaMP soma: does a component end on ANOTHER visible soma?
    Fcov = np.stack([cover_of(snaps[-1]["Rr"], disk_index(x["r"], x["c"], 512, R_FULL)[0]) for x in som])
    for ii, (kind, x) in enumerate(items):
        r, c = x["r"], x["c"]
        r2, c2 = r / 2.0, c / 2.0
        idx_ds, m_ds = disk_index(r2, c2, 256, R_DS)
        idx_f, _ = disk_index(r, c, 512, R_FULL)
        cov = [cover_of(s["Rr"], idx_ds if s["d"] == 256 else idx_f) for s in snaps]
        covered = np.array([cv.max() >= COVER_FRAC if cv.size else False for cv in cov])
        fin_cov = bool(covered[-1])
        # seeding
        gate_raw = bool(np.any((np.nan_to_num(zi["Cn0"][m_ds]) >= zi["min_corr"]) & (np.nan_to_num(zi["PNR0"][m_ds]) >= zi["min_pnr"])))
        elig = m_ds & (zi["v1"] >= min_v)
        elig1 = bool(elig.any())
        blocked = elig1 and bool(halfmax[np.nonzero(elig.ravel(order="F"))[0]].all())
        a_init = seeds_near(zi, r2, c2, R_DS)
        a_pick = seeds_near(seeds["pick"], r2, c2, R_DS)
        a_init5 = seeds_near(zi, r2, c2, 5.0)
        own = [(0, int(k) - 1, "init") for k in a_init[a_init[:, 3] == 5][:, 4]] + \
              [(t_pick, Kpre_pick + int(k) - 1, "pick") for k in a_pick[a_pick[:, 3] == 5][:, 4]]
        fates = [(st, follow(maps, t0, k)) for (t0, k, st) in own]
        # final category
        sub, detail = "", ""
        if own:
            # a surviving own component wins; otherwise report the first own component
            iw = next((i for i, (_, f) in enumerate(fates) if f["fate"] == "survived"), 0)
            st, f = fates[iw]
            t0, k0, _ = own[iw]
            cat = "D seeded"
            if f["fate"] == "survived":
                k_end = f["k"]
                oc = float(cov[-1][k_end])
                sub = "survived, covers soma" if oc >= COVER_FRAC else "survived, footprint off soma"
                rc = centroid(snaps[-1]["R"], k_end, 512)
                detail = f"born {st}; final #{k_end}; own cover {oc:.2f}; centroid {np.hypot(rc[0]-r, rc[1]-c)*um:.1f} um away"
                # where did its own cover first drop below the rule?
                kk, lost = k0, None
                for t in range(t0, len(snaps)):
                    if lost is None and cov[t][kk] < COVER_FRAC:
                        lost = snaps[t]["tag"]
                    if t < len(maps):
                        kk = int(maps[t][0][kk])
                if lost:
                    detail += f"; own cover first < {COVER_FRAC} at {lost}"
            else:
                step = snaps[f["t"]]["tag"]
                if f["fate"] == "merged":
                    typ = "quickMerge" if step_type(step) == "qm" else "MergeNeighbors"
                    sub = f"merged ({typ})"
                    k2 = f["into"]
                    sc = centroid(snaps[f["t"]]["R"], k2, snaps[f["t"]]["d"])
                    scale = 512 / snaps[f["t"]]["d"]
                    dist = np.hypot(sc[0] * scale - r, sc[1] * scale - c) * um
                    fs = follow(maps, f["t"], k2)
                    detail = (f"born {st}; merged at {step} into a component centred {dist:.1f} um away "
                              f"(survivor cover of soma {float(cov[f['t']][k2]):.2f}; survivor then {fs['fate']})")
                else:
                    sub = f"deleted without a merge ({step_type(step)})"
                    detail = f"born {st}; deleted at {step}"
        elif len(a_init) or len(a_pick):
            cat = "C attempted, rejected"
            o = Counter(OUTCOME[int(v)] for v in np.concatenate([a_init[:, 3], a_pick[:, 3]]))
            sub = ", ".join(sorted(o))
            detail = "; ".join(f"{k} x{n}" for k, n in o.items()) + f" (init {len(a_init)}, pick {len(a_pick)})"
        elif elig1:
            cat = "B eligible, never attempted"
            sub = "blocked by a neighbour's half-max footprint" if blocked else "dropped below min after a neighbour was subtracted"
        elif gate_raw:
            cat = "A never eligible"
            sub = "passes raw Cn/PNR gate, lost to the median filter"
        else:
            cat = "A never eligible"
            sub = "fails the Cn/PNR gate"
        ever = bool(covered.any())
        last = int(np.max(np.nonzero(covered)[0])) if ever else -1
        lost_at = snaps[last + 1]["tag"] if ever and last + 1 < len(snaps) else ("" if not ever else "never lost")
        # ---- coverage-first fate: what happened to the component that last covered the soma
        fate, fate_sub, cov_seed_um, cov_cent_um, n_cov_last, cov_birth = "", "", np.nan, np.nan, 0, ""
        retained01, ends_other = np.nan, ""

        def on_other_soma(k_final):
            others = [j for j in range(len(som)) if not (kind == "gcamp" and j == ii)]
            return "yes" if any(Fcov[j, k_final] >= COVER_FRAC for j in others) else "no"
        if not ever:
            fate, fate_sub = "1 never covered", cat[0] + ": " + sub
        else:
            ks = int(np.argmax(cov[last]))
            n_cov_last = int((cov[last] >= COVER_FRAC).sum())
            b = birth[last][ks]
            cov_birth = b[0]
            sr, sc_ = seed_full(b)
            cov_seed_um = float(np.hypot(sr - r, sc_ - c) * um)
            cov_cent_um = cdist_um(last, ks, r, c)
            if last == len(snaps) - 1:
                fate, fate_sub = "0 covered at the end", ""
            else:
                fwd, mi, _ = maps[last]
                stp = step_type(snaps[last + 1]["tag"])
                if fwd[ks] < 0 and ks in mi:
                    fate = "2 covering component merged away"
                    fate_sub = f"{'quickMerge' if stp == 'qm' else 'MergeNeighbors'}"
                    cov_birth += f"; into a component centred {cdist_um(last + 1, mi[ks], r, c):.1f} um away"
                    fo = follow(maps, last + 1, int(mi[ks]))
                    if fo["fate"] == "survived":
                        ends_other = on_other_soma(fo["k"])
                elif fwd[ks] < 0:
                    fate, fate_sub = "3 covering component deleted", stp
                else:
                    fate, fate_sub = "4 covering footprint retracted off the soma", stp
                    k1 = int(fwd[ks])
                    nx = snaps[last + 1]
                    idx1 = idx_ds if nx["d"] == 256 else idx_f
                    retained01 = float((nx["Rr"][idx1, :][:, k1].toarray().ravel() > 0).mean())
                    fo = follow(maps, last + 1, k1)
                    cov_birth += f"; that component then {fo['fate']}"
                    if fo["fate"] == "survived":
                        cov_birth += f" (final candidate centred {cdist_um(len(snaps) - 1, fo['k'], r, c):.1f} um away)"
                        ends_other = on_other_soma(fo["k"])
        gate_px = (np.nan_to_num(zi["Cn0"]) >= zi["min_corr"]) & (np.nan_to_num(zi["PNR0"]) >= zi["min_pnr"])
        n_gate_px, disk_px = int(gate_px[m_ds].sum()), int(m_ds.sum())
        # red somata: nearest GCaMP mean-image soma (a red/green offset would make one cell look like two)
        ng_um, ng_cat = np.nan, ""
        if kind == "red":
            dg = [np.hypot(g["r"] - r, g["c"] - c) * um for g in som]
            if dg:
                j = int(np.argmin(dg))
                ng_um, ng_cat = float(dg[j]), som[j]["cat"]
        # nearest initialised seed (other than its own) -- who took the neighbourhood
        cen = zi["center"] - 1.0
        dn = np.hypot(cen[:, 0] - r2, cen[:, 1] - c2) * 2 * um if len(cen) else np.array([np.inf])
        out_rows.append(dict(
            session=sess, kind=kind, r=r, c=c, prod_cat=x["cat"], prod_gate=x["gate"], rerun_final_covered=fin_cov,
            gate_raw=gate_raw, eligible_pass1=elig1, blocked=blocked, n_att_init=len(a_init), n_att_init_r5=len(a_init5),
            n_att_pick=len(a_pick), n_own=len(own), category=cat, sub=sub, detail=detail, ever_covered=ever,
            first_cov=snaps[int(np.argmax(covered))]["tag"] if ever else "", cover_lost_at=lost_at,
            fate=fate, fate_sub=fate_sub, n_covering_at_last=n_cov_last, covering_birth=cov_birth,
            covering_seed_um=round(cov_seed_um, 2), covering_centroid_um=round(cov_cent_um, 2),
            retract_disk_left_nonzero=round(retained01, 2), covering_ends_on_other_soma=ends_other,
            n_gate_px_in_disk=n_gate_px, disk_px_ds=disk_px, nearest_gcamp_soma_um=round(ng_um, 2),
            nearest_gcamp_soma_cat=ng_cat,
            nearest_init_seed_um=round(float(dn.min()), 2), dog_z=round(float(x.get("dog_z", np.nan)), 2)))
    return exact


def summarise(rows, title, sel):
    S = [r for r in rows if sel(r)]
    print(f"\n--- {title}: n = {len(S)}")
    if not S:
        return
    print("  [coverage-first] fate of the component that last covered the soma")
    fz = Counter(r["fate"] for r in S)
    for fa in sorted(fz):
        T = [r for r in S if r["fate"] == fa]
        print(f"  {fa}: {fz[fa]} ({100 * fz[fa] / len(S):.0f}%)")
        for s, n in Counter(r["fate_sub"] for r in T).most_common():
            if s:
                print(f"      {s}: {n}")
        if not fa.startswith("1"):
            d = np.array([r["covering_seed_um"] for r in T])
            dc = np.array([r["covering_centroid_um"] for r in T])
            print(f"      covering component's birth seed: median {np.median(d):.1f} um from the soma centre "
                  f"(<= 5 um {int((d <= 5).sum())}/{len(d)}); its centroid then median {np.median(dc):.1f} um; "
                  f"born at init {sum(r['covering_birth'].startswith('init') for r in T)}, pick "
                  f"{sum(r['covering_birth'].startswith('pick') for r in T)}")
            eo = Counter(r["covering_ends_on_other_soma"] for r in T if r["covering_ends_on_other_soma"])
            if eo:
                print(f"      that component ends as a candidate covering ANOTHER visible GCaMP soma: {dict(eo)}")
            rt = [r["retract_disk_left_nonzero"] for r in T if np.isfinite(r["retract_disk_left_nonzero"])]
            if rt:
                rt = np.array(rt)
                print(f"      after the step, share of the soma disk still >= 0.1 of that footprint's peak (snapshots keep >= 0.1): "
                      f"median {np.median(rt):.2f}; none left {int((rt == 0).sum())}/{len(rt)}, >= half left {int((rt >= 0.5).sum())}/{len(rt)}")
    nv = [r for r in S if r["fate"].startswith("1")]
    if nv:
        g = np.array([r["n_gate_px_in_disk"] for r in nv])
        print(f"  never covered: raw-gate pixels inside the soma's ds disk ({nv[0]['disk_px_ds']} px): median {np.median(g):.0f}, "
              f"0 px {int((g == 0).sum())}, 1-12 px {int(((g >= 1) & (g <= 12)).sum())}, >= 13 px {int((g >= 13).sum())}; "
              f"nearest initialised seed median {np.median([r['nearest_init_seed_um'] for r in nv]):.1f} um")
    rd = [r for r in S if r["kind"] == "red" and np.isfinite(r["nearest_gcamp_soma_um"])]
    if rd:
        d = np.array([r["nearest_gcamp_soma_um"] for r in rd])
        cov_near = sum(1 for r in rd if r["nearest_gcamp_soma_um"] <= 3 and r["nearest_gcamp_soma_cat"] != "none")
        print(f"  red: nearest GCaMP mean-image soma median {np.median(d):.1f} um; within 3 um AND that GCaMP soma has a "
              f"candidate (possible red/green offset, same cell) {cov_near}/{len(rd)}")
    print("  [seed-based] attempts inside the soma's own 7-px disk")
    cats = Counter(r["category"] for r in S)
    for cat in sorted(cats):
        print(f"  {cat}: {cats[cat]} ({100 * cats[cat] / len(S):.0f}%)")
        subs = Counter(r["sub"] for r in S if r["category"] == cat)
        for s, n in subs.most_common():
            print(f"      {s}: {n}")
    ev = sum(r["ever_covered"] for r in S)
    lost = Counter(step_type(r["cover_lost_at"]) if r["cover_lost_at"] != "never lost" else "never lost"
                   for r in S if r["ever_covered"])
    print(f"  ever covered by some component at some stage: {ev}/{len(S)}; step type where coverage was last lost: {dict(lost)}")
    print(f"  nearest initialised seed (any) median {np.median([r['nearest_init_seed_um'] for r in S]):.1f} um")
    per = defaultdict(Counter)
    for r in S:
        per[r["session"]][r["category"][0]] += 1
    parts = []
    for s in sorted(per):
        an = re.search(r"pnb\d+", s).group(0)
        parts.append(an + " " + "/".join(str(per[s][k]) for k in "ABCD"))
    print("  per session (A/B/C/D): " + "; ".join(parts))


def main():
    sessions = sorted(p.parent.name for p in (HERE / "seedtrace").glob("*/seedtrace.mat"))
    rows, exact = [], {}
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("analyze_seedtrace.py -- " + __doc__.strip().splitlines()[0])
        print(f"sessions with a seedtrace: {len(sessions)}")
        for s in sessions:
            exact[s] = analyse_session(s, rows)
        print("\n=================== SUMMARY (all sessions pooled)")
        print(f"reproduction exact: {sum(exact.values())}/{len(exact)}")
        cons = [r for r in rows if (r["prod_cat"] == "none") != (not r["rerun_final_covered"])]
        print(f"somata whose re-run coverage differs from production category: {len(cons)} of {len(rows)}")
        gm = [r for r in rows if r["kind"] == "gcamp"]
        print(f"raw-gate check vs 2026-11-04 gate (production Cn/pnr files): agree {sum(r['gate_raw'] == r['prod_gate'] for r in gm)}/{len(gm)}")
        summarise(rows, "GCaMP somata with NO candidate, passing both gates (2026-11-04 gate)",
                  lambda r: r["kind"] == "gcamp" and r["prod_cat"] == "none" and r["prod_gate"])
        summarise(rows, "same, EXCLUDING pnb106 (zoom 4: the DoG detector picks sub-cellular texture there, see QC figure)",
                  lambda r: r["kind"] == "gcamp" and r["prod_cat"] == "none" and r["prod_gate"] and "pnb106" not in r["session"])
        summarise(rows, "GCaMP somata with NO candidate, failing a gate",
                  lambda r: r["kind"] == "gcamp" and r["prod_cat"] == "none" and not r["prod_gate"])
        summarise(rows, "red somata with NO candidate", lambda r: r["kind"] == "red" and r["prod_cat"] == "none")
        summarise(rows, "CONTROL: GCaMP somata WITH a candidate (review or auto-rejected)",
                  lambda r: r["kind"] == "gcamp" and r["prod_cat"] != "none")
    out = buf.getvalue()
    print(out)
    (HERE / "seedtrace_analysis.txt").write_text(out, encoding="utf-8")
    with open(HERE / "seedtrace_somata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
