"""
prep_corpus_geometry.py -> fixtures/geometry_{area}.npz + results/corpus_table.json

For every bootstrap session (202): per matched pair, per duplicate candidate
and per unrecovered curated neuron, the geometry the label-correctness attacks
(#5, #6, #7) need -- computed from bootstrap_candidates.npz (candidate rows,
C-order pixels), the curated spatial_footprints.mat stack and Cn.mat.  Also
the corpus recovery table (docs/BOOTSTRAP_MATCHING_BUG_2026-08.md section 4
has no backing artifact; this is it).  Read-only.

Per pair: sim, centroid distance (px), MIRROR distance (candidate centroid vs
the transposed curated centroid -- a residual-transpose detector), IoU at 20%
and 50% of max, area ratio.  Per duplicate: best curated neuron, its matched
candidate, IoU/offset/area ratio vs that candidate, trace correlation with it,
IoU vs the curated neuron.  Per unrecovered neuron: best sim, best candidate,
distance, curated max weight / area, Cn at centroid, partner status.
"""
import sys
from collections import defaultdict

import numpy as np

import rt_lib as rt


def row_geom(csr, j, d1, d2):
    """Centroid (y, x), 20% mask pixel set, 50% mask pixel set, area20 for one
    CSR candidate row (C-order pixel indices)."""
    s, e = csr.indptr[j], csr.indptr[j + 1]
    idx, val = csr.indices[s:e], csr.data[s:e].astype(float)
    if len(idx) == 0 or val.max() <= 0:
        return (np.nan, np.nan), set(), set(), 0
    ys, xs = np.divmod(idx, d2)
    w = val.sum()
    cent = (float((ys * val).sum() / w), float((xs * val).sum() / w))
    m20 = set(idx[val > 0.2 * val.max()].tolist())
    m50 = set(idx[val >= 0.5 * val.max()].tolist())
    return cent, m20, m50, len(m20)


def img_geom(img):
    """Same for a dense (H, W) curated image; pixel sets in C-order indices."""
    if img.max() <= 0:
        return (np.nan, np.nan), set(), set(), 0, 0.0
    d1, d2 = img.shape
    ys, xs = np.mgrid[0:d1, 0:d2]
    w = img.sum()
    cent = (float((ys * img).sum() / w), float((xs * img).sum() / w))
    m20 = set(np.flatnonzero(img > 0.2 * img.max()).tolist())
    m50 = set(np.flatnonzero(img >= 0.5 * img.max()).tolist())
    return cent, m20, m50, len(m20), float(img.max())


def iou_sets(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def dist(c1, c2):
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def process(area, sd, js):
    bc = rt.load_bootstrap_candidates(sd)
    csr, d1, d2 = bc["A_rows_C"], bc["d1"], bc["d2"]
    C_raw = bc["C_raw"]
    stack = rt.load_curated_stack(sd)
    assert stack.shape == (js["n_curated"], d1, d2), (rt.rel(sd), stack.shape)
    Cn = rt.load_cn(sd / "Cn.mat")
    cn_ok = Cn is not None and Cn.shape == (d1, d2)
    gsig = float(js.get("cnmfe_params", {}).get("gSig", np.nan))
    gsiz = float(js.get("cnmfe_params", {}).get("gSiz", np.nan))
    n_c, n_k = js["n_candidates"], js["n_curated"]
    n_m = js["n_matched"]
    cand_idx, cur_idx, sims = js["candidate_indices"], js["curated_indices"], js["pair_similarities"]
    matched = {int(c): (int(r), float(s)) for r, c, s in zip(cand_idx[:n_m], cur_idx[:n_m], sims[:n_m])}
    partner = {int(c): (int(r), float(s)) for r, c, s in zip(cand_idx[n_m:], cur_idx[n_m:], sims[n_m:])}
    amb = set(int(i) for i in js.get("ambiguous_candidate_indices", []))
    dups = [int(i) for i in js.get("duplicate_candidate_indices", [])]
    sim = bc["sim_matrix"]

    cand_geom = {}

    def cg(j):
        if j not in cand_geom:
            cand_geom[j] = row_geom(csr, j, d1, d2)
        return cand_geom[j]

    cur_geom = {}

    def kg(k):
        if k not in cur_geom:
            cur_geom[k] = img_geom(stack[k])
        return cur_geom[k]

    pairs = []
    for k, (j, s) in matched.items():
        cc, cm20, cm50, ca = cg(j)
        kc, km20, km50, ka, kmax = kg(k)
        pairs.append((j, k, s, dist(cc, kc), dist(cc, (kc[1], kc[0])),
                      iou_sets(cm20, km20), iou_sets(cm50, km50),
                      ca / ka if ka else np.nan, ca, ka, kmax))

    dup_rows = []
    for j in dups:
        k = int(np.argmax(sim[j]))
        s_best = float(sim[j, k])
        cc, cm20, cm50, ca = cg(j)
        kc, km20, km50, ka, kmax = kg(k)
        if k in matched:
            jm = matched[k][0]
            mc, mm20, mm50, ma = cg(jm)
            iou20_m, iou50_m = iou_sets(cm20, mm20), iou_sets(cm50, mm50)
            off_m = dist(cc, mc)
            ar_m = ca / ma if ma else np.nan
            a, b = C_raw[j], C_raw[jm]
            tr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        else:
            jm, iou20_m, iou50_m, off_m, ar_m, tr = -1, np.nan, np.nan, np.nan, np.nan, np.nan
        dup_rows.append((j, k, s_best, jm, iou20_m, iou50_m, off_m, ar_m, tr,
                         iou_sets(cm20, km20), iou_sets(cm50, km50), dist(cc, kc), ca))

    unrec = []
    best = np.asarray(js["per_curated_best_similarity"], float)
    cand_cents = None
    for k in range(n_k):
        if k in matched:
            continue
        jb = int(np.argmax(sim[:, k])) if n_c else -1
        s_b = float(sim[jb, k]) if n_c else np.nan
        kc, km20, km50, ka, kmax = kg(k)
        cc = cg(jb)[0] if jb >= 0 else (np.nan, np.nan)
        pj, ps = partner.get(k, (-1, np.nan))
        cn_at = float(Cn[int(round(kc[0])), int(round(kc[1]))]) if cn_ok and np.isfinite(kc[0]) else np.nan
        if cand_cents is None:
            cand_cents = np.array([cg(j)[0] for j in range(n_c)]) if n_c else np.zeros((0, 2))
        if n_c:
            dd = np.hypot(cand_cents[:, 0] - kc[0], cand_cents[:, 1] - kc[1])
            near_j = int(np.nanargmin(dd))
            near_d = float(dd[near_j])
        else:
            near_j, near_d = -1, np.nan
        unrec.append((k, float(best[k]), s_b, jb, dist(cc, kc) if jb >= 0 else np.nan,
                      near_j, near_d, float(sim[near_j, k]) if near_j >= 0 else np.nan,
                      ka, kmax, cn_at, pj, ps, int(pj in amb)))

    summary = {"rel": rt.rel(sd), "n_candidates": n_c, "n_curated": n_k, "n_matched": n_m,
               "recovery": n_m / n_k if n_k else np.nan, "n_ambiguous": len(amb), "n_duplicates": len(dups),
               "gSig": gsig, "gSiz": gsiz, "d1": d1, "d2": d2, "cn_same_res": bool(cn_ok),
               "recovery_by_threshold": js.get("recovery_by_threshold", {}),
               "median_pair_sim": float(np.median(sims[:n_m])) if n_m else np.nan,
               "best_pair_sim": float(max(sims)) if sims else np.nan}
    return pairs, dup_rows, unrec, summary


PAIR_COLS = ["cand", "cur", "sim", "dist", "mirror_dist", "iou20", "iou50",
             "area_ratio", "cand_area20", "cur_area20", "cur_max"]
DUP_COLS = ["cand", "best_cur", "best_sim", "matched_cand", "iou20_vs_matched",
            "iou50_vs_matched", "offset_vs_matched", "area_ratio_vs_matched",
            "trace_corr_vs_matched", "iou20_vs_cur", "iou50_vs_cur", "dist_vs_cur", "cand_area20"]
UNREC_COLS = ["cur", "best_sim_json", "best_sim_matrix", "best_cand", "best_dist",
              "nearest_cand", "nearest_dist", "nearest_sim", "cur_area20", "cur_max",
              "cn_at_centroid", "partner", "partner_sim", "partner_in_ambiguous"]


def main():
    t = rt.Timer()
    rt.assert_pinned()
    rt.FIXTURES.mkdir(exist_ok=True)
    table = {}
    for area in ("vCA1", "BLA"):
        P, D, U, S = [], [], [], []
        sess_of = {"pairs": [], "dups": [], "unrec": []}
        for sd in rt.labeled_sessions(area):
            if not rt.is_bootstrap(sd):
                continue
            js = rt.load_json(sd)
            pairs, dups, unrec, summ = process(area, sd, js)
            i = len(S)
            S.append(summ)
            P += pairs
            D += dups
            U += unrec
            sess_of["pairs"] += [i] * len(pairs)
            sess_of["dups"] += [i] * len(dups)
            sess_of["unrec"] += [i] * len(unrec)
            rt.log(f"  {area} {summ['rel']}: cand {summ['n_candidates']} cur {summ['n_curated']} "
                   f"matched {summ['n_matched']} dup {summ['n_duplicates']} amb {summ['n_ambiguous']} "
                   f"unrec {len(unrec)}")
        np.savez(rt.FIXTURES / f"geometry_{area}.npz",
                 pairs=np.array(P, dtype=float), pairs_session=np.array(sess_of["pairs"]),
                 dups=np.array(D, dtype=float) if D else np.zeros((0, len(DUP_COLS))),
                 dups_session=np.array(sess_of["dups"]),
                 unrec=np.array(U, dtype=float) if U else np.zeros((0, len(UNREC_COLS))),
                 unrec_session=np.array(sess_of["unrec"]),
                 sessions=np.array([s["rel"] for s in S]),
                 pair_cols=np.array(PAIR_COLS), dup_cols=np.array(DUP_COLS), unrec_cols=np.array(UNREC_COLS))
        n_cur = sum(s["n_curated"] for s in S)
        n_mat = sum(s["n_matched"] for s in S)
        rbt = defaultdict(int)
        for s in S:
            for k, v in s["recovery_by_threshold"].items():
                rbt[k] += int(v)
        table[area] = {
            "sessions": len(S), "curated": n_cur, "matched": n_mat, "recovery": n_mat / n_cur,
            "sessions_below_0.40": sum(1 for s in S if s["recovery"] < 0.40),
            "ambiguous": sum(s["n_ambiguous"] for s in S), "duplicates": sum(s["n_duplicates"] for s in S),
            "masked": sum(s["n_ambiguous"] + s["n_duplicates"] for s in S),
            "unrecovered": n_cur - n_mat, "candidates": sum(s["n_candidates"] for s in S),
            "recovery_by_threshold": {k: rbt[k] for k in sorted(rbt)},
            "median_session_recovery": float(np.median([s["recovery"] for s in S])),
            "median_best_pair_sim": float(np.median([s["best_pair_sim"] for s in S])),
            "sessions_best_pair_ge_0.99": sum(1 for s in S if s["best_pair_sim"] >= 0.99),
            "cn_same_res": sum(1 for s in S if s["cn_same_res"]),
            "n_pairs": len(P), "n_dup_rows": len(D), "n_unrec_rows": len(U),
            "per_session": S,
        }
        rt.log(f"{area}: {len(S)} sessions, matched {n_mat}/{n_cur} = {n_mat/n_cur:.4f}, "
               f"ambiguous {table[area]['ambiguous']} + duplicates {table[area]['duplicates']}, "
               f"sessions<0.40 {table[area]['sessions_below_0.40']}, pairs {len(P)}, dups {len(D)}, unrec {len(U)}")
    rt.write_result("corpus_table", {"table": table}, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
