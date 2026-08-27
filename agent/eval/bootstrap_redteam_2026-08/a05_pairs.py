"""
a05_pairs.py -- attack #5: are the matched pairs the right cells?

Automated over every matched pair in the corpus (fixtures/geometry_*.npz):
centroid distance, MIRROR distance (candidate centroid vs the transposed
curated centroid: a residual-transpose detector), IoU at 20%/50% of max,
area ratio, similarity.  Flags: distance > gSiz, or IoU20 < 0.2, or
mirror-closer-than-true.  Contact sheets (curated | candidate | overlay |
Cn crop) for 30 sessions x 5 pairs go to contact_sheets/ for the user's
visual check -- that part is marked pending in the verdict.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

N_SHEET_SESSIONS = 15          # per area: 10 random + 5 lowest-median-similarity
N_SHEET_PAIRS = 5              # per session: 3 random + 2 lowest-similarity
RNG = np.random.default_rng(20260826)


def crop(img, c, half=24):
    y, x = int(round(c[0])), int(round(c[1]))
    d1, d2 = img.shape
    y0, y1 = max(0, y - half), min(d1, y + half)
    x0, x1 = max(0, x - half), min(d2, x + half)
    return img[y0:y1, x0:x1], (y0, x0)


def contact_sheets(area, sessions, pairs, pairs_session, geom_sessions, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rt.SHEETS.mkdir(exist_ok=True)
    per = table["per_session"]
    med = {s["rel"]: s["median_pair_sim"] for s in per}
    names = list(geom_sessions)
    lowest = sorted(names, key=lambda n: med.get(n, 1.0))[:5]
    others = [n for n in names if n not in lowest]
    chosen = lowest + list(RNG.choice(others, size=min(10, len(others)), replace=False))
    made = []
    for relp in chosen:
        i = names.index(relp)
        rows = np.flatnonzero(pairs_session == i)
        if len(rows) == 0:
            continue
        sims = pairs[rows, 2]
        low2 = rows[np.argsort(sims)[:2]]
        rest = [r for r in rows if r not in set(low2)]
        pick = list(low2) + list(RNG.choice(rest, size=min(3, len(rest)), replace=False))
        sd = rt.DATA_ROOT[area] / relp
        bc = rt.load_bootstrap_candidates(sd)
        cur = rt.load_curated_stack(sd)
        Cn = rt.load_cn(sd / "Cn.mat")
        d1, d2 = bc["d1"], bc["d2"]
        fig, axes = plt.subplots(len(pick), 4, figsize=(10, 2.5 * len(pick)))
        axes = np.atleast_2d(axes)
        for r_i, prow in enumerate(pick):
            j, k = int(pairs[prow, 0]), int(pairs[prow, 1])
            cimg = np.asarray(bc["A_rows_C"][j].todense()).reshape(d1, d2)
            kimg = cur[k]
            kc = rt.centroids(kimg[None])[0]
            cc = rt.centroids(cimg[None])[0]
            a, (y0, x0) = crop(kimg, kc)
            b, _ = crop(cimg, kc)
            axes[r_i, 0].imshow(a, cmap="gray"); axes[r_i, 0].set_title(f"curated #{k}", fontsize=8)
            axes[r_i, 1].imshow(b, cmap="gray"); axes[r_i, 1].set_title(f"candidate #{j}", fontsize=8)
            ov = np.zeros(a.shape + (3,))
            ov[..., 0] = a / (a.max() or 1); ov[..., 1] = b / (b.max() or 1)
            axes[r_i, 2].imshow(ov); axes[r_i, 2].set_title(
                f"sim {pairs[prow, 2]:.3f} d {pairs[prow, 3]:.1f}px IoU20 {pairs[prow, 5]:.2f}", fontsize=8)
            if Cn is not None and Cn.shape == (d1, d2):
                c, _ = crop(Cn, kc)
                axes[r_i, 3].imshow(c, cmap="viridis")
            axes[r_i, 3].set_title(f"Cn (cand centroid {cc[0]:.0f},{cc[1]:.0f})", fontsize=8)
            for ax in axes[r_i]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{area} {relp} -- red=curated green=candidate", fontsize=9)
        out = rt.SHEETS / f"a05_{area}_{rt.key(relp)}.png"
        fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
        made.append(str(out.relative_to(rt.SP)))
    return made


def main():
    t = rt.Timer()
    rt.assert_pinned()
    ct = rt.read_result("corpus_table")["table"]
    res = {"areas": {}, "contact_sheets": {}}
    for area in ("vCA1", "BLA"):
        g = np.load(rt.FIXTURES / f"geometry_{area}.npz", allow_pickle=True)
        P, ps, sessions = g["pairs"], g["pairs_session"], list(g["sessions"])
        cols = {c: i for i, c in enumerate(g["pair_cols"])}
        per = {s["rel"]: s for s in ct[area]["per_session"]}
        gsiz = np.array([per[sessions[i]]["gSiz"] for i in ps])
        gsig = np.array([per[sessions[i]]["gSig"] for i in ps])
        d, md = P[:, cols["dist"]], P[:, cols["mirror_dist"]]
        iou20, iou50, ar, sim = P[:, cols["iou20"]], P[:, cols["iou50"]], P[:, cols["area_ratio"]], P[:, cols["sim"]]
        flag_dist = d > gsiz
        flag_iou = iou20 < 0.2
        mirror_closer = (md < d) & (d > gsig) & (md < 0.5 * d)   # far from true, near the mirror
        diag_ambiguous = (md < d) & ~mirror_closer                 # cells on the diagonal: mirror == true
        flagged = flag_dist | flag_iou | mirror_closer
        by_sess = {}
        for i in np.flatnonzero(flagged):
            by_sess.setdefault(sessions[ps[i]], []).append(
                {"cand": int(P[i, 0]), "cur": int(P[i, 1]), "sim": float(sim[i]), "dist": float(d[i]),
                 "mirror_dist": float(md[i]), "iou20": float(iou20[i]), "gSiz": float(gsiz[i])})
        res["areas"][area] = {
            "n_pairs": int(len(P)), "n_sessions": len(sessions),
            "dist_px": {"p50": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())},
            "dist_over_gsig": {"p50": float(np.median(d / gsig)), "p95": float(np.percentile(d / gsig, 95))},
            "mirror_dist_px": {"p05": float(np.percentile(md, 5)), "p50": float(np.median(md))},
            "iou20": {"p05": float(np.percentile(iou20, 5)), "p50": float(np.median(iou20))},
            "iou50": {"p05": float(np.percentile(iou50, 5)), "p50": float(np.median(iou50))},
            "area_ratio": {"p05": float(np.nanpercentile(ar, 5)), "p50": float(np.nanmedian(ar)), "p95": float(np.nanpercentile(ar, 95))},
            "sim": {"p05": float(np.percentile(sim, 5)), "p50": float(np.median(sim))},
            "n_flag_dist_gt_gsiz": int(flag_dist.sum()), "n_flag_iou20_lt_0.2": int(flag_iou.sum()),
            "n_mirror_closer": int(mirror_closer.sum()), "n_diagonal_ambiguous": int(diag_ambiguous.sum()),
            "n_flagged": int(flagged.sum()),
            "flagged_pct": float(100 * flagged.mean()),
            "sessions_with_mirror_closer": sorted({sessions[ps[i]] for i in np.flatnonzero(mirror_closer)}),
            "flagged_by_session": by_sess,
        }
        a = res["areas"][area]
        rt.log(f"{area}: {a['n_pairs']} pairs; dist p50 {a['dist_px']['p50']:.2f} p95 {a['dist_px']['p95']:.2f} px; "
               f"mirror p05 {a['mirror_dist_px']['p05']:.1f}; IoU20 p05 {a['iou20']['p05']:.2f}; "
               f"flagged {a['n_flagged']} ({a['flagged_pct']:.2f}%): dist>gSiz {a['n_flag_dist_gt_gsiz']}, "
               f"IoU20<0.2 {a['n_flag_iou20_lt_0.2']}, mirror-closer {a['n_mirror_closer']}")
        res["contact_sheets"][area] = contact_sheets(area, sessions, P, ps, sessions, ct[area])
    checks = {f"{a}_flagged_lt_1pct": res["areas"][a]["flagged_pct"] < 1.0 for a in res["areas"]}
    checks.update({f"{a}_zero_mirror_closer": res["areas"][a]["n_mirror_closer"] == 0 for a in res["areas"]})
    res["checks"] = checks
    res["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    res["human_check"] = "PENDING (user): contact_sheets/a05_*.png, 15 sessions x 5 pairs per area"
    res["criterion"] = ("flagged pairs (dist > gSiz or IoU20 < 0.2 or residual-transpose: mirror closer AND true dist > gSig "
                        "AND mirror < half the true dist) < 1% per area; 0 residual-transpose pairs. Cells on the image "
                        "diagonal (mirror == true within noise) are counted separately, not flagged")
    rt.log(f"checks {checks}\nVERDICT {res['verdict']} (human spot-check pending)")
    rt.write_result("a05", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
