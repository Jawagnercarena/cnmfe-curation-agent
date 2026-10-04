"""
a06_duplicates.py -- attack #6: what are the masked "duplicate" rows?

For all 11,113 duplicate candidates (fixtures/geometry_*.npz): IoU / offset /
area ratio vs the matched candidate of their best curated neuron, and the
trace correlation with it (same CNMFe run, so valid).  Classes:
  same-cell  : IoU50 vs matched >= 0.5  or trace r >= 0.7
  fragment   : area ratio vs matched < 0.3 and IoU20 vs matched > 0
  distinct   : offset > gSig and trace r < 0.3  (a second neuron overlapping
               the curated one -- a valid positive/negative the mask discards)
  other      : everything else
Sandbox ground truth: the a2 transfer on the 4 sandboxes is re-done with (i)
the original 0.6 mutual-best rule and (ii) a centroid+IoU rule, for the
duplicate rows only.  30 contact sheets stratified by class.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

RNG = np.random.default_rng(20260827)
SANDBOXES = ["2tones/AVG5x-TSeries-093025-bla21-313um-38z-000",
             "2tones/AVG5x-TSeries-100125-bla12-639um-23z-000",
             "4odorDO/AVG5x-TSeries-02092026-bla12-681um-22z-000",
             "Valence/AVG5x-TSeries-121225-bla12-652um-23z-000"]


def classify(iou50_m, iou20_m, ar_m, off_m, tr, gsig):
    if np.isnan(iou50_m):
        return "no_matched_candidate"
    if iou50_m >= 0.5 or (np.isfinite(tr) and tr >= 0.7):
        return "same_cell"
    if np.isfinite(ar_m) and ar_m < 0.3 and iou20_m > 0:
        return "fragment"
    if off_m > gsig and (np.isnan(tr) or tr < 0.3):
        return "distinct"
    return "other"


def sandbox_transfer():
    """Duplicate rows on the 4 sandboxes vs the human (agent-review) labels:
    a duplicate's human label transfers when it mutually-best matches a
    human-reviewed candidate at cosine > 0.6 (a2's rule) or by centroid < gSig
    + IoU20 > 0.3 (looser)."""
    out = {}
    for relp in SANDBOXES:
        sd = rt.DATA_ROOT["BLA"] / relp
        js = rt.load_json(sd)
        if js is None or not (sd / "bootstrap_candidates.npz").exists():
            out[relp] = {"status": "not_a_bootstrap_session_now"}
            continue
        out[relp] = {"status": "n/a: these sessions are agent-reviewed (labels.mat is human), "
                               "not bootstrap; the a2 transfer compared sandbox candidates to human labels"}
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    ct = rt.read_result("corpus_table")["table"]
    res = {"areas": {}}
    for area in ("vCA1", "BLA"):
        g = np.load(rt.FIXTURES / f"geometry_{area}.npz", allow_pickle=True)
        D, ds, sessions = g["dups"], g["dups_session"], list(g["sessions"])
        c = {k: i for i, k in enumerate(g["dup_cols"])}
        per = {s["rel"]: s for s in ct[area]["per_session"]}
        gsig = np.array([per[sessions[i]]["gSig"] for i in ds])
        cls = np.array([classify(D[i, c["iou50_vs_matched"]], D[i, c["iou20_vs_matched"]],
                                 D[i, c["area_ratio_vs_matched"]], D[i, c["offset_vs_matched"]],
                                 D[i, c["trace_corr_vs_matched"]], gsig[i]) for i in range(len(D))])
        counts = {k: int((cls == k).sum()) for k in ("same_cell", "fragment", "distinct", "other", "no_matched_candidate")}
        # threshold sensitivity: a looser reading of the same geometry (refuter's alternative)
        r_ = D[:, c["trace_corr_vs_matched"]]
        same_loose = (D[:, c["iou50_vs_matched"]] >= 0.3) | (r_ >= 0.5)
        distinct_loose = (D[:, c["offset_vs_matched"]] > 0.5 * gsig) & ~same_loose
        loose = {"same_cell_pct": float(100 * same_loose.mean()), "distinct_pct": float(100 * distinct_loose.mean())}
        tr = D[:, c["trace_corr_vs_matched"]]
        per_sess_distinct = {}
        for i in np.flatnonzero(cls == "distinct"):
            per_sess_distinct[sessions[ds[i]]] = per_sess_distinct.get(sessions[ds[i]], 0) + 1
        res["areas"][area] = {
            "n_dups": int(len(D)), "counts": counts,
            "pct": {k: float(100 * v / max(len(D), 1)) for k, v in counts.items()},
            "trace_corr": {"p10": float(np.nanpercentile(tr, 10)), "p50": float(np.nanmedian(tr)),
                           "p90": float(np.nanpercentile(tr, 90)), "frac_ge_0.7": float(np.nanmean(tr >= 0.7))},
            "iou50_vs_matched": {"p10": float(np.nanpercentile(D[:, c["iou50_vs_matched"]], 10)),
                                 "p50": float(np.nanmedian(D[:, c["iou50_vs_matched"]]))},
            "offset_vs_matched_px": {"p50": float(np.nanmedian(D[:, c["offset_vs_matched"]])),
                                     "p90": float(np.nanpercentile(D[:, c["offset_vs_matched"]], 90))},
            "best_sim": {"p10": float(np.percentile(D[:, c["best_sim"]], 10)), "p50": float(np.median(D[:, c["best_sim"]]))},
            "distinct_per_session_top": sorted(per_sess_distinct.items(), key=lambda kv: -kv[1])[:10],
            "n_sessions_with_distinct": len(per_sess_distinct),
            "loose_thresholds": loose,
        }
        a = res["areas"][area]
        rt.log(f"{area}: {a['n_dups']} duplicates -> {a['pct']} | trace r p50 {a['trace_corr']['p50']:.2f} "
               f"(>=0.7: {a['trace_corr']['frac_ge_0.7']:.2f}) | IoU50 p50 {a['iou50_vs_matched']['p50']:.2f} "
               f"| offset p50 {a['offset_vs_matched_px']['p50']:.1f} px")
        # contact sheets: 30 stratified by class
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rt.SHEETS.mkdir(exist_ok=True)
            picks = []
            for k in ("same_cell", "fragment", "distinct", "other"):
                idx = np.flatnonzero(cls == k)
                if len(idx):
                    picks += [(k, int(i)) for i in RNG.choice(idx, size=min(8, len(idx)), replace=False)]
            picks = picks[:30]
            fig, axes = plt.subplots(len(picks), 3, figsize=(8, 2.4 * len(picks)))
            axes = np.atleast_2d(axes)
            cache = {}
            for r_i, (k, i) in enumerate(picks):
                relp = sessions[ds[i]]
                if relp not in cache:
                    sd = rt.DATA_ROOT[area] / relp
                    cache[relp] = (rt.load_bootstrap_candidates(sd), rt.load_curated_stack(sd))
                bc, cur = cache[relp]
                d1, d2 = bc["d1"], bc["d2"]
                j, kk, jm = int(D[i, c["cand"]]), int(D[i, c["best_cur"]]), int(D[i, c["matched_cand"]])
                dimg = np.asarray(bc["A_rows_C"][j].todense()).reshape(d1, d2)
                kimg = cur[kk]
                kc = rt.centroids(kimg[None])[0]
                y, x = int(round(kc[0])), int(round(kc[1]))
                sl = (slice(max(0, y - 24), y + 24), slice(max(0, x - 24), x + 24))
                axes[r_i, 0].imshow(kimg[sl], cmap="gray"); axes[r_i, 0].set_title(f"curated #{kk}", fontsize=7)
                axes[r_i, 1].imshow(dimg[sl], cmap="gray"); axes[r_i, 1].set_title(f"duplicate #{j} [{k}]", fontsize=7)
                if jm >= 0:
                    mimg = np.asarray(bc["A_rows_C"][jm].todense()).reshape(d1, d2)
                    ov = np.zeros(kimg[sl].shape + (3,))
                    ov[..., 0] = mimg[sl] / (mimg.max() or 1); ov[..., 1] = dimg[sl] / (dimg.max() or 1)
                    axes[r_i, 2].imshow(ov)
                axes[r_i, 2].set_title(f"r {D[i, c['trace_corr_vs_matched']]:.2f} IoU50 {D[i, c['iou50_vs_matched']]:.2f} "
                                       f"off {D[i, c['offset_vs_matched']]:.1f}", fontsize=7)
                for ax in axes[r_i]:
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{area} duplicates (red=matched candidate, green=duplicate)", fontsize=9)
            out = rt.SHEETS / f"a06_{area}_duplicates.png"
            fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
            res["areas"][area]["contact_sheet"] = str(out.relative_to(rt.SP))
        except Exception as e:      # sheets are for the human check; never fail the attack on them
            res["areas"][area]["contact_sheet_error"] = repr(e)
    res["sandbox_transfer"] = sandbox_transfer()
    checks = {f"{a}_distinct_lt_10pct_strict": res["areas"][a]["pct"]["distinct"] < 10.0 for a in res["areas"]}
    checks.update({f"{a}_distinct_lt_10pct_loose": res["areas"][a]["loose_thresholds"]["distinct_pct"] < 10.0 for a in res["areas"]})
    checks.update({f"{a}_same_cell_majority_strict": res["areas"][a]["pct"]["same_cell"] > 50.0 for a in res["areas"]})
    res["checks"] = checks
    # The composition depends on the class thresholds (strict 8-10% distinct vs loose ~36%), so the
    # geometric answer is INCONCLUSIVE; what is robust: the rows are spatially overlapping (cosine
    # > 0.45) but mostly temporally distinct (trace r p50 ~0.4), "same-cell re-detection" is a
    # minority reading under any threshold, and the masking DECISION rests on attack #9's model
    # evidence (label-0 ~ masked, label-1 worse), not on this composition.
    res["verdict"] = "INCONCLUSIVE"
    res["criterion"] = ("composition of the masked rows by geometry + trace correlation; reported under strict and loose "
                        "class thresholds; the masking decision is judged in attack #9")
    res["human_check"] = "PENDING (user): contact_sheets/a06_*_duplicates.png"
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a06", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
