"""
a07_unrecovered.py -- attack #7: the unrecovered curated neurons + the section-4 table.

Every curated neuron whose best similarity <= 0.45 (fixtures/geometry_*.npz
unrec table): best sim, nearest candidate by centroid and its distance/sim,
curated brightness (max weight) and area, Cn at the centroid, and whether its
Hungarian partner is in the ambiguous set.  Classes:
  detection_miss : no candidate within gSiz of the curated centroid
  merge_split    : a candidate within gSiz but similarity <= 0.45
  partner_taken  : the nearest candidate is matched to another curated neuron
The corpus table (results/corpus_table.json) is checked against the documented
numbers.  Contact sheets for all unrecovered neurons.  Read-only.
"""
import sys

import numpy as np

import rt_lib as rt

DOC = {"vCA1": {"curated": 5187, "matched": 5148, "ambiguous": 39, "duplicates": 6623, "below_040": 0},
       "BLA": {"curated": 2741, "matched": 2737, "ambiguous": 4, "duplicates": 4490, "below_040": 0}}


def main():
    t = rt.Timer()
    rt.assert_pinned()
    ct = rt.read_result("corpus_table")["table"]
    res = {"areas": {}, "table_check": {}}
    for area in ("vCA1", "BLA"):
        g = np.load(rt.FIXTURES / f"geometry_{area}.npz", allow_pickle=True)
        U, us, sessions = g["unrec"], g["unrec_session"], list(g["sessions"])
        c = {k: i for i, k in enumerate(g["unrec_cols"])}
        per = {s["rel"]: s for s in ct[area]["per_session"]}
        rows = []
        for i in range(len(U)):
            s = per[sessions[us[i]]]
            gsiz, gsig = s["gSiz"], s["gSig"]
            near_d, near_sim = U[i, c["nearest_dist"]], U[i, c["nearest_sim"]]
            best_sim, best_d = U[i, c["best_sim_matrix"]], U[i, c["best_dist"]]
            # is the nearest candidate matched elsewhere?  (matched candidates are the JSON positives)
            js = rt.load_json(rt.DATA_ROOT[area] / sessions[us[i]])
            matched_c = set(int(x) for x in js["candidate_indices"][:js["n_matched"]])
            nearest_matched_elsewhere = int(U[i, c["nearest_cand"]]) in matched_c
            if near_d > gsiz:
                klass = "detection_miss"
            elif nearest_matched_elsewhere:
                klass = "partner_taken"
            else:
                klass = "merge_split"
            rows.append({"session": sessions[us[i]], "cur": int(U[i, c["cur"]]),
                         "best_sim": float(best_sim), "best_dist_px": float(best_d),
                         "nearest_dist_px": float(near_d), "nearest_sim": float(near_sim),
                         "nearest_matched_elsewhere": nearest_matched_elsewhere,
                         "cur_max_weight": float(U[i, c["cur_max"]]), "cur_area20": int(U[i, c["cur_area20"]]),
                         "cn_at_centroid": float(U[i, c["cn_at_centroid"]]),
                         "partner_in_ambiguous": bool(U[i, c["partner_in_ambiguous"]]),
                         "gSig": gsig, "gSiz": gsiz, "class": klass})
        # brightness context: compare curated max weight of unrecovered vs recovered (pairs table cur_max)
        P = g["pairs"]
        pc = {k: i for i, k in enumerate(g["pair_cols"])}
        rec_max = P[:, pc["cur_max"]]
        rec_area = P[:, pc["cur_area20"]]
        unrec_max = np.array([r["cur_max_weight"] for r in rows]) if rows else np.zeros(0)
        unrec_area = np.array([r["cur_area20"] for r in rows]) if rows else np.zeros(0)
        counts = {k: sum(1 for r in rows if r["class"] == k) for k in ("detection_miss", "merge_split", "partner_taken")}
        res["areas"][area] = {
            "n_unrecovered": len(rows), "counts": counts,
            "n_partner_in_ambiguous": sum(1 for r in rows if r["partner_in_ambiguous"]),
            "unrec_max_weight": {"p50": float(np.median(unrec_max)) if len(unrec_max) else None,
                                 "recovered_p50": float(np.median(rec_max)),
                                 "frac_below_recovered_p10": float(np.mean(unrec_max < np.percentile(rec_max, 10))) if len(unrec_max) else None},
            "unrec_area20": {"p50": float(np.median(unrec_area)) if len(unrec_area) else None,
                             "recovered_p50": float(np.median(rec_area))},
            "rows": rows,
        }
        a = res["areas"][area]
        rt.log(f"{area}: {len(rows)} unrecovered -> {counts}; partner-in-ambiguous {a['n_partner_in_ambiguous']}; "
               f"max weight p50 {a['unrec_max_weight']['p50']} vs recovered {a['unrec_max_weight']['recovered_p50']:.3g}")
        tc = ct[area]
        res["table_check"][area] = {
            "computed": {"curated": tc["curated"], "matched": tc["matched"], "ambiguous": tc["ambiguous"],
                         "duplicates": tc["duplicates"], "below_040": tc["sessions_below_0.40"]},
            "documented": DOC[area],
            "match": all(tc[k1] == DOC[area][k2] for k1, k2 in (("curated", "curated"), ("matched", "matched"),
                                                                ("ambiguous", "ambiguous"), ("duplicates", "duplicates"),
                                                                ("sessions_below_0.40", "below_040"))),
        }
    # contact sheets for all unrecovered neurons
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rt.SHEETS.mkdir(exist_ok=True)
        for area in ("vCA1", "BLA"):
            rows = res["areas"][area]["rows"]
            if not rows:
                continue
            fig, axes = plt.subplots(len(rows), 3, figsize=(8, 2.4 * len(rows)))
            axes = np.atleast_2d(axes)
            cache = {}
            for r_i, r in enumerate(rows):
                relp = r["session"]
                if relp not in cache:
                    sd = rt.DATA_ROOT[area] / relp
                    cache[relp] = (rt.load_bootstrap_candidates(sd), rt.load_curated_stack(sd), rt.load_cn(sd / "Cn.mat"))
                bc, cur, Cn = cache[relp]
                d1, d2 = bc["d1"], bc["d2"]
                kimg = cur[r["cur"]]
                kc = rt.centroids(kimg[None])[0]
                y, x = int(round(kc[0])), int(round(kc[1]))
                sl = (slice(max(0, y - 24), y + 24), slice(max(0, x - 24), x + 24))
                axes[r_i, 0].imshow(kimg[sl], cmap="gray"); axes[r_i, 0].set_title(f"{relp.split('/')[-1][:28]} cur#{r['cur']} [{r['class']}]", fontsize=6)
                nj = int(np.load(rt.FIXTURES / f"geometry_{area}.npz")["unrec"][r_i, 5]) if False else None
                # nearest candidate image
                g2 = np.load(rt.FIXTURES / f"geometry_{area}.npz", allow_pickle=True)
                un = g2["unrec"]; c2 = {k: i for i, k in enumerate(g2["unrec_cols"])}
                nj = int(un[r_i, c2["nearest_cand"]])
                if nj >= 0:
                    nimg = np.asarray(bc["A_rows_C"][nj].todense()).reshape(d1, d2)
                    axes[r_i, 1].imshow(nimg[sl], cmap="gray")
                axes[r_i, 1].set_title(f"nearest cand #{nj} d {r['nearest_dist_px']:.1f} sim {r['nearest_sim']:.2f}", fontsize=6)
                if Cn is not None and Cn.shape == (d1, d2):
                    axes[r_i, 2].imshow(Cn[sl], cmap="viridis")
                axes[r_i, 2].set_title(f"Cn; max w {r['cur_max_weight']:.3g}", fontsize=6)
                for ax in axes[r_i]:
                    ax.set_xticks([]); ax.set_yticks([])
            out = rt.SHEETS / f"a07_{area}_unrecovered.png"
            fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
            res["areas"][area]["contact_sheet"] = str(out.relative_to(rt.SP))
    except Exception as e:
        res["contact_sheet_error"] = repr(e)
    checks = {"table_matches_docs": all(v["match"] for v in res["table_check"].values()),
              "every_unrecovered_classified": all(r["class"] in ("detection_miss", "merge_split", "partner_taken")
                                                  for a in res["areas"].values() for r in a["rows"]),
              "total_unrecovered_43": sum(a["n_unrecovered"] for a in res["areas"].values()) == 43}
    res["checks"] = checks
    res["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    res["criterion"] = "section-4 table reproduces from the 202 JSONs; every unrecovered neuron classified (43 expected)"
    rt.log(f"checks {checks}\nVERDICT {res['verdict']}")
    rt.write_result("a07", {k: v for k, v in res.items()}, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
