"""
prep_bootstrap_oos_scores.py -> fixtures/boot_oos_{area}.npz

Out-of-sample scores for every bootstrap row: leave-one-bootstrap-session-out,
training on ALL agent sessions (deployed weight recipe: vCA1 fixed 5.0; BLA
max(sqrt(n_bs/n_ag), 4.0) over the full pool = the trainer's value) plus every
other bootstrap session at its deployed row weight, StandardScaler on the
training block, XGB seed 42.  BLA is scored under both the 35-column contract
(the deployed model's view) and the first 13 columns (the companion first-pass
view, which is the hiconf source attack #14 needs); vCA1 under 13 columns.
Used by #13, #14, #16.  Read-only; ~1 fit per bootstrap session per width.
"""
import sys

import numpy as np

import rt_lib as rt


def lso(area, width):
    recs = rt.load_pool(area, width=rt.EXPECTED_WIDTH[area])
    ag = [r for r in recs if not r["is_bootstrap"]]
    bs = [r for r in recs if r["is_bootstrap"]]
    cols = slice(0, width)
    X_ag = np.vstack([r["X"] for r in ag])[:, cols]
    y_ag = np.concatenate([r["y"] for r in ag])
    n_bs_rows = sum(len(r["y"]) for r in bs)
    agw = rt.fold_agent_weight(n_bs_rows, len(y_ag), rt.AGENT_WEIGHT_OVERRIDE.get(area))
    rt.log(f"{area} width {width}: {len(ag)} agent sessions / {len(y_ag)} rows at weight {agw:.2f}; "
           f"{len(bs)} bootstrap sessions / {n_bs_rows} rows")
    scores = np.full(n_bs_rows, np.nan)
    sess = np.empty(n_bs_rows, dtype=object)
    row_in = np.zeros(n_bs_rows, int)
    y_all = np.zeros(n_bs_rows, int)
    w_all = np.zeros(n_bs_rows)
    off = 0
    offsets = {}
    for r in bs:
        offsets[r["name"]] = off
        off += len(r["y"])
    for i, held in enumerate(bs):
        others = [r for r in bs if r is not held]
        X_tr = np.vstack([X_ag] + [r["X"][:, cols] for r in others])
        y_tr = np.concatenate([y_ag] + [r["y"] for r in others])
        w_tr = np.concatenate([np.full(len(y_ag), agw)] + [r["w"] for r in others])
        s = rt.fit_predict(X_tr, y_tr, w_tr, held["X"][:, cols])
        o = offsets[held["name"]]
        n = len(held["y"])
        scores[o:o + n] = s
        sess[o:o + n] = held["name"]
        row_in[o:o + n] = np.arange(n)
        y_all[o:o + n] = held["y"]
        w_all[o:o + n] = held["w"]
        if (i + 1) % 10 == 0:
            rt.log(f"   {i + 1}/{len(bs)}")
    return {"scores": scores, "session": sess.astype(str), "row": row_in, "y": y_all, "w": w_all,
            "agent_weight": agw}


def main():
    t = rt.Timer()
    rt.assert_pinned()
    rt.FIXTURES.mkdir(exist_ok=True)
    meta = {}
    for area, widths in (("vCA1", (13,)), ("BLA", (35, 13))):
        out = {}
        for w in widths:
            r = lso(area, w)
            out[f"score_{w}"] = r["scores"]
            out.update({"session": r["session"], "row": r["row"], "y": r["y"], "w": r["w"]})
            meta[f"{area}_{w}"] = {"agent_weight": r["agent_weight"], "n_rows": int(len(r["y"])),
                                   "n_pos": int(r["y"].sum()), "n_masked": int((r["w"] == 0).sum()),
                                   "auc_all_rows": rt.auc(r["y"], r["scores"]),
                                   "auc_unmasked": rt.auc(r["y"][r["w"] > 0], r["scores"][r["w"] > 0])}
            rt.log(f"   {area} w{w}: AUC over bootstrap rows (all) {meta[f'{area}_{w}']['auc_all_rows']:.4f}, "
                   f"unmasked {meta[f'{area}_{w}']['auc_unmasked']:.4f}")
        np.savez(rt.FIXTURES / f"boot_oos_{area}.npz", **out)
    rt.write_result("boot_oos_meta", meta, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
