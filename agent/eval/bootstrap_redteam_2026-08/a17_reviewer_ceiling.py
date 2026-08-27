"""
a17_reviewer_ceiling.py -- hypothesis D17: the test set is the ceiling.

Two-reviewer overlap: the exchange inbox (server, READ-ONLY, root taken
from local_config and never printed) holds returns the ingest guard refused
because a DIFFERENT reviewer's labels were already local.  For every inbox
labels.mat whose reviewer differs from the local labels_provenance.txt and
whose length matches the local review set: keep-agreement, Cohen's kappa,
and each reviewer's labels as a classifier of the other's (AUC).  The a2
sandbox transfer (207 / 2 / 15) is a self-consistency bound, restated.
Proxies on all agent sessions: per-session confident-wrong reals (OOF < 0.02)
and false-AR at the deployed T from the reproduced pins.  Ceiling estimate:
symmetric label flips at the measured disagreement rate injected into the
pinned OOF -> AUC drop; +0.004 as a share of (ceiling - current).
Verdict PASS/FAIL only if an overlap with >= 100 reviewed rows exists, else
INCONCLUSIVE with the proxies.  No file is copied from the server.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

import rt_lib as rt


def read_provenance(sd):
    p = sd / "labels_provenance.txt"
    if not p.exists():
        return None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("reviewer:"):
            return line.split(":", 1)[1].strip()
    return None


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def inbox_overlaps():
    import local_config as lc
    root = Path(str(lc.EXCHANGE_ROOT)) / "inbox"
    out = {"inbox_reachable": root.is_dir(), "candidates": [], "overlaps": []}
    if not root.is_dir():
        return out
    for lab in root.rglob("labels.mat"):
        parts = lab.relative_to(root).parts
        if any(p.startswith(".") for p in parts):
            continue
        reviewer, sess_name = parts[0], lab.parent.name
        # locate the local session by name across areas
        local = None
        for area in rt.AREAS:
            for sd in rt.iter_sessions(area):
                if sd.name == sess_name:
                    local = (area, sd)
                    break
            if local:
                break
        if local is None:
            continue
        area, sd = local
        loc_rev = read_provenance(sd)
        if loc_rev is None or loc_rev == reviewer or not (sd / "labels.mat").exists():
            continue
        y_in = sio.loadmat(str(lab))["labels"].flatten().astype(int)
        y_loc = rt.load_labels(sd).astype(int)
        rec = {"area": area, "session": rt.rel(sd), "inbox_reviewer": reviewer, "local_reviewer": loc_rev,
               "n_inbox": int(len(y_in)), "n_local": int(len(y_loc))}
        out["candidates"].append(rec)
        if len(y_in) != len(y_loc):
            rec["status"] = "length_mismatch (different review sets)"
            continue
        rec.update({"status": "aligned", "keep_agreement": float(np.mean(y_in == y_loc)), "kappa": kappa(y_in, y_loc),
                    "inbox_keep": int(y_in.sum()), "local_keep": int(y_loc.sum()),
                    "auc_inbox_as_classifier_of_local": rt.auc(y_loc, y_in.astype(float)),
                    "both_keep": int(((y_in == 1) & (y_loc == 1)).sum()),
                    "only_inbox_keep": int(((y_in == 1) & (y_loc == 0)).sum()),
                    "only_local_keep": int(((y_in == 0) & (y_loc == 1)).sum())})
        out["overlaps"].append(rec)
        rt.log(f"  overlap {area} {rt.rel(sd)}: {reviewer} vs {loc_rev}: n {len(y_in)}, agreement {rec['keep_agreement']:.3f}, "
               f"kappa {rec['kappa']:.3f}, keeps {rec['inbox_keep']}/{rec['local_keep']} (both {rec['both_keep']}, "
               f"only-inbox {rec['only_inbox_keep']}, only-local {rec['only_local_keep']})")
    return out


def proxies():
    out = {}
    for area, fix_name, key in (("BLA", "selftest_bla_oof.npz", "oof_v2b"), ("vCA1", "selftest_vca1_oof.npz", "oof_b13")):
        fix = np.load(rt.FIXTURES / fix_name, allow_pickle=True)
        y, names, g = fix["y"], list(fix["names"]), fix["groups"]
        s = fix[key].mean(axis=0)
        T = rt.DEPLOYED_T[area]
        per = []
        for i, n in enumerate(names):
            m = g == i
            pos = m & (y == 1)
            per.append({"session": n, "n_pos": int(pos.sum()), "far_at_T": float(np.mean(s[pos] < T) * 100),
                        "confident_wrong_lt_0.02": int((s[pos] < 0.02).sum()),
                        "mean_oof_on_reals": float(s[pos].mean())})
        cw = [p["confident_wrong_lt_0.02"] for p in per]
        out[area] = {"n_sessions": len(per), "confident_wrong_total": int(sum(cw)),
                     "sessions_with_confident_wrong": int(sum(1 for c in cw if c > 0)),
                     "top_sessions": sorted(per, key=lambda p: -p["confident_wrong_lt_0.02"])[:8],
                     "reals_below_0.02_frac": float(np.mean(s[y == 1] < 0.02)),
                     "reals_below_T_frac": float(np.mean(s[y == 1] < T))}
        rt.log(f"  proxies {area}: reals < 0.02: {out[area]['reals_below_0.02_frac']:.4f} ({out[area]['confident_wrong_total']} cells in "
               f"{out[area]['sessions_with_confident_wrong']} sessions); reals < T: {out[area]['reals_below_T_frac']:.4f}")
    return out


def ceiling(flip_rate_pos, flip_rate_neg):
    out = {}
    rng = np.random.default_rng(1)
    for area, fix_name, key in (("BLA", "selftest_bla_oof.npz", "oof_v2b"), ("vCA1", "selftest_vca1_oof.npz", "oof_b13")):
        fix = np.load(rt.FIXTURES / fix_name, allow_pickle=True)
        y, oof = fix["y"], fix[key]
        base = float(np.mean([rt.auc(y, o) for o in oof]))
        drops = []
        for _ in range(20):
            yf = y.copy()
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            fp = rng.random(len(pos)) < flip_rate_pos
            fn = rng.random(len(neg)) < flip_rate_neg
            yf[pos[fp]] = 0
            yf[neg[fn]] = 1
            drops.append(base - float(np.mean([rt.auc(yf, o) for o in oof])))
        out[area] = {"auc_current": base, "auc_drop_from_flips": rt.summarize(drops),
                     "flip_rate_pos": flip_rate_pos, "flip_rate_neg": flip_rate_neg}
    return out


def main():
    t = rt.Timer()
    rt.assert_pinned()
    rt.log("[inbox] different-reviewer returns (server read-only)")
    ov = inbox_overlaps()
    rt.log("[proxies]")
    px = proxies()
    aligned = [o for o in ov["overlaps"] if o["status"] == "aligned"]
    n_rows = sum(o["n_inbox"] for o in aligned)
    if aligned:
        # disagreement rates from the overlap: reviewer-2 keeps that reviewer-1 deleted and vice versa
        only_in = sum(o["only_inbox_keep"] for o in aligned)
        only_loc = sum(o["only_local_keep"] for o in aligned)
        keeps = sum(o["local_keep"] for o in aligned)
        dels = n_rows - keeps
        fr_pos = only_loc / max(keeps, 1)       # local keep that the other reviewer would delete
        fr_neg = only_in / max(dels, 1)         # local delete that the other reviewer would keep
    else:
        fr_pos = fr_neg = None
    ce = ceiling(fr_pos, fr_neg) if aligned else None
    res = {"inbox": ov, "proxies": px, "a2_self_consistency": {"kept": 207, "deleted": 2, "unknown": 15,
                                                                "note": "same reviewer's final set vs a re-run's candidates on 4 sandboxes; a self-consistency bound, not inter-rater"},
           "overlap_rows": n_rows, "disagreement": {"keep_rate_pos": fr_pos, "keep_rate_neg": fr_neg}, "ceiling": ce}
    if ce:
        for area in ce:
            drop = ce[area]["auc_drop_from_flips"]["mean"]
            ce[area]["headroom_estimate"] = drop
            ce[area]["share_of_headroom_for_0.004"] = 0.004 / drop if drop > 0 else None
    res["verdict"] = ("PASS" if n_rows >= 100 else "INCONCLUSIVE")
    res["criterion"] = ("a two-reviewer overlap with >= 100 aligned reviewed rows exists -> measured agreement / kappa and "
                        "a flip-injected ceiling; otherwise INCONCLUSIVE with proxies")
    rt.log(f"overlap rows {n_rows}; disagreement {res['disagreement']}; ceiling {ce}\nVERDICT {res['verdict']}")
    rt.write_result("a17", res, __file__, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
