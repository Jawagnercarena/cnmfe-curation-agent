"""
test_motion_onset.py -- FRAME-PRECISE onset-coincidence motion test.

The prior lead (motion_vec.mat) measures the AVERAGE local motion level in a
candidate's patch ("motion cells live in high-motion neighbourhoods") and
plateaued at ~0.648. This asks the sharper, causal question the handoff specs:

    does local MOTION and/or STRUCTURE-CHANGE spike right AT each transient ONSET?

A motion artifact's "transient" IS a motion event (the patch jumps in x-y, or a
z plane-change restructures it), so motion/structure should be elevated exactly
at the onset. A real cell brightens IN PLACE with the patch structurally still.
The average-level features can't tell those apart; onset-locking can.

Per candidate, from motion_series.mat (per-frame mot(t) = LK |disp|, sdec(t) =
1 - corr(patch,rest), traces = C_raw):
  - detect transient ONSETS (rising edges of C_raw across baseline + k*sigma),
  - for each motion series x in {mot, sdec}, measure elevation of x in a tight
    window around the onsets, and
  - a CIRCULAR-SHIFT NULL: shift x's phase vs the FIXED onsets many times to get
    a per-cell z-score. This controls for the "busy session" confound (a cell in
    a globally high-motion patch has a high null too) that capped the level feats.

Scored identically to test_motion_vectors.py: existing-13 vs motionvec-6 vs
onset-7 vs combinations; Q1/Q2; grouped CV + leave-one-animal-out. Plus a QC
enrichment self-check comparing severity signals (lateral vs +structure/onset).

Read-only. Run after run_motion_series.m has produced motion_series.mat for all 8.
    python test_motion_onset.py
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))
import train_classifier as tc
from config import DATA_ROOT

# Sessions are discovered dynamically (any (m)-tagged session whose per-frame
# motion .mat outputs have been extracted) -- see discover_motion_sessions().
# FAILING = the 2 sessions the LATERAL-only QC flag scored worse-than-chance on
# in the original 8; kept as a print marker in the QC self-check.
FAILING = {"AVG5x-TSeries-101525-bla12-660um-23z-000",
           "AVG5x-TSeries-052826-bla37-216um-37z-000"}

# --- onset / coincidence params ---
K_ONSET   = 2.5      # onset threshold in robust sigmas above baseline
REFRACT   = 4        # min frames between onsets (one long transient != many)
WIN       = (-1, 2)  # coincidence window around onset: frames [o-1 .. o+2]
MIN_ONSET = 3        # need this many onsets to trust the coincidence stats
N_NULL    = 200      # circular-shift null draws
ONSET_NAMES = ["mot_elev", "mot_z", "mot_hit", "s_elev", "s_z", "s_hit", "n_onset"]
RNG = np.random.default_rng(42)


def log(m=""):
    print(m, flush=True)


def animal_of(nm):
    m = re.search(r"-[Tt][Ss]eries-[0-9]+-([A-Za-z]*\d+)", nm)
    return m.group(1) if m else "?"


def discover_motion_sessions():
    """Every (m)-tagged session under DATA_ROOT whose per-frame motion outputs
    (motion_series.mat + motion_vec.mat) have been extracted. Sessions that are
    tagged but not yet extracted are skipped with a note (run run_motion_all.m
    first). Returns [(task, name), ...]."""
    found = []
    for td in sorted(DATA_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("."):
            continue
        for sd in sorted(td.iterdir()):
            if not (sd.is_dir() and (sd / "labels.mat").exists()):
                continue
            try:
                L = sio.loadmat(str(sd / "labels.mat"))
            except Exception:
                continue
            if "motion_delete" not in L or int((L["motion_delete"].ravel() > 0).sum()) == 0:
                continue
            if (sd / "motion_series.mat").exists() and (sd / "motion_vec.mat").exists():
                found.append((td.name, sd.name))
            else:
                log(f"  [skip] {td.name}/{sd.name}: motion .mat not extracted yet")
    return found


def detect_onsets(trace, baseline, sigma):
    thr = baseline + K_ONSET * sigma
    above = trace > thr
    ons = np.where(above[1:] & ~above[:-1])[0] + 1
    if REFRACT > 0 and len(ons):
        keep = [ons[0]]
        for o in ons[1:]:
            if o - keep[-1] >= REFRACT:
                keep.append(o)
        ons = np.asarray(keep)
    return ons


def coincidence(x, onsets, T):
    """Elevation of x at onsets vs a circular-shift null; plus event hit-rate.

    Returns (elev, z, hit). Windows wrap circularly (mod T) so observed and null
    are computed the same way; edge wrap is rare and affects both identically.
    """
    woff = np.arange(WIN[0], WIN[1] + 1)                 # window offsets
    base_idx = (onsets[:, None] + woff[None, :]) % T     # n_on x wlen
    base = np.nanmedian(x)
    p80 = np.nanpercentile(x, 80)

    obs_wv = np.nanmax(x[base_idx], axis=1)              # per-onset window max
    obs_elev = float(np.nanmean(obs_wv) - base)
    hit = float(np.mean(obs_wv >= p80))

    shifts = RNG.integers(1, T, size=N_NULL)
    idx = (base_idx[None, :, :] + shifts[:, None, None]) % T  # n_null x n_on x wlen
    null_wv = np.nanmax(x[idx], axis=2)                  # n_null x n_on
    null_elev = np.nanmean(null_wv, axis=1) - base       # n_null
    mu, sd = float(np.nanmean(null_elev)), float(np.nanstd(null_elev))
    z = (obs_elev - mu) / (sd + 1e-9)
    return obs_elev, z, hit


def onset_features_session(sd):
    ms = sio.loadmat(str(sd / "motion_series.mat"))
    mot = ms["mot"].astype(float)      # N x T
    sdec = ms["sdec"].astype(float)    # N x T
    traces = ms["traces"].astype(float)
    baseln = ms["baseln"].astype(float).ravel()
    sigma_ = ms["sigma_"].astype(float).ravel()
    scored = ms["scored"].astype(bool).ravel()
    N, T = mot.shape

    feats = np.full((N, len(ONSET_NAMES)), np.nan)
    for i in range(N):
        ons = detect_onsets(traces[i], baseln[i], sigma_[i])
        n_on = len(ons)
        feats[i, 6] = n_on
        if not scored[i] or n_on < MIN_ONSET or not np.isfinite(mot[i]).any():
            continue
        me, mz, mh = coincidence(mot[i], ons, T)
        se, sz, sh = coincidence(sdec[i], ons, T)
        feats[i, :6] = [me, mz, mh, se, sz, sh]
    return feats


def load_session(task, nm):
    sd = DATA_ROOT / task / nm
    onset = onset_features_session(sd)

    mvd = sio.loadmat(str(sd / "motion_vec.mat"))
    mv = mvd["feats"].astype(float)
    mvnames = [str(x[0]) for x in mvd["feature_names"][0]]

    lab = sio.loadmat(str(sd / "labels.mat"))
    y = lab["labels"].flatten().astype(int)
    ym = lab["motion_delete"].flatten().astype(int)

    npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
    X = npz["feature_matrix"].astype(float)
    auto = set(npz["auto_rejected"].flatten().astype(int).tolist())
    ridx = [i for i in range(len(X)) if i not in auto]
    X13 = X[ridx]
    enames = [str(n) for n in npz["feature_names"]]

    assert len(X13) == len(y) == mv.shape[0] == onset.shape[0], (
        f"{nm}: align X13={len(X13)} y={len(y)} mv={mv.shape[0]} onset={onset.shape[0]}")
    return dict(X13=X13, mv=mv, onset=onset, y=y, ym=ym, animal=animal_of(nm),
                enames=enames, mvnames=mvnames, name=f"{task}/{nm}", short=nm)


def impute(recs, key, newkey):
    allv = np.vstack([r[key] for r in recs])
    med = np.nanmedian(allv, axis=0)
    for r in recs:
        m = r[key].copy()
        bad = np.where(np.isnan(m))
        m[bad] = np.take(med, bad[1])
        r[newkey] = m


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]


def grouped_auc(X, y, g):
    k = max(2, min(5, len(set(g[y == 1].tolist()))))
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y, g):
        oof[te] = fit_predict(X[tr], y[tr], X[te])
    v = ~np.isnan(oof)
    return roc_auc_score(y[v], oof[v]) if len(np.unique(y[v])) > 1 else float("nan")


def loao(X, y, a):
    out = {}
    for held in sorted(set(a[y == 1].tolist())):
        te = a == held
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            out[held] = (int((y[te] == 1).sum()), None)
        else:
            p = fit_predict(X[tr], y[tr], X[te])
            out[held] = (int((y[te] == 1).sum()), roc_auc_score(y[te], p))
    return out


def build(recs, which, kind):
    Xs, ys, gs, ans = [], [], [], []
    for gi, r in enumerate(recs):
        pos = r["ym"] == 1
        neg = (r["y"] == 1) if which == "Q1" else ((r["y"] == 0) & (r["ym"] == 0))
        sel = pos | neg
        if not sel.any():
            continue
        parts = {"e13": [r["X13"]], "mv": [r["mv_imp"]], "on": [r["onset_imp"]],
                 "e13on": [r["X13"], r["onset_imp"]],
                 "allmot": [r["mv_imp"], r["onset_imp"]],
                 "all": [r["X13"], r["mv_imp"], r["onset_imp"]]}[kind]
        M = np.hstack(parts)
        lab = np.zeros(int(sel.sum()), dtype=int)
        lab[pos[sel]] = 1
        Xs.append(M[sel]); ys.append(lab)
        gs.append(np.full(int(sel.sum()), gi))
        ans.append(np.array([r["animal"]] * int(sel.sum())))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ans)


def zscore(v):
    v = np.asarray(v, float)
    mu = np.nanmean(v); sd = np.nanstd(v)
    return (v - mu) / (sd + 1e-9)


def _pct(v):
    """Within-session percentile rank in [0,1]; NaN preserved."""
    v = np.asarray(v, float)
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return out
    ranks = np.argsort(np.argsort(v[ok]))
    out[ok] = ranks / max(ok.sum() - 1, 1)
    return out


def qc_enrichment(recs):
    """Per-session flag enrichment for several severity signals (top quartile).

    Channels: LAT = lateral displacement level (mot_mean + mot_p90, the CURRENT
    flag); STR = onset-locked structure-change (s_z, the z channel that lateral
    misses). Tests whether combining them -- especially a MAX of within-session
    percentile ranks (flag if extreme on EITHER axis) -- fixes the 2 sessions
    where lateral-only is worse than chance, without hurting the lateral-driven
    ones. At a fixed 25% flag rate, recall ~= 0.25 * enrichment, so enrichment
    alone is the sufficient statistic.
    """
    def LAT(r):
        return zscore(r["mv"][:, 0]) + zscore(r["mv"][:, 1])

    def STR(r):
        return zscore(r["onset_imp"][:, 4])          # s_z (onset-locked structure)

    def top(sev, ok, q=75):
        thr = np.nanpercentile(sev[ok], q)
        return ok & (sev >= thr)

    # Each variant returns a boolean flag given (r, ok). Single-channel flags use
    # the top quartile; 'union' flags cells extreme on EITHER axis (recall-first).
    variants = {
        "lateral": lambda r, ok: top(LAT(r), ok),
        "struct":  lambda r, ok: top(STR(r), ok),
        "sum":     lambda r, ok: top(LAT(r) + STR(r), ok),
        "union":   lambda r, ok: top(LAT(r), ok) | top(STR(r), ok),
    }
    log("\n" + "=" * 100)
    log("QC self-check: enrichment = precision / base-rate (>1 better than chance).")
    log("Each cell shows  enrichment / recall  (flag% in parens on the summary rows).")
    log("The success metric: consistent, no <1 enrichment rows -- while covering both motion types.")
    log("=" * 100)
    log(f"  {'session':<40}{'nMot':>5}  " + "".join(f"{k:>16}" for k in variants))
    log("  " + "-" * 104)
    agg = {k: [] for k in variants}
    fracs = {k: [] for k in variants}
    for r in recs:
        ym = r["ym"] == 1
        nM = int(ym.sum())
        ok = ~np.isnan(LAT(r)) & ~np.isnan(STR(r))
        row = f"  {r['short']:<40}{nM:>5}  "
        for k, fn in variants.items():
            cell = "        n/a     "
            if nM > 0 and ok.sum() > 3:
                flag = fn(r, ok)
                base = nM / ok.sum()
                prec = (flag & ym).sum() / max(flag.sum(), 1)
                enr = prec / max(base, 1e-9)
                rec = (flag & ym).sum() / nM
                cell = f"{enr:>7.2f}/{rec:<7.2f}"
                if r["short"] != "AVG5x-TSeries-101525-bla16-278um-36z-000":
                    agg[k].append(enr)
                    fracs[k].append(flag.sum() / ok.sum())
            row += cell
        mark = "  <-- QC FAILS(lateral)" if r["short"] in FAILING else ""
        log(row + mark)
    log("  " + "-" * 104)
    for label, fn in [("MEDIAN enr (excl bla16)", np.nanmedian),
                      ("WORST enr (excl bla16)", np.nanmin)]:
        line = "  " + f"{label:<40}{'':>5}  "
        for k in variants:
            line += f"{fn(agg[k]):>9.2f} ({np.mean(fracs[k])*100:>3.0f}%)"
        log(line)


def main():
    sess = discover_motion_sessions()
    log(f"Loading + computing onset features (circular-shift null) for "
        f"{len(sess)} sessions...")
    recs = [load_session(t, n) for t, n in sess]
    impute(recs, "mv", "mv_imp")
    impute(recs, "onset", "onset_imp")

    n_motion = int(sum(r["ym"].sum() for r in recs))
    by_an = {}
    for r in recs:
        by_an[r["animal"]] = by_an.get(r["animal"], 0) + int(r["ym"].sum())
    log("=" * 100)
    log(f"ONSET-COINCIDENCE MOTION TEST  ({len(recs)} sessions, {n_motion} motion cells)")
    log("by animal: " + ", ".join(f"{a}={n}" for a, n in sorted(by_an.items(), key=lambda x: -x[1])))
    log("=" * 100)

    on_all = np.vstack([r["onset_imp"] for r in recs])
    y_all = np.concatenate([r["y"] for r in recs])
    ym_all = np.concatenate([r["ym"] for r in recs])
    keep = y_all == 1
    od = (y_all == 0) & (ym_all == 0)
    mo = ym_all == 1
    log("\nOnset-feature group means  (do motion cells show onset-locked motion/structure?)")
    log(f"  {'feature':<12}{'keep':>9}{'other-del':>11}{'MOTION':>9}")
    for j, s in enumerate(ONSET_NAMES):
        log(f"  {s:<12}{on_all[keep, j].mean():>9.3f}{on_all[od, j].mean():>11.3f}{on_all[mo, j].mean():>9.3f}")

    for which, desc in [("Q2", "motion vs OTHER deletes  (the decision)"),
                        ("Q1", "motion vs keeps  (floor)")]:
        log("\n" + "=" * 100)
        log(f"{which}: {desc}")
        log("=" * 100)
        log(f"  {'feature set':<22}{'grouped':>9}   leave-one-animal-out AUC (n motion)")
        for kind, label in [("e13", "existing-13"), ("mv", "motionvec-6"),
                            ("on", "onset-7"), ("e13on", "existing-13 + onset-7"),
                            ("allmot", "motionvec-6 + onset-7"),
                            ("all", "existing-13 + mv + onset")]:
            X, y, g, a = build(recs, which, kind)
            gauc = grouped_auc(X, y, g)
            lo = loao(X, y, a)
            lo_s = "  ".join(
                f"{an}:{('%.3f' % v[1]) if v[1] is not None else ' n/a '}(n={v[0]})"
                for an, v in sorted(lo.items(), key=lambda kv: -kv[1][0]))
            log(f"  {label:<22}{gauc:>9.3f}   {lo_s}")

    # Importance of the onset feats in the full combined fit (Q2)
    X, y, g, a = build(recs, "Q2", "all")
    names = recs[0]["enames"] + recs[0]["mvnames"] + ONSET_NAMES
    sc = StandardScaler().fit(X)
    spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    clf = tc._make_clf("xgboost", scale_pos_weight=spw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X), y)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    log("\n" + "=" * 100)
    log("Combined feature importance (Q2 full fit) -- top 12")
    log("=" * 100)
    for rank, i in enumerate(order[:12], 1):
        tag = "  [ONSET]" if names[i] in ONSET_NAMES else (
            "  [motion-vec]" if names[i] in recs[0]["mvnames"] else "")
        log(f"  {rank:>2}. {names[i]:<16} {imp[i]:.4f}{tag}")

    qc_enrichment(recs)


if __name__ == "__main__":
    main()
