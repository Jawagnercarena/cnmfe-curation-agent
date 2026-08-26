"""
Step 3b.2: high-confidence neighbour scores for every row of every session.

nb_corr_max (features.py:429-430) needs to know which neighbours are
high-confidence BEFORE the 35-column model exists.  Production solves this with
the companion 13-column first-pass model in the same joblib (curator.py:585-601).
The backfill has no such model yet, and Step 2 originally used the deployed
model's IN-SAMPLE scores -- a label leak the red team caught, fixed there by
using grouped-OOF scores instead.  vCA1 has no OOF pin at all, so we build one
score source per row class, none of them in-sample on that row's own labels:

  CV agent (16)        8-seed mean grouped OOF from PIN/baseline_oof.npz.
  train-only agent (7) leave-session-out fit: train on the other 22 agent
                       sessions + all bootstrap, score the held-out one.
  bootstrap (111)      leave-session-out over bootstrap: train on all 23 agent
                       + the other 110 bootstrap, score the held-out one.
  pending (29)         the deployed 13-column joblib.  These rows carry no
                       labels, so there is nothing to leak, and this is exactly
                       what production would compute (backfill_v2.py:174).

Second-order path, stated plainly: the leave-session-out models are trained on
OTHER sessions' labels, so a held-out session's neighbour mask is informed by
the corpus at large.  That is the same bounded exposure Step 4 accepted for its
4 non-OOF sessions; it is not the in-sample leak that was fixed.

Resumable: partial results are keyed on a hash of the pool manifest, so a
reviewer return that moves the pool discards stale partials instead of mixing
regimes.  ~120 fits, minutes.

Writes PIN/hiconf_scores.npz: {rel}__scores (N_full,), {rel}__source (str).
Read-only on session dirs.
"""
import hashlib
import json
import sys
import warnings

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

import vca1_common as vc

PART = vc.PIN / "hiconf_partial.npz"
OUT  = vc.PIN / "hiconf_scores.npz"


def pool_hash(records):
    """Identity of the pool these scores belong to."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r["name"]):
        h.update(f"{r['name']}:{len(r['y'])}:{int(r['y'].sum())}".encode())
    return h.hexdigest()[:16]


def fit_score(train_recs, held, agent_weight):
    """Train a 13-col xgboost on train_recs, score `held`. Deployed pipeline shape."""
    import diagnose_model as dm
    X = np.vstack([r["X"] for r in train_recs])
    y = np.concatenate([r["y"] for r in train_recs])
    w = np.concatenate([(r["w"] if r["is_bootstrap"] else np.ones(len(r["y"])) * agent_weight)
                        for r in train_recs])
    keep = w > 0
    sc = StandardScaler()
    Xs = sc.fit_transform(X[keep])
    clf = dm.make_clf("xgb", dm.compute_spw(y[keep], w[keep]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xs, y[keep], sample_weight=w[keep])
        return clf.predict_proba(sc.transform(held["X"]))[:, 1]


def main():
    vc.configure()
    vc.PIN.mkdir(parents=True, exist_ok=True)
    records = vc.load_pool(v2=False, require_width=13)
    cv, rest, bs = vc.split_pool(records)
    ph = pool_hash(records)
    print(f"pool hash {ph}: {len(cv)} CV agent / {len(rest)} train-only agent / {len(bs)} bootstrap")

    out, src = {}, {}
    if PART.exists():
        p = np.load(PART, allow_pickle=True)
        if str(p["pool_hash"]) == ph:
            for k in p.files:
                if k.endswith("__scores"):
                    out[k[:-8]] = p[k]
                    src[k[:-8]] = str(p[k[:-8] + "__source"])
            print(f"  resuming: {len(out)} sessions already scored")
        else:
            print(f"  discarding stale partial (pool hash {p['pool_hash']} != {ph})")

    def save_partial():
        d = {"pool_hash": ph}
        for k, v in out.items():
            d[k + "__scores"] = v
            d[k + "__source"] = np.array(src[k])
        np.savez(PART, **d)

    # ---- 1. CV agent: 8-seed mean grouped OOF from the pin ----
    oof = np.load(vc.PIN / "baseline_oof.npz", allow_pickle=True)
    mean_oof = oof["oof_seeds"].mean(axis=0)
    by_sess = {}
    for i in range(len(mean_oof)):
        by_sess.setdefault(str(oof["session"][i]), {})[int(oof["idx_in_session"][i])] = float(mean_oof[i])
    for r in cv:
        if r["name"] in out:
            continue
        d = by_sess[r["name"]]
        assert len(d) == len(r["y"]), f"{r['name']}: OOF covers {len(d)} of {len(r['y'])} rows"
        out[r["name"]] = np.array([d[i] for i in range(len(r["y"]))])
        src[r["name"]] = "oof8"
    print(f"  CV agent: {len(cv)} sessions from the 8-seed OOF pin")
    save_partial()

    # ---- 2. train-only agent: leave-session-out ----
    agent_all = cv + rest
    for r in rest:
        if r["name"] in out:
            continue
        train = [x for x in agent_all if x["name"] != r["name"]] + bs
        out[r["name"]] = fit_score(train, r, vc.AGENT_WEIGHT)
        src[r["name"]] = "loso"
        print(f"    loso  {r['name']}  (N={len(r['y'])})", flush=True)
        save_partial()

    # ---- 3. bootstrap: leave-session-out over bootstrap ----
    todo = [r for r in bs if r["name"] not in out]
    print(f"  bootstrap leave-session-out: {len(todo)} to fit "
          f"({len(bs) - len(todo)} cached)")
    for i, r in enumerate(todo, 1):
        train = agent_all + [x for x in bs if x["name"] != r["name"]]
        out[r["name"]] = fit_score(train, r, vc.AGENT_WEIGHT)
        src[r["name"]] = "lsobs"
        if i % 10 == 0 or i == len(todo):
            save_partial()
            print(f"    [{i}/{len(todo)}] {r['name']}", flush=True)

    # ---- 4. pending: the deployed 13-column model (production-realistic) ----
    _, _, pend, _ = vc.classify_sessions()
    model = joblib.load(str(vc.JOBLIB_LIVE))
    assert model["scaler"].n_features_in_ == 13, "deployed vCA1 joblib is not 13-column"
    n_new = 0
    for sd in pend:
        name = vc.rel(sd)
        if name in out:
            continue
        X = np.load(sd / vc.V1, allow_pickle=True)["feature_matrix"].astype(float)
        out[name] = model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]
        src[name] = "deployed"
        n_new += 1
    print(f"  pending: {len(pend)} sessions from the deployed 13-col joblib")

    d = {"pool_hash": np.array(ph)}
    for k, v in out.items():
        d[k + "__scores"] = v
        d[k + "__source"] = np.array(src[k])
    np.savez(OUT, **d)
    PART.unlink(missing_ok=True)

    from collections import Counter
    tally = Counter(src.values())
    hi = {k: int((v >= vc.HICONF).sum()) for k, v in out.items()}
    print(f"\nwrote {OUT}")
    print(f"  {len(out)} sessions, sources {dict(tally)}")
    print(f"  rows scored: {sum(len(v) for v in out.values())}")
    print(f"  high-confidence (>= {vc.HICONF}) rows: {sum(hi.values())} "
          f"({100 * sum(hi.values()) / sum(len(v) for v in out.values()):.1f}%)")
    for s in ("oof8", "loso", "lsobs", "deployed"):
        ks = [k for k in out if src[k] == s]
        if ks:
            tot = sum(len(out[k]) for k in ks)
            print(f"    {s:<9} {len(ks):>3} sessions, {sum(hi[k] for k in ks):>6}/{tot:<6} hi-conf "
                  f"({100 * sum(hi[k] for k in ks) / tot:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
