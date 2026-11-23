"""
stack_xgb.py -- plan step (c) / README "Round 2": the CNN's out-of-fold score
and its within-session percentile rank appended as columns 36-37 to the
35-column contract, then the DEPLOYED XGBoost config re-run on the control's
folds (bakeoff_tabular.run_oof_generic: same seeds, same bootstrap block, same
animal-grouping mask, threads pinned).

Bootstrap rows carry NaN in both new columns: the CNN trained on all of them,
so no out-of-fold CNN score exists for them; xgboost routes missing values
natively.  Level-0 scores are out-of-fold w.r.t. the outer split only (not
nested; the README records the caveat and the planned leak measurement).

The 35-column deployed config is also re-run through the identical generic
code path ("ctrl35") so one paired delta is code-path-clean; both deltas are
reported.

Usage:
  python stack_xgb.py --area BLA --grouping session,animal            # round-1 CNN OOF
  python stack_xgb.py --area BLA --grouping session --cnn results/r2/valfix_BLA_session.npz --name valfix
Outputs results/r2/stack_<name>_<AREA>_<grouping>.{npz,json} and appends to results/r2/stack_<name>.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bakeoff_tabular as B  # noqa: E402
import stack_eval as SE  # noqa: E402

rt = B.rt
RESULTS = HERE / os.environ.get("BAKEOFF_RESULTS", "results")
R2 = RESULTS / "r2"


def log(msg=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def session_rank(score, g_sess):
    """features.compute_ranks convention (average rank / n) among the session's scored rows."""
    r = np.full(len(score), np.nan)
    for s in np.unique(g_sess):
        m = (g_sess == s) & np.isfinite(score)
        if m.any():
            r[m] = rankdata(score[m], method="average") / m.sum()
    return r


def run_cell(area, grouping, cnn_path, name, use_rank=True):
    recs = B.load_area_pool(area)
    P = B.prepare(area, recs, grouping)
    cnn = np.load(cnn_path, allow_pickle=True)
    if not (np.array_equal(cnn["y"].astype(int), P["y"].astype(int))
            and [str(n) for n in cnn["names"]] == [str(n) for n in P["names"]]):
        raise SystemExit(f"{cnn_path.name}: rows differ from the harness pool")
    ctrl = np.load(RESULTS / f"{area}_{grouping}_xgb_deployed_t8.npz", allow_pickle=True)
    seeds = [int(s) for s in cnn["seeds"]]
    ci = [[int(t) for t in ctrl["seeds"]].index(s) for s in seeds]
    y, rev, g_sess = P["y"].astype(int), P["rev"].astype(bool), P["g_sess"]
    n_bs = len(P["y_bs"])
    n_new = 2 if use_rank else 1
    oof_stack = np.full((len(seeds), len(y)), np.nan)
    oof_c35 = np.full((len(seeds), len(y)), np.nan)
    t0 = time.time()
    for si, s in enumerate(seeds):
        sc = cnn["oof"][si].astype(float)
        cols = [sc[:, None]] + ([session_rank(sc, g_sess)[:, None]] if use_rank else [])
        P2 = dict(P)
        P2["X_ag"] = np.hstack([P["X_ag"]] + cols)
        P2["X_bs"] = np.hstack([P["X_bs"], np.full((n_bs, n_new), np.nan)]) if n_bs else P["X_bs"]
        oof_stack[si] = B.run_oof_generic("xgb", B.DEPLOYED_XGB, P2, s)[0]
        oof_c35[si] = B.run_oof_generic("xgb", B.DEPLOYED_XGB, P, s)[0]
        log(f"  {area}/{grouping} seed {s}: stack reviewed AUC {rt.auc(y, oof_stack[si], rev):.4f} | ctrl35 "
            f"{rt.auc(y, oof_c35[si], rev):.4f} | pinned control {rt.auc(y, ctrl['oof'][ci[si]], rev):.4f} ({time.time() - t0:.0f}s)")
    T = B.T_DEPLOY[area]
    ref = ctrl["oof"][ci]
    new = cnn["oof"].astype(float)
    blend = np.where(np.isnan(new), ref, 0.5 * (ref + np.nan_to_num(new)))
    blocks = {
        "stack_vs_pinned": SE.block(y, rev, ref, oof_stack, T, "stack (35 + CNN + rank) vs pinned control"),
        "stack_vs_ctrl35": SE.block(y, rev, oof_c35, oof_stack, T, "stack vs ctrl35 (same code path)"),
        "ctrl35_vs_pinned": SE.block(y, rev, ref, oof_c35, T, "ctrl35 vs pinned control (code-path check)"),
        "blend_vs_pinned": SE.block(y, rev, ref, blend, T, "blend 50/50 vs pinned control (same seeds)"),
    }
    animals = np.asarray(ctrl["animals"]).astype(str)
    per_animal = {}
    for an in sorted(set(animals)):
        m = animals == an
        if y[m].sum() >= 5 and (y[m] == 0).sum() >= 5:
            per_animal[an] = {"n": int(m.sum()), "n_real": int(y[m].sum()),
                              "control": float(np.mean([rt.auc(y[m], o[m], rev[m]) for o in ref])),
                              "blend": float(np.mean([rt.auc(y[m], o[m], rev[m]) for o in blend])),
                              "stack": float(np.mean([rt.auc(y[m], o[m], rev[m]) for o in oof_stack]))}
    b = blocks["stack_vs_pinned"]
    gate_auc = b["d_auc_reviewed"]["mean"] >= 0.005 and b["d_auc_reviewed"]["all_positive"]
    gate_junk = b["d_junk_pp"] >= 3.0
    gate_rule = (grouping != "animal") or (b["rule_T"] is not None and b["rule_detail"].get("far_max", 9) <= 1.0)
    out = {"area": area, "grouping": grouping, "cnn": str(cnn_path), "name": name, "seeds": seeds, "T": T,
           "use_rank": use_rank, "blocks": blocks, "per_animal": per_animal,
           "gate": {"auc": bool(gate_auc), "junk": bool(gate_junk), "rule": bool(gate_rule),
                    "pass": bool(gate_auc and gate_junk and gate_rule)},
           "runtime_s": time.time() - t0, "written_at": dt.datetime.now().isoformat(timespec="seconds")}
    R2.mkdir(exist_ok=True)
    stem = f"stack_{name}_{area}_{grouping}"
    np.savez_compressed(R2 / f"{stem}.npz", oof=oof_stack, oof_ctrl35=oof_c35, y=y, reviewed=rev,
                        names=np.asarray(P["names"]), groups=g_sess, groups_cv=P["groups"], seeds=np.array(seeds))
    (R2 / f"{stem}.json").write_text(json.dumps(out, indent=1, default=rt._jsonable))
    lines = [f"## {area} {grouping}-grouped, CNN = {cnn_path.name}, {len(seeds)} seeds ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
             "| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [SE.fmt(blocks[k]) for k in ("stack_vs_pinned", "stack_vs_ctrl35", "ctrl35_vs_pinned", "blend_vs_pinned")]
    lines += ["", f"gate (stack vs pinned control): AUC {'pass' if gate_auc else 'fail'}, junk {'pass' if gate_junk else 'fail'}, "
              f"rule {'pass' if gate_rule else 'fail'} -> **{'PASS' if out['gate']['pass'] else 'FAIL'}**", ""]
    if per_animal:
        lines += ["| animal | rows | reals | control | blend | stack |", "|---|---|---|---|---|---|"]
        lines += [f"| {an} | {v['n']} | {v['n_real']} | {v['control']:.4f} | {v['blend']:.4f} | {v['stack']:.4f} |"
                  for an, v in per_animal.items()]
        lines.append("")
    md = R2 / f"stack_{name}.md"
    with md.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="BLA")
    ap.add_argument("--grouping", default="session,animal")
    ap.add_argument("--cnn", default="", help="CNN OOF npz (default: round-1 results/<AREA>_<grouping>_cnn_t8.npz)")
    ap.add_argument("--name", default="r1")
    ap.add_argument("--no-rank", action="store_true", help="CNN score only (no within-session rank column)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    for grouping in [g.strip() for g in a.grouping.split(",") if g.strip()]:
        cnn_path = Path(a.cnn) if a.cnn else RESULTS / f"{a.area}_{grouping}_cnn_t8.npz"
        if not cnn_path.exists():
            log(f"missing {cnn_path}")
            continue
        run_cell(a.area, grouping, cnn_path, a.name, use_rank=not a.no_rank)


if __name__ == "__main__":
    main()
