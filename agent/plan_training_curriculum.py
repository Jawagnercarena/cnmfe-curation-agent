"""
plan_training_curriculum.py -- which sessions to train on, in what order.

Reads every training key (DATA_PARENT/.training/keys/<area>/...), describes
each session by what its REFERENCE-KEPT cells look like and how ambiguous its
candidates are, ranks sessions from easy to hard within each area, and
proposes a small curriculum spanning that range with distinct animals and
tasks.  Nothing here reads a video; everything comes from the feature rows
stored in the keys.

Per-session descriptors (over the reference keeps unless stated):
  keep_frac           reference keeps / review candidates
  med_transient_s     median mean-transient length (s) of the kept cells:
                      transient_freq (fraction of frames above baseline + 2.5 sd)
                      divided by events_per_min, at 3.75 Hz ->
                      transient_freq * 225 / events_per_min
  med_events_per_min  median events per minute of the kept cells
  med_ev_asym         median decay/rise ratio of their qualified transients
  med_peak_snr, med_ev_snr   brightness / clarity of the kept cells
  med_nb_corr_max     crowding: trace correlation with a confident neighbour
  med_ring_contrast   footprint contrast with its surround (Cn sd units)
  motion_frac         reference motion deletes / review candidates
  unexplained_del     reference deletes carrying no feature-based hint / deletes
  frac_near_boundary  model score within CLEAR_MARGIN of CONTEST_SCORE
  frac_contested      reference and model disagree

difficulty (HEURISTIC, 0 = easiest in the area, 1 = hardest): mean of the
within-area percentile ranks of frac_near_boundary, unexplained_del,
motion_frac, med_nb_corr_max (higher = harder) and of -med_ev_snr (dimmer =
harder).  tier = tertile of that score (easy / medium / hard).
clarity (HEURISTIC, 1 = clearest video in the area): mean of the within-area
percentile ranks of med_ev_snr, med_peak_snr and med_ring_contrast of the
kept cells.
typical_dynamics = med_transient_s AND med_events_per_min both inside the
area's inter-quartile range -- a flag to read, not a rule.

--pick K proposes K sessions per area spread over the tiers; within a tier,
--prefer clarity (default) takes the clearest typical session, --prefer
centre the one nearest the tier's centre.  Sessions with fewer than
--min-keeps reference-kept cells (ASSUMED default 10) are not proposed.
Distinct animal and task are preferred, then relaxed.  It is a starting point
for the supervisor.

Usage:
  python agent/plan_training_curriculum.py [--area BLA] [--pick 3] [--prefer clarity|centre]
                                           [--min-keeps 10] [--mark <session>]
Writes DATA_PARENT/.training/curriculum_plan.csv and curriculum_plan.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import rankdata

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
import training_common as tc
from push_review_bundle import animal_of

FRAME_HZ = 3.75                         # features.temporal_features assumes it
TIERS = ["easy", "medium", "hard"]
HARDER_UP = ["frac_near_boundary", "unexplained_del", "motion_frac", "med_nb_corr_max"]
HARDER_DOWN = ["med_ev_snr"]
CLEARER_UP = ["med_ev_snr", "med_peak_snr", "med_ring_contrast"]
MIN_KEEPS_DEFAULT = 10            # ASSUMED: fewer real cells than this teaches little

COLUMNS = ["area", "task", "session", "animal", "reviewer", "n_review", "n_keep", "keep_frac",
           "med_transient_s", "med_events_per_min", "med_ev_asym", "med_peak_snr", "med_ev_snr",
           "med_nb_corr_max", "med_ring_contrast", "motion_frac", "has_motion_field",
           "unexplained_del", "frac_near_boundary", "frac_contested", "difficulty", "tier",
           "clarity", "typical_dynamics", "pick", "key_path"]


def load_key(path: Path) -> dict:
    m = sio.loadmat(str(path))
    k = {}
    for name in ("ref_keep", "ref_motion", "model_score", "contested", "clear_cut", "hint_code"):
        k[name] = np.asarray(m[name], dtype=float).ravel()
    k["features"] = np.asarray(m["features"], dtype=float)
    k["names"] = [str(x[0]) for x in np.asarray(m["feature_names"]).ravel()]
    for s in ("area", "task", "session", "reference_reviewer"):
        k[s] = str(np.asarray(m[s]).ravel()[0]) if m[s].size else ""
    k["n_review"] = int(np.asarray(m["n_review"]).ravel()[0])
    k["has_motion_field"] = int(np.asarray(m["has_motion_field"]).ravel()[0])
    k["params"] = {kk: float(np.asarray(m["params"][kk][0, 0]).ravel()[0]) for kk in m["params"].dtype.names}
    k["path"] = str(path)
    return k


def nanmedian(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def describe(k: dict) -> dict:
    col = {n: i for i, n in enumerate(k["names"])}
    F = k["features"]
    keep = k["ref_keep"] == 1
    dele = ~keep
    n = k["n_review"]
    P = k["params"]
    sc = k["model_score"]
    epm = F[:, col["events_per_min"]]
    tf = F[:, col["transient_freq"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        tlen = np.where(epm > 0, tf * (FRAME_HZ * 60.0) / epm, np.nan)
    unexpl = float(np.mean(k["hint_code"][dele] == 0)) if dele.any() else float("nan")
    near = float(np.mean(np.abs(sc - P["CONTEST_SCORE"]) < P["CLEAR_MARGIN"])) if np.isfinite(sc).any() else float("nan")
    return {
        "area": k["area"], "task": k["task"], "session": k["session"],
        "animal": animal_of(k["session"]) or "?", "reviewer": k["reference_reviewer"],
        "n_review": n, "n_keep": int(keep.sum()), "keep_frac": float(keep.mean()),
        "med_transient_s": nanmedian(tlen[keep]),
        "med_events_per_min": nanmedian(epm[keep]),
        "med_ev_asym": nanmedian(F[keep, col["ev_asym"]]),
        "med_peak_snr": nanmedian(F[keep, col["peak_snr"]]),
        "med_ev_snr": nanmedian(F[keep, col["ev_snr"]]),
        "med_nb_corr_max": nanmedian(F[keep, col["nb_corr_max"]]),
        "med_ring_contrast": nanmedian(F[keep, col["ring_contrast"]]),
        "motion_frac": float(np.mean(k["ref_motion"] == 1)),
        "has_motion_field": k["has_motion_field"],
        "unexplained_del": unexpl,
        "frac_near_boundary": near,
        "frac_contested": float(np.mean(k["contested"] == 1)),
        "key_path": k["path"],
    }


def pct_rank(v: np.ndarray) -> np.ndarray:
    """Within-group percentile rank in (0, 1]; NaN -> the group median first."""
    v = np.asarray(v, dtype=float).copy()
    if np.isnan(v).all():
        return np.full(len(v), 0.5)
    v[np.isnan(v)] = np.nanmedian(v)
    return rankdata(v, method="average") / len(v)


def score_area(rows: list[dict]) -> None:
    """Adds difficulty, tier and typical_dynamics in place (one area)."""
    if not rows:
        return
    parts = [pct_rank([r[c] for r in rows]) for c in HARDER_UP]
    parts += [1.0 - pct_rank([r[c] for r in rows]) + 1.0 / len(rows) for c in HARDER_DOWN]
    diff = np.mean(np.vstack(parts), axis=0)
    diff = (diff - diff.min()) / (diff.max() - diff.min()) if diff.max() > diff.min() else np.zeros(len(rows))
    clarity = np.mean(np.vstack([pct_rank([r[c] for r in rows]) for c in CLEARER_UP]), axis=0)
    order = rankdata(diff, method="ordinal")
    n = len(rows)
    tl = np.array([r["med_transient_s"] for r in rows], dtype=float)
    ep = np.array([r["med_events_per_min"] for r in rows], dtype=float)
    q1t, q3t = np.nanpercentile(tl, 25), np.nanpercentile(tl, 75)
    q1e, q3e = np.nanpercentile(ep, 25), np.nanpercentile(ep, 75)
    for i, r in enumerate(rows):
        r["difficulty"] = float(diff[i])
        r["tier"] = TIERS[min(2, int((order[i] - 1) * 3 // n))]
        r["clarity"] = float(clarity[i])
        r["typical_dynamics"] = int(np.isfinite(tl[i]) and np.isfinite(ep[i])
                                    and q1t <= tl[i] <= q3t and q1e <= ep[i] <= q3e)
        r["pick"] = ""


def pick_sessions(rows: list[dict], k: int, prefer: str = "clarity",
                  min_keeps: int = MIN_KEEPS_DEFAULT) -> list[dict]:
    """K sessions spread over the tiers; within a tier the clearest (or the most
    central) typical session; distinct animals/tasks where possible."""
    if k <= 0 or not rows:
        return []
    want = {t: 0 for t in TIERS}
    for i in range(k):
        want[TIERS[i % 3]] += 1
    chosen, used_animals, used_tasks = [], set(), set()
    for tier in TIERS:
        pool = [r for r in rows if r["tier"] == tier and r["n_keep"] >= min_keeps]
        if not pool:
            pool = [r for r in rows if r["tier"] == tier]
        if not pool:
            continue
        centre = float(np.median([r["difficulty"] for r in pool]))

        def key_fn(r):
            last = -r["clarity"] if prefer == "clarity" else abs(r["difficulty"] - centre)
            return (0 if r["typical_dynamics"] else 1, 0 if r["has_motion_field"] else 1, last)
        for r in sorted(pool, key=key_fn):
            if want[tier] <= 0:
                break
            if r["animal"] in used_animals or r["task"] in used_tasks:
                continue
            chosen.append(r); used_animals.add(r["animal"]); used_tasks.add(r["task"]); want[tier] -= 1
        for r in sorted(pool, key=key_fn):           # relax the diversity rule if needed
            if want[tier] <= 0:
                break
            if r in chosen:
                continue
            chosen.append(r); want[tier] -= 1
    for i, r in enumerate(sorted(chosen, key=lambda r: r["difficulty"]), 1):
        r["pick"] = str(i)
    return chosen


def fmt(v, nd=2):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def short(session: str) -> str:
    return session.replace("AVG5x-TSeries-", "")


def print_area(area: str, rows: list[dict], mark: str) -> list[str]:
    hdr = ["#", "pick", "tier", "diff", "session", "task", "animal", "rev", "n", "keep%", "tr_s", "ev/min",
           "asym", "evSNR", "nbcorr", "ring", "mot%", "unexpl%", "near%", "clear", "typ"]
    lines = [f"\n== {area}: {len(rows)} keyed session(s), easy -> hard ==",
             " ".join(f"{h:>7}" if i else f"{h:>3}" for i, h in enumerate(hdr))]
    for i, r in enumerate(sorted(rows, key=lambda r: r["difficulty"]), 1):
        flag = "*" if r["session"] == mark else ""
        vals = [str(i), r["pick"] or "", r["tier"], fmt(r["difficulty"]), short(r["session"]) + flag, r["task"][:12],
                r["animal"], r["reviewer"][:6], str(r["n_review"]), fmt(100 * r["keep_frac"], 0),
                fmt(r["med_transient_s"], 1), fmt(r["med_events_per_min"], 1), fmt(r["med_ev_asym"], 1),
                fmt(r["med_ev_snr"], 1), fmt(r["med_nb_corr_max"]), fmt(r["med_ring_contrast"]),
                fmt(100 * r["motion_frac"], 0), fmt(100 * r["unexplained_del"], 0),
                fmt(100 * r["frac_near_boundary"], 0), fmt(r["clarity"]), "y" if r["typical_dynamics"] else ""]
        lines.append(f"{vals[0]:>3} " + " ".join(f"{v:>7}" if j != 4 else f"{v:<34}" for j, v in enumerate(vals[1:], 1)))
    tl = [r["med_transient_s"] for r in rows if np.isfinite(r["med_transient_s"])]
    ep = [r["med_events_per_min"] for r in rows if np.isfinite(r["med_events_per_min"])]
    lines.append(f"   transient length (s) of kept cells, session medians: IQR {np.percentile(tl, 25):.1f}-{np.percentile(tl, 75):.1f}, "
                 f"range {min(tl):.1f}-{max(tl):.1f};  events/min IQR {np.percentile(ep, 25):.1f}-{np.percentile(ep, 75):.1f}")
    return lines


def write_md(path: Path, areas: dict, picks: dict, mark: str, k: int) -> None:
    out = ["# Training curriculum plan", "", f"Generated {tc.now_str()} from the training keys. "
           "Difficulty is a HEURISTIC rank within the area (0 easiest, 1 hardest); tiers are tertiles. "
           "Columns: tr_s = median mean-transient length (s) of the reference-kept cells, ev/min = their "
           "median events per minute, asym = decay/rise, evSNR = median event SNR, nbcorr = crowding, "
           "ring = contrast with surround, mot% = motion deletes, unexpl% = deletes without a feature hint, "
           "near% = candidates near the model boundary, clear = clarity rank (1 = clearest video in the area: "
           "event SNR, peak SNR, contrast), typ = typical dynamics (inside the area's IQR).", ""]
    if mark:
        out.append(f"`*` marks {mark}.")
        out.append("")
    for area, rows in areas.items():
        out.append(f"## {area}")
        out.append("")
        if picks.get(area):
            out.append(f"Proposed {k} session(s), easy to hard (clearest typical session per tier, at least "
                       f"{MIN_KEEPS_DEFAULT} kept cells, distinct animals/tasks preferred):")
            for r in sorted(picks[area], key=lambda r: r["difficulty"]):
                out.append(f"- pick {r['pick']} [{r['tier']}, difficulty {r['difficulty']:.2f}, clarity {r['clarity']:.2f}]: "
                           f"`{area}\\{r['task']}\\{r['session']}` (animal {r['animal']}, reviewer {r['reviewer']}, "
                           f"{r['n_review']} candidates, {r['n_keep']} kept, transients ~{fmt(r['med_transient_s'],1)} s, "
                           f"{fmt(r['med_events_per_min'],1)} ev/min, motion {100*r['motion_frac']:.0f}%)")
            out.append("")
        out.append("| # | pick | tier | diff | clear | session | task | animal | reviewer | n | keep% | tr_s | ev/min | asym | evSNR | nbcorr | ring | mot% | unexpl% | near% | contested% | typ |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(sorted(rows, key=lambda r: r["difficulty"]), 1):
            flag = " *" if r["session"] == mark else ""
            out.append(f"| {i} | {r['pick']} | {r['tier']} | {r['difficulty']:.2f} | {r['clarity']:.2f} | {short(r['session'])}{flag} | {r['task']} | "
                       f"{r['animal']} | {r['reviewer']} | {r['n_review']} | {100*r['keep_frac']:.0f} | "
                       f"{fmt(r['med_transient_s'],1)} | {fmt(r['med_events_per_min'],1)} | {fmt(r['med_ev_asym'],1)} | "
                       f"{fmt(r['med_ev_snr'],1)} | {fmt(r['med_nb_corr_max'])} | {fmt(r['med_ring_contrast'])} | "
                       f"{100*r['motion_frac']:.0f} | {fmt(100*r['unexplained_del'],0)} | {fmt(100*r['frac_near_boundary'],0)} | "
                       f"{100*r['frac_contested']:.0f} | {'y' if r['typical_dynamics'] else ''} |")
        out.append("")
    path.write_text("\n".join(out) + "\n", encoding="ascii", errors="replace")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", help="BLA or vCA1 (default both)")
    ap.add_argument("--pick", type=int, default=3, help="sessions to propose per area (default 3; 0 = none)")
    ap.add_argument("--prefer", choices=["clarity", "centre"], default="clarity",
                    help="within a tier: the clearest typical session (default) or the most central one")
    ap.add_argument("--min-keeps", type=int, default=MIN_KEEPS_DEFAULT,
                    help=f"do not propose sessions with fewer reference-kept cells (ASSUMED default {MIN_KEEPS_DEFAULT})")
    ap.add_argument("--mark", default="", help="session name to flag with * in the tables")
    ap.add_argument("--training-root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.training_root) if args.training_root else tc.training_root()
    areas_list = [tc.check_area(args.area)] if args.area else list(tc.TRAINING_AREAS)
    areas, picks = {}, {}
    for a in areas_list:
        rows = []
        for kp in sorted((root / "keys" / a).rglob(tc.KEY_NAME)):
            try:
                rows.append(describe(load_key(kp)))
            except Exception as e:
                print(f"  WARN cannot read {kp}: {e}")
        if not rows:
            print(f"{a}: no keys under {root / 'keys' / a}")
            continue
        score_area(rows)
        picks[a] = pick_sessions(rows, args.pick, args.prefer, args.min_keeps)
        areas[a] = rows
        print("\n".join(print_area(a, rows, args.mark)))
        if picks[a]:
            print(f"   proposed {len(picks[a])} session(s) (prefer {args.prefer}, min {args.min_keeps} kept cells): " + ", ".join(
                f"#{r['pick']} {short(r['session'])} ({r['tier']}, clarity {r['clarity']:.2f}, {r['n_keep']} kept)"
                for r in sorted(picks[a], key=lambda r: r["difficulty"])))
    if not areas:
        sys.exit("ERROR: no keys found; run build_training_key.py first")
    all_rows = [r for a in areas for r in areas[a]]
    tc.write_csv_rows(root / "curriculum_plan.csv", COLUMNS, all_rows)
    write_md(root / "curriculum_plan.md", areas, picks, args.mark, args.pick)
    print(f"\nwritten: {root / 'curriculum_plan.csv'} and curriculum_plan.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
