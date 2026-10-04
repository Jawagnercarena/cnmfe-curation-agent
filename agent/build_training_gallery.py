"""
build_training_gallery.py -- the "what does a real cell look like" example gallery.

Reads every training key under DATA_PARENT/.training/keys/ (BLA and vCA1),
picks example candidates per category from the reference labels, model
scores and hints, has headless MATLAB render each one the way the drill shows
it (footprint, zoom, Cn crop + contour, trace) and writes a static
index.html with a DRAFT explanation per category.  Every caption carries a
TODO(Julian) marker: the text is a first draft for the expert to edit before
any trainee sees it.

Output: DATA_PARENT/.training/gallery/{manifest.json, rendered.json, img/*.png,
index.html}.  push_training_bundle.py --gallery copies the folder to
<exchange>/training/_shared/gallery/.

Usage:
  python agent/build_training_gallery.py --all [--per-category 12] [--max-per-session 2]
  python agent/build_training_gallery.py --area BLA --no-render --dry-run

Categories (from key fields; contested items only in their own category):
  clear_real      reference keep, clear-cut, no hint
  dim_real        reference keep with a low-SNR hint (code 7)
  motion          reference motion delete (m)
  duplicate       delete hinted as duplicate / split (code 4)
  diffuse         delete hinted as diffuse / neuropil-like (code 3)
  low_snr         delete hinted as low SNR / noise (code 2)
  few_transients  delete hinted as few plausible transients (code 5)
  cn_mismatch     delete hinted as not matching the correlation image (code 6)
  contested       reference and model disagree
Counts per category and per session are ASSUMED defaults (flags).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, REPO_ROOT
import training_common as tc

GALLERY_PER_CATEGORY = 12      # ASSUMED default
GALLERY_MAX_PER_SESSION = 2    # ASSUMED default

CATEGORIES = [
    ("clear_real", "Clear real cells",
     "A real cell is a compact, roughly round footprint that sits on a bright blob in the "
     "correlation image (Cn), with a trace that shows calcium transients: a fast rise, a slower "
     "decay over a few seconds, and a flat baseline in between. The reference reviewer kept "
     "every example here and the model agrees with a wide margin."),
    ("dim_real", "Dim but real cells",
     "These were KEPT by the reference reviewer even though their signal-to-noise ratio is in "
     "the bottom of the session. Look for the calcium-shaped transients and a footprint that "
     "still lines up with a blob in Cn. Dim cells are the most common thing a new reviewer "
     "deletes by mistake; the auto-rejection threshold is set low precisely so that these "
     "survive to the human."),
    ("motion", "Motion artefacts (press m, not d)",
     "The reference reviewer deleted these with the m key: the 'transient' happens when the "
     "frame moves sideways (x-y drift) or the imaging plane changes (z), not when a cell fires. "
     "In the video pass the whole neighbourhood jumps or restructures at the moment of the "
     "transient; a real cell brightens in place with its surroundings unchanged. Tag motion "
     "with m so the model learns to catch it."),
    ("duplicate", "Duplicates and split cells",
     "A second footprint on a cell that already has one, or one cell cut into two pieces. The "
     "trace follows a confident neighbour almost exactly (nb_corr_max near the top of the "
     "session) or the footprint overlaps another candidate. Keep ONE of them: the cleaner "
     "footprint with the better trace. In the real tool you can also merge them."),
    ("diffuse", "Diffuse, neuropil-like footprints",
     "Large, irregular footprints with little contrast between the footprint and its "
     "surround (ring_contrast low) or a low circularity. They pick up neuropil or out-of-focus "
     "signal rather than one cell body. The trace is often slow and low-amplitude."),
    ("low_snr", "Low SNR / noise",
     "Weak, noisy traces with no convincing transients (peak_snr or ev_snr in the bottom of "
     "the session). The footprint may look plausible, but there is nothing cell-like in time."),
    ("few_transients", "Too few plausible transients",
     "Candidates with hardly any events, or whose events do not have the fast-rise / slow-decay "
     "shape of a calcium transient (ev_rate or ev_frac_plausible low). Over a long recording a "
     "real cell usually fires at least a few times."),
    ("cn_mismatch", "Footprint does not match the correlation image",
     "The footprint sits where Cn shows no blob (cn_correlation low): the spatial component "
     "was fitted to something that is not a coherent source in the movie."),
    ("contested", "Contested: reference and model disagree",
     "The reference reviewer and the classifier disagree on these. They are shown so you can "
     "form your own opinion; they are never counted against you in a drill. Some of them are "
     "reference mistakes, some are genuinely hard."),
]


def log(msg):
    print(msg, flush=True)


def load_key(path: Path) -> dict:
    m = sio.loadmat(str(path))
    k = {}
    for name in ("ref_keep", "ref_motion", "model_score", "contested", "clear_cut",
                 "hint_code", "overlap_max", "overlap_partner"):
        if name in m:
            k[name] = np.asarray(m[name], dtype=float).ravel()
    n = int(np.asarray(m["n_review"]).ravel()[0])
    for name in ("overlap_max", "overlap_partner"):
        if name not in k:
            k[name] = np.zeros(n)
    k["n"] = n
    k["hint"] = [str(x[0]) if np.size(x) else "" for x in np.asarray(m["hint"]).ravel()]
    for s in ("area", "task", "session", "reference_reviewer", "score_kind"):
        k[s] = str(np.asarray(m[s]).ravel()[0]) if m[s].size else ""
    k["spatial_ok"] = float(np.asarray(m.get("spatial_ok", 0)).ravel()[0]) if "spatial_ok" in m else 0.0
    k["params"] = {kk: float(np.asarray(m["params"][kk][0, 0]).ravel()[0]) for kk in m["params"].dtype.names}
    return k


def category_masks(k: dict) -> dict:
    keep = k["ref_keep"] == 1
    con = k["contested"] == 1
    hc = k["hint_code"]
    return {
        "clear_real": keep & (k["clear_cut"] == 1) & (hc == 0) & ~con,
        "dim_real": keep & (hc == 7) & ~con,
        "motion": (k["ref_motion"] == 1) & ~con,
        "duplicate": ~keep & (hc == 4) & ~con,
        "diffuse": ~keep & (hc == 3) & ~con,
        "low_snr": ~keep & (hc == 2) & ~con,
        "few_transients": ~keep & (hc == 5) & ~con,
        "cn_mismatch": ~keep & (hc == 6) & ~con,
        "contested": con,
    }


def collect(keys: list[tuple[Path, dict]]) -> dict:
    pool = {c: [] for c, _, _ in CATEGORIES}
    for kpath, k in keys:
        masks = category_masks(k)
        sd = DATA_PARENT / k["area"] / k["task"] / k["session"]
        for cat, mask in masks.items():
            for i in np.flatnonzero(mask):
                col = int(i) + 1
                partner = 0
                if (k["hint_code"][i] == 4 and k["spatial_ok"] == 1
                        and k["overlap_max"][i] >= k["params"].get("DUP_OVERLAP", 0.5)):
                    partner = int(k["overlap_partner"][i])
                pool[cat].append({
                    "category": cat, "area": k["area"], "task": k["task"], "session": k["session"],
                    "session_dir": str(sd), "review_col": col, "ref_keep": int(k["ref_keep"][i]),
                    "ref_motion": int(k["ref_motion"][i]),
                    "model_score": (None if np.isnan(k["model_score"][i]) else round(float(k["model_score"][i]), 3)),
                    "contested": int(k["contested"][i]), "hint": k["hint"][i],
                    "partner_col": partner, "reference_reviewer": k["reference_reviewer"],
                })
    return pool


def sample(pool: dict, per_cat: int, max_per_session: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen = []
    for cat, _, _ in CATEGORIES:
        items = list(pool[cat])
        rng.shuffle(items)
        per_sess = {}
        picked = []
        for it in items:
            key = (it["area"], it["task"], it["session"])
            if per_sess.get(key, 0) >= max_per_session:
                continue
            per_sess[key] = per_sess.get(key, 0) + 1
            picked.append(it)
            if len(picked) >= per_cat:
                break
        picked.sort(key=lambda it: (it["area"], it["session"], it["review_col"]))
        for j, it in enumerate(picked, 1):
            it["png"] = f"img/{cat}_{j:02d}.png"
            it["label"] = f"{cat} example {j}"
        chosen.extend(picked)
    return chosen


def run_render(manifest: Path) -> bool:
    import run_cnmfe
    repo = str(REPO_ROOT).replace("\\", "/")
    man = str(manifest).replace("\\", "/")
    script = (f"addpath(genpath('{repo}/ca_source_extraction')); "
              f"addpath('{repo}/training'); "
              f"acorn_training_render_gallery('{man}');")
    return run_cnmfe._run_matlab(script, log, timeout_hours=1.0)


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_index(out: Path, items: list[dict], rendered: dict, params: dict, n_keys: int, areas: list[str]):
    by_cat = {c: [] for c, _, _ in CATEGORIES}
    for it in items:
        by_cat[it["category"]].append(it)
    lines = []
    w = lines.append
    w("<!DOCTYPE html>\n<html><head><meta charset=\"ascii\"><title>ACORN training gallery</title>")
    w("<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;max-width:1300px;color:#222}"
      ".card{border:1px solid #ccc;margin:12px 0;padding:8px}.card img{max-width:100%}"
      ".cap{font-size:13px}.hint{background:#fffbe6;padding:3px 6px;display:inline-block;font-size:12px}"
      ".intro{background:#eef5ff;padding:10px;border-left:5px solid #4a7fc1}"
      ".todo{color:#a00;font-size:12px}nav a{margin-right:12px}</style></head><body>")
    w("<h1>ACORN training gallery: what a real cell looks like, and what is not one</h1>")
    w(f"<p>Examples drawn from {n_keys} reviewed session(s) in {', '.join(areas)}. Each panel is exactly what the "
      "review tool shows: the full-frame footprint, a zoom on it, the correlation image (Cn) with the "
      "candidate's outline in red (a dashed cyan outline is an overlapping partner), and the raw trace "
      "(blue) with the fitted calcium trace (red). The reference decision is the experienced reviewer's; "
      f"the model score is the classifier's probability of a real cell (in-sample). Items are "
      "'contested' when the two disagree at the midpoint "
      f"{params.get('CONTEST_SCORE', '?')}.</p>")
    w("<p class=\"todo\">DRAFT captions written by Claude from the feature definitions and the motion "
      "notes; every category text below needs Julian's review (search for TODO).</p>")
    w("<nav>" + " ".join(f"<a href=\"#{c}\">{esc(t)} ({len(by_cat[c])})</a>" for c, t, _ in CATEGORIES) + "</nav>")
    for cat, title, intro in CATEGORIES:
        its = by_cat[cat]
        w(f"<h2 id=\"{cat}\">{esc(title)} ({len(its)})</h2>")
        w(f"<!-- TODO(Julian): confirm or rewrite this description of '{cat}' -->")
        w(f"<p class=\"intro\">{esc(intro)}</p>")
        if not its:
            w("<p>No examples found in the current keys.</p>")
        for it in its:
            r = rendered.get(it["png"], {})
            shown = r.get("shown_pos")
            ok = r.get("ok", False)
            ref = "keep" if it["ref_keep"] else "delete"
            if it["ref_motion"]:
                ref += " (motion)"
            ms = "n/a" if it["model_score"] is None else f"{it['model_score']:.2f}"
            w("<div class=\"card\">")
            if ok:
                w(f"<img src=\"{esc(it['png'])}\" alt=\"{esc(it['label'])}\">")
            else:
                w("<p class=\"todo\">(image not rendered)</p>")
            where = f"{it['area']} / {it['task']} / {it['session']}, candidate {it['review_col']}"
            if shown:
                where += f" (shown as Neuron {shown} in the drill)"
            w(f"<div class=\"cap\"><b>Reference: {esc(ref)}</b> &nbsp; Model score: {ms}"
              f"{' &nbsp; <b>contested</b>' if it['contested'] else ''}<br>{esc(where)}<br>"
              f"<span class=\"hint\">{esc(it['hint'])}</span></div></div>")
    w("<hr><p style=\"color:#666;font-size:12px\">Built by agent/build_training_gallery.py. Parameters: "
      + ", ".join(f"{k}={v}" for k, v in params.items()) + ".</p></body></html>")
    (out / "index.html").write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", help="BLA or vCA1 (default: both)")
    ap.add_argument("--all", action="store_true", help="both areas (same as omitting --area)")
    ap.add_argument("--per-category", type=int, default=GALLERY_PER_CATEGORY,
                    help=f"examples per category (ASSUMED default {GALLERY_PER_CATEGORY})")
    ap.add_argument("--max-per-session", type=int, default=GALLERY_MAX_PER_SESSION,
                    help=f"max examples from one session per category (ASSUMED default {GALLERY_MAX_PER_SESSION})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true", help="skip the MATLAB rendering step")
    ap.add_argument("--out", help="output folder (default DATA_PARENT/.training/gallery)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    areas = [tc.check_area(args.area)] if args.area else list(tc.TRAINING_AREAS)
    root = tc.training_root()
    out = Path(args.out) if args.out else root / "gallery"
    keys = []
    for a in areas:
        for kp in sorted((root / "keys" / a).rglob(tc.KEY_NAME)):
            try:
                keys.append((kp, load_key(kp)))
            except Exception as e:
                log(f"  WARN cannot read {kp}: {e}")
    if not keys:
        sys.exit(f"ERROR: no keys under {root / 'keys'} for {areas}; run build_training_key.py first")
    log(f"{len(keys)} key(s) from {', '.join(areas)}")
    pool = collect(keys)
    for cat, _, _ in CATEGORIES:
        log(f"  {cat:<15} {len(pool[cat])} candidate(s)")
    items = sample(pool, args.per_category, args.max_per_session, args.seed)
    log(f"selected {len(items)} example(s)")
    if args.dry_run:
        for it in items:
            log(f"  {it['png']}: {it['area']}/{it['session']} col {it['review_col']} ref_keep={it['ref_keep']} score={it['model_score']}")
        log("DRY RUN: nothing written.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    params = dict(keys[0][1]["params"])
    params.update({"GALLERY_PER_CATEGORY": args.per_category, "GALLERY_MAX_PER_SESSION": args.max_per_session,
                   "seed": args.seed})
    manifest = {"built_at": tc.now_str(), "out_dir": str(out), "params": params,
                "categories": [{"id": c, "title": t} for c, t, _ in CATEGORIES], "items": items}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="ascii")
    rendered = {}
    if not args.no_render:
        ok = run_render(out / "manifest.json")
        rj = out / "rendered.json"
        if rj.exists():
            for r in json.loads(rj.read_text(encoding="utf-8", errors="replace")):
                rendered[r["png"]] = r
        if not ok:
            log("  WARN: MATLAB rendering did not complete; index lists un-rendered items")
    write_index(out, items, rendered, params, len(keys), areas)
    n_ok = sum(1 for r in rendered.values() if r.get("ok"))
    log(f"gallery: {out / 'index.html'} ({n_ok} of {len(items)} images rendered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
