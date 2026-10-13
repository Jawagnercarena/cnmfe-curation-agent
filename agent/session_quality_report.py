"""
session_quality_report.py - one fixed table of per-session numbers for
reviewer oversight and recording quality, written under
DATA_PARENT/<area>/.quality/ (a dot folder: no scanner reads it).

Every row is a labelled session (labels.mat + candidate_features.npz).  The
columns are fixed and only ever APPENDED (older CSVs read as a prefix):

  identity      area, task, session, animal, date, depth_um, reviewer,
                review_version, kind (agent | bootstrap | retro)
  review        n_candidates, n_auto_rejected, n_review, n_keep, n_delete,
                keep_rate, n_motion
  model         model_type, model_n_sessions, score_kind (in_sample |
                out_of_sample | width_mismatch | no_model), model_keep_rate
                (reviewed candidates the deployed model scores >= 0.5),
                keep_rate_minus_model, auc_model_vs_labels,
                n_kept_below_threshold (kept cells the model would have
                auto-rejected = this session's false-auto-reject contribution),
                n_confident_keep_low (kept, score < 0.05),
                n_confident_delete_high (deleted, not motion, score > 0.90)
  labels        within_session_auc (5-fold logistic regression on this
                session's own features and labels: can its features separate
                its keeps from its deletes at all?)
  features      feature_nan, feature_max_abs_z (vs the area corpus)
  curator       curator_motion_flags, curator_merge_pairs (review_summary.txt)
  field         fov_group, fov_source (sheet | depth_proxy | none),
                fov_n_sessions, fov_keep_median, keep_vs_fov
  animal        animal_param_flag, gsig_animal (agent/config/animal_params.json)
  process       review_duration_min, video_passes, merges_logged,
                resumed_from_stage, reviewer_os_user, review_machine
                (review_meta.txt, written by CNMFe_final_save.m since 2026-10-12)
  bookkeeping   labels_saved, provenance_ingested, n_flags, flags

What the numbers can and cannot say: every "model" column compares a reviewer
with a classifier trained on the pool of reviewers, so it flags a reviewer who
is FAR FROM THE POOL, not one who is wrong; a reviewer biased the same way as
everyone else is invisible here.  The field columns are the only
reviewer-independent reference and depend on the imaging sheets.  Flags are
named, ASSUMED thresholds (FLAG_PARAMS) printed in the summary; they pick
sessions to look at, they do not adjudicate.

Writes:  <area>/.quality/sessions.csv, <area>/.quality/summary.md
Reads:   session folders (never modified), agent/model/<area>/classifier.joblib,
         agent/config/animal_params.json, agent/config/fov_sheets.json and the
         imaging sheets it names.
--backfill-provenance writes labels_provenance.txt (reviewer: Julian) into
labelled, non-bootstrap sessions that have none (operator decision 2026-10-12);
with --dry-run it only lists them.  That is the only write into a session folder
this script can make.

Usage:
  python agent/session_quality_report.py [--area BLA] [--no-model] [--no-cv]
  python agent/session_quality_report.py --backfill-provenance [--dry-run]
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT
from ingest_returns import read_provenance, PROVENANCE_NAME
import review_prep as rp

QUALITY_DIRNAME = ".quality"
BACKFILL_REVIEWER = "Julian"

COLUMNS = [
    "area", "task", "session", "animal", "date", "depth_um", "reviewer",
    "review_version", "kind",
    "n_candidates", "n_auto_rejected", "n_review", "n_keep", "n_delete", "keep_rate",
    "n_motion",
    "model_type", "model_n_sessions", "score_kind", "model_keep_rate",
    "keep_rate_minus_model", "auc_model_vs_labels", "n_kept_below_threshold",
    "n_confident_keep_low", "n_confident_delete_high",
    "within_session_auc",
    "feature_nan", "feature_max_abs_z",
    "curator_motion_flags", "curator_merge_pairs",
    "fov_group", "fov_source", "fov_n_sessions", "fov_keep_median", "keep_vs_fov",
    "animal_param_flag", "gsig_animal",
    "review_duration_min", "video_passes", "merges_logged", "resumed_from_stage",
    "reviewer_os_user", "review_machine",
    "labels_saved", "provenance_ingested", "n_flags", "flags",
]

# ASSUMED thresholds (2026-10-12).  Printed in summary.md.
FLAG_PARAMS = {
    "KEEP_RATE_GAP": 0.20,        # |(keep_rate - model_keep_rate) - area median of that gap| above this
    "CONFIDENT_MIN": 5,           # confident disagreements (keep_low + delete_high)
    "CONFIDENT_FRAC": 0.05,       # ... or this fraction of the review set
    "WITHIN_AUC_LOW": 0.70,       # within-session separability below this
    "FOV_RATIO_LOW": 0.5,         # keep count vs field median outside [low, high]
    "FOV_RATIO_HIGH": 2.0,
    "FOV_MIN_SESSIONS": 3,        # ... only when the field has at least this many
    "Z_SHIFT_MAX": 6.0,           # a feature mean this many corpus SDs away
    "CONTEST_SCORE": 0.5,
    "CONFIDENT_LOW": 0.05,
    "CONFIDENT_HIGH": 0.90,
    "CV_MIN_PER_CLASS": 5,        # within-session CV needs this many keeps and deletes
}

# animal token: 'bla12', 'pnb97', 'DG6D', or a bare number ('962' in the 2023
# bootstrap sessions); depth = the first '<n>um' token after it.
SESSION_RE = re.compile(r"-(\d{6}|\d{8})-([A-Za-z]*\s?\d+[A-Za-z]?(?:-[A-Z])?)-(\d+)um", re.IGNORECASE)


def log(msg=""):
    print(msg, flush=True)


def parse_session_name(name):
    """(animal, date, depth_um) from AVG5x-TSeries-<MMDDYY>-<animal>-<depth>um-..."""
    m = SESSION_RE.search(name)
    if not m:
        return None, None, None
    date = None
    for fmt in ("%m%d%y", "%m%d%Y"):
        try:
            date = datetime.strptime(m.group(1), fmt).date()
            break
        except ValueError:
            pass
    return m.group(2).lower().replace(" ", ""), date, int(m.group(3))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def iter_sessions(data_parent: Path, area=None):
    areas = [data_parent / area] if area else [d for d in sorted(data_parent.iterdir())
                                                if d.is_dir() and not d.name.startswith(".")]
    for a in areas:
        if not a.is_dir():
            continue
        for t in sorted(a.iterdir()):
            if not t.is_dir() or t.name.startswith("."):
                continue
            for s in sorted(t.iterdir()):
                if s.is_dir() and not s.name.startswith("."):
                    yield a.name, t.name, s


def session_kind(sd: Path) -> str:
    if (sd / "bootstrap_match_stats.json").exists():
        return "bootstrap"
    if (sd / "ROIs_candidates.jpg").exists():
        return "agent"
    return "retro"


def load_session(sd: Path):
    """npz + labels -> dict or None (reason logged)."""
    try:
        npz = np.load(str(sd / "candidate_features.npz"), allow_pickle=True)
        X = np.asarray(npz["feature_matrix"], dtype=float)
        n = int(npz["n_candidates"].ravel()[0]) if "n_candidates" in npz.files else len(X)
        auto = np.asarray(npz["auto_rejected"]).ravel().astype(int)
        d = sio.loadmat(str(sd / "labels.mat"))
        labels = np.asarray(d["labels"]).ravel().astype(int)
        motion = (np.asarray(d["motion_delete"]).ravel().astype(int)
                  if "motion_delete" in d else np.zeros(len(labels), dtype=int))
    except Exception as e:
        log(f"  skip {sd.name}: {e}")
        return None
    if len(labels) == n:
        rows = np.arange(n)
    elif len(labels) == n - len(set(auto.tolist())):
        auto_set = set(auto.tolist())
        rows = np.array([i for i in range(n) if i not in auto_set], dtype=int)
    else:
        log(f"  skip {sd.name}: labels {len(labels)} vs candidates {n} minus "
            f"{len(auto)} auto-rejected do not line up")
        return None
    return {"X": X, "n": n, "auto": auto, "labels": labels, "motion": motion, "rows": rows}


def read_summary_counts(sd: Path):
    """(motion flags, merge pairs) from review_summary.txt, or (None, None)."""
    f = sd / "review_summary.txt"
    if not f.exists():
        return None, None
    mf = mp = None
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*Motion artifact flags:\s*(\d+)", line)
        if m:
            mf = int(m.group(1))
        m = re.match(r"\s*Merge candidates:\s*(\d+)", line)
        if m:
            mp = int(m.group(1))
    return mf, mp


def load_models(areas, agent_dir: Path):
    out = {}
    try:
        import joblib
    except Exception:
        return out
    for a in areas:
        mp = agent_dir / "model" / a / "classifier.joblib"
        if mp.exists():
            try:
                m = joblib.load(str(mp))
                m["_mtime"] = mp.stat().st_mtime
                try:
                    m["clf"].set_params(n_jobs=1)
                except Exception:
                    pass
                out[a] = m
            except Exception as e:
                log(f"  model {a}: not loaded ({e})")
    return out


def load_animal_params(agent_dir: Path):
    p = agent_dir / "config" / "animal_params.json"
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Field of view
# ---------------------------------------------------------------------------

def load_fov_sheets(agent_dir: Path):
    """agent/config/fov_sheets.json: {area: {task: [sheet paths]}}."""
    p = agent_dir / "config" / "fov_sheets.json"
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _sheet_rows(path: str):
    """[(animal, date, notes)] from one imaging sheet; tolerant of the header
    variant whose first cell is not 'Animal ID' (column 0 is always the animal,
    column 1 the date, 'Notes' found by name or taken as column 20)."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    hdr = [h.strip() for h in rows[0]]
    idx = {h: i for i, h in enumerate(hdr)}
    i_an, i_dt = 0, idx.get("Date", 1)
    i_no = idx.get("Notes", 20)
    out = []
    for r in rows[1:]:
        if len(r) <= max(i_an, i_dt):
            continue
        r = r + [""] * (max(i_no, i_dt) + 1 - len(r))
        animal = r[i_an].strip().lower().replace(" ", "")
        date = None
        for fmt in ("%m/%d/%y", "%m/%d/%Y"):
            try:
                date = datetime.strptime(r[i_dt].strip(), fmt).date()
                break
            except ValueError:
                pass
        if not animal or date is None:
            continue
        out.append({"animal": animal, "date": date, "notes": r[i_no].strip(),
                    "day": None, "fov": None, "why": ""})
    return out


def resolve_sheet_fovs(area, task, paths):
    """{(animal, date): fov} using plan_fov_assignment's resolver when present."""
    try:
        import plan_fov_assignment as pfa
    except Exception:
        return {}, "resolver unavailable (agent/plan_fov_assignment.py)"
    lookup = {}
    missing = []
    for path in paths:
        rows = _sheet_rows(path)
        if rows is None:
            missing.append(path)
            continue
        for r in rows:
            m = pfa.DAY_RE.match(r["notes"])
            r["day"] = int(m.group(1)) if m else None
        rows.sort(key=lambda x: (x["animal"], x["date"]))
        for r in pfa.resolve_fovs(task, rows):
            if r["fov"]:
                lookup[(r["animal"], r["date"])] = r["fov"]
    note = f"missing sheet(s): {missing}" if missing else ""
    return lookup, note


def assign_fov(records, fov_sheets, depth_tol=25):
    """Fill fov_group / fov_source per record.  Sheet first; otherwise sessions
    of one (area, task, animal) are clustered by depth (+-depth_tol um) as a
    labelled proxy."""
    notes = {}
    by_task = defaultdict(list)
    for rec in records:
        by_task[(rec["area"], rec["task"])].append(rec)
    for (area, task), recs in by_task.items():
        paths = (fov_sheets.get(area) or {}).get(task) or []
        lookup, note = resolve_sheet_fovs(area, task, paths) if paths else ({}, "no sheet configured")
        if note:
            notes[(area, task)] = note
        for rec in recs:
            key = (rec["animal"], rec["_date"])
            if rec["animal"] and key in lookup:
                rec["fov_group"] = f"{rec['animal']}/{task}/{lookup[key]}"
                rec["fov_source"] = "sheet"
        # depth proxy for the rest, per animal, sorted by depth
        rest = defaultdict(list)
        for rec in recs:
            if not rec.get("fov_group") and rec["animal"] and rec["depth_um"] is not None:
                rest[rec["animal"]].append(rec)
        for animal, rs in rest.items():
            rs.sort(key=lambda r: r["depth_um"])
            cluster, start = 0, None
            for r in rs:
                if start is None or r["depth_um"] - start > depth_tol:
                    cluster += 1
                    start = r["depth_um"]
                r["fov_group"] = f"{animal}/{task}/depth~{start}um"
                r["fov_source"] = "depth_proxy"
        for rec in recs:
            if not rec.get("fov_group"):
                rec["fov_group"], rec["fov_source"] = "", "none"
    groups = defaultdict(list)
    for rec in records:
        if rec["fov_group"]:
            groups[rec["fov_group"]].append(rec)
    for g, rs in groups.items():
        keeps = [r["n_keep"] for r in rs]
        med = statistics.median(keeps)
        for r in rs:
            r["fov_n_sessions"] = len(rs)
            r["fov_keep_median"] = med
            r["keep_vs_fov"] = (r["n_keep"] / med) if med > 0 else ""
    return notes


# ---------------------------------------------------------------------------
# Per-session metrics
# ---------------------------------------------------------------------------

def within_session_auc(X, y, min_per_class):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return ""
    if (y == 1).sum() < min_per_class or (y == 0).sum() < min_per_class:
        return ""
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    k = int(min(5, (y == 1).sum(), (y == 0).sum()))
    oof = np.zeros(len(y))
    try:
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(Xc, y):
            sc = StandardScaler().fit(Xc[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(Xc[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(Xc[te]))[:, 1]
        return round(float(roc_auc_score(y, oof)), 4)
    except Exception:
        return ""


def session_record(area, task, sd, data, models, corpus_stats, animal_params, use_cv, P):
    X, rows, labels, motion = data["X"], data["rows"], data["labels"], data["motion"]
    Xr = X[rows]
    animal, date, depth = parse_session_name(sd.name)
    prov = rp.read_kv(sd / PROVENANCE_NAME)
    kind = session_kind(sd)
    reviewer = prov.get("reviewer") or ("bootstrap" if kind == "bootstrap" else BACKFILL_REVIEWER)
    rec = {c: "" for c in COLUMNS}
    rec.update({
        "area": area, "task": task, "session": sd.name, "animal": animal or "",
        "date": date.isoformat() if date else "", "depth_um": depth if depth is not None else "",
        "reviewer": reviewer, "review_version": prov.get("version", "1") if prov else "1",
        "kind": kind, "n_candidates": data["n"], "n_auto_rejected": int(len(set(data["auto"].tolist()))),
        "n_review": int(len(labels)), "n_keep": int(labels.sum()),
        "n_delete": int((labels == 0).sum()),
        "keep_rate": round(float(labels.mean()), 4) if len(labels) else "",
        "n_motion": int(motion.sum()),
        "labels_saved": datetime.fromtimestamp((sd / "labels.mat").stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "provenance_ingested": prov.get("ingested", ""),
    })
    rec["_date"] = date
    flags = []

    # --- model ---
    model = models.get(area)
    scores = None
    if model is None:
        rec["score_kind"] = "no_model"
    elif Xr.shape[1] != model.get("n_features", Xr.shape[1]):
        rec["score_kind"] = "width_mismatch"
        rec["model_type"] = model.get("model_type", "")
        rec["model_n_sessions"] = model.get("n_sessions", "")
    else:
        try:
            scores = model["clf"].predict_proba(model["scaler"].transform(
                np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)))[:, 1]
        except Exception as e:
            log(f"  {sd.name}: scoring failed ({e})")
        rec["model_type"] = model.get("model_type", "")
        rec["model_n_sessions"] = model.get("n_sessions", "")
        rec["score_kind"] = ("in_sample" if (sd / "labels.mat").stat().st_mtime < model["_mtime"]
                             else "out_of_sample")
    if scores is not None and len(labels):
        mk = scores >= P["CONTEST_SCORE"]
        rec["model_keep_rate"] = round(float(mk.mean()), 4)
        rec["keep_rate_minus_model"] = round(float(labels.mean() - mk.mean()), 4)
        if (labels == 1).any() and (labels == 0).any():
            try:
                from sklearn.metrics import roc_auc_score
                rec["auc_model_vs_labels"] = round(float(roc_auc_score(labels, scores)), 4)
            except Exception:
                pass
        thr = float(model.get("reject_threshold", 0.0))
        rec["n_kept_below_threshold"] = int(((labels == 1) & (scores < thr)).sum())
        rec["n_confident_keep_low"] = int(((labels == 1) & (scores < P["CONFIDENT_LOW"])).sum())
        rec["n_confident_delete_high"] = int(((labels == 0) & (motion == 0)
                                              & (scores > P["CONFIDENT_HIGH"])).sum())
        # keep_rate_far_from_pool is decided in finish_flags(): the classifier
        # is trained with a positive-class weight, so its probabilities sit
        # above the reviewers' keep rates for everyone; the reference is the
        # area's median keep_rate_minus_model, not zero.
        nconf = rec["n_confident_keep_low"] + rec["n_confident_delete_high"]
        if nconf >= P["CONFIDENT_MIN"] and nconf >= P["CONFIDENT_FRAC"] * len(labels):
            flags.append("confident_disagreements")

    # --- labels coherence ---
    if use_cv:
        rec["within_session_auc"] = within_session_auc(Xr, labels, P["CV_MIN_PER_CLASS"])
        if rec["within_session_auc"] != "" and rec["within_session_auc"] < P["WITHIN_AUC_LOW"]:
            flags.append("low_within_session_auc")

    # --- feature health ---
    rec["feature_nan"] = int((~np.isfinite(Xr)).sum())
    cs = corpus_stats.get(area)
    if cs is not None and cs[0].shape[0] == Xr.shape[1] and len(Xr):
        mu, sd_ = cs
        z = np.abs((np.nanmean(np.where(np.isfinite(Xr), Xr, np.nan), axis=0) - mu) / sd_)
        rec["feature_max_abs_z"] = round(float(np.nanmax(z)), 2) if np.isfinite(z).any() else ""
        if rec["feature_max_abs_z"] != "" and rec["feature_max_abs_z"] > P["Z_SHIFT_MAX"]:
            flags.append("feature_shift")
    if rec["feature_nan"] > 0:
        flags.append("feature_nan")

    # --- curator / animal ---
    mf, mp = read_summary_counts(sd)
    rec["curator_motion_flags"] = "" if mf is None else mf
    rec["curator_merge_pairs"] = "" if mp is None else mp
    # animal_params.json is keyed by the numeric part of the animal id ('12'
    # for bla12); try the full token first, then the digits.
    ap = None
    if animal:
        ap = animal_params.get(animal) or animal_params.get(re.sub(r"\D", "", animal))
    if isinstance(ap, dict):
        rec["animal_param_flag"] = ap.get("flag", "")
        rec["gsig_animal"] = ap.get("gSig", "")

    # --- process (review_meta.txt) ---
    meta = rp.read_kv(sd / "review_meta.txt")
    if meta:
        rec["review_duration_min"] = meta.get("duration_min", "")
        rec["video_passes"] = meta.get("video_passes", "")
        rec["merges_logged"] = meta.get("merges_logged", "")
        rec["resumed_from_stage"] = meta.get("resumed_from_stage", "")
        rec["reviewer_os_user"] = meta.get("reviewer_os_user", "")
        rec["review_machine"] = meta.get("machine", "")
        if meta.get("video_passes", "") == "0":
            flags.append("no_video_pass")
    if not prov and kind != "bootstrap":
        flags.append("no_reviewer_record")
    rec["_flags"] = flags
    return rec


def finish_flags(rec, P, pool_median_gap=None):
    flags = list(rec.pop("_flags", []))
    gap = rec.get("keep_rate_minus_model", "")
    if gap != "" and pool_median_gap is not None and abs(gap - pool_median_gap) > P["KEEP_RATE_GAP"]:
        flags.append("keep_rate_far_from_pool")
    r = rec.get("keep_vs_fov", "")
    if r != "" and rec.get("fov_n_sessions", 0) >= P["FOV_MIN_SESSIONS"] \
            and (r < P["FOV_RATIO_LOW"] or r > P["FOV_RATIO_HIGH"]):
        flags.append("keep_count_vs_field")
    rec["n_flags"] = len(flags)
    rec["flags"] = ";".join(flags)
    rec.pop("_date", None)
    if rec.get("keep_vs_fov", "") != "":
        rec["keep_vs_fov"] = round(float(rec["keep_vs_fov"]), 3)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _med(vals):
    vals = [v for v in vals if v != "" and v is not None]
    return f"{statistics.median(vals):.3f}" if vals else "-"


def write_summary(path: Path, area, rows, fov_notes, P, models):
    lines = [f"# Session quality -- {area}", f"_generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    m = models.get(area)
    if m:
        lines.append(f"Model: {m.get('model_type', '?')}, n_sessions {m.get('n_sessions', '?')}, "
                     f"reject_threshold {m.get('reject_threshold', '?')}, "
                     f"n_features {m.get('n_features', '?')} (read from the joblib).")
    lines += ["", f"Sessions: {len(rows)}", "",
              "## Per reviewer", "",
              "| reviewer | sessions | median keep rate | median keep - model | "
              "confident disagreements | kept below threshold | median within AUC | "
              "flagged | median duration min | with review_meta |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[r["reviewer"]].append(r)
    for who in sorted(by, key=lambda k: (-len(by[k]), k)):
        rs = by[who]
        n_meta = sum(1 for r in rs if r["review_duration_min"] != "")
        conf = sum((r["n_confident_keep_low"] or 0) + (r["n_confident_delete_high"] or 0)
                   for r in rs if r["n_confident_keep_low"] != "")
        below = sum(r["n_kept_below_threshold"] or 0 for r in rs if r["n_kept_below_threshold"] != "")
        lines.append(f"| {who} | {len(rs)} | {_med([r['keep_rate'] for r in rs])} | "
                     f"{_med([r['keep_rate_minus_model'] for r in rs])} | {conf} | {below} | "
                     f"{_med([r['within_session_auc'] for r in rs])} | "
                     f"{sum(1 for r in rs if r['n_flags'])} | "
                     f"{_med([float(r['review_duration_min']) for r in rs if r['review_duration_min'] != ''])} | "
                     f"{n_meta} |")
    flagged = sorted((r for r in rows if r["n_flags"]), key=lambda r: (-r["n_flags"], r["session"]))
    lines += ["", f"## Flagged sessions ({len(flagged)})", ""]
    if flagged:
        lines += ["| session | task | reviewer | keep rate | model keep rate | within AUC | "
                  "keep vs field | flags |", "|---|---|---|---|---|---|---|---|"]
        for r in flagged:
            lines.append(f"| {r['session']} | {r['task']} | {r['reviewer']} | {r['keep_rate']} | "
                         f"{r['model_keep_rate']} | {r['within_session_auc']} | "
                         f"{r['keep_vs_fov']} | {r['flags']} |")
    else:
        lines.append("_none_")
    lines += ["", "## Field of view", ""]
    src = defaultdict(int)
    for r in rows:
        src[r["fov_source"]] += 1
    lines.append("Sessions by field source: " + ", ".join(f"{k}: {v}" for k, v in sorted(src.items())))
    for (a, t), note in sorted(fov_notes.items()):
        if a == area:
            lines.append(f"- {t}: {note}")
    lines += ["", "## Flag thresholds (ASSUMED)", ""]
    lines += [f"- {k} = {v}" for k, v in P.items()]
    lines += ["", "## What these numbers cannot say", "",
              "- Every model column compares a reviewer with a classifier trained on the "
              "pool of reviewers: it flags a reviewer far from the pool, not a wrong one. "
              "A bias shared by everyone is invisible. score_kind = in_sample means the "
              "model was trained after these labels existed; on a small area (DG_AL) an "
              "in-sample model reproduces its own labels and the model columns say nothing.",
              "- model_keep_rate counts candidates the model scores >= 0.5; the classifier "
              "is trained with a positive-class weight, so this sits above every reviewer's "
              "keep rate. keep_rate_far_from_pool therefore compares each session's gap "
              "with the area's median gap over human-reviewed sessions, not with zero.",
              "- The field columns are the only reviewer-independent reference and are "
              "only as good as the imaging sheet; depth_proxy groups are a guess.",
              "- A low within-session AUC can mean noisy labels OR a degenerate recording.",
              "- Process columns exist only for reviews run with the review_meta.txt "
              "version of CNMFe_final_save.m (2026-10-12 or later).", ""]
    path.write_text("\n".join(lines), encoding="ascii", errors="replace")


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill_provenance(data_parent: Path, area, dry_run):
    todo = []
    for a, t, sd in iter_sessions(data_parent, area):
        if (sd / "labels.mat").exists() and not (sd / PROVENANCE_NAME).exists() \
                and not (sd / "bootstrap_match_stats.json").exists():
            todo.append((a, t, sd))
    log(f"{'Would write' if dry_run else 'Writing'} {PROVENANCE_NAME} (reviewer: {BACKFILL_REVIEWER}) "
        f"into {len(todo)} labelled session(s) without one:")
    for a, t, sd in todo:
        log(f"  {a}/{t}/{sd.name}")
        if not dry_run:
            rp.write_kv(sd / PROVENANCE_NAME, {
                "reviewer": BACKFILL_REVIEWER,
                "ingested": "-",
                "source": "backfill: labels present before provenance tracking; "
                          "attributed by the operator",
                "backfilled": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
    return len(todo)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="per-session quality table for oversight")
    ap.add_argument("--area", default=None)
    ap.add_argument("--no-model", action="store_true", help="skip classifier scoring")
    ap.add_argument("--no-cv", action="store_true", help="skip the within-session AUC")
    ap.add_argument("--backfill-provenance", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-parent", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--agent-dir", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    data_parent = Path(args.data_parent) if args.data_parent else DATA_PARENT
    agent_dir = Path(args.agent_dir) if args.agent_dir else AGENT_DIR
    P = dict(FLAG_PARAMS)

    if args.backfill_provenance:
        backfill_provenance(data_parent, args.area, args.dry_run)
        return 0

    areas = [args.area] if args.area else [d.name for d in sorted(data_parent.iterdir())
                                           if d.is_dir() and not d.name.startswith(".")]
    models = {} if args.no_model else load_models(areas, agent_dir)
    animal_params = load_animal_params(agent_dir)
    fov_sheets = load_fov_sheets(agent_dir)

    # pass 1: load
    loaded = []
    for a, t, sd in iter_sessions(data_parent, args.area):
        if not ((sd / "labels.mat").exists() and (sd / "candidate_features.npz").exists()):
            continue
        d = load_session(sd)
        if d is not None:
            loaded.append((a, t, sd, d))
    log(f"Labelled sessions loaded: {len(loaded)}")

    # corpus feature stats per area (for the z-shift column)
    corpus_stats = {}
    for a in areas:
        mats = [d["X"][d["rows"]] for aa, _, _, d in loaded if aa == a]
        widths = {m.shape[1] for m in mats}
        if len(widths) == 1 and mats:
            Xa = np.vstack(mats)
            Xa = np.where(np.isfinite(Xa), Xa, np.nan)
            mu, sd_ = np.nanmean(Xa, axis=0), np.nanstd(Xa, axis=0)
            sd_[~np.isfinite(sd_) | (sd_ == 0)] = 1.0
            corpus_stats[a] = (mu, sd_)

    # pass 2: per-session metrics
    records = []
    for i, (a, t, sd, d) in enumerate(loaded, 1):
        if i % 50 == 0:
            log(f"  ... {i}/{len(loaded)}")
        records.append(session_record(a, t, sd, d, models, corpus_stats, animal_params,
                                      not args.no_cv, P))
    fov_notes = assign_fov(records, fov_sheets)
    # pool reference for the keep-rate flag: the area's median gap between the
    # reviewers' keep rate and the model's (human-reviewed sessions only)
    pool_gap = {}
    for a in areas:
        gaps = [r["keep_rate_minus_model"] for r in records
                if r["area"] == a and r["keep_rate_minus_model"] != "" and r["kind"] != "bootstrap"]
        pool_gap[a] = statistics.median(gaps) if gaps else None
    for r in records:
        finish_flags(r, P, pool_gap.get(r["area"]))

    # pass 3: write per area
    if args.dry_run:
        log("(dry run) nothing written")
    for a in areas:
        rows = [r for r in records if r["area"] == a]
        if not rows:
            continue
        qdir = data_parent / a / QUALITY_DIRNAME
        if not args.dry_run:
            write_csv(qdir / "sessions.csv", rows)
            write_summary(qdir / "summary.md", a, rows, fov_notes, P, models)
            log(f"{a}: {len(rows)} rows -> {qdir / 'sessions.csv'}, {qdir / 'summary.md'}")
        flagged = [r for r in rows if r["n_flags"]]
        per_rev = defaultdict(int)
        for r in rows:
            per_rev[r["reviewer"]] += 1
        log(f"{a}: {len(rows)} sessions, {len(flagged)} flagged; by reviewer: "
            + ", ".join(f"{k} {v}" for k, v in sorted(per_rev.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
