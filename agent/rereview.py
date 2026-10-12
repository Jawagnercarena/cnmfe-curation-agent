"""
rereview.py - a second, independent review of an already labelled session.

ACORN infers a session's state from the files in its folder (ROIs.jpg = reviewed,
labels.mat + candidate_features.npz = in training), so "re-opening" a reviewed
session by moving files aside is unsafe: without neuron.mat the watcher re-runs
CNMF-E, without labels.mat the trainer's retrospective path brings the old
decisions back, and the reviewer's return would hit ingest's same-size skip.
This tool therefore treats a second review as a NEW VERSION that sits next to
the live one and replaces it only through an explicit, logged adoption:

  request      stage the normal input bundle (review_neuron.mat, Cn/pnr/Ybg,
               report, summary, raw video, a fresh run_final_review.m) into
               outbox/<assignee>/<area>/<task>/<session>/ plus a
               REREVIEW_SESSION.txt marker.  BLIND: no first-review output
               travels, and the marker does not say why or who reviewed before.
               Locally it writes <session>/.rereview/request.txt and nothing
               else (never review_assigned.txt).
  (ingest)     ingest_returns.py parks the return under
               <session>/.rereview/<stamp>_<reviewer>/ -- the live files are
               never overwritten by a return.
  status       list sessions with an open request or an undecided return.
  compare      candidate-by-candidate agreement between the live labels and a
               parked version (same review_neuron.mat, so the columns line up),
               kappa, the candidates only one of them kept, motion tags, and
               the deployed model's score on each disagreement.
  adopt        make a parked version the live one: the current outputs are
               MOVED (same-volume rename, never deleted) into
               .rereview/archive_<stamp>_<reviewer>/, the parked outputs are
               moved into place file by file so neuron.mat and ROIs.jpg are
               never absent (the watcher polls), labels_provenance.txt gets a
               version number and the reason, REVIEW_VERSION.txt is written in
               the session, agent_run.log and <area>/REREVIEW_LOG.md are
               appended, and labels.mat is touched so the watcher retrains.
  keep-current keep the live labels, record the decision on the parked version.

Usage (central machine, Python by full path from local_config.PYTHON_EXE):
  python agent/rereview.py request  vCA1\\3odor\\AVG5x-... --assignee Taylor --reason "..." [--dry-run] [--force]
  python agent/rereview.py status   [--area vCA1] [vCA1\\3odor\\AVG5x-...]
  python agent/rereview.py compare  vCA1\\3odor\\AVG5x-... [--version <stamp>_<reviewer>]
  python agent/rereview.py adopt    vCA1\\3odor\\AVG5x-... --version <stamp>_<reviewer> --reason "..." [--dry-run]
  python agent/rereview.py keep-current vCA1\\3odor\\AVG5x-... --version <stamp>_<reviewer> [--note "..."]

Invariant: never regenerate review_neuron.mat / candidate_features.npz of a
session under re-review (recurate_sessions.py already refuses labelled
sessions); both versions and the training rows are positional against them.

SAFETY: the only server writes are shutil.copy2 into outbox/<assignee>/ plus the
generated run_final_review.m and the marker.  Locally, adopt renames files
within the session folder (os.replace); there is no delete call in this file.
"""
import argparse
import getpass
import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT, REPO_ROOT
import review_prep as rp
import push_review_bundle as P
from ingest_returns import read_provenance, read_label_counts, PROVENANCE_NAME

# Files CNMFe_final_save.m writes at the end of a review (checked against the
# script on 2026-10-12: labels + the dlmwrite traces + Coor/neuron/footprints +
# ROIs.jpg + the per-neuron image folder + merge logs; review_meta.txt is the
# metadata record added the same day).  Cn.mat is re-saved too, with the SAME
# content as the headless run's, so it stays with the inputs and is never moved.
REVIEW_OUTPUT_FILES = ["labels.mat", "neuron.mat", "ROIs.jpg", "Coor.mat",
                       "spatial_footprints.mat", "A.txt", "C.txt", "C_df.txt",
                       "C_raw.txt", "Cnn.txt", "S.txt", "review_meta.txt",
                       PROVENANCE_NAME]
REVIEW_OUTPUT_GLOBS = ["merge_log_*.mat"]
VERSION_FILE = "REVIEW_VERSION.txt"
AREA_LOG = "REREVIEW_LOG.md"
CONTEST_SCORE = 0.5      # model keep/delete midpoint for the "model says" column

# Thresholds used only to LABEL disagreements in compare's listing.  ASSUMED.
CONFIDENT_LOW = 0.05
CONFIDENT_HIGH = 0.90


def log(msg=""):
    print(msg, flush=True)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp_str():
    return datetime.now().strftime("%Y%m%d-%H%M")


def operator_name():
    try:
        return os.environ.get("USERNAME") or getpass.getuser()
    except Exception:
        return "operator"


def sha256_of(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def resolve_session(arg: str, data_parent: Path = None) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else (data_parent or DATA_PARENT) / arg


def session_parts(session_dir: Path, data_parent: Path = None):
    rel = Path(session_dir).resolve().relative_to((data_parent or DATA_PARENT).resolve())
    if len(rel.parts) != 3:
        raise ValueError(f"{session_dir} is not at DATA_PARENT/<area>/<task>/<session> depth")
    return rel.parts


def output_items(folder: Path) -> list:
    """Review outputs present in folder: files, merge logs, the _neurons dir."""
    items = []
    for name in REVIEW_OUTPUT_FILES:
        if (folder / name).exists():
            items.append(folder / name)
    for pat in REVIEW_OUTPUT_GLOBS:
        items += sorted(p for p in folder.glob(pat) if p.is_file())
    for d in folder.glob("*_neurons"):
        if d.is_dir():
            items.append(d)
    return items


def review_rows(session_dir: Path):
    """npz row index of every candidate in the review set, in review order
    (mirrors curator.py / train_classifier.load_prospective_session)."""
    npz = np.load(str(session_dir / "candidate_features.npz"), allow_pickle=True)
    n = int(npz["n_candidates"].ravel()[0]) if "n_candidates" in npz.files \
        else int(len(npz["feature_matrix"]))
    auto = set(npz["auto_rejected"].ravel().astype(int).tolist())
    return [i for i in range(n) if i not in auto], npz


def load_labels(path: Path):
    """(labels, motion) as int arrays; motion is zeros when the file predates it."""
    d = sio.loadmat(str(path))
    labels = np.asarray(d["labels"]).ravel().astype(int)
    motion = (np.asarray(d["motion_delete"]).ravel().astype(int)
              if "motion_delete" in d else np.zeros(len(labels), dtype=int))
    return labels, motion


def load_area_model(area: str):
    """The area's deployed classifier dict, or None."""
    try:
        import joblib
        mp = AGENT_DIR / "model" / area / "classifier.joblib"
        if not mp.exists():
            return None
        m = joblib.load(str(mp))
        m["_path"] = mp
        m["_mtime"] = mp.stat().st_mtime
        return m
    except Exception as e:
        log(f"  (model not loaded: {e})")
        return None


def score_with(model, X: np.ndarray):
    """Scores for rows of X, or None if the feature width does not match."""
    nf = model.get("n_features")
    if nf is not None and X.shape[1] != nf:
        return None
    clf = model["clf"]
    try:
        clf.set_params(n_jobs=1)       # thread count changes xgboost scores slightly
    except Exception:
        pass
    return clf.predict_proba(model["scaler"].transform(X))[:, 1]


def kappa(a: np.ndarray, b: np.ndarray):
    """Cohen's kappa for two binary vectors; None when undefined."""
    if len(a) == 0:
        return None
    po = float(np.mean(a == b))
    pa, pb = float(np.mean(a == 1)), float(np.mean(b == 1))
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def append_text(path: Path, text: str):
    with open(path, "a", encoding="ascii", errors="replace") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------

def cmd_request(args):
    sd = resolve_session(args.session, args.data_parent)
    if not sd.is_dir():
        sys.exit(f"ERROR: session folder not found: {sd}")
    area, task, session = session_parts(sd, args.data_parent)
    assignee = P.safe_name(args.assignee or "")
    if not assignee:
        sys.exit("ERROR: --assignee <reviewer> is required.")
    if not args.reason:
        sys.exit("ERROR: --reason is required (it stays local; the reviewer never sees it).")
    if not (sd / "labels.mat").exists():
        sys.exit("ERROR: this session has no labels.mat -- it has not been reviewed yet. "
                 "Use push_review_bundle.py for a first review.")
    for name in ("candidate_features.npz", "review_neuron.mat"):
        if not (sd / name).exists():
            sys.exit(f"ERROR: {name} missing; a re-review needs the frozen review set.")
    exchange = args.exchange
    if not exchange:
        sys.exit("ERROR: exchange root not set. Set CNMFE_EXCHANGE_ROOT (or agent/.env), "
                 "or pass --exchange.")
    rd = rp.rereview_dir(sd)
    req_file = rd / rp.REREVIEW_REQUEST
    if req_file.exists() and not args.force:
        old = rp.read_kv(req_file)
        sys.exit(f"ERROR: a re-review is already out (to {old.get('assignee', '?')} since "
                 f"{old.get('requested', '?')}). Decide it first, or --force to re-push.")
    rows, _ = review_rows(sd)
    prev = read_provenance(sd) or "unknown"
    live_labels, _ = load_labels(sd / "labels.mat")
    if len(live_labels) != len(rows):
        log(f"  WARNING: live labels.mat has {len(live_labels)} entries but the review set "
            f"holds {len(rows)}; the live version itself is misaligned.")
    dest = Path(exchange) / "outbox" / assignee / area / task / session

    files = P.collect_files(sd)              # inputs only (+ the raw video)
    leaked = [f.name for f in files if f.name in REVIEW_OUTPUT_FILES]
    if leaked:
        sys.exit(f"ERROR: bundle would carry review outputs {leaked}; refusing (blind).")

    log(f"[{area}/{task}/{session}]")
    log(f"  second review -> {assignee}  (current labels by {prev}; blind)")
    log(f"  reason (local only): {args.reason}")
    log(f"  dest: {dest}")
    total = 0
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        size = f.stat().st_size
        target = dest / f.name
        if args.dry_run:
            log(f"  would copy {f.name} ({size/1e6:.1f} MB)")
        elif target.exists() and target.stat().st_size == size:
            log(f"  skip (already staged): {f.name}")
        else:
            log(f"  copy {f.name} ({size/1e6:.1f} MB) ...")
            shutil.copy2(str(f), str(target))
        total += size
    marker = {
        "session": f"{area}/{task}/{session}",
        "kind": "second independent review",
        "assignee": assignee,
        "staged": now_str(),
        "staged_by": "rereview.py",
        "blind": "yes",
        "instructions": ("Pull this folder to a NEW local folder (never on top of an "
                         "earlier copy of this session), run run_final_review.m as for "
                         "any review, and return the finished folder to "
                         f"inbox/{assignee}/{area}/{task}/{session}/ with this file "
                         "still in it."),
    }
    request = {
        "requested": now_str(),
        "requested_by": operator_name(),
        "assignee": assignee,
        "blind": "yes",
        "reason": args.reason,
        "current_reviewer": prev,
        "current_labels_sha256": sha256_of(sd / "labels.mat"),
        "expected_n_review": len(rows),
        "bundle": str(dest),
    }
    if args.dry_run:
        log(f"  would generate run_final_review.m + {rp.REREVIEW_MARKER}")
        log(f"  would write {req_file}")
        log(f"  -> {len(files) + 2} items, {total/1e6:.1f} MB")
        return 0
    rp.write_launcher(dest, REPO_ROOT)
    rp.write_kv(dest / rp.REREVIEW_MARKER, marker)
    rd.mkdir(exist_ok=True)
    rp.write_kv(req_file, request)
    append_text(sd / "agent_run.log",
                f"{now_str()}  [REREVIEW] second review requested -> {assignee} "
                f"(blind; current labels by {prev}); reason: {args.reason}\n")
    log(f"  generated run_final_review.m + {rp.REREVIEW_MARKER}; request recorded locally")
    log(f"  -> staged {total/1e6:.1f} MB to {dest}")
    log("  Tell the reviewer: a second-review bundle is in their outbox; pull it to a "
        "fresh folder. Do not tell them who reviewed it before or why.")
    return 0


# ---------------------------------------------------------------------------
# status
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
                    yield s


def cmd_status(args):
    dp = args.data_parent
    sessions = [resolve_session(args.session, dp)] if args.session else list(iter_sessions(dp, args.area))
    n = 0
    for sd in sessions:
        st = rp.rereview_state(sd)
        hist = [v for v in rp.parked_versions(sd) if v[2]["decision"]]
        if not st and not hist and not args.session:
            continue
        n += 1
        try:
            rel = "/".join(session_parts(sd, dp))
        except ValueError:
            rel = str(sd)
        log(f"{rel}")
        log(f"  live labels by {read_provenance(sd) or 'unknown'}"
            + (f"; {VERSION_FILE} present" if (sd / VERSION_FILE).exists() else ""))
        if st:
            log(f"  {st['summary']}")
            if st["request"]:
                r = st["request"]
                log(f"    request: {r.get('requested', '?')} by {r.get('requested_by', '?')}; "
                    f"reason: {r.get('reason', '')}")
            for name, _, info in st["versions"]:
                log(f"    parked: {name}  labels {info.get('n_labels', '?')} "
                    f"(expected {info.get('expected_n_review', '?')}, aligned {info.get('aligned', '?')})")
        for name, _, info in hist:
            d = info["decision"]
            log(f"  decided: {name} -> {d.get('decision', '?')} on {d.get('at', '?')} "
                f"by {d.get('by', '?')}")
        archives = sorted(p.name for p in rp.rereview_dir(sd).glob(rp.REREVIEW_ARCHIVE_PREFIX + "*")
                          if p.is_dir()) if rp.rereview_dir(sd).is_dir() else []
        for a in archives:
            log(f"  archived: {a}")
    if n == 0:
        log("No sessions with an open re-review" + (f" in {args.area}" if args.area else "") + ".")
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def pick_version(sd: Path, version):
    parked = rp.parked_versions(sd)
    if version:
        for name, p, info in parked:
            if name == version:
                return name, p, info
        sys.exit(f"ERROR: no parked version '{version}' under {rp.rereview_dir(sd)}. "
                 f"Known: {[n for n, _, _ in parked] or 'none'}")
    undecided = [v for v in parked if not v[2]["decision"]]
    if len(undecided) == 1:
        return undecided[0]
    if not undecided:
        sys.exit("ERROR: no undecided parked version; pass --version to name one.")
    sys.exit(f"ERROR: {len(undecided)} undecided versions; pass --version: "
             f"{[n for n, _, _ in undecided]}")


def compare_versions(sd: Path, name: str, pdir: Path, info: dict, write_record=True):
    """Print (and record in the parked folder) the live-vs-parked comparison.
    Returns a dict of the headline numbers."""
    area, task, session = session_parts(sd, None) if sd.resolve().is_relative_to(DATA_PARENT.resolve()) \
        else (sd.parent.parent.name, sd.parent.name, sd.name)
    rows, npz = review_rows(sd)
    live_lab, live_mot = load_labels(sd / "labels.mat")
    new_lab, new_mot = load_labels(pdir / "labels.mat")
    live_who = read_provenance(sd) or "unknown"
    new_who = info.get("reviewer", "?")
    out = []
    out.append(f"Re-review comparison  {area}/{task}/{session}")
    out.append(f"  live   : {live_who:10s} {int(live_lab.sum()):4d} keep / {int((live_lab == 0).sum()):4d} delete"
               f"  ({int(live_mot.sum())} motion tags)  [{len(live_lab)} entries]")
    out.append(f"  parked : {new_who:10s} {int(new_lab.sum()):4d} keep / {int((new_lab == 0).sum()):4d} delete"
               f"  ({int(new_mot.sum())} motion tags)  [{len(new_lab)} entries]  version {name}")
    out.append(f"  review set holds {len(rows)} candidates (candidate_features.npz minus auto-rejected)")
    res = {"aligned": len(live_lab) == len(new_lab) == len(rows)}
    if not res["aligned"]:
        out.append("  !! NOT ALIGNED: the two label vectors and the review set differ in length; "
                   "no per-candidate comparison is possible and adopt will refuse this version.")
        text = "\n".join(out)
        log(text)
        if write_record:
            append_text(pdir / "compare.txt", f"# {now_str()}\n{text}\n\n")
        return res
    eq = live_lab == new_lab
    k = kappa(live_lab, new_lab)
    res.update({"n": int(len(eq)), "agree": int(eq.sum()), "agreement": float(eq.mean()),
                "kappa": k, "live_keep": int(live_lab.sum()), "new_keep": int(new_lab.sum()),
                "only_live_keep": int(((live_lab == 1) & (new_lab == 0)).sum()),
                "only_new_keep": int(((live_lab == 0) & (new_lab == 1)).sum())})
    out.append(f"  agreement: {res['agree']}/{res['n']} = {res['agreement']*100:.1f}%   "
               f"kappa = {'n/a' if k is None else f'{k:.3f}'}")
    out.append(f"  kept only by {live_who}: {res['only_live_keep']}    kept only by {new_who}: {res['only_new_keep']}")
    if res["only_live_keep"] > res["only_new_keep"]:
        out.append(f"  direction: {new_who} is the stricter of the two on this session")
    elif res["only_new_keep"] > res["only_live_keep"]:
        out.append(f"  direction: {new_who} is the more liberal of the two on this session")

    # Model opinion on the disagreements.  In-sample for the live labels if the
    # model was trained after they existed (the watcher retrains on every
    # prospective session), so it is a tie-breaker hint, not a referee.
    model = load_area_model(area)
    scores = None
    if model is not None:
        X = np.asarray(npz["feature_matrix"], dtype=float)[rows]
        scores = score_with(model, X)
        if scores is None:
            out.append(f"  model: feature width {X.shape[1]} does not match the deployed "
                       f"{model.get('n_features')}-column model; no scores")
        else:
            in_sample = (sd / "labels.mat").stat().st_mtime < model["_mtime"]
            out.append(f"  model: {model.get('model_type', '?')} (n_sessions {model.get('n_sessions', '?')}, "
                       f"threshold {model.get('reject_threshold', '?')}); "
                       f"{'IN-SAMPLE for the live labels' if in_sample else 'out-of-sample for the live labels'}; "
                       f"out-of-sample for the parked labels")
            mk = scores >= CONTEST_SCORE
            out.append(f"  model agrees with live on {int((mk == (live_lab == 1)).sum())}/{len(rows)}, "
                       f"with parked on {int((mk == (new_lab == 1)).sum())}/{len(rows)}")
    dis = np.where(~eq)[0]
    if len(dis):
        out.append("")
        out.append(f"  disagreements ({len(dis)}): review # = column of review_neuron.mat (1-based), "
                   f"npz row = candidate_features.npz row (0-based)")
        out.append(f"  {'review #':>8} {'npz row':>7} {live_who[:10]:>10} {new_who[:10]:>10} {'model':>6}  note")
        for i in dis:
            lv = "keep" if live_lab[i] else ("del(m)" if live_mot[i] else "delete")
            nv = "keep" if new_lab[i] else ("del(m)" if new_mot[i] else "delete")
            sc = "  -   " if scores is None else f"{scores[i]:.2f}"
            note = ""
            if scores is not None:
                if scores[i] < CONFIDENT_LOW:
                    note = "model confident junk"
                elif scores[i] > CONFIDENT_HIGH:
                    note = "model confident real"
            out.append(f"  {i + 1:>8} {rows[i]:>7} {lv:>10} {nv:>10} {sc:>6}  {note}")
    text = "\n".join(out)
    log(text)
    if write_record:
        append_text(pdir / "compare.txt", f"# {now_str()}\n{text}\n\n")
    return res


def cmd_compare(args):
    sd = resolve_session(args.session, args.data_parent)
    if not (sd / "labels.mat").exists():
        sys.exit(f"ERROR: {sd} has no live labels.mat to compare against.")
    name, pdir, info = pick_version(sd, args.version)
    compare_versions(sd, name, pdir, info)
    return 0


# ---------------------------------------------------------------------------
# adopt / keep-current
# ---------------------------------------------------------------------------

def _move(src: Path, dst: Path, dry: bool, plan: list):
    plan.append(f"    move {src.name:32s} -> {dst}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(src), str(dst))


def cmd_adopt(args):
    sd = resolve_session(args.session, args.data_parent)
    area, task, session = session_parts(sd, args.data_parent)
    if not args.reason:
        sys.exit("ERROR: --reason is required; it goes into labels_provenance.txt and the logs.")
    name, pdir, info = pick_version(sd, args.version)
    if info["decision"]:
        sys.exit(f"ERROR: version {name} was already decided: {info['decision']}")
    if not (pdir / "labels.mat").exists():
        sys.exit(f"ERROR: parked version {name} has no labels.mat.")
    rows, _ = review_rows(sd)
    new_lab, _ = load_labels(pdir / "labels.mat")
    if len(new_lab) != len(rows):
        sys.exit(f"ERROR: parked labels.mat has {len(new_lab)} entries but the review set holds "
                 f"{len(rows)}; the reviewer worked from a stale bundle. Refusing.")
    for name_req in ("neuron.mat", "ROIs.jpg"):
        if not (pdir / name_req).exists():
            sys.exit(f"ERROR: parked version lacks {name_req}; the return is incomplete. Refusing.")
    live_has_labels = (sd / "labels.mat").exists()
    prev = read_provenance(sd) or "unknown"
    prev_version = int(rp.read_kv(sd / PROVENANCE_NAME).get("version", "1") or 1) if live_has_labels else 0
    new_version = prev_version + 1
    stamp = stamp_str()
    archive = rp.rereview_dir(sd) / f"{rp.REREVIEW_ARCHIVE_PREFIX}{stamp}_{P.safe_name(prev)}"
    if live_has_labels:
        log("Comparison before adopting:")
        compare_versions(sd, name, pdir, info, write_record=not args.dry_run)
        log()

    plan = []
    live_items = {p.name: p for p in output_items(sd)}
    new_items = {p.name: p for p in output_items(pdir) if p.name != PROVENANCE_NAME}
    plan.append(f"  archive current outputs ({prev}, version {prev_version}) -> {archive}")
    plan.append(f"  adopt {name} ({info.get('reviewer', '?')}) as version {new_version}")
    # File by file: the incoming file is first renamed to a temp name in the
    # session folder (same volume, instant), the live file renamed into the
    # archive, then the temp renamed into place -- neuron.mat / ROIs.jpg are
    # absent for microseconds only, so a concurrent watcher poll cannot read
    # the session as unfinished or unreviewed.
    for nm in sorted(set(live_items) | set(new_items)):
        src = new_items.get(nm)
        live = live_items.get(nm)
        if src is not None:
            tmp = sd / (nm + ".rereview_tmp")
            _move(src, tmp, args.dry_run, plan)
            if live is not None:
                _move(live, archive / nm, args.dry_run, plan)
            _move(tmp, sd / nm, args.dry_run, plan)
        elif live is not None:
            _move(live, archive / nm, args.dry_run, plan)
    for line in plan:
        log(line)
    if args.dry_run:
        log("  (dry run) would then write labels_provenance.txt, REVIEW_VERSION.txt, "
            "decision.txt, the logs, and touch labels.mat")
        return 0

    operator = operator_name()
    rp.write_kv(sd / PROVENANCE_NAME, {
        "reviewer": info.get("reviewer", "unknown"),
        "ingested": info.get("parked_at", now_str()),
        "source": info.get("source", str(pdir)),
        "version": new_version,
        "supersedes": f"{prev} (version {prev_version}, archived at {archive.relative_to(sd)})",
        "adopted": now_str(),
        "adopted_by": operator,
        "reason": args.reason,
    })
    append_text(sd / VERSION_FILE,
                f"version: {new_version}\n"
                f"reviewer: {info.get('reviewer', 'unknown')}\n"
                f"adopted: {now_str()} by {operator}\n"
                f"reason: {args.reason}\n"
                f"supersedes: version {prev_version} by {prev}, archived at {archive.relative_to(sd)}\n"
                f"note: the final outputs in this folder (neuron.mat, C_df.txt, ROIs.jpg, ...) "
                f"changed with this version; downstream analyses that read them before "
                f"{datetime.now():%Y-%m-%d} used the archived version.\n\n")
    rp.write_kv(pdir / rp.REREVIEW_DECISION, {
        "decision": "adopted", "at": now_str(), "by": operator,
        "version": new_version, "reason": args.reason, "archive": str(archive.relative_to(sd)),
    })
    req = rp.rereview_dir(sd) / rp.REREVIEW_REQUEST
    if req.exists():
        os.replace(str(req), str(pdir / rp.REREVIEW_REQUEST))
    now = datetime.now().timestamp()
    os.utime(sd / "labels.mat", (now, now))
    line = (f"{now_str()}  [REREVIEW] adopted version {new_version} by "
            f"{info.get('reviewer', '?')} (parked {name}); version {prev_version} by {prev} "
            f"archived at {archive.relative_to(sd)}; reason: {args.reason}")
    append_text(sd / "agent_run.log", line + "\n")
    append_text(DATA_PARENT_OF(sd) / area / AREA_LOG,
                f"- {now_str()}  {task}/{session}: ADOPTED version {new_version} "
                f"({info.get('reviewer', '?')}) over version {prev_version} ({prev}); "
                f"reason: {args.reason}; archive {archive.relative_to(sd)}; by {operator}\n")
    log()
    log(f"Adopted. {sd / PROVENANCE_NAME} now says version {new_version} by {info.get('reviewer', '?')}.")
    log("Follow-ups (nothing detects these for you):")
    log("  1. labels.mat was touched: an alive watcher retrains on its next poll; otherwise "
        "run /retrain-area. n_sessions will NOT change -- check the trainer log line for "
        "this session instead.")
    log("  2. The training key for this session (if any) was built from the old labels: "
        f"python agent/build_training_key.py {area}\\{task}\\{session} --force")
    log("  3. Evaluation pools that pin label hashes will refuse this session; note the "
        "change in the active eval LOG.md.")
    log(f"  4. {VERSION_FILE} in the session records the change for downstream readers of "
        "the trace files; tell whoever analyses this session.")
    log("  5. The reviewer's bundle and return stay on the server; deleting them is an "
        "operator action.")
    return 0


def DATA_PARENT_OF(sd: Path) -> Path:
    """DATA_PARENT for a session (works for temp-dir tests too)."""
    return sd.resolve().parent.parent.parent


def cmd_keep_current(args):
    sd = resolve_session(args.session, args.data_parent)
    area, task, session = session_parts(sd, args.data_parent)
    name, pdir, info = pick_version(sd, args.version)
    if info["decision"]:
        sys.exit(f"ERROR: version {name} was already decided: {info['decision']}")
    if (sd / "labels.mat").exists() and (pdir / "labels.mat").exists():
        compare_versions(sd, name, pdir, info)
        log()
    operator = operator_name()
    rp.write_kv(pdir / rp.REREVIEW_DECISION, {
        "decision": "kept-current", "at": now_str(), "by": operator,
        "note": args.note or "",
    })
    req = rp.rereview_dir(sd) / rp.REREVIEW_REQUEST
    if req.exists():
        os.replace(str(req), str(pdir / rp.REREVIEW_REQUEST))
    prev = read_provenance(sd) or "unknown"
    line = (f"{now_str()}  [REREVIEW] kept current labels by {prev}; parked version {name} "
            f"({info.get('reviewer', '?')}) recorded, not adopted"
            + (f"; note: {args.note}" if args.note else ""))
    append_text(sd / "agent_run.log", line + "\n")
    append_text(DATA_PARENT_OF(sd) / area / AREA_LOG,
                f"- {now_str()}  {task}/{session}: KEPT current ({prev}); parked {name} "
                f"({info.get('reviewer', '?')}) not adopted"
                + (f"; note: {args.note}" if args.note else "") + f"; by {operator}\n")
    log(f"Kept the current labels ({prev}). Parked version {name} stays under "
        f"{rp.rereview_dir(sd).relative_to(sd)} for the record.")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="second, independent review of a labelled session")
    ap.add_argument("--data-parent", default=None, help=argparse.SUPPRESS)   # tests only
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("request", help="stage a blind second-review bundle")
    r.add_argument("session", help="area\\task\\session (or absolute path)")
    r.add_argument("--assignee", required=True)
    r.add_argument("--reason", required=True, help="why (stays local)")
    r.add_argument("--exchange", default=EXCHANGE_ROOT)
    r.add_argument("--force", action="store_true", help="re-push although a request is open")
    r.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("status", help="open requests and undecided returns")
    s.add_argument("session", nargs="?")
    s.add_argument("--area", default=None)

    c = sub.add_parser("compare", help="live labels vs a parked version")
    c.add_argument("session")
    c.add_argument("--version", default=None, help="<stamp>_<reviewer> under .rereview/")

    a = sub.add_parser("adopt", help="make a parked version the live one (archives the current)")
    a.add_argument("session")
    a.add_argument("--version", default=None)
    a.add_argument("--reason", required=True)
    a.add_argument("--dry-run", action="store_true")

    k = sub.add_parser("keep-current", help="record that the live labels stay")
    k.add_argument("session")
    k.add_argument("--version", default=None)
    k.add_argument("--note", default="")

    args = ap.parse_args(argv)
    args.data_parent = Path(args.data_parent) if args.data_parent else DATA_PARENT
    return {"request": cmd_request, "status": cmd_status, "compare": cmd_compare,
            "adopt": cmd_adopt, "keep-current": cmd_keep_current}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
