"""
watcher.py
Main agent loop. Monitors D:\\Julian_CNMFe\\BLA\\ for new .tif files sitting loose
inside task folders, then orchestrates the full processing pipeline.

Usage:
    C:\\ProgramData\\anaconda3\\envs\\valence\\python.exe watcher.py

Keep this running in the background. It polls every 60 seconds.

Transfer safety: a .tif file is only processed once its file size has been
stable for at least MIN_STABLE_SECS (default 120 s) across two consecutive
polls. This prevents processing partially-transferred files from the server.

Logs are written to agent/logs/watcher.log and printed to the console.
"""

import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# -- local modules (same directory) --
import params as param_estimator
import run_cnmfe

from config import DATA_ROOT
from local_config import REPO_ROOT, PYTHON_EXE
AGENT_DIR       = Path(__file__).parent
LOG_DIR         = AGENT_DIR / "logs"
POLL_SECS       = 60    # seconds between scans
MIN_STABLE_SECS = 120   # file must be size-stable for this long before processing

LOG_DIR.mkdir(exist_ok=True)

# ---- Logging setup ----
logger = logging.getLogger("cnmfe_agent")
logger.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler(LOG_DIR / "watcher.log", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

log = logger.info


# ---- Transfer-safety state ----
# Tracks files we've seen but aren't ready yet.
# Key: Path, Value: {"size": int, "first_seen": float, "last_size_change": float}
_pending: dict[Path, dict] = {}


def find_new_tifs() -> list[tuple[Path, Path]]:
    """
    Find .tif files that need processing. Two cases:

    1. Loose .tif directly in a task folder — new file, not yet moved.
       These go through the transfer-stability check before processing.

    2. .tif inside an existing session subfolder with no neuron.mat —
       incomplete/failed session that needs to be retried.
       These skip the stability check (file was already fully transferred
       and moved on a previous run).

    Returns (task_dir, tif_path) for every .tif found in either case.
    """
    found = []
    if not DATA_ROOT.exists():
        log(f"[WATCH] Data root not found: {DATA_ROOT}")
        return found

    for task_dir in DATA_ROOT.iterdir():
        if not task_dir.is_dir():
            continue
        for item in task_dir.iterdir():
            if item.is_file() and item.suffix.lower() in (".tif", ".tiff"):
                # Case 1: loose file in task folder
                found.append((task_dir, item))
            elif item.is_dir():
                # Case 2: session folder that was started but not finished
                session_dir = item
                if (session_dir / "neuron.mat").exists():
                    continue  # already done
                # Look for a .tif inside matching the session name
                for sub in session_dir.iterdir():
                    if sub.is_file() and sub.suffix.lower() in (".tif", ".tiff"):
                        found.append((task_dir, sub))
                        break
    return found


def is_transfer_complete(tif_path: Path) -> bool:
    """
    Returns True only if:
      - The file has been seen for at least MIN_STABLE_SECS, AND
      - Its size has not changed since the last poll.

    Exception: if the .tif is already inside a session subfolder (incomplete
    retry case), it is considered immediately ready — it was fully transferred
    on a previous run.

    Updates _pending state on every call.
    """
    # Retry case: tif is inside a session folder, i.e. 3 levels deep:
    # DATA_ROOT / task_dir / session_dir / file.tif
    # A loose new file is only 2 levels deep: DATA_ROOT / task_dir / file.tif
    if tif_path.parent.parent.parent == DATA_ROOT:
        return True  # already in session folder — was fully transferred, skip check
    now = time.time()

    try:
        current_size = tif_path.stat().st_size
    except FileNotFoundError:
        _pending.pop(tif_path, None)
        return False

    if tif_path not in _pending:
        # First time we see this file — start tracking, don't process yet
        _pending[tif_path] = {
            "size": current_size,
            "first_seen": now,
            "last_size_change": now,
        }
        log(f"  [WATCH] New file detected (waiting for transfer to complete): "
            f"{tif_path.name} ({current_size / 1e9:.2f} GB)")
        return False

    entry = _pending[tif_path]

    if current_size != entry["size"]:
        # Still growing — reset the stability timer
        log(f"  [WATCH] {tif_path.name} still transferring "
            f"({entry['size']/1e9:.2f} → {current_size/1e9:.2f} GB)")
        entry["size"] = current_size
        entry["last_size_change"] = now
        return False

    # Size is unchanged since last poll — check how long it's been stable
    stable_secs = now - entry["last_size_change"]
    if stable_secs < MIN_STABLE_SECS:
        log(f"  [WATCH] {tif_path.name} size stable for {stable_secs:.0f}s "
            f"(need {MIN_STABLE_SECS}s). Waiting...")
        return False

    # File is stable and old enough — ready to process
    log(f"  [WATCH] {tif_path.name} transfer complete "
        f"(stable for {stable_secs:.0f}s, size={current_size/1e9:.2f} GB).")
    return True


def session_name_from_tif(tif_path: Path) -> str:
    return tif_path.stem


def setup_session_folder(task_dir: Path, tif_path: Path) -> Path:
    """
    Create session folder and move tif inside it.
    If the tif is already inside the correct session folder (retry case),
    skip the move.
    """
    session_name = session_name_from_tif(tif_path)
    session_dir  = task_dir / session_name
    session_dir.mkdir(exist_ok=True)

    dest = session_dir / tif_path.name
    if tif_path.resolve() != dest.resolve():
        if not dest.exists():
            shutil.move(str(tif_path), str(dest))
            log(f"  [SETUP] Moved {tif_path.name} → {session_dir.name}/")
    else:
        log(f"  [SETUP] {tif_path.name} already in session folder (retry).")

    # Remove from pending tracker now that we've committed to processing
    _pending.pop(tif_path, None)
    return session_dir


def write_session_log(session_dir: Path, content: str):
    with open(session_dir / "agent_run.log", "a", encoding="utf-8") as f:
        f.write(content + "\n")


def session_log(session_dir: Path):
    """Log function that writes to both the main log and session-specific log."""
    def _log(msg: str):
        log(msg)
        write_session_log(session_dir, f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}")
    return _log


def is_already_processed(session_dir: Path) -> bool:
    """A session is done if neuron.mat exists (written at end of headless run)."""
    return (session_dir / "neuron.mat").exists()


def write_review_launcher(session_dir: Path, session_name: str):
    """
    Write the self-locating run_final_review.m into the session folder.
    User opens MATLAB, navigates to session folder, runs this to start review.
    """
    from review_prep import write_launcher
    write_launcher(session_dir, REPO_ROOT)


def process_session(task_dir: Path, tif_path: Path):
    """Full pipeline for one new, transfer-complete .tif file."""
    session_name = session_name_from_tif(tif_path)
    log(f"\n{'='*60}")
    log(f"[AGENT] Starting: {task_dir.name}/{session_name}")
    log(f"{'='*60}")

    # 1. Create session folder, move tif
    session_dir = setup_session_folder(task_dir, tif_path)
    slog = session_log(session_dir)
    slog(f"[AGENT] Processing started")

    # 2. gSig / gSiz from animal config
    gSig, gSiz = param_estimator.get_gsig_gsiz(session_name, slog)

    # 3. Pre-pass: MATLAB computes Cn + pnr only (~1-2 min)
    pre_ok = run_cnmfe.run_pre_pass(session_name, session_dir, gSig, gSiz, slog)
    if not pre_ok:
        slog("[AGENT] Pre-pass failed. Check agent_run.log and MATLAB output.")
        return

    # 4. Estimate min_corr, min_pnr, bd from saved Cn and pnr images
    all_params = param_estimator.estimate_all_params(session_name, session_dir, slog)

    # 5. Full headless CNMFe run
    full_ok = run_cnmfe.run_full_cnmfe(session_name, session_dir, all_params, slog)
    if not full_ok:
        slog("[AGENT] Full CNMFe run failed. Check agent_run.log.")
        return

    # Count candidates — use spatial_footprints.mat (fast binary) not A.txt (slow text)
    n_candidates = "unknown"
    sf_file = session_dir / "spatial_footprints.mat"
    if sf_file.exists():
        import scipy.io as sio
        n_candidates = sio.loadmat(str(sf_file))["spatial_footprints"].shape[0]
    slog(f"[AGENT] Headless run complete: {n_candidates} candidate neurons.")

    # 6. Feature extraction + auto-curation (Phase 2)
    try:
        import curator
        curator.prepare_review_package(session_name, session_dir, slog)
    except ImportError:
        slog("[AGENT] curator.py not yet available — all candidates passed to review.")
        _write_passthrough_review(session_dir, n_candidates, slog)
    except Exception as e:
        import traceback
        slog(f"[AGENT] Curator error: {e}\n{traceback.format_exc()}")
        slog("[AGENT] Falling back to passthrough review (all candidates).")
        _write_passthrough_review(session_dir, n_candidates, slog)

    # 7. Write MATLAB launcher for final review
    write_review_launcher(session_dir, session_name)

    # 8. Pre-compute background weights so run_final_review.m starts fast
    _precompute_bg(session_dir, slog)

    slog(f"[AGENT] Ready. Open MATLAB and run: run_final_review.m")
    slog(f"[AGENT] See review_summary.txt for details.")
    log(f"[AGENT] Session ready for your review: {session_name}\n")


def _write_passthrough_review(session_dir: Path, n_candidates, slog):
    """
    Before classifier is trained: copy neuron.mat → review_neuron.mat
    and write a basic summary. User reviews all candidates.
    """
    src = session_dir / "neuron.mat"
    dst = session_dir / "review_neuron.mat"
    if src.exists() and not dst.exists():
        shutil.copy(str(src), str(dst))
        slog("[AGENT] review_neuron.mat = all candidates (classifier not yet trained).")

    summary = (
        f"Session: {session_dir.name}\n"
        f"{'='*50}\n"
        f"Candidate neurons:  {n_candidates}\n"
        f"Auto-rejected:      0  (classifier not yet trained)\n"
        f"Awaiting review:    {n_candidates}  (all candidates)\n"
        f"\n"
        f"NOTE: This is your first headless session. After you complete\n"
        f"your review in MATLAB (run_final_review.m), the agent will use\n"
        f"your keep/delete decisions to train the classifier. Future sessions\n"
        f"will have auto-rejected neurons pre-filtered for you.\n"
    )
    (session_dir / "review_summary.txt").write_text(summary)
    slog("[AGENT] Wrote review_summary.txt")


# ---- Background pre-computation ----

_PRECOMPUTE_SCRIPT = REPO_ROOT / "CNMFe_precompute_BG.m"


def _precompute_bg(session_dir: Path, slog):
    """
    Run CNMFe_precompute_BG.m to cache Ybg_weights.mat for this session.
    Saves ~10 min from the user's interactive run_final_review session.
    Non-fatal: logs warning on failure; user falls back to slow path.
    """
    sd = str(session_dir).replace("\\", "/")
    script = f"session_dir='{sd}'; run('{_PRECOMPUTE_SCRIPT.as_posix()}')"
    slog("[PRECOMPUTE] Pre-computing background weights (~10-15 min)...")
    ok = run_cnmfe._run_matlab(script, slog, timeout_hours=0.5)
    if ok:
        slog("[PRECOMPUTE] Ybg_weights.mat saved — run_final_review will start fast.")
    else:
        slog("[PRECOMPUTE] WARNING: precompute failed — run_final_review will estimate "
             "background interactively (~10 min).")


def find_precompute_pending() -> list[tuple[Path, Path]]:
    """
    Find watcher-processed sessions that are awaiting user review and do not yet
    have pre-computed background weights.

    Criteria (all must be true):
      - ROIs_candidates.jpg exists  → watcher/headless processed this session
      - neuron.mat exists           → headless run complete
      - review_neuron.mat exists    → curator done
      - ROIs.jpg does NOT exist     → user hasn't completed final review yet
      - Ybg_weights.mat NOT exists  → pre-computation not done
    """
    found = []
    for task_dir in DATA_ROOT.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in task_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if not (session_dir / "ROIs_candidates.jpg").exists():
                continue  # not a watcher-processed session
            if not (session_dir / "neuron.mat").exists():
                continue  # headless not done
            if not (session_dir / "review_neuron.mat").exists():
                continue  # curator not done
            if (session_dir / "ROIs.jpg").exists():
                continue  # user already completed final review
            if (session_dir / "Ybg_weights.mat").exists():
                continue  # already pre-computed
            found.append((task_dir, session_dir))
    return found


# ---- Animal params auto-refresh ----

_PARAMS_JSON = AGENT_DIR / "config" / "animal_params.json"
_EXTRACT_SCRIPT = AGENT_DIR.parent / "agent" / "extract_params.m"


def _needs_params_update() -> bool:
    """
    Return True if any neuron.mat under DATA_ROOT is newer than animal_params.json.
    Pure Python filesystem stat — essentially free.
    """
    if not _PARAMS_JSON.exists():
        return True
    params_mtime = _PARAMS_JSON.stat().st_mtime
    for neuron_mat in DATA_ROOT.rglob("neuron.mat"):
        if neuron_mat.stat().st_mtime > params_mtime:
            return True
    return False


def _refresh_animal_params():
    """Run extract_params.m to rebuild animal_params.json from completed sessions."""
    log("[PARAMS] New neuron.mat detected — refreshing animal_params.json...")
    script = f"run('{_EXTRACT_SCRIPT.as_posix()}')"
    ok = run_cnmfe._run_matlab(script, log, timeout_hours=0.25)  # 15 min max
    if ok:
        log("[PARAMS] animal_params.json updated.")
    else:
        log("[PARAMS] WARNING: extract_params.m failed — keeping existing animal_params.json.")


# ---- Curator retry (for sessions where MATLAB succeeded but curator failed) ----

def find_curator_retries() -> list[tuple[Path, Path]]:
    """
    Find headless sessions where the full curator step hasn't completed yet.

    Criteria (all must be true):
      - neuron.mat exists           → MATLAB headless run finished
      - ROIs_candidates.jpg exists  → confirms this was a headless session
                                      (old interactive sessions have ROIs.jpg, not this)
      - review_report.pdf missing   → full curator never wrote its PDF
                                      (passthrough fallback does NOT write a PDF,
                                      so passthrough-only sessions are retried here)
      - ROIs.jpg missing            → user hasn't done final MATLAB review yet
    """
    found = []
    if not DATA_ROOT.exists():
        return found
    for task_dir in DATA_ROOT.iterdir():
        if not task_dir.is_dir():
            continue
        for item in task_dir.iterdir():
            if not item.is_dir():
                continue
            session_dir = item
            if not (session_dir / "neuron.mat").exists():
                continue
            if not (session_dir / "ROIs_candidates.jpg").exists():
                continue  # not a headless session
            if (session_dir / "review_report.pdf").exists():
                continue  # full curator already ran
            if (session_dir / "ROIs.jpg").exists():
                continue  # final MATLAB review already done
            found.append((task_dir, session_dir))
    return found


def _retry_curator_session(task_dir: Path, session_dir: Path):
    """
    Re-run only the curator step for a session where MATLAB already completed.
    Called when neuron.mat exists but review_neuron.mat is missing.
    """
    session_name = session_dir.name
    log(f"\n[AGENT] Curator retry: {task_dir.name}/{session_name}")
    slog = session_log(session_dir)
    slog("[AGENT] Curator retry: neuron.mat found but review_neuron.mat missing.")

    n_candidates = "unknown"
    sf_file = session_dir / "spatial_footprints.mat"
    if sf_file.exists():
        import scipy.io as sio
        n_candidates = sio.loadmat(str(sf_file))["spatial_footprints"].shape[0]

    try:
        import curator
        curator.prepare_review_package(session_name, session_dir, slog)
    except ImportError:
        slog("[AGENT] curator.py not available — passing all candidates to review.")
        _write_passthrough_review(session_dir, n_candidates, slog)
    except Exception as e:
        import traceback
        slog(f"[AGENT] Curator error: {e}\n{traceback.format_exc()}")
        slog("[AGENT] Falling back to passthrough review (all candidates).")
        _write_passthrough_review(session_dir, n_candidates, slog)

    write_review_launcher(session_dir, session_name)
    slog(f"[AGENT] Ready. Open MATLAB and run: run_final_review.m")
    log(f"[AGENT] Curator retry complete: {session_name}")


# ---- Classifier auto-retraining ----

_PYTHON_EXE = str(PYTHON_EXE)
from config import MODEL_DIR as _MODEL_DIR
_TRAIN_SCRIPT = AGENT_DIR / "train_classifier.py"


def _needs_classifier_update() -> bool:
    """
    Return True if any session has both candidate_features.npz and labels.mat
    and the labels.mat is newer than the current classifier.joblib.
    This is a cheap filesystem-stat-only check.
    """
    clf_path  = _MODEL_DIR / "classifier.joblib"
    clf_mtime = clf_path.stat().st_mtime if clf_path.exists() else 0.0
    if not DATA_ROOT.exists():
        return False
    for task_dir in DATA_ROOT.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in task_dir.iterdir():
            if not session_dir.is_dir():
                continue
            labels_f   = session_dir / "labels.mat"
            features_f = session_dir / "candidate_features.npz"
            if labels_f.exists() and features_f.exists():
                if labels_f.stat().st_mtime > clf_mtime:
                    return True
    return False


def _retrain_classifier():
    """
    Retrain the classifier on all prospective labeled sessions.
    Runs train_classifier.py --prospective-only as a subprocess so retro
    MATLAB extraction (which is slow) is skipped here; the user triggers
    that manually via `python train_classifier.py`.
    """
    import subprocess
    log("[CLASSIFIER] New labels detected — retraining classifier "
        "(prospective sessions only)...")
    try:
        result = subprocess.run(
            [_PYTHON_EXE, str(_TRAIN_SCRIPT), "--prospective-only",
             "--model", "xgboost"],   # skip 3-way CV; run train_classifier.py manually for full comparison
            capture_output=True,
            text=True,
            timeout=600,   # XGBoost on 105 sessions needs more headroom than LR did
        )
        for line in result.stdout.splitlines():
            if line.strip():
                log(f"[CLASSIFIER] {line}")
        if result.returncode == 0:
            log("[CLASSIFIER] Classifier updated successfully.")
        else:
            log(f"[CLASSIFIER] Training failed (exit code {result.returncode}).")
            for line in result.stderr.splitlines():
                if line.strip():
                    log(f"[CLASSIFIER] {line}")
    except subprocess.TimeoutExpired:
        log("[CLASSIFIER] WARNING: Training timed out after 10 minutes.")
    except Exception as e:
        import traceback
        log(f"[CLASSIFIER] ERROR: {e}\n{traceback.format_exc()}")


# ---- Review queue ----

QUEUE_FILE = DATA_ROOT / "REVIEW_QUEUE.md"


def _read_neuron_count(session_dir: Path) -> str:
    """Extract the awaiting-review neuron count from review_summary.txt."""
    summary = session_dir / "review_summary.txt"
    if not summary.exists():
        return "?"
    try:
        for line in summary.read_text(encoding="utf-8").splitlines():
            if "Awaiting" in line and "review" in line and ":" in line:
                val = line.split(":", 1)[1].strip().split()[0]
                return val
    except Exception:
        pass
    return "?"


def _write_review_queue():
    """
    Write REVIEW_QUEUE.md to DATA_ROOT listing all watcher-processed sessions
    grouped by status.  Called every main loop iteration — cheap filesystem scan.
    """
    from datetime import datetime

    pending         = []   # review_neuron.mat exists, ROIs.jpg absent
    complete_agent  = []   # ROIs.jpg + ROIs_candidates.jpg (watcher-reviewed)
    complete_manual = []   # ROIs.jpg only, no ROIs_candidates.jpg (pre-agent)
    running         = []   # headless or curator still in progress

    if not DATA_ROOT.exists():
        return

    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for session_dir in sorted(task_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            is_watcher = (session_dir / "ROIs_candidates.jpg").exists()
            is_done    = (session_dir / "ROIs.jpg").exists() and \
                         (session_dir / "neuron.mat").exists()
            if is_done and is_watcher:
                complete_agent.append((task_dir.name, session_dir))
            elif is_done:
                complete_manual.append((task_dir.name, session_dir))
            elif is_watcher and (session_dir / "review_neuron.mat").exists():
                pending.append((task_dir.name, session_dir))
            elif is_watcher:
                running.append((task_dir.name, session_dir))

    # Sessions handed out to reviewers carry a marker (push_review_bundle.py) so
    # the same file is not reviewed by two machines at once.
    from review_prep import read_assignment
    out_for_review = []
    _still_pending = []
    for task_name, sd in pending:
        info = read_assignment(sd)
        if info:
            out_for_review.append((task_name, sd, info))
        else:
            _still_pending.append((task_name, sd))
    pending = _still_pending

    lines = [
        "# CNMFe Review Queue",
        f"_Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
    ]

    if running:
        lines += [f"## Pipeline running  ({len(running)})", ""]
        for task_name, sd in running:
            lines.append(f"- {sd.name}  ({task_name})")
        lines.append("")

    if pending:
        lines += [
            f"## Awaiting your review  ({len(pending)} session(s))",
            "",
            "| Session | Task | Neurons | BG cache |",
            "|---------|------|---------|----------|",
        ]
        for task_name, sd in pending:
            n  = _read_neuron_count(sd)
            bg = "Ready" if (sd / "Ybg_weights.mat").exists() else "pre-computing"
            lines.append(f"| {sd.name} | {task_name} | {n} | {bg} |")
        lines += [
            "",
            "_To review: open MATLAB, cd to the session folder, run `run_final_review.m`_",
            "",
        ]
    else:
        lines += ["## Awaiting your review", "", "No sessions pending review.", ""]

    if out_for_review:
        lines += [
            f"## Out for review  ({len(out_for_review)} session(s))",
            "",
            "_Sent to a reviewer; don't hand these to another machine._",
            "",
            "| Session | Task | Assigned | To |",
            "|---------|------|----------|----|",
        ]
        for task_name, sd, info in out_for_review:
            lines.append(f"| {sd.name} | {task_name} | {info.get('assigned', '?')} | {info.get('to', '-')} |")
        lines.append("")

    # Agent-reviewed complete sessions — most recent first
    complete_agent.sort(key=lambda x: (x[1] / "ROIs.jpg").stat().st_mtime, reverse=True)
    lines += [f"## Complete — agent pipeline  ({len(complete_agent)} session(s))", ""]
    for task_name, sd in complete_agent:
        lines.append(f"- {sd.name}  ({task_name})")
    if not complete_agent:
        lines.append("_None yet._")
    lines.append("")

    # Pre-agent sessions — split by bootstrap status
    # A pre-agent session is "bootstrapped" when bootstrap_preagent.py has run on it:
    # it will have candidate_features.npz + labels.mat and will be included in
    # classifier training automatically (same prospective path as agent sessions).
    complete_manual.sort(key=lambda x: x[1].name)
    bootstrapped     = [(t, sd) for t, sd in complete_manual
                        if (sd / "candidate_features.npz").exists()
                        and (sd / "labels.mat").exists()]
    not_bootstrapped = [(t, sd) for t, sd in complete_manual
                        if not ((sd / "candidate_features.npz").exists()
                                and (sd / "labels.mat").exists())]

    lines += [f"## Pre-agent — in training set  ({len(bootstrapped)} session(s))", ""]
    for task_name, sd in bootstrapped:
        lines.append(f"- {sd.name}  ({task_name})")
    if not bootstrapped:
        lines.append("_None yet.  Run `python bootstrap_preagent.py` to generate training labels._")
    lines.append("")

    lines += [f"## Pre-agent — not yet bootstrapped  ({len(not_bootstrapped)} session(s))", ""]
    for task_name, sd in not_bootstrapped:
        tif_present = any(sd.glob("*.tif")) or any(sd.glob("*.tiff"))
        marker = "" if tif_present else "  *(no .tif — skip)*"
        lines.append(f"- {sd.name}  ({task_name}){marker}")
    if not not_bootstrapped:
        lines.append("_All pre-agent sessions bootstrapped._")
    lines.append("")

    try:
        QUEUE_FILE.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        log(f"[QUEUE] WARNING: Could not write REVIEW_QUEUE.md: {e}")


# ---- Main loop ----

def main():
    log("=" * 60)
    log("[AGENT] CNMFe Agent started.")
    log(f"[AGENT] Watching: {DATA_ROOT}")
    log(f"[AGENT] Poll interval: {POLL_SECS}s | "
        f"Transfer stability threshold: {MIN_STABLE_SECS}s")
    log(f"[AGENT] Review queue: {QUEUE_FILE}")
    log("=" * 60)

    while True:
        # Auto-refresh animal_params.json if any neuron.mat is newer (cheap check).
        if _needs_params_update():
            _refresh_animal_params()

        all_tifs = find_new_tifs()

        ready    = []
        waiting  = []

        for task_dir, tif_path in all_tifs:
            if is_transfer_complete(tif_path):
                ready.append((task_dir, tif_path))
            else:
                waiting.append(tif_path.name)

        if waiting:
            log(f"[WATCH] Waiting on transfer: {', '.join(waiting)}")

        if ready:
            log(f"[WATCH] {len(ready)} file(s) ready to process.")
            for task_dir, tif_path in ready:
                try:
                    process_session(task_dir, tif_path)
                except Exception as e:
                    import traceback
                    log(f"[ERROR] {tif_path.name}: {e}\n{traceback.format_exc()}")
        elif not waiting:
            log(f"[WATCH] No new .tif files. Next check in {POLL_SECS}s.")

        # Check for sessions where MATLAB ran but curator failed (no review_neuron.mat)
        retries = find_curator_retries()
        if retries:
            log(f"[WATCH] {len(retries)} session(s) need curator retry.")
            for task_dir, session_dir in retries:
                try:
                    _retry_curator_session(task_dir, session_dir)
                except Exception as e:
                    import traceback
                    log(f"[ERROR] Curator retry {session_dir.name}: {e}\n{traceback.format_exc()}")

        # Post-hoc: pre-compute BG weights for sessions already awaiting review
        pending_bg = find_precompute_pending()
        if pending_bg:
            log(f"[WATCH] {len(pending_bg)} session(s) need background pre-computation.")
            for task_dir, session_dir in pending_bg:
                slog = session_log(session_dir)
                try:
                    _precompute_bg(session_dir, slog)
                except Exception as e:
                    import traceback
                    log(f"[ERROR] Precompute BG {session_dir.name}: {e}\n{traceback.format_exc()}")

        # Auto-retrain classifier if any labels.mat is newer than the model
        if _needs_classifier_update():
            _retrain_classifier()

        # Update the human-readable review queue file
        _write_review_queue()

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
