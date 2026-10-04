"""
training_common.py -- shared constants and helpers for the ACORN reviewer
training program (central-machine side).

The training program lets a trainee re-review sessions that an experienced
reviewer has already labelled, and scores the trainee against that reviewer's
labels (with the deployed classifier as a second opinion).  Everything the
central machine produces for it lives under DATA_PARENT/.training/ -- a
dot-prefixed folder that every pipeline scanner (watcher, ingest, trainer,
push) already skips.  Nothing in the training program writes into a
production session folder, into inbox/ or outbox/ on the exchange, or a
review_assigned.txt marker.

Central scripts: build_training_key.py, build_training_gallery.py,
push_training_bundle.py, summarize_training.py.  Trainee-side MATLAB code
lives in <repo>/training/ (pure MATLAB, no Python on reviewer machines).

Areas: BLA and vCA1 only (decided 2026-10-01).  Both run the 35-column v2
feature contract; the key builder refuses anything else.
"""
import csv
import importlib
import re
from datetime import datetime
from pathlib import Path

from local_config import DATA_PARENT, REPO_ROOT
from ingest_returns import read_provenance

TRAINING_DIRNAME = ".training"
KEY_NAME = "training_key.mat"
SPATIAL_NAME = "training_key_spatial.mat"     # written by the MATLAB augment step
TRAINING_MARKER = "TRAINING_SESSION.txt"       # must match ingest_returns.TRAINING_MARKER
TRAINING_AREAS = ("BLA", "vCA1")
_AREA_CONFIG_MODULE = {"BLA": "config", "vCA1": "config_vCA1"}

# A session qualifies for the answer-key pool only with all four files.
POOL_FILES = ("review_neuron.mat", "labels.mat",
              "candidate_features.npz", "labels_provenance.txt")

# The 13 base feature names in contract order (measured from every pool npz on
# 2026-10-01; features.extract_all returns them in this dict order).
BASE13 = ["area", "circularity", "eccentricity", "compactness", "max_weight",
          "weight_spread", "peak_snr", "transient_freq", "events_per_min",
          "baseline_stability", "skewness", "motion_correlation", "cn_correlation"]

# One row per attempt.  Single definition: the MATLAB side writes these columns
# in this order (training/acorn_training_progress_columns.m) and
# summarize_training.py validates returned files against it.  New columns are
# only ever APPENDED: the MATLAB appender pads an older file's rows and the
# summariser reads an older header as a prefix.
PROGRESS_COLUMNS = [
    "timestamp", "trainee", "area", "task", "session", "stage", "mode", "pass",
    "subset", "n_items", "n_visited", "n_scored", "n_contested",
    "n_ref_keep_scored", "n_ref_delete_scored", "agreement", "kappa",
    "false_keep", "false_delete", "motion_ref_n", "motion_tagged_m",
    "motion_deleted_any", "motion_false_tags", "n_flagged", "median_sec",
    "total_sec", "result_file", "repo_hash",
    "contested_with_ref", "contested_with_model", "n_corrected",
]

CURRICULUM_COLUMNS = [
    "area", "task", "session", "reviewer", "n_candidates", "n_review", "n_keep",
    "n_motion", "has_motion_field", "frac_contested", "frac_clear",
    "frac_near_boundary", "has_video", "score_kind", "spatial_ok", "key_path",
    "built_at",
]

PUSHED_COLUMNS = ["timestamp", "trainee", "area", "task", "session", "stage",
                  "with_video", "dest", "bytes"]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(name: str) -> str:
    """Folder-safe trainee name (same rule as push_review_bundle.safe_name)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("_")


def training_root(data_parent: Path | None = None) -> Path:
    return (data_parent or DATA_PARENT) / TRAINING_DIRNAME


def check_area(area: str) -> str:
    """Canonical area name, or ValueError naming the supported areas."""
    for a in TRAINING_AREAS:
        if a.lower() == (area or "").strip().lower():
            return a
    raise ValueError(f"area '{area}' is not part of the training program; "
                     f"supported areas: {', '.join(TRAINING_AREAS)}")


def area_config(area: str):
    """The area's config module (MODEL_DIR, AREA, FEATURE_VERSION)."""
    return importlib.import_module(_AREA_CONFIG_MODULE[check_area(area)])


def key_dir(area: str, task: str, session: str, root: Path | None = None) -> Path:
    return (root or training_root()) / "keys" / check_area(area) / task / session


def key_path(area: str, task: str, session: str, root: Path | None = None) -> Path:
    return key_dir(area, task, session, root) / KEY_NAME


def session_parts(session_dir: Path, data_parent: Path | None = None):
    """(area, task, session) of a DATA_PARENT/{area}/{task}/{session} folder."""
    rel = Path(session_dir).resolve().relative_to((data_parent or DATA_PARENT).resolve())
    if len(rel.parts) != 3:
        raise ValueError(f"{session_dir} is not at DATA_PARENT/<area>/<task>/<session> depth")
    return rel.parts[0], rel.parts[1], rel.parts[2]


def resolve_session(arg: str, data_parent: Path | None = None) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else ((data_parent or DATA_PARENT) / arg)


def iter_area_sessions(area: str, data_parent: Path | None = None):
    """Every DATA_PARENT/<area>/<task>/<session> folder, dot folders skipped."""
    area_dir = (data_parent or DATA_PARENT) / check_area(area)
    if not area_dir.is_dir():
        return
    for task_dir in sorted(area_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        for sd in sorted(task_dir.iterdir()):
            if sd.is_dir() and not sd.name.startswith("."):
                yield sd


def is_pool_session(sd: Path) -> bool:
    return all((sd / f).exists() for f in POOL_FILES)


def pool_sessions(area: str | None = None, reviewer: str | None = None,
                  data_parent: Path | None = None) -> list[Path]:
    """Answer-key candidates: pool sessions of the supported areas, optionally
    restricted to one reference reviewer (case-insensitive on provenance)."""
    areas = [check_area(area)] if area else list(TRAINING_AREAS)
    out = []
    for a in areas:
        for sd in iter_area_sessions(a, data_parent):
            if not is_pool_session(sd):
                continue
            if reviewer:
                who = read_provenance(sd) or ""
                if who.lower() != reviewer.strip().lower():
                    continue
            out.append(sd)
    return out


def training_launcher_text(repo_root=None) -> str:
    """Self-locating run_training.m (modelled on review_prep.launcher_text).
    ASCII only; no '%' needed."""
    repo = str(repo_root or REPO_ROOT).replace("\\", "/")
    return (
        "session_dir = fileparts(mfilename('fullpath'));\n"
        "if ~isempty(which('acorn_training'))\n"
        "    acorn_training(session_dir);\n"
        f"elseif isfile('{repo}/training/acorn_training.m')\n"
        f"    addpath(genpath('{repo}'));\n"
        "    acorn_training(session_dir);\n"
        "else\n"
        "    error(['acorn_training.m not found. Add the ACORN repo to your MATLAB path: ' ...\n"
        "           'addpath(genpath(<repo_root>)) -- see docs/TRAINING.md.']);\n"
        "end\n"
    )


def write_training_marker(dest_dir: Path, fields: dict) -> Path:
    """TRAINING_SESSION.txt: the bundle is a sandbox and must never be ingested."""
    lines = [f"{k}: {v}" for k, v in fields.items()]
    lines.append("NOTE: training sandbox -- never copy this folder into inbox/; "
                 "return training_results/ to training/<trainee>/returns/<session>/")
    p = Path(dest_dir) / TRAINING_MARKER
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    return p


def read_training_marker(path: Path) -> dict:
    info = {}
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line and not line.startswith("NOTE:"):
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
    except OSError:
        pass
    return info


def append_csv_row(path: Path, columns: list[str], row: dict) -> None:
    """Append one row (header written if the file is new).  ASCII, LF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})


def read_csv_rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
