"""
prune_movie_mats.py -- delete the movie .mat of FINISHED sessions (local data root only).

<session>/<session>.mat (Y, Ysiz; 1-4 GB) is a frame-exact re-encoding of
<session>/<session>.tif, written by tif2mat at the start of every headless run.
Once a session is finished, everything that still needs it rebuilds it from the
.tif on demand (CNMFe_final_save.m, push_review_bundle.py, rereview.py request,
push_training_bundle.py -- see movie_mat.py), so it can go.

DRY RUN BY DEFAULT.  Nothing is deleted without --apply.

A session movie DATA_PARENT/<area>/<task>/<session>/<session>.mat (dot folders
skipped at every level) is deleted only when ALL of these hold:
  1. finished: labels.mat and neuron.mat are present;
  2. not a training sandbox (TRAINING_SESSION.txt);
  3. its source <session>.tif (or .tiff, exact folder name) is present;
  4. the watcher would not pick the session up for a background pre-compute
     (ROIs_candidates.jpg + review_neuron.mat, no ROIs.jpg, no Ybg_weights.mat):
     CNMFe_precompute_BG.m reads the movie and does not rebuild it;
  5. the animal is not tracked by the retro project (a folder under its data/):
     that project's review GUI and projections read the movie .mat directly --
     unless --include-retro;
  6. the movie was last written more than --min-age-hours ago (default 24), so a
     movie rebuilt for a push in progress is left alone;
  7. the .tif reproduces it: same frame count, frame size and pixel type, and
     --frames sampled frames (default 32, first and last included) identical;
     --full compares every frame;
  8. it is unchanged since step 7 and not open in another process (Windows
     refuses the delete; reported, not retried).
Training-cache movies (DATA_PARENT/.training/movies/<area>/<task>/<session>/
<session>.mat, written by push_training_bundle.py) are cleared by the same rules
6-8 against the production session's .tif.

Only a file named exactly <folder>/<folder>.mat is ever deleted -- never a .tif,
never any other file, never a folder.  A big .mat under another name (a renamed
session's movie) is listed and left alone.  Refuses to run when DATA_PARENT is a
network path or a mapped network drive.  Every run writes a manifest CSV, one row
per movie considered, to DATA_PARENT/.movie_prune/.

Usage:
  python prune_movie_mats.py                          # dry run, everything
  python prune_movie_mats.py --area vCA1 --task 3odor # dry run, one task
  python prune_movie_mats.py --apply --limit 3        # first outing: three movies
  python prune_movie_mats.py --apply
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, RETRO_RESULTS
import movie_mat
import training_common as tc
from push_review_bundle import animal_of

OTHER_MOVIE_BYTES = 200_000_000     # a .mat this big under another name: listed, never touched
MANIFEST_DIRNAME = ".movie_prune"
FIELDS = ["path", "decision", "reason", "movie_gb", "tif", "verify", "movie_written"]


# ---------------------------------------------------------------- guards ----

def is_local_dir(p: Path) -> bool:
    """True only for a folder on a local fixed/removable drive (never UNC, never a
    mapped network drive)."""
    raw = str(p)
    if raw.startswith("\\\\") or raw.startswith("//"):
        return False
    try:
        s = str(Path(p).resolve())
    except OSError:
        return False
    if s.startswith("\\\\"):
        return False
    if os.name == "nt":
        import ctypes
        drive = os.path.splitdrive(s)[0]
        if not drive:
            return False
        kind = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")
        return kind in (2, 3)           # DRIVE_REMOVABLE, DRIVE_FIXED (4 = DRIVE_REMOTE)
    return True


def retro_animals(results_root) -> tuple[set, str]:
    """Animals the retro project tracks (folders under <retro root>/data), and a note."""
    if not results_root:
        return set(), "retro results root not set: no retro exclusion"
    data = Path(results_root).parent / "data"
    if not data.is_dir():
        return set(), f"retro data tree {data} not found: no retro exclusion"
    names = {d.name.lower() for d in data.iterdir() if d.is_dir() and not d.name.startswith(".")}
    return names, f"retro-tracked animals ({data}): {', '.join(sorted(names)) or 'none'}"


# ----------------------------------------------------------------- scan -----

def iter_sessions(data_parent: Path, area=None, task=None):
    for a in sorted(data_parent.iterdir()):
        if not a.is_dir() or a.name.startswith("."):
            continue
        if area and a.name.lower() != area.lower():
            continue
        for t in sorted(a.iterdir()):
            if not t.is_dir() or t.name.startswith("."):
                continue
            if task and t.name.lower() != task.lower():
                continue
            for sd in sorted(t.iterdir()):
                if sd.is_dir() and not sd.name.startswith("."):
                    yield sd


def iter_training_cache(data_parent: Path, area=None, task=None):
    """(cached movie, production session dir) pairs under .training/movies."""
    root = tc.training_root(data_parent) / "movies"
    if not root.is_dir():
        return
    for mat in sorted(root.glob("*/*/*/*.mat")):
        a, t, s = mat.parent.parent.parent.name, mat.parent.parent.name, mat.parent.name
        if mat.name != f"{s}.mat":
            continue
        if area and a.lower() != area.lower():
            continue
        if task and t.lower() != task.lower():
            continue
        yield mat, data_parent / a / t / s


def other_movies(sd: Path):
    """Big .mat files in the session folder under a name other than the folder's."""
    out = []
    for f in sd.iterdir():
        if (f.is_file() and f.suffix.lower() == ".mat" and f.stem != sd.name
                and f.stat().st_size >= OTHER_MOVIE_BYTES):
            out.append(f)
    return sorted(out)


def session_skip_reason(sd: Path, now: float, min_age_s: float, retro: set,
                        include_retro: bool) -> str | None:
    """Why this session's movie must stay (cheap checks only), or None."""
    has = lambda n: (sd / n).exists()   # noqa: E731
    if not has("labels.mat"):
        return "not finished (no labels.mat)"
    if not has("neuron.mat"):
        return "no neuron.mat"
    if has(tc.TRAINING_MARKER):
        return "training sandbox"
    if movie_mat.movie_tif(sd) is None:
        return f"no {sd.name}.tif to rebuild it from"
    if (has("ROIs_candidates.jpg") and has("review_neuron.mat")
            and not has("ROIs.jpg") and not has("Ybg_weights.mat")):
        return "watcher would pre-compute Ybg_weights.mat from it (no ROIs.jpg, no Ybg_weights.mat)"
    animal = animal_of(sd.name)
    if animal and animal in retro and not include_retro:
        return f"retro-tracked animal {animal} (its review GUI reads the movie; --include-retro)"
    return age_skip_reason(movie_mat.movie_mat(sd), now, min_age_s)


def age_skip_reason(mat: Path, now: float, min_age_s: float) -> str | None:
    age = now - mat.stat().st_mtime
    if age < min_age_s:
        return f"written {age / 3600:.1f} h ago (younger than --min-age-hours)"
    return None


def delete_movie(mat: Path, data_parent: Path, tif: Path, verified_stat) -> str | None:
    """Delete exactly this movie; return None on success, else why not."""
    rp, root = mat.resolve(), data_parent.resolve()
    if root not in rp.parents:
        return "outside the data root"
    if rp.suffix.lower() != ".mat" or rp.name != f"{rp.parent.name}.mat":
        return "not a <folder>/<folder>.mat movie"
    st = mat.stat()
    if (st.st_size, st.st_mtime_ns) != verified_stat:
        return "changed since it was verified"
    if not tif.is_file():
        return "its .tif disappeared"
    try:
        mat.unlink()
    except OSError as e:
        return f"could not delete ({type(e).__name__}: {e})"
    return None


# ------------------------------------------------------------------ run -----

def run(data_parent: Path, apply: bool = False, area=None, task=None, sessions=None,
        limit=None, min_age_hours: float = 24.0, n_frames: int = movie_mat.VERIFY_FRAMES,
        full: bool = False, include_retro: bool = False, retro_root=None,
        log=print) -> tuple[list[dict], Path]:
    """Scan, verify and (with apply) delete.  Returns (manifest rows, manifest path).
    retro_root defaults to local_config.RETRO_RESULTS."""
    data_parent = Path(data_parent)
    if retro_root is None:
        retro_root = RETRO_RESULTS
    if not data_parent.is_dir():
        raise SystemExit(f"ERROR: data root not found: {data_parent}")
    if not is_local_dir(data_parent):
        raise SystemExit(f"ERROR: {data_parent} is not on a local drive; this script only "
                         f"deletes on local storage.")
    retro, retro_note = retro_animals(retro_root)
    wanted = None
    if sessions:
        wanted = set()
        for s in sessions:
            p = Path(s)
            wanted.add(str((p if p.is_absolute() else data_parent / p).resolve()).lower())
    now = time.time()
    min_age_s = min_age_hours * 3600

    mdir = data_parent / MANIFEST_DIRNAME
    mdir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = mdir / f"{'apply' if apply else 'dryrun'}_{stamp}.csv"

    mode = "APPLY (deletes)" if apply else "DRY RUN (deletes nothing)"
    log(f"{mode}  data root {data_parent}")
    log(f"  {retro_note}{' -- included (--include-retro)' if include_retro and retro else ''}")
    log(f"  min age {min_age_hours:g} h; verify {'every frame' if full else f'{n_frames} sampled frames'}"
        f"{f'; limit {limit}' if limit else ''}")
    log(f"  manifest {manifest}\n")

    rows = []
    done = 0
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()

        def record(**r):
            row = {k: r.get(k, "") for k in FIELDS}
            rows.append(row)
            w.writerow(row)
            fh.flush()
            return row

        def handle(mat: Path, tif: Path | None, reason: str | None, rel: str):
            nonlocal done
            st = mat.stat()
            base = dict(path=rel, movie_gb=f"{st.st_size / 1e9:.3f}",
                        tif=tif.name if tif else "",
                        movie_written=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"))
            if reason is None and limit is not None and done >= limit:
                reason = "beyond --limit"
            if reason is not None:
                record(decision="keep", reason=reason, **base)
                return
            ok, detail = movie_mat.verify_against_tif(mat, tif, n_frames=n_frames, full=full)
            if not ok:
                record(decision="keep", reason="tif does not reproduce it", verify=detail, **base)
                log(f"  KEEP {rel}: tif does not reproduce it ({detail})")
                return
            if not apply:
                done += 1
                record(decision="would delete", verify=detail, **base)
                return
            err = delete_movie(mat, data_parent, tif, (st.st_size, st.st_mtime_ns))
            if err:
                record(decision="keep", reason=err, verify=detail, **base)
                log(f"  KEEP {rel}: {err}")
            else:
                done += 1
                record(decision="deleted", verify=detail, **base)
                log(f"  deleted {rel} ({st.st_size / 1e9:.2f} GB; {detail})")

        for sd in iter_sessions(data_parent, area, task):
            if wanted is not None and str(sd.resolve()).lower() not in wanted:
                continue
            rel_dir = sd.relative_to(data_parent)
            for f in other_movies(sd):
                record(path=str(rel_dir / f.name), decision="keep",
                       reason="movie under another name: never touched",
                       movie_gb=f"{f.stat().st_size / 1e9:.3f}")
            if (sd / ".movie_rebuild").is_dir():
                record(path=str(rel_dir / ".movie_rebuild"), decision="keep",
                       reason="leftover of an interrupted rebuild: inspect by hand")
            mat = movie_mat.movie_mat(sd)
            if not mat.is_file():
                continue
            reason = session_skip_reason(sd, now, min_age_s, retro, include_retro)
            handle(mat, movie_mat.movie_tif(sd), reason, str(rel_dir / mat.name))

        for mat, sd in iter_training_cache(data_parent, area, task):
            if wanted is not None and str(sd.resolve()).lower() not in wanted:
                continue
            tif = movie_mat.movie_tif(sd) if sd.is_dir() else None
            reason = (None if tif is not None else "production session has no .tif")
            reason = reason or age_skip_reason(mat, now, min_age_s)
            handle(mat, tif, reason, str(mat.relative_to(data_parent)))

    # ---- summary ----
    gb = defaultdict(float)
    n = Counter()
    reasons = Counter()
    for r in rows:
        n[r["decision"]] += 1
        gb[r["decision"]] += float(r["movie_gb"] or 0)
        if r["decision"] == "keep":
            key = r["reason"]
            if key.startswith("written ") and key.endswith("(younger than --min-age-hours)"):
                key = "younger than --min-age-hours"
            if key.startswith("retro-tracked animal"):
                key = "retro-tracked animal (--include-retro)"
            reasons[key] += 1
    log("\nSummary")
    for d in ("deleted", "would delete", "keep"):
        if n[d]:
            log(f"  {d:<13} {n[d]:>4} movie(s)  {gb[d]:8.1f} GB")
    for k, c in reasons.most_common():
        log(f"      keep: {c:>4}  {k}")
    log(f"\nmanifest: {manifest}")
    return rows, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="really delete (default: dry run)")
    ap.add_argument("--area", help="only this area (BLA, vCA1, DG_AL)")
    ap.add_argument("--task", help="only this task folder")
    ap.add_argument("--session", action="append",
                    help="only this session (area\\task\\session, or absolute); repeatable")
    ap.add_argument("--limit", type=int, help="stop after this many deletions (dry run: candidates)")
    ap.add_argument("--min-age-hours", type=float, default=24.0,
                    help="keep a movie written more recently than this (default 24)")
    ap.add_argument("--frames", type=int, default=movie_mat.VERIFY_FRAMES,
                    help=f"sampled frames compared with the tif (default {movie_mat.VERIFY_FRAMES})")
    ap.add_argument("--full", action="store_true", help="compare every frame (reads both files fully)")
    ap.add_argument("--include-retro", action="store_true",
                    help="also prune retro-tracked animals (their review GUI then has no movie)")
    ap.add_argument("--data-parent", help="data root (default: CNMFE_DATA_PARENT / local_config)")
    args = ap.parse_args(argv)
    rows, _ = run(Path(args.data_parent) if args.data_parent else DATA_PARENT,
                  apply=args.apply, area=args.area, task=args.task, sessions=args.session,
                  limit=args.limit, min_age_hours=args.min_age_hours, n_frames=args.frames,
                  full=args.full, include_retro=args.include_retro)
    failed = [r for r in rows if r["decision"] == "keep" and r["reason"].startswith(
        ("could not delete", "changed since", "its .tif disappeared"))]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
