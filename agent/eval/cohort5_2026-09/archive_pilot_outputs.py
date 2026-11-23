"""
archive_pilot_outputs.py -- prepare the 12 cohort-5 pilot sessions for a re-run.

The vCA1 watcher retries any session folder that holds its .tif but no
neuron.mat (watcher.find_new_tifs, case 2).  This script moves the outputs of
the first run (estimated min_corr, 85th percentile) into a dot-prefixed archive
folder inside each session so the watcher re-runs the session with the current
parameters (param_overrides.json: min_corr 0.75) while the old candidate set
stays available for the re-evaluation.  Dot folders are invisible to every
scanner (watcher, curator, queue, trainer, push).

Never touched: the .tif, the converted <session>.mat movie (so the re-run
skips the 2-min conversion), review_assigned.txt (scripts never delete it),
and agent_run.log (copied into the archive, the original keeps appending).
neuron.mat is moved LAST so the watcher cannot pick up a half-archived folder.

Dry run by default.  --go performs the moves.  Local disk only.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import local_config  # noqa: E402

TASK_DIR = Path(local_config.DATA_PARENT) / "vCA1" / "6odorDualDiffRew"
ARCHIVE = ".prev_est85_2026-10-05"
PILOT = [
    "AVG5x-TSeries-082626-pnb102-479um-3z-000",
    "AVG5x-TSeries-082626-pnb104-404um-3z-000",
    "AVG5x-TSeries-082626-pnb105-397um-35z-000",
    "AVG5x-TSeries-082626-pnb106-205um-4z-000",
    "AVG5x-TSeries-082626-pnb109-398um-26z-000",
    "AVG5x-TSeries-082626-pnb99-399um-3z-000",
    "AVG5x-TSeries-082726-pnb102-497um-3z-000",
    "AVG5x-TSeries-082726-pnb104-416um-3z-000",
    "AVG5x-TSeries-082726-pnb105-391um-35z-000",
    "AVG5x-TSeries-082726-pnb106-201um-4z-000",
    "AVG5x-TSeries-082726-pnb109-405um-25z-000",
    "AVG5x-TSeries-082826-pnb99-436um-3z-000-000",
]
MOVE_FILES = [
    "A.txt", "C.txt", "C_df.txt", "C_raw.txt", "S.txt", "Cnn.txt",
    "Cn.mat", "pnr.mat", "Coor.mat", "ROIs_candidates.jpg",
    "Ybg_mean.mat", "Ybg_weights.mat", "candidate_features.npz",
    "review_neuron.mat", "review_report.pdf", "review_summary.txt",
    "run_final_review.m", "spatial_footprints.mat", "clamp_warning.txt",
]
COPY_FILES = ["agent_run.log"]
LAST = "neuron.mat"
REVIEWED_MARKERS = ["labels.mat", "ROIs.jpg", "labels_provenance.txt"]


def plan(sd: Path):
    """Return (actions, reason_to_skip)."""
    name = sd.name
    if not sd.is_dir():
        return [], "session folder missing"
    if not (sd / f"{name}.tif").exists():
        return [], "no .tif inside the session folder (watcher could not retry it)"
    if not (sd / f"{name}.mat").exists():
        return [], "converted .mat movie missing (re-run would redo the conversion; check first)"
    if not (sd / LAST).exists():
        return [], "no neuron.mat (already archived or a run is in progress)"
    if (sd / ARCHIVE).exists():
        return [], f"{ARCHIVE} already exists"
    present = [m for m in REVIEWED_MARKERS if (sd / m).exists()]
    if present:
        return [], f"looks reviewed ({', '.join(present)}) -- refusing"
    actions = []
    for f in MOVE_FILES:
        if (sd / f).exists():
            actions.append(("move", sd / f))
    neurons_dir = sd / f"{name}_neurons"
    if neurons_dir.is_dir():
        actions.append(("move", neurons_dir))
    for f in COPY_FILES:
        if (sd / f).exists():
            actions.append(("copy", sd / f))
    actions.append(("move", sd / LAST))
    return actions, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="perform the moves (default: dry run)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these session names")
    args = ap.parse_args()
    names = args.only if args.only else PILOT
    unknown = [n for n in names if n not in PILOT]
    if unknown:
        sys.exit(f"ERROR: not pilot sessions: {unknown}")
    mode = "GO" if args.go else "DRY RUN"
    print(f"[{mode}] task folder: {TASK_DIR}")
    done = skipped = 0
    for name in names:
        sd = TASK_DIR / name
        actions, why = plan(sd)
        if why:
            print(f"\n[{name}] SKIP: {why}")
            skipped += 1
            continue
        total = sum(p.stat().st_size for _, p in actions if p.is_file())
        print(f"\n[{name}] {len(actions)} item(s), {total/1e6:.0f} MB -> {ARCHIVE}/")
        for verb, p in actions:
            print(f"   {verb:4s} {p.name}")
        if args.go:
            dst = sd / ARCHIVE
            dst.mkdir()
            for verb, p in actions:
                if verb == "copy":
                    shutil.copy2(str(p), str(dst / p.name))
                else:
                    shutil.move(str(p), str(dst / p.name))
            print(f"   archived; neuron.mat gone -> watcher will retry {name}")
        done += 1
    print(f"\n{mode}: {done} session(s) {'archived' if args.go else 'would be archived'}, {skipped} skipped.")
    if not args.go:
        print("Re-run with --go to perform the moves.")


if __name__ == "__main__":
    main()
