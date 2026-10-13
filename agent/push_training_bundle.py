"""
push_training_bundle.py -- stage TRAINING bundles on the lab-server exchange.

Copy-only.  For each session it copies, into
    <exchange>/training/<trainee>/<area>/<task>/<session>/
the files the trainee-side MATLAB needs:
    review_neuron.mat          (required: the candidate set the reviewer saw)
    Cn.mat, pnr.mat, Ybg_weights.mat   (optional; Ybg_weights for the dress rehearsal)
    {session}.mat              (raw video; the default drill is the VIDEO drill, so keep it.
                                --no-video leaves only the static warm-up drill usable)
    training_key.mat           (required; from DATA_PARENT/.training/keys, built by
                                build_training_key.py)
    run_training.m             (generated, self-locating launcher)
    TRAINING_SESSION.txt       (marker: sandbox, never ingested)
--gallery also copies DATA_PARENT/.training/gallery/** to <exchange>/training/_shared/gallery/.

What it never does: write review_assigned.txt (training does not take a session out
of the review queue), write into the production session folder, write into inbox/
or outbox/, delete or move anything.  <exchange>/training/ must already exist --
the operator creates it by hand once; this script exits if it is missing.

Usage (central machine):
  python agent/push_training_bundle.py --from-plan --trainee Alice --gallery --dry-run
      (every session picked in .training/curriculum_plan.csv, stage from its tier:
       easy 1, medium 2, hard 3; --stage overrides all)
  python agent/push_training_bundle.py BLA\\<task>\\<session> --trainee Alice --stage 1 --dry-run
  python agent/push_training_bundle.py BLA\\<task>\\<session> vCA1\\<task>\\<session> --trainee Alice --stage 3
  python agent/push_training_bundle.py --gallery --trainee Alice     (gallery only)
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import DATA_PARENT, EXCHANGE_ROOT, REPO_ROOT
import training_common as tc

REQUIRED = ["review_neuron.mat"]
OPTIONAL = ["Cn.mat", "pnr.mat", "Ybg_weights.mat"]
TIER_STAGE = {"easy": 1, "medium": 2, "hard": 3}   # curriculum tier -> stage hint


def planned_sessions(plan_csv: Path, data_parent: Path) -> list[tuple[Path, int, str]]:
    """(session_dir, stage, label) for every row of curriculum_plan.csv with a
    non-empty 'pick', ordered by area then pick number."""
    rows = [r for r in tc.read_csv_rows(plan_csv) if (r.get("pick") or "").strip()]
    if not rows:
        raise ValueError(f"no picked sessions in {plan_csv}; run plan_training_curriculum.py first")
    rows.sort(key=lambda r: (r["area"], int(r["pick"])))
    out = []
    for r in rows:
        tier = (r.get("tier") or "").strip().lower()
        stage = TIER_STAGE.get(tier, 2)
        sd = data_parent / r["area"] / r["task"] / r["session"]
        out.append((sd, stage, f"{r['area']} pick {r['pick']} ({tier or '?'})"))
    return out


def parts_of(sd: Path, data_parent: Path):
    try:
        return tc.session_parts(sd, data_parent)
    except ValueError:
        p = sd.resolve().parts
        if len(p) < 3:
            raise
        return p[-3], p[-2], p[-1]


def reference_reviewer_of(key_path: Path) -> str:
    try:
        import scipy.io as sio
        m = sio.loadmat(str(key_path), variable_names=["reference_reviewer"])
        v = m.get("reference_reviewer")
        return str(v.ravel()[0]) if v is not None and v.size else ""
    except Exception:
        return ""


def copy_file(src: Path, dst_dir: Path, dry_run: bool) -> int:
    size = src.stat().st_size
    target = dst_dir / src.name
    if dry_run:
        print(f"  would copy {src.name} ({size/1e6:.1f} MB)")
        return size
    if target.exists() and target.stat().st_size == size:
        print(f"  skip (already staged): {src.name}")
        return 0
    print(f"  copy {src.name} ({size/1e6:.1f} MB) ...")
    shutil.copy2(str(src), str(target))
    return size


def stage_one(sd: Path, exchange: Path, trainee: str, stage, with_video: bool,
              dry_run: bool, training_root: Path, data_parent: Path, repo_root: Path):
    area, task, session = parts_of(sd, data_parent)
    area = tc.check_area(area)
    key = tc.key_path(area, task, session, training_root)
    if not key.exists():
        raise ValueError(f"no training key at {key}; build it first: "
                         f"python agent/build_training_key.py {area}\\{task}\\{session}")
    files = []
    for name in REQUIRED:
        f = sd / name
        if not f.exists():
            raise ValueError(f"required file missing: {f}")
        files.append(f)
    for name in OPTIONAL:
        f = sd / name
        if f.exists():
            files.append(f)
        else:
            print(f"  (optional) missing: {name}")
    video = sd / f"{sd.name}.mat"
    has_video = False
    if with_video:
        if video.exists():
            files.append(video)
            has_video = True
        else:
            print(f"  WARNING: raw video {video.name} not found -- no video pass / rehearsal possible")
    else:
        print("  --no-video: raw video NOT included -- only the static warm-up drill works on this bundle; "
              "the video drill (the training default), the workflow drill and the rehearsal need it.")
    files.append(key)

    dest = exchange / "training" / trainee / area / task / session
    print(f"  -> {dest}")
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in files:
        total += copy_file(f, dest, dry_run)
    gallery_index = exchange / "training" / "_shared" / "gallery" / "index.html"
    marker = {
        "trainee": trainee,
        "session": f"{area}/{task}/{session}",
        "reference_reviewer": reference_reviewer_of(key),
        "stage_hint": stage if stage is not None else "-",
        "has_video": int(has_video),
        "gallery": str(gallery_index),
        "pushed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "pushed_by": "push_training_bundle.py",
    }
    if dry_run:
        print("  would generate run_training.m and TRAINING_SESSION.txt")
    else:
        (dest / "run_training.m").write_text(tc.training_launcher_text(repo_root), encoding="ascii")
        tc.write_training_marker(dest, marker)
        tc.append_csv_row(training_root / trainee / "pushed.csv", tc.PUSHED_COLUMNS, {
            "timestamp": tc.now_str(), "trainee": trainee, "area": area, "task": task,
            "session": session, "stage": marker["stage_hint"], "with_video": int(has_video),
            "dest": str(dest), "bytes": total})
        print("  generated run_training.m + TRAINING_SESSION.txt")
    print(f"  {'would stage' if dry_run else 'staged'} {total/1e6:.1f} MB")
    return dest, total


def push_gallery(exchange: Path, training_root: Path, dry_run: bool) -> int:
    src = training_root / "gallery"
    if not (src / "index.html").exists():
        raise ValueError(f"no gallery at {src}; run build_training_gallery.py first")
    dest = exchange / "training" / "_shared" / "gallery"
    print(f"[gallery] {src} -> {dest}")
    total = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        d = dest / rel.parent
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)
        total += copy_file(f, d, dry_run)
    print(f"  {'would stage' if dry_run else 'staged'} {total/1e6:.1f} MB of gallery")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="area\\task\\session (relative to DATA_PARENT) or absolute")
    ap.add_argument("--trainee", required=True, help="trainee name -> training/<name>/")
    ap.add_argument("--stage", type=int, help="stage hint written into TRAINING_SESSION.txt (1-4); "
                                              "with --from-plan it overrides the tier-derived stage")
    ap.add_argument("--from-plan", action="store_true",
                    help="push every session picked in .training/curriculum_plan.csv (stage from tier)")
    ap.add_argument("--plan", default=None, help="plan csv (default .training/curriculum_plan.csv)")
    ap.add_argument("--no-video", action="store_true", help="omit the raw {session}.mat")
    ap.add_argument("--gallery", action="store_true", help="also push .training/gallery to training/_shared/gallery/")
    ap.add_argument("--exchange", default=EXCHANGE_ROOT, help="exchange root (default CNMFE_EXCHANGE_ROOT / .env)")
    ap.add_argument("--training-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--data-parent", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.exchange:
        sys.exit("ERROR: exchange root not set. Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.")
    exchange = Path(args.exchange)
    if not exchange.is_dir():
        sys.exit(f"ERROR: exchange root not reachable: {exchange}")
    tdir = exchange / "training"
    if not tdir.is_dir():
        sys.exit(f"ERROR: {tdir} does not exist. The operator must create it (and training\\_shared\\) "
                 f"by hand on the server before the first push; this script never creates it.")
    trainee = tc.safe_name(args.trainee)
    if not trainee:
        sys.exit("ERROR: --trainee must be a usable folder name")
    training_root = Path(args.training_root) if args.training_root else tc.training_root()
    data_parent = Path(args.data_parent) if args.data_parent else DATA_PARENT
    jobs = [(tc.resolve_session(s, data_parent), args.stage, s) for s in args.sessions]
    if args.from_plan:
        plan = Path(args.plan) if args.plan else training_root / "curriculum_plan.csv"
        if not plan.exists():
            sys.exit(f"ERROR: {plan} not found; run agent/plan_training_curriculum.py first")
        try:
            planned = planned_sessions(plan, data_parent)
        except ValueError as e:
            sys.exit(f"ERROR: {e}")
        print(f"plan: {plan} -> {len(planned)} picked session(s)")
        for sd, stage, label in planned:
            if args.stage is not None:
                stage = args.stage
            print(f"  [plan] {label}: {sd.parent.name}\\{sd.name} -> stage {stage}")
            jobs.append((sd, stage, label))
    if not jobs and not args.gallery:
        sys.exit("ERROR: give at least one session, --from-plan, or --gallery")

    print(f"{'DRY RUN: ' if args.dry_run else ''}trainee {trainee}; exchange {exchange}")
    failures = []
    for sd, stage, label in jobs:
        if not sd.is_dir():
            failures.append((label, "session folder not found"))
            print(f"[{label}]\n  ERROR: session folder not found: {sd}")
            continue
        print(f"[{label}]")
        try:
            stage_one(sd, exchange, trainee, stage, not args.no_video, args.dry_run,
                      training_root, data_parent, REPO_ROOT)
        except Exception as e:
            failures.append((label, str(e)))
            print(f"  ERROR: {e}")
        print()
    if args.gallery:
        try:
            push_gallery(exchange, training_root, args.dry_run)
        except Exception as e:
            failures.append(("gallery", str(e)))
            print(f"  ERROR: {e}")
    if failures:
        print(f"{len(failures)} item(s) not staged:")
        for s, e in failures:
            print(f"  - {s}: {e}")
        return 1
    print("Done. Tell the trainee: pull training\\<name>\\... to a local disk, run run_training.m; "
          "return training_results\\ to training\\<name>\\returns\\<session>\\.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
