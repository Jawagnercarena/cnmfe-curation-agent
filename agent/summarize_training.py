"""
summarize_training.py -- bring trainee results home and summarise them.

Server READ-ONLY.  Scans <exchange>/training/<trainee>/returns/ for
training_results/ folders (what the trainee copied back), copies their result
files (*.csv, *.mat, *.html, *.png, *.txt -- never a raw video) into
    DATA_PARENT/.training/<trainee>/returns/<session>/training_results/
then concatenates every progress.csv into .training/progress_all.csv, writes
.training/summary.md and prints one table per trainee.  Numbers only -- the
supervisor decides what counts as progress.  Exits gracefully when the
exchange has no training/ folder yet.

Usage:
  python agent/summarize_training.py [--trainee NAME] [--exchange ROOT] [--dry-run]
"""
import argparse
import shutil
import statistics
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))
from local_config import EXCHANGE_ROOT
import training_common as tc

RESULT_EXT = {".csv", ".mat", ".html", ".png", ".txt"}
# mode tags are drill_<video|workflow|static>_<coach|exam> or rehearsal; the
# summary lists whichever appear, in this preference order
MODE_ORDER = ["drill_video_coach", "drill_video_exam", "drill_workflow_coach", "drill_workflow_exam",
              "drill_static_coach", "drill_static_exam", "drill_coach", "drill_exam", "rehearsal"]


def modes_in(rows: list[dict]) -> list[str]:
    seen = {r["mode"] for r in rows}
    return [m for m in MODE_ORDER if m in seen] + sorted(m for m in seen if m not in MODE_ORDER)


def fetch_returns(exchange: Path, training_root: Path, only: str | None, dry_run: bool) -> list[tuple[str, Path]]:
    """Copy result files home.  Returns [(trainee, local training_results dir)]."""
    tdir = exchange / "training"
    found = []
    for trainee_dir in sorted(tdir.iterdir()):
        if not trainee_dir.is_dir() or trainee_dir.name.startswith((".", "_")):
            continue
        trainee = trainee_dir.name
        if only and trainee.lower() != only.lower():
            continue
        returns = trainee_dir / "returns"
        if not returns.is_dir():
            continue
        for tr in sorted(returns.rglob("training_results")):
            if not tr.is_dir():
                continue
            session = tr.parent.name
            local = training_root / trainee / "returns" / session / "training_results"
            n_copied = 0
            for f in sorted(tr.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in RESULT_EXT:
                    continue
                rel = f.relative_to(tr)
                target = local / rel
                if target.exists() and target.stat().st_size == f.stat().st_size:
                    continue
                if dry_run:
                    print(f"  would copy {trainee}/{session}/{rel}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(target))
                n_copied += 1
            print(f"  {trainee}/{session}: {n_copied} file(s) {'to copy' if dry_run else 'copied'}")
            found.append((trainee, local))
    return found


def read_progress(training_root: Path, only: str | None) -> list[dict]:
    rows = []
    for csv in sorted((training_root).rglob("progress.csv")):
        try:
            rel = csv.relative_to(training_root)
        except ValueError:
            continue
        if len(rel.parts) < 2 or rel.parts[1] != "returns":
            continue
        trainee = rel.parts[0]
        if only and trainee.lower() != only.lower():
            continue
        with open(csv, encoding="utf-8", errors="replace", newline="") as fh:
            header = fh.readline().strip().split(",")
        is_prefix = (len(header) < len(tc.PROGRESS_COLUMNS)
                     and header == tc.PROGRESS_COLUMNS[:len(header)])
        if header != tc.PROGRESS_COLUMNS and not is_prefix:
            print(f"  WARN {csv}: header differs from PROGRESS_COLUMNS; skipped")
            continue
        if is_prefix:
            print(f"  note {csv}: older header ({len(header)} columns); missing columns read as empty")
        for r in tc.read_csv_rows(csv):
            r["_file"] = str(csv)
            rows.append(r)
    seen = set()
    out = []
    for r in rows:
        k = (r["trainee"], r["result_file"], r["pass"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    out.sort(key=lambda r: (r["trainee"], r["timestamp"]))
    return out


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def attempt_rows(rows: list[dict]) -> list[dict]:
    """One row per attempt: the 'final' row when present, else pass 1."""
    by = {}
    for r in rows:
        by.setdefault((r["trainee"], r["result_file"]), []).append(r)
    out = []
    for k, rs in by.items():
        fin = [r for r in rs if r["pass"] == "final"]
        p1 = [r for r in rs if r["pass"] == "1"]
        out.append((fin or p1 or rs)[0])
    out.sort(key=lambda r: (r["trainee"], r["timestamp"]))
    return out


def fmt(v, nd=3):
    if v != v:
        return "-"
    return f"{v:.{nd}f}"


def summarise(rows: list[dict]) -> tuple[str, str]:
    att = attempt_rows(rows)
    trainees = sorted({r["trainee"] for r in att})
    md = ["# ACORN reviewer training -- summary", "", f"Generated {tc.now_str()}. "
          "One row per attempt (final row of a video drill, else pass 1). "
          "Rates: false keep = kept / reference deletes among scored; false delete = deleted / reference keeps among scored.", ""]
    txt = []
    for t in trainees:
        rs = [r for r in att if r["trainee"] == t]
        sessions = sorted({f"{r['area']}/{r['session']}" for r in rs})
        stages = [fnum(r["stage"]) for r in rs]
        stages = [s for s in stages if s == s]
        md.append(f"## {t}")
        md.append(f"- attempts: {len(rs)}; sessions: {len(sessions)}; highest stage attempted: "
                  f"{int(max(stages)) if stages else '-'}")
        med = [fnum(r["median_sec"]) for r in rs]
        med = [m for m in med if m == m and m > 0]
        md.append(f"- median seconds per decision (over attempts): {fmt(statistics.median(med), 1) if med else '-'}")
        ag = [fnum(r["agreement"]) for r in rs]
        ag = [a for a in ag if a == a]
        if len(ag) >= 2:
            md.append(f"- agreement first -> latest: {fmt(ag[0])} -> {fmt(ag[-1])}; mean of last 3: {fmt(statistics.mean(ag[-3:]))}")
        for mode in modes_in(rs):
            ms = [r for r in rs if r["mode"] == mode]
            if not ms:
                continue
            r = ms[-1]
            fk = fnum(r["false_keep"]) / max(1.0, fnum(r["n_ref_delete_scored"]))
            fd = fnum(r["false_delete"]) / max(1.0, fnum(r["n_ref_keep_scored"]))
            g = lambda k: r.get(k, "") or "-"
            md.append(f"- latest {mode} ({r['timestamp']}, {r['area']}/{r['session']}, stage {r['stage'] or '-'}): "
                      f"agreement {fmt(fnum(r['agreement']))}, kappa {fmt(fnum(r['kappa']))}, "
                      f"false keep {r['false_keep']}/{r['n_ref_delete_scored']} ({fmt(fk)}), "
                      f"false delete {r['false_delete']}/{r['n_ref_keep_scored']} ({fmt(fd)}), "
                      f"m-tags on motion {r['motion_tagged_m']}/{r['motion_ref_n']}, flagged {r['n_flagged']}; "
                      f"contested (uncounted): with reference {g('contested_with_ref')}, with model "
                      f"{g('contested_with_model')} of {r['n_contested']}; changed after feedback {g('n_corrected')}")
        md.append("")
        md.append("| # | finished | area/session | stage | mode | shown | scored | contested | agreement | kappa | false keep | false delete | m-tags | contested ref/model | corrected | median s |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(rs, 1):
            g = lambda k: r.get(k, "") or "-"
            md.append(f"| {i} | {r['timestamp']} | {r['area']}/{r['session']} | {r['stage'] or '-'} | {r['mode']} | "
                      f"{r['n_items']} | {r['n_scored']} | {r['n_contested']} | {fmt(fnum(r['agreement']))} | "
                      f"{fmt(fnum(r['kappa']))} | {r['false_keep']}/{r['n_ref_delete_scored']} | "
                      f"{r['false_delete']}/{r['n_ref_keep_scored']} | {r['motion_tagged_m']}/{r['motion_ref_n']} | "
                      f"{g('contested_with_ref')}/{g('contested_with_model')} | {g('n_corrected')} | "
                      f"{fmt(fnum(r['median_sec']), 1)} |")
        md.append("")
        txt.append(f"{t}: {len(rs)} attempt(s), {len(sessions)} session(s)" +
                   (f", latest agreement {fmt(ag[-1])}" if ag else ""))
    if not trainees:
        md.append("No attempts found.")
        txt.append("No attempts found.")
    return "\n".join(md) + "\n", "\n".join(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainee", help="only this trainee")
    ap.add_argument("--exchange", default=EXCHANGE_ROOT)
    ap.add_argument("--training-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.exchange:
        sys.exit("ERROR: exchange root not set. Set CNMFE_EXCHANGE_ROOT (or agent/.env), or pass --exchange.")
    exchange = Path(args.exchange)
    training_root = Path(args.training_root) if args.training_root else tc.training_root()
    tdir = exchange / "training"
    if not tdir.is_dir():
        print(f"No training folder yet at {tdir} -- nothing to summarise. "
              f"(The operator creates it by hand before the first push.)")
        return 0
    print(f"{'DRY RUN: ' if args.dry_run else ''}reading returns under {tdir}")
    found = fetch_returns(exchange, training_root, args.trainee, args.dry_run)
    print(f"{len(found)} returned session folder(s)")
    if args.dry_run:
        print("DRY RUN: nothing copied or written.")
        return 0
    rows = read_progress(training_root, args.trainee)
    tc.write_csv_rows(training_root / "progress_all.csv", tc.PROGRESS_COLUMNS, rows)
    md, txt = summarise(rows)
    (training_root / "summary.md").write_text(md, encoding="ascii", errors="replace")
    print(txt)
    print(f"\nprogress_all.csv ({len(rows)} rows) and summary.md written under {training_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
