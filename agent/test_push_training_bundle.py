"""Temp-dir test for push_training_bundle.py and summarize_training.py.

Everything lives under one temp folder: a fake session, a fake .training root
with a key and a gallery, and a fake exchange.  The real exchange is never
touched.
"""
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.io as sio
import push_training_bundle as pb
import summarize_training as st
import training_common as tc

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def run(mod, argv):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = mod.main(argv)
    except SystemExit as e:
        rc = e.code
        if isinstance(e.code, str):          # sys.exit("message") -> message on stderr, code 1
            buf.write(e.code)
            rc = 1
    return rc, buf.getvalue()


tmp = Path(tempfile.mkdtemp(prefix="push_training_"))
try:
    data = tmp / "data"
    sd = data / "BLA" / "task1" / "sessA"
    sd.mkdir(parents=True)
    for name, size in [("review_neuron.mat", 3000), ("Cn.mat", 200), ("pnr.mat", 100),
                       ("Ybg_weights.mat", 500), ("sessA.mat", 10000), ("labels.mat", 50),
                       ("candidate_features.npz", 60), ("labels_provenance.txt", 20)]:
        (sd / name).write_bytes(b"x" * size)
    troot = tmp / ".training"
    kp = tc.key_path("BLA", "task1", "sessA", troot)
    kp.parent.mkdir(parents=True)
    sio.savemat(str(kp), {"reference_reviewer": "Taylor", "n_review": 5.0})
    gal = troot / "gallery"
    (gal / "img").mkdir(parents=True)
    (gal / "index.html").write_text("<html></html>", encoding="ascii")
    (gal / "img" / "a.png").write_bytes(b"png")
    exchange = tmp / "exchange"
    exchange.mkdir()

    # 1. missing training/ -> refuse, create nothing
    rc, out = run(pb, [str(sd), "--trainee", "Alice", "--exchange", str(exchange), "--training-root", str(troot)])
    check(rc not in (0, None) and "does not exist" in out, "missing <exchange>/training -> non-zero exit with instruction")
    check(not (exchange / "training").exists(), "nothing created on the exchange")

    (exchange / "training").mkdir()
    # 2. dry run copies nothing
    rc, out = run(pb, [str(sd), "--trainee", "Alice", "--stage", "1", "--no-video", "--dry-run",
                       "--exchange", str(exchange), "--training-root", str(troot)])
    check(rc == 0 and "would copy review_neuron.mat" in out, "dry run lists the copies")
    check(not (exchange / "training" / "Alice").exists(), "dry run writes nothing")

    # 3. real push, no video
    rc, out = run(pb, [str(sd), "--trainee", "Alice", "--stage", "1", "--no-video",
                       "--exchange", str(exchange), "--training-root", str(troot)])
    dest = exchange / "training" / "Alice" / "BLA" / "task1" / "sessA"
    names = sorted(p.name for p in dest.iterdir()) if dest.exists() else []
    check(rc == 0 and names == sorted(["review_neuron.mat", "Cn.mat", "pnr.mat", "Ybg_weights.mat",
                                        "training_key.mat", "run_training.m", "TRAINING_SESSION.txt"]),
          f"bundle without video has exactly the expected files (got {names})")
    marker = tc.read_training_marker(dest / tc.TRAINING_MARKER)
    check(marker.get("trainee") == "Alice" and marker.get("stage_hint") == "1"
          and marker.get("has_video") == "0" and marker.get("reference_reviewer") == "Taylor",
          "marker fields (trainee, stage, has_video, reference reviewer)")
    launcher = (dest / "run_training.m").read_text(encoding="ascii")
    check("acorn_training(session_dir)" in launcher and "%" not in launcher, "launcher generated, no % comments")
    check(not (sd / "review_assigned.txt").exists(), "no review_assigned.txt written in the source session")
    check(sorted(p.name for p in sd.iterdir()) == sorted(["review_neuron.mat", "Cn.mat", "pnr.mat", "Ybg_weights.mat",
                                                           "sessA.mat", "labels.mat", "candidate_features.npz",
                                                           "labels_provenance.txt"]),
          "source session folder untouched")
    pushed = tc.read_csv_rows(troot / "Alice" / "pushed.csv")
    check(len(pushed) == 1 and pushed[0]["with_video"] == "0", "pushed.csv row logged")

    # 4. re-push with video: same-size skip + video added + marker updated
    rc, out = run(pb, [str(sd), "--trainee", "Alice", "--stage", "3",
                       "--exchange", str(exchange), "--training-root", str(troot)])
    check(rc == 0 and (dest / "sessA.mat").exists() and "skip (already staged): review_neuron.mat" in out,
          "second push adds the video and skips same-size files")
    check(tc.read_training_marker(dest / tc.TRAINING_MARKER).get("has_video") == "1", "marker has_video updated")

    # 5. gallery push
    rc, out = run(pb, ["--gallery", "--trainee", "Alice", "--exchange", str(exchange), "--training-root", str(troot)])
    check(rc == 0 and (exchange / "training" / "_shared" / "gallery" / "img" / "a.png").exists(),
          "gallery copied to training/_shared/gallery/")

    # 6. missing key -> failure reported, nothing staged
    sd2 = data / "vCA1" / "task1" / "sessB"; sd2.mkdir(parents=True); (sd2 / "review_neuron.mat").write_bytes(b"x")
    rc, out = run(pb, [str(sd2), "--trainee", "Alice", "--exchange", str(exchange), "--training-root", str(troot)])
    check(rc == 1 and "build it first" in out and not (exchange / "training" / "Alice" / "vCA1").exists(),
          "missing key -> not staged")

    # 7. unsupported area refused
    sd3 = data / "DG_AL" / "task1" / "sessC"; sd3.mkdir(parents=True); (sd3 / "review_neuron.mat").write_bytes(b"x")
    rc, out = run(pb, [str(sd3), "--trainee", "Alice", "--exchange", str(exchange), "--training-root", str(troot)])
    check(rc == 1 and "not part of the training program" in out, "DG_AL session refused")

    # ---- summariser ----
    cols = tc.PROGRESS_COLUMNS
    def prow(**kw):
        base = {c: "" for c in cols}
        base.update({"timestamp": "2026-10-01 10:00:00", "trainee": "Alice", "area": "BLA", "task": "task1",
                     "session": "sessA", "stage": "1", "mode": "drill_coach", "pass": "1", "subset": "all",
                     "n_items": "10", "n_visited": "10", "n_scored": "9", "n_contested": "1",
                     "n_ref_keep_scored": "4", "n_ref_delete_scored": "5", "agreement": "0.7", "kappa": "0.4",
                     "false_keep": "2", "false_delete": "1", "motion_ref_n": "1", "motion_tagged_m": "0",
                     "motion_deleted_any": "1", "motion_false_tags": "0", "n_flagged": "0", "median_sec": "3",
                     "total_sec": "30", "result_file": "drill_sessA_1.mat", "repo_hash": "abc"})
        base.update(kw)
        return base
    ret = exchange / "training" / "Alice" / "returns" / "sessA"
    tr = ret / "training_results"; (tr / "report_x_files").mkdir(parents=True)
    tc.write_csv_rows(tr / "progress.csv", cols, [
        prow(), prow(timestamp="2026-10-01 11:00:00", agreement="0.9", kappa="0.8", result_file="drill_sessA_2.mat",
                     mode="drill_exam", pass_="1")])
    (tr / "drill_sessA_1.mat").write_bytes(b"m"); (tr / "report_x.html").write_text("<html>", encoding="ascii")
    (tr / "report_x_files" / "a.png").write_bytes(b"p")
    (ret / "sessA.mat").write_bytes(b"VIDEO")                   # stray video at session level
    (ret / "review_neuron.mat").write_bytes(b"big")
    bob = exchange / "training" / "Bob" / "returns" / "sessA" / "training_results"; bob.mkdir(parents=True)
    tc.write_csv_rows(bob / "progress.csv", cols, [prow(trainee="Bob", agreement="0.5", result_file="drill_sessA_9.mat")])
    bad = exchange / "training" / "Bob" / "returns" / "sessZ" / "training_results"; bad.mkdir(parents=True)
    (bad / "progress.csv").write_text("wrong,header\n1,2\n", encoding="ascii")
    old = exchange / "training" / "Bob" / "returns" / "sessOld" / "training_results"; old.mkdir(parents=True)
    old_cols = tc.PROGRESS_COLUMNS[:-3]                                  # a file from before the last 3 columns
    tc.write_csv_rows(old / "progress.csv", old_cols,
                      [{c: v for c, v in prow(trainee="Bob", session="sessOld", agreement="0.6",
                                               result_file="drill_old.mat").items() if c in old_cols}])

    rc, out = run(st, ["--exchange", str(exchange), "--training-root", str(troot), "--dry-run"])
    check(rc == 0 and not (troot / "Alice" / "returns").exists(), "summariser dry run copies nothing")
    rc, out = run(st, ["--exchange", str(exchange), "--training-root", str(troot)])
    local = troot / "Alice" / "returns" / "sessA" / "training_results"
    check(rc == 0 and (local / "progress.csv").exists() and (local / "report_x_files" / "a.png").exists(),
          "result files copied home (incl. report images)")
    check(not (troot / "Alice" / "returns" / "sessA" / "sessA.mat").exists()
          and not (troot / "Alice" / "returns" / "sessA" / "review_neuron.mat").exists(),
          "files outside training_results (video, review_neuron) not copied")
    rows = tc.read_csv_rows(troot / "progress_all.csv")
    check(len(rows) == 4 and {r["trainee"] for r in rows} == {"Alice", "Bob"}, "progress_all.csv has the 4 valid rows")
    check("header differs" in out and "older header" in out, "bad header skipped with a warning; older prefix header accepted")
    old_row = [r for r in rows if r["result_file"] == "drill_old.mat"]
    check(len(old_row) == 1 and old_row[0]["n_corrected"] == "" and old_row[0]["agreement"] == "0.6",
          "older file's missing columns read as empty, values intact")
    md = (troot / "summary.md").read_text(encoding="ascii")
    check("## Alice" in md and "## Bob" in md and "0.700 -> 0.900" in md, "summary.md per trainee with first->latest agreement")
    check("contested (uncounted)" in md and "changed after feedback" in md, "summary.md carries the contested and corrected numbers")
    rc, out = run(st, ["--exchange", str(exchange), "--training-root", str(troot)])
    check(len(tc.read_csv_rows(troot / "progress_all.csv")) == 4, "re-run is idempotent (no duplicate rows)")
    empty = tmp / "exchange_empty"; empty.mkdir()
    rc, out = run(st, ["--exchange", str(empty), "--training-root", str(troot)])
    check(rc == 0 and "No training folder yet" in out, "absent training/ -> graceful exit")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
