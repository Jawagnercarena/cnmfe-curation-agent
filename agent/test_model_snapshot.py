"""Temp-dir test for train_classifier._snapshot_previous_model (no real model,
no training).

The trainer overwrites classifier.joblib in place; the snapshot taken just
before must be byte-identical, keep the old file's mtime, be named after it,
never overwrite an existing snapshot, never touch or remove the live model,
and leave no .partial file behind.
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_classifier as tcl

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def set_mtime(p: Path, y, mo, d, h, mi, s):
    t = datetime(y, mo, d, h, mi, s).timestamp()
    os.utime(str(p), (t, t))
    return t


tmp = Path(tempfile.mkdtemp(prefix="model_snapshot_"))
try:
    mdir = tmp / "model" / "vCA1"
    mdir.mkdir(parents=True)
    live = mdir / "classifier.joblib"

    # 1. no model yet -> nothing to do, nothing created
    check(tcl._snapshot_previous_model(live) is None, "no live model -> returns None")
    check(not (mdir / "history").exists(), "no live model -> no history folder created")

    # 2. first snapshot
    live.write_bytes(b"MODEL-A" * 1000)
    tA = set_mtime(live, 2026, 11, 6, 17, 9, 15)
    s1 = tcl._snapshot_previous_model(live)
    check(s1 == mdir / "history" / "classifier_2026-11-06_170915.joblib",
          f"snapshot is named after the old model's mtime ({s1.name})")
    check(s1.read_bytes() == b"MODEL-A" * 1000, "snapshot is byte-identical")
    check(abs(s1.stat().st_mtime - tA) < 2, "snapshot keeps the old model's mtime")
    check(live.read_bytes() == b"MODEL-A" * 1000 and abs(live.stat().st_mtime - tA) < 2,
          "live model untouched (bytes and mtime)")
    check(sorted(p.name for p in (mdir / "history").iterdir()) == ["classifier_2026-11-06_170915.joblib"],
          "history holds exactly the snapshot, no .partial left")

    # 3. same model again (a retrain that crashed before the save) -> reused, not duplicated
    s2 = tcl._snapshot_previous_model(live)
    check(s2 == s1, "identical model -> existing snapshot reused")
    check(len(list((mdir / "history").iterdir())) == 1, "no second copy written")

    # 4. the retrain replaces the live model; the next retrain snapshots the new one
    live.write_bytes(b"MODEL-B" * 1200)
    set_mtime(live, 2026, 11, 20, 13, 22, 17)
    s3 = tcl._snapshot_previous_model(live)
    check(s3.name == "classifier_2026-11-20_132217.joblib", f"second model gets its own snapshot ({s3.name})")
    check(s1.read_bytes() == b"MODEL-A" * 1000, "first snapshot still intact")
    check(s3.read_bytes() == b"MODEL-B" * 1200, "second snapshot byte-identical")

    # 5. different bytes under the same timestamp -> never overwrite, suffix instead
    live.write_bytes(b"MODEL-C" * 900)
    set_mtime(live, 2026, 11, 20, 13, 22, 17)
    s4 = tcl._snapshot_previous_model(live)
    check(s4.name == "classifier_2026-11-20_132217_2.joblib", f"name collision with different bytes -> _2 ({s4.name})")
    check(s3.read_bytes() == b"MODEL-B" * 1200, "colliding snapshot was not overwritten")
    check(s4.read_bytes() == b"MODEL-C" * 900, "suffixed snapshot holds the new bytes")
    s5 = tcl._snapshot_previous_model(live)
    check(s5 == s4, "same bytes again -> the _2 snapshot is reused")

    # 6. a stale .partial from an interrupted copy does not block or corrupt
    live.write_bytes(b"MODEL-D" * 500)
    set_mtime(live, 2026, 11, 21, 8, 0, 0)
    stale = mdir / "history" / "classifier_2026-11-21_080000.joblib.partial"
    stale.write_bytes(b"half")
    s6 = tcl._snapshot_previous_model(live)
    check(s6.name == "classifier_2026-11-21_080000.joblib" and s6.read_bytes() == b"MODEL-D" * 500,
          "stale .partial is replaced by a complete snapshot")
    check(not stale.exists(), "no .partial left afterwards")

    # 7. the copy cannot be completed -> raises, live model intact, no snapshot under the final name
    live.write_bytes(b"MODEL-E" * 700)
    set_mtime(live, 2026, 11, 22, 9, 0, 0)
    real_cmp = tcl.filecmp.cmp
    calls = {"n": 0}

    def fake_cmp(a, b, shallow=True):
        calls["n"] += 1
        return False if str(b).endswith(".partial") else real_cmp(a, b, shallow=shallow)

    tcl.filecmp.cmp = fake_cmp
    try:
        raised = False
        try:
            tcl._snapshot_previous_model(live)
        except OSError:
            raised = True
    finally:
        tcl.filecmp.cmp = real_cmp
    check(raised, "unverifiable copy -> OSError (the caller never reaches joblib.dump)")
    check(live.read_bytes() == b"MODEL-E" * 700, "live model intact after the failure")
    check(not (mdir / "history" / "classifier_2026-11-22_090000.joblib").exists(),
          "no snapshot under the final name after the failure")

    # 8. nothing in the helper deletes: every earlier snapshot is still there
    names = sorted(p.name for p in (mdir / "history").iterdir() if p.suffix == ".joblib")
    check(names == ["classifier_2026-11-06_170915.joblib", "classifier_2026-11-20_132217.joblib",
                    "classifier_2026-11-20_132217_2.joblib", "classifier_2026-11-21_080000.joblib"],
          f"all snapshots kept ({len(names)})")

    # 9. main() calls the helper immediately before the save
    src = Path(tcl.__file__).read_text(encoding="utf-8", errors="replace")
    i_snap = src.find("_prev = _snapshot_previous_model(model_path)")
    i_dump = src.find("joblib.dump(", i_snap)
    check(i_snap > 0 and 0 < i_dump - i_snap < 200, "snapshot call sits directly before joblib.dump in main()")
    check(src.count("joblib.dump(") == 1, "train_classifier has exactly one joblib.dump")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
