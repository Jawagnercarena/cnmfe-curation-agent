"""Temp-dir test for the pre-ingest snapshot, the first-review length guard,
the provenance write rule and the "already live" no-op.

Reproduces the 2026-10-12 situation: a session fresh from the headless run
(neuron.mat with all candidates, footprints, traces, PNGs), then a reviewer's
return.  The first ingest must save every file it overwrites into
<session>/.pre_ingest/ exactly once; a return whose labels.mat does not match
the review-set length must be skipped with nothing written; a whole-inbox
re-run must be a no-op (no re-park, no provenance rewrite).
"""
import io
import os
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.io as sio

import ingest_returns as ir
import review_prep as rp

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def run_ingest(argv):
    old = sys.argv
    sys.argv = ["ingest_returns.py"] + argv
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            ir.main()
    except SystemExit as e:
        buf.write(str(e.code))
    finally:
        sys.argv = old
    return buf.getvalue()


def write_labels(path, labels):
    labels = np.asarray(labels, dtype=float).reshape(-1, 1)
    sio.savemat(str(path), {"labels": labels, "motion_delete": np.zeros_like(labels)},
                do_compression=False)


tmp = Path(tempfile.mkdtemp(prefix="ingest_snapshot_"))
try:
    data = tmp / "data"
    sd = data / "BLA" / "task1" / "AVG5x-TSeries-010126-bla99-500um-3z-000"
    sd.mkdir(parents=True)
    N = 10
    np.savez(sd / "candidate_features.npz", feature_matrix=np.zeros((N, 35)),
             feature_names=np.array([f"f{i}" for i in range(35)]),
             auto_rejected=np.array([1, 2], dtype=int), n_candidates=np.array([N]))
    headless = {"neuron.mat": b"H" * 4000, "spatial_footprints.mat": b"H" * 900,
                "Coor.mat": b"H" * 50, "A.txt": b"H" * 3000, "C.txt": b"H" * 700,
                "C_df.txt": b"H" * 700, "C_raw.txt": b"H" * 700, "S.txt": b"H" * 700,
                "Cnn.txt": b"H" * 300}
    for name, content in headless.items():
        (sd / name).write_bytes(content)
    (sd / f"{sd.name}_neurons").mkdir()
    for i in range(1, 4):
        (sd / f"{sd.name}_neurons" / f"neuron_{i}.png").write_bytes(b"H" * (100 + i))
    inputs = {"review_neuron.mat": b"I" * 3000, "Cn.mat": b"I" * 200, "pnr.mat": b"I" * 100,
              "Ybg_weights.mat": b"I" * 500, "review_report.pdf": b"I" * 400,
              "review_summary.txt": b"I" * 50, "run_final_review.m": b"I" * 30,
              "ROIs_candidates.jpg": b"I" * 80}
    for name, content in inputs.items():
        (sd / name).write_bytes(content)
    video = b"V" * 20000
    (sd / f"{sd.name}.mat").write_bytes(video)
    (sd / f"{sd.name}.tif").write_bytes(b"T" * 30000)
    exchange = tmp / "exchange"
    (exchange / "outbox").mkdir(parents=True)
    ret = exchange / "inbox" / "Taylor" / "BLA" / "task1" / sd.name
    ret.mkdir(parents=True)
    for name, content in inputs.items():             # stale copies, different sizes
        (ret / name).write_bytes(b"S" * (len(content) // 2))
    (ret / f"{sd.name}.mat").write_bytes(video)       # same video
    reviewed = {"neuron.mat": b"R" * 1500, "spatial_footprints.mat": b"R" * 300,
                "Coor.mat": b"R" * 40, "A.txt": b"R" * 1000, "C.txt": b"R" * 200,
                "C_df.txt": b"R" * 200, "C_raw.txt": b"R" * 200, "S.txt": b"R" * 200,
                "Cnn.txt": b"R" * 350, "ROIs.jpg": b"R" * 80, "merge_log_101010.mat": b"R" * 60}
    # (a same-size file is skipped by the size rule, so it is neither overwritten
    #  nor snapshotted; every reviewed file above therefore differs in size)
    for name, content in reviewed.items():
        (ret / name).write_bytes(content)
    (ret / f"{sd.name}_neurons").mkdir()
    (ret / f"{sd.name}_neurons" / "neuron_1.png").write_bytes(b"R" * 90)
    write_labels(ret / "labels.mat", [1, 1, 0, 0, 0, 0, 0, 1])      # 8 = 10 - 2 auto-rejected
    ir.DATA_PARENT = data

    # ---- dry run -----------------------------------------------------------
    print("dry run")
    out = run_ingest(["--exchange", str(exchange), "--dry-run"])
    check("would snapshot" in out and "would copy neuron.mat" in out, "dry run announces snapshot + copies")
    check(not (sd / ".pre_ingest").exists() and (sd / "neuron.mat").read_bytes() == headless["neuron.mat"],
          "dry run writes nothing")

    # ---- first ingest ------------------------------------------------------
    print("first ingest")
    out = run_ingest(["--exchange", str(exchange)])
    snap = sd / ".pre_ingest"
    check("pre-ingest snapshot:" in out and "NEW" in out, "ingest reports the snapshot")
    check(snap.is_dir() and (snap / "PRE_INGEST.txt").exists(), ".pre_ingest/ created with its note")
    for name, content in headless.items():
        check((snap / name).read_bytes() == content, f"snapshot holds the headless {name}")
        check((sd / name).read_bytes() == reviewed[name], f"live {name} is now the reviewer's")
    check((snap / f"{sd.name}_neurons" / "neuron_1.png").read_bytes() == b"H" * 101,
          "overwritten PNG snapshotted")
    check(not (snap / f"{sd.name}_neurons" / "neuron_2.png").exists(),
          "a PNG the return does not touch is not snapshotted (nothing overwrote it)")
    check(not (snap / f"{sd.name}.mat").exists() and not (snap / f"{sd.name}.tif").exists(),
          "raw video never snapshotted")
    check(not any((snap / n).exists() for n in inputs), "bundle inputs not snapshotted (never overwritten)")
    check(all((sd / n).read_bytes() == c for n, c in inputs.items()), "bundle inputs untouched")
    note = rp.read_kv(snap / "PRE_INGEST.txt")
    check(note.get("had_labels") == "no" and note.get("n_files") == str(len(headless) + 1),
          f"PRE_INGEST.txt: had_labels={note.get('had_labels')} n_files={note.get('n_files')}")
    check((sd / "labels.mat").exists() and rp.read_kv(sd / "labels_provenance.txt").get("reviewer") == "Taylor",
          "labels and provenance landed")
    prov_bytes = (sd / "labels_provenance.txt").read_bytes()
    snap_mtimes = {p: p.stat().st_mtime for p in snap.rglob("*") if p.is_file()}
    check("1 new" in out and "files/0 MB" in out, "summary table shows the snapshot column")

    # ---- whole-inbox re-run: a no-op ---------------------------------------
    print("re-run")
    time.sleep(1.1)
    out = run_ingest(["--exchange", str(exchange)])
    check("unchanged" in out and "PARKED" not in out and "Park " not in out,
          "the return that is already live is reported unchanged, not parked")
    check(not (sd / ".rereview").exists(), "no .rereview/ created by the re-run")
    check((sd / "labels_provenance.txt").read_bytes() == prov_bytes, "provenance not rewritten on a no-op run")
    check({p: p.stat().st_mtime for p in snap.rglob("*") if p.is_file()} == snap_mtimes, "snapshot untouched")

    # ---- --replace-labels with nothing to copy: provenance still untouched --
    print("replace-labels no-op")
    out = run_ingest(["--exchange", str(exchange), "--replace-labels"])
    check((sd / "labels_provenance.txt").read_bytes() == prov_bytes,
          "provenance not rewritten when --replace-labels copies nothing")

    # ---- --force on a changed return: snapshot write-once -------------------
    print("force")
    (ret / "neuron.mat").write_bytes(b"R2" * 800)
    write_labels(ret / "labels.mat", [1, 0, 0, 0, 0, 0, 0, 1])
    out = run_ingest(["--exchange", str(exchange), "--replace-labels", "--force"])
    check("snapshot kept as is" in out and "kept" in out, "--force reports the snapshot kept")
    check((snap / "neuron.mat").read_bytes() == headless["neuron.mat"], "snapshot still holds the headless neuron.mat")
    check((sd / "neuron.mat").read_bytes() == b"R2" * 800, "live neuron.mat replaced by --force --replace-labels")
    check(not (snap / "ROIs.jpg").exists(), "ROIs.jpg (created by the first ingest) was never snapshotted")

    # ---- stale return on a fresh session: skipped, nothing written ---------
    print("stale first return")
    sd2 = data / "BLA" / "task1" / "AVG5x-TSeries-010226-bla99-500um-3z-000"
    sd2.mkdir()
    np.savez(sd2 / "candidate_features.npz", feature_matrix=np.zeros((20, 35)),
             feature_names=np.array([f"f{i}" for i in range(35)]),
             auto_rejected=np.array([], dtype=int), n_candidates=np.array([20]))
    (sd2 / "neuron.mat").write_bytes(b"H" * 500)
    (sd2 / "review_neuron.mat").write_bytes(b"I" * 900)
    ret2 = exchange / "inbox" / "Alisia" / sd2.name          # area + task levels dropped
    ret2.mkdir(parents=True)
    write_labels(ret2 / "labels.mat", [1] * 12)                # 12, review set holds 20
    (ret2 / "neuron.mat").write_bytes(b"R" * 300)
    (ret2 / "ROIs.jpg").write_bytes(b"R")
    out = run_ingest(["--exchange", str(exchange)])
    check("labels.mat has 12 entries but the live review set expects 20" in out, "stale first return skipped loudly")
    check(not (sd2 / "labels.mat").exists() and not (sd2 / "ROIs.jpg").exists()
          and (sd2 / "neuron.mat").read_bytes() == b"H" * 500 and not (sd2 / ".pre_ingest").exists(),
          "nothing written to the fresh session")
    check("SKIPPED" in out and sd2.name in out, "summary row SKIPPED")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
