"""Temp-dir test for the re-review lifecycle.

Builds a fake DATA_PARENT with one labelled session, a fake exchange, and runs
    rereview.py request  ->  reviewer return in inbox  ->  ingest (parks)
    -> rereview.py compare / adopt / keep-current  ->  REVIEW_QUEUE.md note
The real data tree, the real exchange and the real watcher log are never
touched: ingest_returns.DATA_PARENT, watcher.DATA_ROOT and watcher.QUEUE_FILE
are pointed at the temp folder for the duration of the test.
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
import rereview as rr
import review_prep as rp
import count_final_neurons as cfn

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def run(fn, *a, **kw):
    buf = io.StringIO()
    rc = None
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = fn(*a, **kw)
    except SystemExit as e:
        rc = e.code
        if isinstance(e.code, str):
            buf.write(e.code)
            rc = 1
    return rc, buf.getvalue()


def run_ingest(argv):
    old = sys.argv
    sys.argv = ["ingest_returns.py"] + argv
    try:
        return run(ir.main)
    finally:
        sys.argv = old


def write_labels(path, labels, motion=None):
    labels = np.asarray(labels, dtype=float).reshape(-1, 1)
    motion = (np.zeros_like(labels) if motion is None
              else np.asarray(motion, dtype=float).reshape(-1, 1))
    sio.savemat(str(path), {"labels": labels, "motion_delete": motion}, do_compression=False)


tmp = Path(tempfile.mkdtemp(prefix="rereview_"))
try:
    data = tmp / "data"
    sd = data / "BLA" / "task1" / "AVG5x-TSeries-010126-bla99-500um-3z-000"
    sd.mkdir(parents=True)
    N = 10
    rng = np.random.default_rng(0)
    np.savez(sd / "candidate_features.npz", feature_matrix=rng.normal(size=(N, 35)),
             feature_names=np.array([f"f{i}" for i in range(35)]),
             auto_rejected=np.array([1, 2], dtype=int), n_candidates=np.array([N]))
    n_review = N - 2
    v1 = [1, 1, 1, 1, 0, 0, 0, 0]
    write_labels(sd / "labels.mat", v1, [0, 0, 0, 0, 1, 0, 0, 0])
    inputs = {"review_neuron.mat": 3000, "Cn.mat": 200, "pnr.mat": 100, "Ybg_weights.mat": 500,
              "review_report.pdf": 400, "review_summary.txt": 50, f"{sd.name}.mat": 20000,
              "ROIs_candidates.jpg": 80}
    for name, size in inputs.items():
        (sd / name).write_bytes(b"i" * size)
    v1_outputs = {"neuron.mat": b"NEURON-V1" * 10, "ROIs.jpg": b"ROIS-V1", "Coor.mat": b"COOR-V1",
                  "spatial_footprints.mat": b"SF-V1", "A.txt": b"A-V1", "C.txt": b"C-V1",
                  "C_df.txt": b"CDF-V1", "C_raw.txt": b"1 2 3\n4 5 6\n", "Cnn.txt": b"CNN-V1",
                  "S.txt": b"S-V1", "merge_log_111111.mat": b"MERGE-V1"}
    for name, content in v1_outputs.items():
        (sd / name).write_bytes(content)
    (sd / f"{sd.name}_neurons").mkdir()
    (sd / f"{sd.name}_neurons" / "n1.png").write_bytes(b"PNG-V1")
    (sd / "labels_provenance.txt").write_text("reviewer: Taylor\ningested: 2026-09-01 10:00:00\nsource: x\n")
    (sd / "review_assigned.txt").write_text("assigned: 2026-08-01 10:00\nto: Taylor\nby: push_review_bundle.py\n")
    (sd / "agent_run.log").write_text("2026-08-01 09:00:00  [AGENT] Ready.\n")
    v1_labels_bytes = (sd / "labels.mat").read_bytes()
    marker_bytes = (sd / "review_assigned.txt").read_bytes()
    exchange = tmp / "exchange"
    (exchange / "outbox").mkdir(parents=True)
    (exchange / "inbox").mkdir()

    # ---- request --------------------------------------------------------
    print("request")
    rc, out = run(rr.main, ["--data-parent", str(data), "request", "BLA\\task1\\" + sd.name,
                            "--assignee", "Aneesh", "--reason", "keep rate looked low",
                            "--exchange", str(exchange), "--dry-run"])
    check(rc == 0 and "would copy" in out, "dry run lists the copies")
    check(not (exchange / "outbox" / "Aneesh").exists(), "dry run stages nothing")
    rc, out = run(rr.main, ["--data-parent", str(data), "request", "BLA\\task1\\" + sd.name,
                            "--assignee", "Aneesh", "--reason", "keep rate looked low",
                            "--exchange", str(exchange)])
    check(rc == 0, f"request succeeds (rc={rc})")
    bundle = exchange / "outbox" / "Aneesh" / "BLA" / "task1" / sd.name
    staged = {p.name for p in bundle.iterdir()}
    check(staged == set(inputs) - {"ROIs_candidates.jpg"} | {"run_final_review.m", rp.REREVIEW_MARKER},
          f"bundle = inputs + launcher + marker, nothing else ({sorted(staged)})")
    for name in ("labels.mat", "ROIs.jpg", "neuron.mat", "labels_provenance.txt"):
        check(name not in staged, f"blind: {name} not in the bundle")
    mk = (bundle / rp.REREVIEW_MARKER).read_text()
    check("Taylor" not in mk and "keep rate looked low" not in mk, "marker names neither the first reviewer nor the reason")
    req = rp.read_kv(sd / ".rereview" / "request.txt")
    check(req.get("assignee") == "Aneesh" and req.get("current_reviewer") == "Taylor"
          and req.get("expected_n_review") == str(n_review), "request.txt records assignee, previous reviewer, n_review")
    check((sd / "review_assigned.txt").read_bytes() == marker_bytes, "review_assigned.txt untouched")
    check((sd / "labels.mat").read_bytes() == v1_labels_bytes, "live labels untouched by request")
    rc, out = run(rr.main, ["--data-parent", str(data), "request", "BLA\\task1\\" + sd.name,
                            "--assignee", "Alisia", "--reason", "x", "--exchange", str(exchange)])
    check(rc == 1 and "already out" in out, "second request refused while one is open")
    st = rp.rereview_state(sd)
    check(st and st["state"] == "out" and "Aneesh" in st["summary"], f"state = out ({st and st['summary']})")

    # ---- reviewer return --------------------------------------------------
    print("return + ingest (park)")
    ret = exchange / "inbox" / "Aneesh" / "BLA" / "task1" / sd.name
    shutil.copytree(bundle, ret)
    v2 = [1, 1, 0, 0, 0, 1, 0, 0]           # same length -> same byte size as v1
    write_labels(ret / "labels.mat", v2, [0, 0, 1, 0, 0, 0, 0, 0])
    check((ret / "labels.mat").stat().st_size == (sd / "labels.mat").stat().st_size,
          "the redone labels.mat has the SAME size as the live one (the size-skip trap)")
    v2_outputs = {"neuron.mat": b"NEURON-V2" * 10, "ROIs.jpg": b"ROIS-V2", "Coor.mat": b"COOR-V2",
                  "spatial_footprints.mat": b"SF-V2", "A.txt": b"A-V2", "C.txt": b"C-V2",
                  "C_df.txt": b"CDF-V2", "C_raw.txt": b"7 8 9\n", "Cnn.txt": b"CNN-V2", "S.txt": b"S-V2",
                  "merge_log_222222.mat": b"MERGE-V2",
                  "review_meta.txt": b"schema: 1\nreviewer_os_user: aneesh\nmachine: PC2\nduration_min: 42.0\nvideo_passes: 1\nmerges_logged: 1\nresumed_from_stage: 0\n"}
    for name, content in v2_outputs.items():
        (ret / name).write_bytes(content)
    (ret / f"{sd.name}_neurons").mkdir()
    (ret / f"{sd.name}_neurons" / "n1.png").write_bytes(b"PNG-V2")
    (ret / "labels_provenance.txt").write_text("reviewer: STALE\n")      # CENTRAL_ONLY echo
    (ret / "candidate_features.npz").write_bytes(b"STALE-NPZ")           # CENTRAL_ONLY echo
    (ret / ".parked_junk").mkdir()
    (ret / ".parked_junk" / "neuron.mat").write_bytes(b"JUNK")
    t_old = time.time() - 3600
    os.utime(ret / "labels.mat", (t_old, t_old))
    v2_labels_bytes = (ret / "labels.mat").read_bytes()

    ir.DATA_PARENT = data
    rc, out = run_ingest(["--exchange", str(exchange), "--dry-run"])
    check(rc is None and "would park" in out, "dry run announces the parking")
    parked_dirs = [p for p in (sd / ".rereview").iterdir() if p.is_dir()]
    check(parked_dirs == [], "dry run parks nothing")
    rc, out = run_ingest(["--exchange", str(exchange)])
    check("PARKED" in out and "Park BLA" in out, "ingest reports PARKED")
    parked_dirs = [p for p in (sd / ".rereview").iterdir() if p.is_dir()]
    check(len(parked_dirs) == 1 and parked_dirs[0].name.endswith("_Aneesh"), f"one parked version ({[p.name for p in parked_dirs]})")
    pv = parked_dirs[0]
    check((sd / "labels.mat").read_bytes() == v1_labels_bytes, "live labels.mat untouched by ingest")
    check((sd / "neuron.mat").read_bytes() == v1_outputs["neuron.mat"], "live neuron.mat untouched by ingest")
    check((sd / "labels_provenance.txt").read_text().startswith("reviewer: Taylor"), "live provenance untouched")
    check((pv / "labels.mat").read_bytes() == v2_labels_bytes, "parked labels.mat is the reviewer's (same-size file WAS copied)")
    check((pv / "neuron.mat").read_bytes() == v2_outputs["neuron.mat"], "parked neuron.mat is the reviewer's")
    check((pv / "review_meta.txt").exists() and (pv / f"{sd.name}_neurons" / "n1.png").read_bytes() == b"PNG-V2",
          "review_meta.txt and the per-neuron images were parked")
    check(not (pv / "candidate_features.npz").exists() and not (pv / "labels_provenance.txt").exists(),
          "CENTRAL_ONLY echoes were not parked")
    check(not (pv / "review_neuron.mat").exists() and not (pv / f"{sd.name}.mat").exists(),
          "unchanged bundle inputs and the raw video were not parked")
    check(not (pv / ".parked_junk").exists(), "dot folders in the return were ignored")
    retd = rp.read_kv(pv / "returned.txt")
    check(retd.get("aligned") == "yes" and retd.get("marker_present") == "yes"
          and retd.get("live_reviewer_at_park") == "Taylor", f"returned.txt: {retd}")
    rc, out = run_ingest(["--exchange", str(exchange)])
    check("already parked" in out, "re-running ingest reports already parked")
    st = rp.rereview_state(sd)
    check(st and st["state"] == "returned" and "Aneesh" in st["summary"], f"state = returned ({st and st['summary']})")

    # ---- queue note -------------------------------------------------------
    print("queue")
    import watcher
    watcher.DATA_ROOT = data / "BLA"
    watcher.QUEUE_FILE = data / "BLA" / "REVIEW_QUEUE.md"
    rc, out = run(watcher._write_review_queue)
    q = watcher.QUEUE_FILE.read_text(encoding="utf-8")
    check("re-review RETURNED by Aneesh" in q and "1 session(s) under re-review" in q,
          "REVIEW_QUEUE.md notes the returned re-review on the complete session")
    check(sd.name in q.split("Complete")[1], "the session is still listed as complete")

    # ---- compare ------------------------------------------------------------
    print("compare")
    rc, out = run(rr.main, ["--data-parent", str(data), "compare", str(sd)])
    # v1 keeps 0-3, v2 keeps 0,1,5 -> disagree on 2, 3, 5 -> 5 of 8 agree
    check(rc == 0 and "agreement: 5/8" in out and "kappa" in out, "compare prints agreement 5/8")
    check("kept only by Taylor: 2" in out and "kept only by Aneesh: 1" in out, "compare counts one-sided keeps")
    check("stricter" in out, "compare names the direction")
    check((pv / "compare.txt").exists(), "compare recorded in the parked folder")

    # ---- adopt ----------------------------------------------------------------
    print("adopt")
    rc, out = run(rr.main, ["--data-parent", str(data), "adopt", str(sd), "--reason", "second opinion stricter", "--dry-run"])
    check(rc == 0 and "(dry run)" in out, "adopt dry run")
    check((sd / "labels.mat").read_bytes() == v1_labels_bytes and (sd / "neuron.mat").read_bytes() == v1_outputs["neuron.mat"],
          "dry run changed nothing")
    rc, out = run(rr.main, ["--data-parent", str(data), "adopt", str(sd), "--reason", "second opinion stricter"])
    check(rc == 0, f"adopt succeeds (rc={rc})\n{out if rc else ''}")
    check((sd / "labels.mat").read_bytes() == v2_labels_bytes, "live labels.mat is now v2")
    check((sd / "neuron.mat").read_bytes() == v2_outputs["neuron.mat"] and (sd / "ROIs.jpg").read_bytes() == b"ROIS-V2",
          "live neuron.mat / ROIs.jpg are v2")
    check((sd / f"{sd.name}_neurons" / "n1.png").read_bytes() == b"PNG-V2", "live per-neuron images are v2")
    check((sd / "merge_log_222222.mat").exists() and not (sd / "merge_log_111111.mat").exists(), "merge logs swapped")
    check((sd / "review_meta.txt").exists(), "review_meta.txt moved into the session")
    check((sd / "Cn.mat").exists() and (sd / "review_neuron.mat").exists() and (sd / "candidate_features.npz").exists(),
          "inputs stayed in place")
    archives = [p for p in (sd / ".rereview").iterdir() if p.name.startswith("archive_")]
    check(len(archives) == 1 and archives[0].name.endswith("_Taylor"), f"one archive ({[a.name for a in archives]})")
    ar = archives[0]
    check((ar / "labels.mat").read_bytes() == v1_labels_bytes and (ar / "neuron.mat").read_bytes() == v1_outputs["neuron.mat"]
          and (ar / "merge_log_111111.mat").exists() and (ar / f"{sd.name}_neurons" / "n1.png").read_bytes() == b"PNG-V1"
          and (ar / "labels_provenance.txt").read_text().startswith("reviewer: Taylor"),
          "archive holds every v1 output including provenance")
    prov = rp.read_kv(sd / "labels_provenance.txt")
    check(prov.get("reviewer") == "Aneesh" and prov.get("version") == "2" and "Taylor" in prov.get("supersedes", "")
          and prov.get("reason") == "second opinion stricter", f"provenance: {prov}")
    check((sd / "REVIEW_VERSION.txt").exists() and "version: 2" in (sd / "REVIEW_VERSION.txt").read_text(), "REVIEW_VERSION.txt written")
    check((data / "BLA" / "REREVIEW_LOG.md").exists() and "ADOPTED version 2" in (data / "BLA" / "REREVIEW_LOG.md").read_text(),
          "area REREVIEW_LOG.md appended")
    check("[REREVIEW] adopted version 2" in (sd / "agent_run.log").read_text(), "agent_run.log appended")
    check(time.time() - (sd / "labels.mat").stat().st_mtime < 60, "labels.mat mtime refreshed (retrain trigger)")
    check(not (sd / ".rereview" / "request.txt").exists() and (pv / "request.txt").exists(), "request closed into the parked folder")
    check(rp.read_kv(pv / "decision.txt").get("decision") == "adopted", "decision.txt = adopted")
    check(rp.rereview_state(sd) is None, "no open re-review after adopt")
    check((sd / "review_assigned.txt").read_bytes() == marker_bytes, "review_assigned.txt still untouched")
    leftovers = list(sd.glob("*.rereview_tmp"))
    check(leftovers == [], "no temp files left")
    rc, out = run(rr.main, ["--data-parent", str(data), "adopt", str(sd), "--reason", "again"])
    check(rc == 1 and "no undecided" in out, "adopting again refused (nothing undecided)")

    # count_final_neurons must not double count the archive
    found = cfn.find_sessions(data)
    check(found == [sd], f"count_final_neurons sees one session, not the archive ({[str(f.relative_to(data)) for f in found]})")

    # ---- second round: keep-current ---------------------------------------
    print("keep-current")
    ret2 = exchange / "inbox" / "Taylor" / "BLA" / "task1" / sd.name
    shutil.copytree(bundle, ret2)
    (ret2 / rp.REREVIEW_MARKER).unlink()          # an UNMARKED return (temp copy only)
    write_labels(ret2 / "labels.mat", [1, 0, 0, 0, 0, 0, 0, 0])
    for name in ("neuron.mat", "ROIs.jpg"):
        (ret2 / name).write_bytes(b"V3")
    t3 = time.time() - 1800
    os.utime(ret2 / "labels.mat", (t3, t3))
    rc, out = run_ingest(["--exchange", str(exchange)])
    check("PARKED" in out and "Taylor" in out, "unmarked return for a labelled session is parked too")
    check((sd / "labels.mat").read_bytes() == v2_labels_bytes, "live labels still v2")
    rc, out = run(rr.main, ["--data-parent", str(data), "keep-current", str(sd), "--note", "too strict"])
    check(rc == 0 and (sd / "labels.mat").read_bytes() == v2_labels_bytes, "keep-current leaves the live labels")
    und = [v for v in rp.parked_versions(sd) if not v[2]["decision"]]
    check(und == [], "no undecided versions after keep-current")
    check("KEPT current" in (data / "BLA" / "REREVIEW_LOG.md").read_text(), "keep-current logged")

    # ---- misaligned return refused at adopt --------------------------------
    print("misaligned")
    ret3 = exchange / "inbox" / "Alisia" / "BLA" / "task1" / sd.name
    shutil.copytree(bundle, ret3)
    write_labels(ret3 / "labels.mat", [1, 0, 0, 0, 0, 0, 0])      # 7, not 8
    for name in ("neuron.mat", "ROIs.jpg"):
        (ret3 / name).write_bytes(b"V4")
    rc, out = run_ingest(["--exchange", str(exchange)])
    check("WARNING: parked labels.mat has 7 entries" in out, "ingest warns about the misaligned return")
    rc, out = run(rr.main, ["--data-parent", str(data), "adopt", str(sd), "--reason", "x"])
    check(rc == 1 and "stale bundle" in out, "adopt refuses a misaligned version")

    # ---- --replace-labels keeps the old overwrite path for unmarked returns
    print("replace-labels escape hatch")
    ret4 = exchange / "inbox" / "Alisia" / "BLA" / "task1" / sd.name
    (ret4 / rp.REREVIEW_MARKER).unlink()          # unmarked (temp copy only)
    (ret4 / "labels.mat").unlink()
    write_labels(ret4 / "labels.mat", [0, 0, 0, 0, 0, 0, 0, 1])
    (ret4 / "neuron.mat").write_bytes(b"V5-NEURON")
    rc, out = run_ingest(["--exchange", str(exchange), "--replace-labels", "--dry-run"])
    # old path: copy_session with its same-size skip (labels.mat is same-size, so
    # only neuron.mat shows up) -- the pinned behaviour, unchanged
    check("would copy neuron.mat" in out and "would park" not in out and "Ingest BLA" in out,
          "--replace-labels on an unmarked return takes the old overwrite path")
    rc, out = run_ingest(["--exchange", str(exchange), "--dry-run"])
    check("would park" in out and "Ingest BLA" not in out, "without the flag the same return is parked")

    # ---- status -------------------------------------------------------------
    rc, out = run(rr.main, ["--data-parent", str(data), "status", "--area", "BLA"])
    check(rc == 0 and ("archived:" in out or "decided:" in out), "status lists history")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
