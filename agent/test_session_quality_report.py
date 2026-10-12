"""Temp-dir test for session_quality_report.py.

A synthetic DATA_PARENT (two sessions, one with provenance, one without) and an
empty agent dir (no model, no animal_params.json, no sheets) so the report
exercises the depth-proxy grouping, the flag logic and the provenance backfill
without touching the real tree.
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

import session_quality_report as sq
import review_prep as rp

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def run(argv):
    buf = io.StringIO()
    rc = None
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = sq.main(argv)
    except SystemExit as e:
        rc = e.code
    return rc, buf.getvalue()


def make_session(sd, n, auto, keep_idx, provenance=None, bootstrap=False):
    sd.mkdir(parents=True)
    rng = np.random.default_rng(len(sd.name))
    X = rng.normal(size=(n, 35))
    n_review = n - len(auto)
    labels = np.zeros(n_review)
    labels[keep_idx] = 1
    # make the keeps separable on feature 0 so the within-session AUC is high
    rows = [i for i in range(n) if i not in set(auto)]
    for k, r in enumerate(rows):
        X[r, 0] = 5.0 if labels[k] else -5.0
    np.savez(sd / "candidate_features.npz", feature_matrix=X,
             feature_names=np.array([f"f{i}" for i in range(35)]),
             auto_rejected=np.array(auto, dtype=int), n_candidates=np.array([n]))
    sio.savemat(str(sd / "labels.mat"), {"labels": labels.reshape(-1, 1),
                                         "motion_delete": np.zeros((n_review, 1))})
    (sd / "ROIs_candidates.jpg").write_bytes(b"x")
    (sd / "review_summary.txt").write_text("Motion artifact flags:  3\nMerge candidates:       2 pairs\n")
    if provenance:
        (sd / "labels_provenance.txt").write_text(f"reviewer: {provenance}\ningested: 2026-09-01 10:00:00\n")
    if bootstrap:
        (sd / "bootstrap_match_stats.json").write_text("{}")


tmp = Path(tempfile.mkdtemp(prefix="sq_"))
try:
    data = tmp / "data"
    agent = tmp / "agent"
    (agent / "config").mkdir(parents=True)
    a = data / "BLA" / "3odor" / "AVG5x-TSeries-030126-bla7-600um-3z-000"
    b = data / "BLA" / "3odor" / "AVG5x-TSeries-030226-bla7-610um-3z-000"
    c = data / "BLA" / "3odor" / "AVG5x-TSeries-030326-bla7-900um-3z-000"
    d = data / "BLA" / "3odor" / "AVG5x-TSeries-030426-bla8-500um-3z-000"
    make_session(a, 30, [0, 1], list(range(10)), provenance="Taylor")
    make_session(b, 30, [], list(range(12)))                       # no provenance -> Julian
    make_session(c, 30, [3], list(range(11)))                      # different depth cluster
    make_session(d, 30, [], list(range(6)), bootstrap=True)        # bootstrap -> reviewer bootstrap
    (b / "review_meta.txt").write_text("video_passes: 0\nduration_min: 12.5\nmerges_logged: 0\n"
                                       "resumed_from_stage: 0\nreviewer_os_user: julian\nmachine: PC1\n")

    rc, out = run(["--data-parent", str(data), "--agent-dir", str(agent)])
    check(rc == 0, f"report runs (rc={rc})\n{out if rc else ''}")
    csv_path = data / "BLA" / ".quality" / "sessions.csv"
    md_path = data / "BLA" / ".quality" / "summary.md"
    check(csv_path.exists() and md_path.exists(), "sessions.csv and summary.md written under BLA/.quality")
    import csv as _csv
    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    check(len(rows) == 4, f"4 rows ({len(rows)})")
    check(list(rows[0].keys()) == sq.COLUMNS, "CSV columns are exactly COLUMNS in order")
    by = {r["session"]: r for r in rows}
    ra, rb, rc_, rd = by[a.name], by[b.name], by[c.name], by[d.name]
    check(ra["reviewer"] == "Taylor" and rb["reviewer"] == "Julian" and rd["reviewer"] == "bootstrap",
          f"reviewer attribution: {ra['reviewer']}, {rb['reviewer']}, {rd['reviewer']}")
    check(ra["n_review"] == "28" and ra["n_keep"] == "10" and ra["n_auto_rejected"] == "2", "review counts")
    check(ra["score_kind"] == "no_model" and ra["model_keep_rate"] == "", "no model -> no model columns")
    check(float(ra["within_session_auc"]) > 0.9, f"within-session AUC high on separable labels ({ra['within_session_auc']})")
    check(ra["fov_source"] == "depth_proxy" and ra["fov_group"] == rb["fov_group"] and ra["fov_group"] != rc_["fov_group"],
          f"depth proxy groups 600 and 610 um together, 900 apart ({ra['fov_group']} / {rc_['fov_group']})")
    check(ra["fov_n_sessions"] == "2" and float(ra["fov_keep_median"]) == 11.0, "field median over the group")
    check("no_reviewer_record" in rb["flags"] and "no_video_pass" in rb["flags"], f"flags on b: {rb['flags']}")
    check("no_reviewer_record" not in rd["flags"], "bootstrap session not flagged for a missing reviewer record")
    check(rb["review_duration_min"] == "12.5" and rb["review_machine"] == "PC1", "review_meta columns read")
    check(ra["curator_motion_flags"] == "3" and ra["curator_merge_pairs"] == "2", "curator counts parsed")
    md = md_path.read_text()
    check("| Taylor | 1 |" in md and "| Julian | 2 |" in md and "KEEP_RATE_GAP" in md, "summary has per-reviewer rows and thresholds")

    rc, out = run(["--data-parent", str(data), "--backfill-provenance", "--dry-run"])
    check(rc == 0 and "Would write" in out and b.name in out and c.name in out and a.name not in out and d.name not in out,
          "backfill dry run lists only the unattributed non-bootstrap sessions")
    check(not (b / "labels_provenance.txt").exists(), "dry run writes nothing")
    rc, out = run(["--data-parent", str(data), "--backfill-provenance"])
    check(rp.read_kv(b / "labels_provenance.txt").get("reviewer") == "Julian"
          and rp.read_kv(c / "labels_provenance.txt").get("reviewer") == "Julian"
          and not (d / "labels_provenance.txt").exists(), "backfill writes reviewer: Julian, skips bootstrap")
    rc, out = run(["--data-parent", str(data), "--agent-dir", str(agent)])
    with open(csv_path, newline="") as f:
        rows = {r["session"]: r for r in _csv.DictReader(f)}
    check("no_reviewer_record" not in rows[b.name]["flags"], "flag cleared after backfill")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
