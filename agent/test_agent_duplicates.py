"""Temp-dir test for train_classifier._agent_duplicate_mask (synthetic sessions,
no real data, no training).

A reviewed delete that is a second copy of a kept cell must be flagged (and
later trained at weight 0); nothing else may be: not a kept cell, not an
auto-rejected candidate, not a motion-tagged delete, not a delete that merely
touches a kept cell.  The cache must be used when valid and recomputed when
labels.mat changes, and a session without the all-candidate footprints must
come back unflagged.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.io as sio

import train_classifier as tcl

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


H = W = 64


def blob(cy, cx, r=5):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * (r / 2.0) ** 2)))


def make_session(sd: Path, foot, labels_review, auto, motion=None, with_pre=True):
    """foot: (N,H,W); labels_review: labels of the non-auto-rejected rows in order."""
    sd.mkdir(parents=True, exist_ok=True)
    n = len(foot)
    np.savez(str(sd / "candidate_features.npz"), feature_matrix=np.zeros((n, 3)),
             auto_rejected=np.array(auto, dtype=int), n_candidates=n)
    d = {"labels": np.array(labels_review, dtype=float).reshape(-1, 1)}
    if motion is not None:
        d["motion_delete"] = np.array(motion, dtype=float).reshape(-1, 1)
    sio.savemat(str(sd / "labels.mat"), d)
    if with_pre:
        (sd / ".pre_ingest").mkdir(exist_ok=True)
        sio.savemat(str(sd / ".pre_ingest" / "spatial_footprints.mat"), {"spatial_footprints": foot})


tmp = Path(tempfile.mkdtemp(prefix="agent_dup_"))
try:
    # candidates:
    # 0 kept cell A            1 exact copy of A, deleted        -> DUPLICATE
    # 2 kept cell B            3 shifted copy of B (IoU ~0.6)     -> DUPLICATE
    # 4 neighbour touching B (IoU < 0.5), deleted                 -> not a duplicate
    # 5 copy of A, deleted with a motion tag                      -> not a duplicate
    # 6 copy of A, AUTO-REJECTED (never shown)                    -> not flagged
    # 7 junk far away, deleted                                    -> not a duplicate
    # 8 kept cell C, alone
    foot = np.stack([blob(15, 15), blob(15, 15), blob(40, 40), blob(40, 42), blob(40, 48),
                     blob(15, 15), blob(15, 15), blob(55, 10), blob(10, 50)])
    auto = [6]
    #        idx: 0  1  2  3  4  5  7  8      (6 is auto-rejected, not in labels.mat)
    labels =     [1, 0, 1, 0, 0, 0, 0, 1]
    motion =     [0, 0, 0, 0, 0, 1, 0, 0]
    s1 = tmp / "vCA1" / "task" / "s1"
    make_session(s1, foot, labels, auto, motion)
    n = len(foot)

    m, st = tcl._agent_duplicate_mask(s1, n)
    check(st == "ok", f"status ok ({st})")
    check(np.where(m)[0].tolist() == [1, 3], f"flags exactly the two copies of kept cells (got {np.where(m)[0].tolist()})")
    check(not m[4], "a neighbour with IoU < 0.5 is not a duplicate")
    check(not m[5], "a motion-tagged delete is not a duplicate")
    check(not m[6], "an auto-rejected copy is never flagged")
    check(not m[[0, 2, 8]].any(), "kept cells are never flagged")

    cache = s1 / tcl.AGENT_DUP_FILE
    js = json.loads(cache.read_text())
    check(js["duplicate_candidate_indices"] == [1, 3] and js["n_candidates"] == n and js["status"] == "ok",
          "cache written with the indices")
    check(not (s1 / (tcl.AGENT_DUP_FILE + ".partial")).exists(), "no .partial left")

    # cache is used: make the footprints unreadable for a recompute, result must still come from the cache
    real = tcl._compute_agent_duplicates
    calls = {"n": 0}

    def counting(sd, nn):
        calls["n"] += 1
        return real(sd, nn)

    tcl._compute_agent_duplicates = counting
    try:
        m2, _ = tcl._agent_duplicate_mask(s1, n)
        check(calls["n"] == 0 and (m2 == m).all(), "valid cache -> no recompute, same mask")

        # labels change (a re-review keeps candidate 1 instead of 0): cache invalid, recomputed
        time.sleep(0.05)
        labels_b = [0, 1, 1, 0, 0, 0, 0, 1]
        d = {"labels": np.array(labels_b, dtype=float).reshape(-1, 1),
             "motion_delete": np.array(motion, dtype=float).reshape(-1, 1)}
        sio.savemat(str(s1 / "labels.mat"), d)
        os.utime(str(s1 / "labels.mat"), None)
        m3, _ = tcl._agent_duplicate_mask(s1, n)
        check(calls["n"] == 1, "labels.mat changed -> recomputed")
        check(np.where(m3)[0].tolist() == [0, 3], f"new labels: copy 0 of kept cell 1 flagged (got {np.where(m3)[0].tolist()})")
        check(json.loads(cache.read_text())["duplicate_candidate_indices"] == [0, 3], "cache replaced")

        # write_cache=False (dry run): computes, writes nothing
        s2 = tmp / "vCA1" / "task" / "s2"
        make_session(s2, foot, labels, auto, motion)
        m4, _ = tcl._agent_duplicate_mask(s2, n, write_cache=False)
        check(np.where(m4)[0].tolist() == [1, 3] and not (s2 / tcl.AGENT_DUP_FILE).exists(),
              "write_cache=False -> same mask, no file written")
    finally:
        tcl._compute_agent_duplicates = real

    # no all-candidate footprints (session ingested before the snapshot existed)
    s3 = tmp / "vCA1" / "task" / "s3"
    make_session(s3, foot, labels, auto, motion, with_pre=False)
    m5, st5 = tcl._agent_duplicate_mask(s3, n)
    check(st5 == "no_footprints" and not m5.any() and not (s3 / tcl.AGENT_DUP_FILE).exists(),
          "no .pre_ingest footprints -> unflagged, status no_footprints, nothing written")

    # footprint count does not match the candidate count
    s4 = tmp / "vCA1" / "task" / "s4"
    make_session(s4, foot[:-1], labels, auto, motion)
    m6, st6 = tcl._agent_duplicate_mask(s4, n)
    check(st6 == "footprint_count_mismatch" and not m6.any(), f"wrong footprint count -> unflagged ({st6})")

    # labels cover every candidate (threshold-0 review, nothing auto-rejected)
    s5 = tmp / "vCA1" / "task" / "s5"
    make_session(s5, foot, [1, 0, 1, 0, 0, 0, 0, 0, 1], [], [0, 0, 0, 0, 0, 1, 0, 0, 0])
    m7, st7 = tcl._agent_duplicate_mask(s5, n)
    check(st7 == "ok" and np.where(m7)[0].tolist() == [1, 3, 6],
          f"threshold-0 review: the reviewed third copy is flagged too (got {np.where(m7)[0].tolist()})")

    # labels.mat without motion_delete (older reviews)
    s6 = tmp / "vCA1" / "task" / "s6"
    make_session(s6, foot, labels, auto, None)
    m8, st8 = tcl._agent_duplicate_mask(s6, n)
    check(st8 == "ok" and np.where(m8)[0].tolist() == [1, 3, 5],
          f"no motion_delete variable -> treated as no motion tags (got {np.where(m8)[0].tolist()})")

    # the helper never touches the inputs
    check((s1 / "labels.mat").exists() and (s1 / ".pre_ingest" / "spatial_footprints.mat").exists()
          and (s1 / "candidate_features.npz").exists(), "inputs still present")

    # main(): duplicates get weight 0 in the fit and are reported
    src = Path(tcl.__file__).read_text(encoding="utf-8", errors="replace")
    check("w[dup] = 0.0" in src and '"n_excluded_agent_duplicates": n_dup_excluded' in src,
          "main() zero-weights the flagged rows and stores the count in the joblib")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nRESULT: {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S)'}")
sys.exit(1 if fails else 0)
