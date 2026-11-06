"""Temp-dir test: a ROIs.jpg older than neuron.mat is a stale final review, and
the watcher never runs a folder holding a final review (ROIs.jpg) headless.

Covers review_prep.final_review_state, the REVIEW_QUEUE.md classification in
watcher._write_review_queue (STALE and BLOCKED sections), the headless guard in
watcher.find_new_tifs, and train_classifier.find_retro_sessions.  The real data
tree and the real watcher log are never touched: watcher.DATA_ROOT,
watcher.QUEUE_FILE and train_classifier.DATA_ROOT point at a temp folder, and
the watcher's file log handler is detached before anything runs.
"""
import io
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_prep as rp

fails = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def touch(p: Path, t: float):
    p.write_bytes(b"x")
    os.utime(p, (t, t))


def make(root: Path, name: str, roi=None, nm=None, cand=False, revnm=False,
         labelled=False):
    sd = root / name
    sd.mkdir(parents=True)
    if nm is not None:
        touch(sd / "neuron.mat", nm)
    if roi is not None:
        touch(sd / "ROIs.jpg", roi)
    if cand:
        touch(sd / "ROIs_candidates.jpg", nm or time.time())
    if revnm:
        touch(sd / "review_neuron.mat", nm or time.time())
    if labelled:
        touch(sd / "labels.mat", time.time())
        touch(sd / "candidate_features.npz", time.time())
    return sd


def section(text: str, header_start: str) -> str:
    """Body of the first '## ' section whose header starts with header_start."""
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line.startswith(header_start)
            continue
        if on:
            out.append(line)
    return "\n".join(out)


now = time.time()
OLD = now - 300 * 86400

with tempfile.TemporaryDirectory() as t:
    task = Path(t) / "BLA" / "3odor"

    print("final_review_state")
    s_none = make(task, "S_none", nm=now, cand=True, revnm=True)
    s_done = make(task, "S_done_agent", nm=now - 3, roi=now, cand=True, revnm=True)
    s_5s = make(task, "S_done_agent_5s", nm=now, roi=now - 5, cand=True, revnm=True)
    s_599 = make(task, "S_done_599", nm=now, roi=now - 599, cand=True, revnm=True,
                 labelled=True)
    s_601 = make(task, "S_stale_601", nm=now, roi=now - 601, cand=True, labelled=True)
    s_stale = make(task, "S_stale_agent", nm=now, roi=OLD, cand=True, revnm=True)
    s_stale_pre = make(task, "S_stale_pre", nm=now, roi=OLD)
    s_pre = make(task, "S_pre_done", nm=OLD, roi=OLD + 3, labelled=True)
    s_noneuron = make(task, "S_roi_only", roi=OLD)
    check(rp.final_review_state(s_none) == "none", "no ROIs.jpg -> none")
    check(rp.final_review_state(s_done) == "done", "ROIs.jpg after neuron.mat -> done")
    check(rp.final_review_state(s_5s) == "done", "neuron.mat 5 s newer -> done")
    check(rp.final_review_state(s_599) == "done", "neuron.mat 599 s newer -> done")
    check(rp.final_review_state(s_601) == "stale", "neuron.mat 601 s newer -> stale")
    check(rp.final_review_state(s_stale) == "stale", "neuron.mat 300 days newer -> stale")
    check(rp.final_review_state(s_noneuron) == "done", "ROIs.jpg without neuron.mat -> done")

    # Folders for the headless guard (find_new_tifs), in a second task folder.
    gtask = Path(t) / "BLA" / "2tones"
    g_blocked = make(gtask, "G_blocked", roi=OLD)               # the 022525-bla2 state
    touch(g_blocked / "G_blocked.tif", OLD)
    g_retry = make(gtask, "G_retry")                            # genuine retry
    touch(g_retry / "G_retry.tif", OLD)
    g_done = make(gtask, "G_done", nm=OLD)
    touch(g_done / "G_done.tif", OLD)
    g_loose_rev = make(gtask, "G_loose_reviewed", nm=OLD, roi=OLD + 3)
    touch(gtask / "G_loose_reviewed.tif", OLD)                  # loose tif, reviewed folder
    touch(gtask / "G_loose_new.tif", OLD)                       # loose tif, no folder yet

    print("REVIEW_QUEUE.md")
    import logging
    import watcher
    watcher.logger.removeHandler(watcher.fh)    # never write the real watcher.log
    watcher.logger.removeHandler(watcher.ch)
    logbuf = io.StringIO()
    watcher.logger.addHandler(logging.StreamHandler(logbuf))
    watcher.DATA_ROOT = Path(t) / "BLA"
    watcher.QUEUE_FILE = Path(t) / "BLA" / "REVIEW_QUEUE.md"
    with redirect_stdout(io.StringIO()):
        watcher._write_review_queue()
    q = watcher.QUEUE_FILE.read_text(encoding="utf-8")
    bl = section(q, "## BLOCKED")
    check("## BLOCKED -- ROIs.jpg without neuron.mat  (2 session(s))" in q,
          "blocked section lists 2 sessions")
    check(g_blocked.name in bl and s_noneuron.name in bl,
          "ROIs.jpg-without-neuron.mat folders in the blocked section")
    check(g_retry.name not in bl and g_done.name not in bl, "retry/done folders not blocked")
    st = section(q, "## STALE final review")
    comp_agent = section(q, "## Complete")
    pre_in = section(q, "## Pre-agent")         # both pre-agent sections
    check("## STALE final review -- not complete  (3 session(s))" in q,
          "stale section lists 3 sessions")
    for sd in (s_stale, s_stale_pre, s_601):
        check(sd.name in st, f"{sd.name} in the stale section")
        check(f"- {sd.name} " not in comp_agent and f"- {sd.name} " not in pre_in,
              f"{sd.name} not listed as complete")
    check("ROIs.jpg " + time.strftime("%Y-%m-%d", time.localtime(OLD)) in st,
          "stale line shows the ROIs.jpg date")
    for sd in (s_done, s_5s, s_599):
        check(f"- {sd.name} " in comp_agent, f"{sd.name} still complete (agent)")
    check(f"- {s_pre.name} " in q and s_pre.name not in st, "pre-agent done session unaffected")
    check(s_none.name in section(q, "## Awaiting your review"), "pending session unaffected")
    check(all(ord(c) < 128 for line in q.splitlines()
              if "STALE" in line or "older than" in line or "BLOCKED" in line
              or "neuron.mat is gone" in line
              for c in line), "new queue lines are ASCII")

    print("watcher.find_new_tifs headless guard")
    found = {p.name for _, p in watcher.find_new_tifs()}
    found2 = {p.name for _, p in watcher.find_new_tifs()}      # second poll
    check("G_blocked.tif" not in found, "ROIs.jpg + no neuron.mat: retry NOT run headless")
    check("G_retry.tif" in found, "genuine retry (no ROIs.jpg, no neuron.mat) still found")
    check("G_done.tif" not in found, "folder with neuron.mat not re-run")
    check("G_loose_reviewed.tif" not in found,
          "loose tif whose session folder holds a review NOT run headless")
    check("G_loose_new.tif" in found, "new loose tif still found")
    check(found == found2, "second poll gives the same result")
    logged = logbuf.getvalue()
    check(logged.count("SKIP 2tones/G_blocked") == 1,
          "blocked folder logged once across two polls")
    check("(neuron.mat MISSING)" in logged and "(neuron.mat present)" in logged,
          "skip line says whether neuron.mat is there")
    check(logged.count("SKIP 2tones/G_loose_reviewed") == 1, "loose-tif skip logged once")
    check(all(ord(c) < 128 for c in logged), "skip log lines are ASCII")

    print("train_classifier.find_retro_sessions")
    import train_classifier as tc
    tc.DATA_ROOT = Path(t) / "BLA"
    buf = io.StringIO()
    with redirect_stdout(buf):
        retro = tc.find_retro_sessions()
    names = {p.name for p in retro}
    check(s_stale.name not in names, "stale agent session not retro-labelled")
    check("[RETRO] SKIP 3odor/S_stale_agent" in buf.getvalue(), "skip is logged")
    check({s_done.name, s_5s.name} <= names, "genuinely reviewed sessions still retro-labelled")
    check(s_none.name not in names, "unreviewed session not retro-labelled")

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
