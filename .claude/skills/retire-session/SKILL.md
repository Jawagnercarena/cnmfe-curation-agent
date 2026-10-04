---
name: retire-session
description: Park an unusable session out of every scanner's sight by moving it to {AREA}/.excluded/{task}/ with an EXCLUDED_REASON.txt - the dot prefix is the whole mechanism; no script does this.
disable-model-invocation: true
argument-hint: "<session_dir> \"<reason>\""
---

# Retire (park) a session

Manual-only. This is a convention, not a script: every scanner in the pipeline
skips dot-prefixed folders, so moving a session under `{AREA}/.excluded/{task}/`
removes it from watching, pushing, re-curation, training and the review queue at
once. First case: bla37 060526-229um, 2026-08-08, whole-FOV motion. Local data
only; nothing on the server is touched.

## 1. Check before moving

```bash
ls "<session_dir>"/review_assigned.txt "<session_dir>"/labels.mat "<session_dir>"/ROIs.jpg 2>/dev/null
```

- `review_assigned.txt` present: the bundle is on the server and a reviewer may
  be working on it. The operator deletes `outbox/<reviewer>/.../<session>` and
  tells the reviewer to skip it; Claude never deletes on the server.
- `labels.mat` present: parking removes a *labeled* session from training. Say so
  explicitly and get the user's yes; that is a data decision, not housekeeping.
- Do not park to fix a bad run: a session processed at the wrong gSig is
  re-run, not retired (see the DG_AL 2026-08-18 case in memory).

Record the file count: `find "<session_dir>" -type f | wc -l`.

## 2. Move (ask first; then same drive, no copy)

The move is within the local data drive, so it is a rename. Ask the user before
running it, then:

```powershell
$src = "<session_dir>"; $dst = "<DATA_PARENT>\<AREA>\.excluded\<task>\<session>"
New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
Move-Item -LiteralPath $src -Destination $dst
```

Then write `EXCLUDED_REASON.txt` inside the parked folder (ASCII only):

```
retired: <YYYY-MM-DD>
by: <operator>
from: <original path>
reason: <one or two sentences>
```

## 3. Verify

- File count after == before (plus the reason file).
- `"<PYTHON_EXE>" agent/push_review_bundle.py --all --area <AREA> --assignee x --dry-run`
  no longer lists it; the watcher's `REVIEW_QUEUE.md` drops it on its next cycle.
- Known non-skippers: `train_classifier_vCA1.py`'s session counter skipped
  dot-dirs only since 2026-09-23, and a few `agent/eval/` scripts hardcode a
  parked path on purpose.

## 4. Afterwards

Add the session to memory `project_excluded_sessions` (path, date, reason) and,
if it had been pushed, note the operator's server delete.
