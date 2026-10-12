---
name: rereview
description: Second, blind review of an already labelled session - request a bundle, let ingest park the return next to the live labels, compare the two reviewers candidate by candidate, then adopt or keep-current with the reason logged. Never overwrites a review.
disable-model-invocation: true
argument-hint: "request <area>\\<task>\\<session> --assignee <name> --reason \"...\" | compare <session> | adopt <session> --reason \"...\" | keep-current <session> | status [--area A]"
---

# Re-review a session (second, independent review)

Manual-only. `$ARGUMENTS` are passed to `agent/rereview.py` (central machine,
Python by full path from `local_config.PYTHON_EXE`). Decisions of 2026-10-12:
blind by default, same or different reviewer allowed, training uses the adopted
labels only, every adoption writes `REVIEW_VERSION.txt` in the session and a
line in `<area>/REREVIEW_LOG.md`.

## 0. Why it is built this way

A session's state is inferred from its files (`ROIs.jpg` = reviewed,
`labels.mat` = in training). Moving files aside to "re-open" a session is unsafe:
without `neuron.mat` the watcher re-runs CNMF-E, without `labels.mat` the
trainer's retrospective path brings the old decisions back, and the return would
hit ingest's same-size skip. So a second review is a NEW VERSION parked under
`<session>/.rereview/` and the live files change only in `adopt`, which archives
the current version first. Never recurate a session under re-review: both label
vectors are positional against the same `review_neuron.mat`.

## 1. request

```bash
"<PYTHON_EXE>" agent/rereview.py request <area>\<task>\<session> --assignee <Name> --reason "<why>" --dry-run
"<PYTHON_EXE>" agent/rereview.py request <area>\<task>\<session> --assignee <Name> --reason "<why>"
```

- Refuses a session without `labels.mat` (that is a first review: `/push-review-bundle`)
  and a session with an open request (`--force` re-pushes).
- Stages the input bundle plus `REREVIEW_SESSION.txt` into
  `outbox/<assignee>/<area>/<task>/<session>/`. The marker tells the reviewer to
  pull into a FRESH folder and says nothing about the previous reviewer or the
  reason; the reason stays in `<session>/.rereview/request.txt`. Server writes
  are copy-only (CLAUDE.md section 1).
- Does not touch `review_assigned.txt`. The queue shows "re-review out to X" on
  the session's Complete line (watcher restart needed for watchers started
  before 2026-10-12).
- Tell the operator what to tell the reviewer: a second-review bundle is in
  their outbox, pull it to a new folder, review as usual, return it to their
  inbox with the marker still inside. Do not say who reviewed it before or why.
  If the same reviewer is re-reviewing, their old outbox copy may still be there
  (scripts never delete on the server); the fresh launcher and marker are added
  to it.

## 2. return -> ingest parks it

`/ingest-returns` as usual. A return carrying the marker, or any return for a
session that already has `labels.mat`, is PARKED under
`<session>/.rereview/<stamp>_<reviewer>/` with every review output copied
regardless of size; the live files are untouched and nothing retrains. The
summary row says `PARKED`. Read the two warnings if they appear: `parked
labels.mat has N entries but the live review set expects M` (stale bundle;
adopt will refuse) and `returned review_neuron.mat differs in size`.
`--replace-labels` still takes the old overwrite path for UNMARKED returns only
(pinned 2026-10-12).

## 3. compare

```bash
"<PYTHON_EXE>" agent/rereview.py compare <area>\<task>\<session> [--version <stamp>_<reviewer>]
```

Prints agreement, kappa, the candidates kept by only one reviewer (review # =
column of `review_neuron.mat`, 1-based, as in the PDF report), motion tags, and
the deployed model's score on each disagreement. The model is IN-SAMPLE for the
live labels when it was trained after they existed; treat it as a hint, never as
the referee. The text is appended to `compare.txt` in the parked folder.

There is no measured inter-reviewer agreement floor yet (zero double-reviewed
sessions before this tool). Until a few comparisons exist, a disagreement is
information, not a verdict. Calibration-first (no adoption on the first few) is
pinned, not decided.

## 4. decide

```bash
"<PYTHON_EXE>" agent/rereview.py adopt <area>\<task>\<session> --reason "<why>" --dry-run
"<PYTHON_EXE>" agent/rereview.py adopt <area>\<task>\<session> --reason "<why>"
"<PYTHON_EXE>" agent/rereview.py keep-current <area>\<task>\<session> [--note "..."]
```

`adopt` refuses a misaligned or incomplete version. It moves the current outputs
(labels, neuron.mat, ROIs.jpg, traces, footprints, per-neuron images, merge
logs, review_meta, provenance) into `.rereview/archive_<stamp>_<reviewer>/`
file by file with a temp rename, so `neuron.mat` / `ROIs.jpg` are never absent
for a watcher poll; writes `labels_provenance.txt` with `version`, `supersedes`,
`reason`; appends `REVIEW_VERSION.txt`, `agent_run.log` and
`<area>/REREVIEW_LOG.md`; touches `labels.mat`. `keep-current` changes nothing
live and records the decision.

## 5. After an adoption (nothing detects these for you)

- Retrain: an alive watcher retrains on the touched `labels.mat`; otherwise
  `/retrain-area`. `n_sessions` will NOT change; confirm from the trainer's
  per-session line for this session.
- Training key: `build_training_key.py <session> --force` if the session has one.
- Evaluation pools that pin label hashes will refuse the session; note it in
  the active eval `LOG.md`.
- Downstream: `REVIEW_VERSION.txt` in the session says the trace files changed;
  tell whoever analyses it.
- Server: the bundle and the return stay; deleting them is an operator action.
- `/session-quality` to refresh the table.
- Update memory: session, reviewers, agreement, decision, reason.
