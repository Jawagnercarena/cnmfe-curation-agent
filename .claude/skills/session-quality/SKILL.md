---
name: session-quality
description: Rebuild the per-session quality table (review vs model, label coherence, field-of-view consistency, recording and process signals) under <area>/.quality/ and read the flagged list - operator only; it picks sessions to look at, it does not adjudicate.
disable-model-invocation: true
argument-hint: "[--area <BLA|vCA1|DG_AL>] [--no-cv] | --backfill-provenance [--dry-run]"
---

# Session quality table

Manual-only. `$ARGUMENTS` are passed to `agent/session_quality_report.py`
(central machine, Python by full path). Read-only on session folders; writes
`DATA_PARENT/<area>/.quality/sessions.csv` and `summary.md`. Operator only
(decision 2026-10-12): reviewers do not see these numbers.

## 1. Run

```bash
"<PYTHON_EXE>" agent/session_quality_report.py [--area vCA1]
```

Takes a few minutes over all areas (scoring plus a 5-fold logistic regression
per session). `--no-cv` skips the within-session AUC, `--no-model` the
classifier columns. Run it after every `/ingest-returns` so new sessions are
scored before the retrain makes them in-sample.

## 2. Read `summary.md`

- **Per reviewer:** sessions, median keep rate, median keep rate minus the
  model's, confident disagreements, kept cells below the auto-reject threshold,
  median within-session AUC, flagged count, median review duration (only for
  reviews run with `review_meta.txt`, 2026-10-12 or later).
- **Flagged sessions**, sorted by number of flags. Flags are named, ASSUMED
  thresholds listed at the bottom (`FLAG_PARAMS` in the script).
- **Field of view:** how many sessions were grouped from an imaging sheet
  (`agent/config/fov_sheets.json`) and how many by the depth proxy. DG_AL has no
  sheet configured.

What a flag can and cannot say (also printed in the file): every model column
compares a reviewer with a classifier trained on the pool of reviewers, so it
flags a reviewer far from the pool, not a wrong one; a bias everyone shares is
invisible. The field columns are the only reviewer-independent reference and
depend on the sheet. A low within-session AUC can be noisy labels or a
degenerate recording. The table picks sessions for `/rereview`; the second
review is the evidence.

## 3. Reviewer attribution

`reviewer` comes from `labels_provenance.txt`; labelled non-bootstrap sessions
without one are reported as `Julian` (decision 2026-10-12) and flagged
`no_reviewer_record` until the file exists:

```bash
"<PYTHON_EXE>" agent/session_quality_report.py --backfill-provenance --dry-run
"<PYTHON_EXE>" agent/session_quality_report.py --backfill-provenance
```

That is the only write into a session folder this script makes (a new
`labels_provenance.txt` with `reviewer: Julian`, `backfilled: <date>`); it never
touches an existing one. Side effect worth knowing: those sessions become
eligible for the training-key pool and for ingest's duplicate-review parking.

## 4. Afterwards

Record in memory which sessions were flagged and what was decided about them;
numbers themselves live in the CSV, do not copy them into memory.
