---
name: ingest-returns
description: Bring reviewers' finished sessions back from the lab-server inbox into the local data tree - dry run first, read every SKIP reason, then make sure the area's classifier actually retrains.
disable-model-invocation: true
argument-hint: "[<reviewer>\<area>\<task>\<session>] [--force] [--replace-labels] [--dry-run]"
---

# Ingest reviewer returns

Manual-only: it writes into the live data tree and is the step that feeds the
classifier. Central machine only. Arguments `$ARGUMENTS` are passed through to
`agent/ingest_returns.py` (flags: optional inbox-relative session path, `--exchange`,
`--force`, `--replace-labels`, `--dry-run`).

## 0. Preconditions

- The server is read-only for Claude (CLAUDE.md section 1). This script only reads
  `<exchange>/inbox/` and writes under `DATA_PARENT`; keep it that way. Exchange
  root comes from `CNMFE_EXCHANGE_ROOT` in `agent/.env`; the script exits if unset.
- If it prints `No inbox yet at ...`, that is the answer. Do not create the folder.
- Python by full path (`local_config.PYTHON_EXE`).

## 1. Dry run and read the table

```bash
"<PYTHON_EXE>" agent/ingest_returns.py --dry-run
```

The summary table has one row per inbox session: reviewer, local destination,
status `NEW` / `unchanged` / `SKIPPED` / `NOT FOUND`, files, MB, and a
`keep/delete/motion` label count read from the reviewer's `labels.mat`. Every
`SKIPPED` row is a decision, never noise:

| Reason printed | What it means | What to do |
|---|---|---|
| `cannot resolve destination ... likely dropped the {area} level` | no local session of that name; ingest refuses to invent an area (CLAUDE.md rule 4) | check for an existing session under a *different* name first: a reviewer rename forks a session (2026-08-07). If it is a rename, tell the operator; do not guess a destination |
| `AMBIGUOUS: N existing folders named ...` | two local sessions share the name | tell the operator which two; do not pick |
| `session already carries <X>'s ingested labels; refusing <Y>'s duplicate review` | another reviewer's labels are already in and trained on | only the operator decides to replace; if they do, re-run with `--replace-labels` and tell the first reviewer |
| `NOT FOUND` | explicit path typo | fix the path |

`unchanged` rows are fine: the multi-GB video is matched by size and not
re-copied; `--force` copies everything regardless.

## 2. Real run

Same command without `--dry-run`. Success per `NEW` session is the block
`Ingest <area>/<task>/<session>`, `copied N files (M MB), skipped K unchanged`,
the label line, and `labels.mat present -> watcher will auto-retrain on its next
poll.` Ingest also writes `labels_provenance.txt` locally (reviewer, source, time);
it never copies that file or `candidate_features.npz` back from a reviewer.

## 3. Make sure the retrain actually happens

`shutil.copy2` preserves the reviewer's save time on `labels.mat`. The watcher
retrains only when some `labels.mat` is newer than `agent/model/{AREA}/classifier.joblib`
(`agent/watcher.py:486`). So a session whose reviewer saved *before* the area's
last retrain (another return triggered it, or a manual `/retrain-area` ran) never
trips the check. For each `NEW` session that carried labels:

```bash
"<PYTHON_EXE>" -c "import sys,os; j=os.path.getmtime(sys.argv[1]); [print(('OLDER than model' if os.path.getmtime(p)<j else 'newer: watcher will retrain'), p) for p in sys.argv[2:]]" agent/model/{AREA}/classifier.joblib "<session>/labels.mat" ...
```

If any line says `OLDER than model`, or no watcher is running for that area
(`Get-CimInstance Win32_Process | ? { $_.Name -match 'python' -and $_.CommandLine -match 'watcher' }`),
run `/retrain-area <AREA>`. Otherwise watch the area's `agent/logs/watcher*.log`
for the `[CLASSIFIER] ... Classifier updated successfully.` line and confirm
`n_sessions` in the joblib went up.

## 4. Afterwards

- The reviewer's bundle stays in the server `outbox/`; deleting it is an operator
  action. `review_assigned.txt` in the local session stays (rule 4).
- Update memory: reviewer, sessions ingested, keep/delete/motion counts, any
  SKIPPED rows and what was decided, and whether a manual retrain was needed.
