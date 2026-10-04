---
name: summarize-training
description: Bring trainee results home from the lab-server training folder (server read-only) and summarise every trainee's attempts under D:\Julian_CNMFe\.training - numbers only, the supervisor decides progression.
disable-model-invocation: true
argument-hint: "[--trainee <name>] [--dry-run]"
---

# Summarise training returns

Manual-only. Reads `<exchange>/training/<trainee>/returns/` on the lab server
and writes only under `D:\Julian_CNMFe\.training\`. `$ARGUMENTS` are passed to
`agent/summarize_training.py`. Full operator procedure: `docs/TRAINING_SETUP.md`.

## 0. What the script does and does not do

- Copies `*.csv`, `*.mat`, `*.html`, `*.png`, `*.txt` from every returned
  `training_results\` folder (never a raw video, never anything outside
  `training_results\`) into `.training\<trainee>\returns\<session>\training_results\`,
  size-skipping unchanged files.
- Concatenates the `progress.csv` files (header must equal
  `PROGRESS_COLUMNS` in `agent/training_common.py`; others are skipped with a
  warning), dedupes on (trainee, result file, pass), writes
  `.training\progress_all.csv` and `.training\summary.md`, prints one line per
  trainee.
- Never writes to the server, never touches `inbox/`, never trains anything,
  never changes production labels (CLAUDE.md s1, s4).
- Exits with a message if `<exchange>/training/` does not exist yet.

## 1. Pre-flight

```bash
"<PYTHON_EXE>" agent/summarize_training.py [--trainee <Name>] --dry-run
```

Read the `would copy` lines: only `training_results` content, from the
expected trainees. A trainee folder that is missing from the list has not
returned anything (or returned it to the wrong place -- check `inbox\` for a
`TRAINING_SESSION.txt` folder and ask them to move it; ingest refuses it
anyway).

## 2. Run

Same command without `--dry-run`. Success: `N returned session folder(s)`,
then one line per trainee and `progress_all.csv (N rows) and summary.md
written`. Open `.training\summary.md`: per trainee, attempts, sessions,
highest stage, latest agreement / kappa / false keep / false delete / m-tags
per mode, and the attempt table.

## 3. Reading it (for the supervisor, not the script)

- False deletes of reference keeps are the costly error (real cells lost).
- Contested items are excluded from every score; `contested_with_ref` /
  `contested_with_model` say which side the trainee took on them (uncounted).
  Flagged items (`n_flagged`) are the trainee saying "the key is wrong here"
  -- look at those sessions' reports under `returns\<session>\training_results\`
  before judging.
- Coach-mode rows score the FIRST decision (before the feedback);
  `n_corrected` counts decisions changed after feedback. An older progress
  file lacking the last three columns is read as a prefix (empty values).
- No pass threshold exists in code. Progression is the supervisor's call.

## 4. Afterwards

- Update memory: which trainees returned what, their latest numbers, and any
  stage decisions taken.
