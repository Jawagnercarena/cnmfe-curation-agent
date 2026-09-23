---
name: retrain-area
description: Retrain one brain area's curation classifier (BLA, vCA1, DG_AL) safely - right wrapper, xgboost only, threshold persistence, model snapshot, watcher race check, post-run verification.
disable-model-invocation: true
argument-hint: "<BLA|vCA1|DG_AL> [--threshold X] [--dry-run]"
---

# Retrain an area's classifier

Manual-only: this overwrites the deployed model in place and nothing backs it up.
Central machine only (CLAUDE.md rule 4: reviewer machines never train).
Arguments: `$ARGUMENTS` = area, then optional `--threshold X` and/or `--dry-run`.

## 0. Resolve the area

| Area | Script | Threshold the script will use with no `--threshold` |
|---|---|---|
| BLA | `agent/train_classifier.py` | `_THRESHOLD_BY_MODEL["xgboost"]` (`agent/train_classifier.py:1208`) |
| vCA1 | `agent/train_classifier_vCA1.py` | wrapper constant, injected always |
| DG_AL | `agent/train_classifier_DG_AL.py` | wrapper constant below 10 agent sessions; **nothing injected above 10** (then the BLA-calibrated default would apply: set one explicitly) |

Refuse any other area token. A bare `train_classifier.py` is BLA; running it for
another area trains BLA's model on BLA's data. Python by full path
(`local_config.PYTHON_EXE`, default `C:\ProgramData\anaconda3\envs\valence\python.exe`).

## 1. Check for a running watcher (race)

There is no lock file. Each watcher polls every cycle with
`_needs_classifier_update()` (`agent/watcher.py:486`: any `labels.mat` newer than
`classifier.joblib`) and then runs
`[PYTHON, _TRAIN_SCRIPT, "--prospective-only", "--model", "xgboost"]`
(`agent/watcher.py:509-522`) with **no `--threshold`**.

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'watcher' } | Select-Object ProcessId, CommandLine
```

(The `Name` filter keeps your own shell, which also contains the word, out of the list.)
If a watcher for this area is running: report it and let the operator decide. Do
not kill it (long-running-job rule). The only real hazard is both processes
writing the joblib at once; a manual retrain that finishes first makes the joblib
newer than every `labels.mat`, so the watcher stays quiet afterwards.

## 2. Snapshot the live model and record the "before" values

`joblib.dump` overwrites in place; no script keeps a backup. Copy
`agent/model/{AREA}/classifier.joblib` to `classifier_pre_<YYYY-MM-DD>.joblib` in
the same folder (local, `*.joblib` is gitignored). Then print the deployed values:

```bash
"/c/ProgramData/anaconda3/envs/valence/python.exe" -c "import joblib,sys; m=joblib.load(sys.argv[1]); print({k:m.get(k) for k in ('model_type','feature_version','n_features','n_sessions','reject_threshold','agent_weight')})" agent/model/{AREA}/classifier.joblib
```

Skip the copy on `--dry-run` (nothing will be written).

## 3. Dry run first

```bash
"<PYTHON_EXE>" <script> --prospective-only --model xgboost --dry-run [--threshold X]
```

Expect `[--dry-run] Model not saved.` (`agent/train_classifier.py:1134`). Read the
wrapper's threshold line: `[vCA1] N agent sessions -> using threshold ...` or the
`[DG_AL] ...` line. Confirm the session count and the operating-point block look
sane before writing anything. If the user asked for `--dry-run`, stop here and
report.

## 4. Real run

Same command without `--dry-run`. Rules:

- **Never `--model auto`.** A CV tie can flip the deployed model to lightgbm; the
  watcher pins xgboost, so the manual run must too.
- **`--threshold` is not persisted.** It lands only inside this joblib. The next
  watcher auto-retrain passes no flag, so the deployed threshold reverts to the
  source constant. A change meant to stick is a source edit (the wrapper constant,
  or `_THRESHOLD_BY_MODEL` for BLA) with its reasoning comment updated; a flag
  alone is a temporary override and must be reported as such.
- **Use the space form `--threshold 0.05`.** The DG_AL wrapper tests
  `"--threshold" not in sys.argv` (`agent/train_classifier_DG_AL.py:53`), which is
  exact-token on a list, so `--threshold=0.05` goes unrecognised, a second flag is
  appended, and argparse keeps the wrapper's value. The vCA1 wrapper handles both
  forms; use the space form everywhere anyway.
- Success lines: `Model saved -> <path>  (type: xgboost)` and `Sessions used: N`.
  Anything else, including a zero exit with no `Model saved` line, is not success.

## 5. Verify against the joblib, not the log

Re-run the one-liner from step 2 and check:

- `model_type == "xgboost"`
- `feature_version` and `n_features` unchanged from the snapshot
- `n_sessions` >= the snapshot's value
- `reject_threshold` equals the intended value (the wrapper constant, the BLA
  default, or the explicit `--threshold`)

Compare the logged agent-pool AUC / false-AR / junk-caught lines with the previous
run's, and report every number as measured, with N. If any check fails, say so
and name the snapshot file; do not restore it without asking.

## 6. Afterwards

- No restart needed: `curator._load_model()` (`agent/curator.py:68`) re-reads the
  joblib on every call.
- Update the area's memory note: date, `n_sessions`, `reject_threshold`, whether the
  threshold came from source or a flag, and the snapshot filename.
- If a watcher was running during the retrain, tell the operator to eyeball the
  next `[CLASSIFIER]` lines in that watcher's log.
