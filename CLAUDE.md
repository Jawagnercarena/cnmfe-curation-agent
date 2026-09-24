# CLAUDE.md -- standing rules for this repo (ACORN, Kheirbek Lab at UCSF)

Loaded every session. Rules marked [explicit] were stated by Julian; [derived]
rules come from bugs that actually happened. Every path/line here was checked
against the working tree on 2026-09-22; re-check before relying on a line number.

## 1. The lab server is read-only for Claude  [explicit, 2026-06-25]

Server root: `\kheirbek-nas.cin.ucsf.edu\kheirbek1` (often mapped to `X:`).
Claude must not create, write, move, rename, or delete anything under it. Not
files, not folders, not "just an empty directory so the script works". Reading is
fine. A human operator performs every server write; Claude may propose the
command, clearly labelled as operator-run. A missing server folder is not a
problem to solve by creating it: exit and say so.

The two exchange scripts have a fixed contract:
- `agent/push_review_bundle.py`: copy-only into `outbox/<assignee>/...`. Its only
  server writes are `shutil.copy2` and the generated `run_final_review.m`; the
  `review_assigned.txt` marker it writes is local.
- `agent/ingest_returns.py`: read-only on the server; writes only under
  `DATA_PARENT`. Exits gracefully when the inbox is absent.
If you edit either, or add exchange tooling, grep it for delete/move verbs before
calling the work done.

## 2. Test path-touching logic against temp directories first  [explicit]

Anything that copies, moves, or resolves destinations is exercised against a
scratch directory or `--dry-run` first, never against the server and never
against the real data root on its first outing.

## 3. Code constraints

- **No `%` comments in MATLAB passed to `_run_matlab()`.** [explicit]
  `agent/run_cnmfe.py:108` joins the script onto one line for `-batch`, so a `%`
  comments out everything after it. MATLAB then appears to do nothing and reports
  no error. Semicolons only. Callers: watcher.py, curator.py, train_classifier.py,
  validate_threshold.py, bootstrap_preagent.py.
- **No non-ASCII in any line you write or edit**, Python or MATLAB, including
  string literals that get logged or written to disk. [explicit] Use `--` not an
  em dash, `->` not an arrow, straight quotes. Windows cp1252 has broken the
  watcher twice and once made a successful training run log "Training failed".
  Existing violations in tracked files stay as they are; do not mass-clean them.
  When reading Windows-generated files use `encoding="utf-8", errors="replace"`
  (the in-tree pattern, e.g. `agent/ingest_returns.py:139`) or `latin-1`.
- **Do not modify `C:\code\CNMF_E_legacy_Biane\`.** [explicit] It is the legacy
  checkout and is still on the saved MATLAB path on the central machine, so it can
  shadow functions. If MATLAB runs a version you did not edit: `which -all <fn>`.
- **Call Python by full path.** [explicit] `conda run` / `conda activate` are not
  reachable from a tool call. Import the path from `agent/local_config.py`
  (`PYTHON_EXE`); never hardcode an absolute path in a script.
- **Naming.** [explicit, 2026-07-30] The project belongs to the Kheirbek Lab at
  UCSF and is called ACORN (Automated CNMFe Of Recording-Networks). Never describe
  it as a Biane-lab project or fork; `LEGACY_BIANE` in the directory name is
  historical only.
- **Review whitespace-heavy MATLAB diffs with `git diff -w`.** [derived] Wrapping
  a loop in try/catch re-indents the block and hides the real change.

## 4. Workflow rules

- **Training is central and single-canonical.** [explicit] One model per brain
  area, trained on the central machine. Reviewer machines produce labels, never
  models; `train_classifier*.py` must not run there.
- **Scripts never delete `review_assigned.txt`.** [explicit] Re-opening a session
  is a deliberate human action. Nothing in `agent/` deletes the marker; keep it so.
- **Ingest never invents a brain-area folder.** [derived, 2026-07-12] It resolves
  the destination by session name against the local tree and skips loudly when it
  cannot. Skipping is correct; guessing is not. It also never copies
  `labels_provenance.txt` or `candidate_features.npz` back from a reviewer
  (`CENTRAL_ONLY`), and refuses to overwrite a different reviewer's labels unless
  `--replace-labels` is given.
- **Ingest matches by name, so a reviewer rename forks the session.** [derived,
  2026-08-07] Before concluding "new session", check for an existing one under a
  new name.
- **A new imaging prep is a new regime.** [derived, 2026-08-05] Do not assume the
  area's model transfers. Start at threshold 0 (a human sees everything) and only
  tighten once a false auto-reject rate has been measured on reviewed sessions.
- **Report faithfully.** [explicit] A zero exit code is not success; a timed-out
  MATLAB job has more than once kept running and finished while the log said
  "failed". If a step was skipped, say so.
- **Keep memory current without being asked.** [explicit] Update the memory files
  at the end of a substantive task or a context switch.
- **Commit only when asked, on a branch, never to main.** [explicit]

## 5. Machine-specific values (re-derive, do not copy)

All resolved by `agent/local_config.py:37-54`; each is overridable by an env var
or an entry in `agent/.env` (gitignored; template in `agent/.env.example`).

| Env var | Central-machine default |
|---|---|
| `CNMFE_REPO_ROOT` | self-locating (parent of `agent/`) |
| `CNMFE_DATA_PARENT` | `D:\Julian_CNMFe` |
| `CNMFE_MATLAB_EXE` | `C:\Program Files\MATLAB\R2023b\bin\matlab.exe` |
| `CNMFE_PYTHON_EXE` | `C:\ProgramData\anaconda3\envs\valence\python.exe` |
| `CNMFE_EXCHANGE_ROOT` | server exchange root, read-only to Claude |

MATLAB reads `getenv('CNMFE_DATA_PARENT')`.

## 6. Orientation (structure, not numbers)

- **Three areas, one config each:** BLA = `agent/config.py`, vCA1 =
  `agent/config_vCA1.py`, DG_AL = `agent/config_DG_AL.py` (display name "DG AL").
  Each has `watcher_/train_classifier_/recurate_sessions_{AREA}.py` wrappers that
  install the config via `sys.modules["config"]` before importing the shared
  module. **A bare script (`watcher.py`, `train_classifier.py`,
  `recurate_sessions.py`) is BLA.** Models live in `agent/model/{AREA}/classifier.joblib`.
- **The feature contract is positional and versioned per config.**
  `FEATURE_VERSION` in the config selects it; absent means v1 (13 columns). The
  first 13 columns of every version are bit-identical (`agent/features.py:291`).
  Changing the contract is an atomic swap of extractor and model.
- **Never quote a column count, threshold, AUC, or session count from docs or
  memory.** Read `n_features`, `feature_version`, `reject_threshold`, `n_sessions`
  from the area's `classifier.joblib`; they are the only deployed values.
- **`candidate_features.npz` holds all N candidates; `labels.mat` covers only the
  non-auto-rejected subset.** Code joining them must reconstruct the full label
  vector. Length mismatches here are a recurring silent bug.
- **Bootstrap sessions are load-bearing for calibration.** Do not "clean up" the
  training set by dropping them.
- **Data are motion-corrected, averaged 2-photon recordings**, despite CNMF-E's
  usual 1-photon association.
- Docs: `docs/SETUP.md` (central role), `docs/REVIEW_SETUP.md` (reviewer role,
  MATLAB only), `agent/eval/README.md` (evaluation harness contract).

## 7. Skills

Operator procedures with side effects live under `.claude/skills/` and are
manual-only (type the slash command; Claude is not shown them otherwise):
`/ingest-returns`, `/retrain-area`, `/recurate-sessions`, `/push-review-bundle`,
`/retire-session`. `/skills` lists them.
