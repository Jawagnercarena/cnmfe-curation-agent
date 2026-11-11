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

The exchange scripts have a fixed contract:
- `agent/push_review_bundle.py`: copy-only into `outbox/<assignee>/...`. Its only
  server writes are `shutil.copy2` and the generated `run_final_review.m`; the
  `review_assigned.txt` marker it writes is local. For pnb* sessions it runs the
  retro project's `guide-for-push` hook (reads X:, writes only inside that
  project) and adds `red_guide.png` when one matches (section 4)
  [2026-11-03]. A file already in the outbox is skipped only when byte-identical,
  never on size alone [explicit, 2026-11-03]; `rereview.py request` uses the
  same check.
- `agent/ingest_returns.py`: read-only on the server; writes only under
  `DATA_PARENT`. Exits gracefully when the inbox is absent.
- `agent/push_training_bundle.py` [2026-10-01]: copy-only into
  `training/<trainee>/...` and `training/_shared/gallery/`, plus the generated
  `run_training.m` and `TRAINING_SESSION.txt` there. Never writes
  `review_assigned.txt`, never touches `inbox/`, `outbox/` or a production
  session folder; exits when `training/` is absent (the operator creates it).
- `agent/summarize_training.py` [2026-10-01]: read-only on the server; writes
  only under `DATA_PARENT/.training`.
- `agent/rereview.py` [2026-10-12]: `request` is copy-only into
  `outbox/<assignee>/...` plus the generated `run_final_review.m` and
  `REREVIEW_SESSION.txt`; it never writes `review_assigned.txt`. `adopt` renames
  files only inside the local session folder (archive under
  `<session>/.rereview/`); nothing in it deletes. Ingest parks any return for a
  labelled session under `<session>/.rereview/` instead of overwriting it, and
  never overwrites an existing bundle input (`review_neuron.mat`, summary,
  report, Cn/pnr/Ybg) with a reviewer's copy, even with `--force` [derived,
  2026-10-12: a stale return overwrote two re-run review sets].
If you edit any of these, or add exchange tooling, grep it for delete/move verbs
before calling the work done.

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
  `labels_provenance.txt`, `candidate_features.npz` or `red_guide.png` back from
  a reviewer (`CENTRAL_ONLY`), and refuses to overwrite a different reviewer's
  labels unless `--replace-labels` is given.
- **A red guide ships only if it matches the live review set.** [derived,
  2026-11-03: 090526-pnb99 was re-run 30 -> 76 candidates after its guide was
  drawn] `red_guide.png` (the tdTomato image with the review candidates
  outlined; retro-labelled pnb* animals) is made by the separate project
  `D:\retro_tdTomato_Alignment`. ACORN code never writes there itself and never
  calls its `red-guide --force`.
- **The push builds the guide.** [explicit, 2026-11-03: the user chose push
  time over after-curation and after-ingest] For pnb* sessions `stage_one` runs
  `python -m retro_align guide-for-push <session> [--dry-run]` from the retro
  project. The hook reuses a current guide or rebuilds it inside that project.
  The guide goes in only if the hook exits 0 AND the push's own check passes
  (same N, same auto-rejected indices in `candidates_shown.npz`); otherwise
  the push says why. A failed, timed-out or missing hook never blocks a push.
  `CNMFE_RETRO_GUIDE_BUILD=0` skips the hook (lookup only). Re-review bundles
  never run the hook (`collect_files` default). X: must be mapped in the shell
  that runs the push. Do not move the call into the watcher or ingest.
- **Ingest matches by name, so a reviewer rename forks the session.** [derived,
  2026-08-07] Before concluding "new session", check for an existing one under a
  new name.
- **Ingest never imports a training sandbox.** [explicit, 2026-10-01] A folder
  carrying `TRAINING_SESSION.txt` is a trainee's practice copy of an already
  labelled session (`docs/TRAINING.md`); `iter_sessions` and the explicit-path
  branch skip it, because name matching would otherwise land practice labels on
  the real session. Trainee results go to `training/<name>/returns/`, never
  `inbox/`; everything central for training lives under `DATA_PARENT/.training/`
  (dot folder, skipped by every scanner). The program covers BLA and vCA1 only.
- **A new imaging prep is a new regime.** [derived, 2026-08-05] Do not assume the
  area's model transfers. Start at threshold 0 (a human sees everything) and only
  tighten once a false auto-reject rate has been measured on reviewed sessions.
- **The reviewer of record is the inbox folder a return was ingested from.**
  [explicit, 2026-10-12] `review_assigned.txt` is routing only and
  `review_meta.txt`'s OS user is informational; no script compares them with the
  inbox folder, and none should. A return dropped in the wrong person's folder is
  that person's review until the operator edits `labels_provenance.txt`.
- **Ingest never overwrites what it cannot get back.** [derived, 2026-10-12] A
  return never replaces an existing bundle input (`review_neuron.mat`, summary,
  report, Cn/pnr/Ybg); the first ingest snapshots the headless outputs it is
  about to overwrite into `<session>/.pre_ingest/` (write-once); a first-review
  `labels.mat` whose length does not match the live review set is skipped; a
  return for a labelled session is parked under `<session>/.rereview/`.
- **A finished session's movie .mat is pruned; whatever needs it rebuilds it.**
  [explicit, 2026-11-11] `<session>/<session>.mat` is a frame-exact copy of
  `<session>.tif`. `agent/prune_movie_mats.py` (dry run unless `--apply`; manifest
  in `DATA_PARENT/.movie_prune/`) deletes it only for finished sessions whose .tif
  reproduces it. Retro-tracked animals are exempt until the user decides
  (`--include-retro`), as are sessions the watcher still pre-computes from.
  `CNMFe_final_save.m`, the push and `rereview.py request` rebuild it into the
  session folder; `push_training_bundle.py` rebuilds into `.training/movies/`.
  All rebuilds go through `rebuild_movie_mat.m` (never overwrites; moves into
  place only when complete). A missing movie in a finished session is expected.
- **Report faithfully.** [explicit] A zero exit code is not success; a timed-out
  MATLAB job has more than once kept running and finished while the log said
  "failed". If a step was skipped, say so.
- **Keep memory current without being asked.** [explicit] Update the memory files
  at the end of a substantive task or a context switch.
- **Commit only when asked, on a branch, never to main.** [explicit]

## 5. Machine-specific values (re-derive, do not copy)

All resolved by `agent/local_config.py:37-66`; each is overridable by an env var
or an entry in `agent/.env` (gitignored; template in `agent/.env.example`).

| Env var | Central-machine default |
|---|---|
| `CNMFE_REPO_ROOT` | self-locating (parent of `agent/`) |
| `CNMFE_DATA_PARENT` | `D:\Julian_CNMFe` |
| `CNMFE_MATLAB_EXE` | `C:\Program Files\MATLAB\R2023b\bin\matlab.exe` |
| `CNMFE_PYTHON_EXE` | `C:\ProgramData\anaconda3\envs\valence\python.exe` |
| `CNMFE_EXCHANGE_ROOT` | server exchange root, read-only to Claude |
| `CNMFE_RETRO_RESULTS` | `D:\retro_tdTomato_Alignment\results` (ACORN never writes there; set empty to send no red guides) |
| `CNMFE_RETRO_GUIDE_BUILD` | on (`0` = the push does not run the retro guide hook) |

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
  MATLAB only), `docs/TRAINING.md` (trainee role, MATLAB only),
  `docs/TRAINING_SETUP.md` (operator side of the training program),
  `agent/eval/README.md` (evaluation harness contract).

## 7. Skills

Operator procedures with side effects live under `.claude/skills/` and are
manual-only (type the slash command; Claude is not shown them otherwise):
`/ingest-returns`, `/retrain-area`, `/recurate-sessions`, `/push-review-bundle`,
`/retire-session`, `/push-training-bundle`, `/summarize-training`, `/rereview`,
`/session-quality`. `/skills` lists them.
