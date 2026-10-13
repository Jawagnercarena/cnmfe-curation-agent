# HANDOFF 2026-10-12: whole-inbox ingest undone; pre-ingest snapshots to build

Written 2026-10-12 ~10:30 by the Claude session that undid the morning's ingest, for the
Claude session working on `agent/rereview.py` / `agent/ingest_returns.py`. Everything below
was measured or read from files this morning; paths are as of 10:25. Full record with every
number: `agent/eval/cohort5_2026-09/LOG.md`, section "2026-10-12 08:45". Memory:
`project_ingest_undo_2026-10-12.md` (already indexed in MEMORY.md).

## 1. Current state (verified 10:15)

- `vCA1/6odorDualDiffRew/AVG5x-TSeries-082626-{pnb102,pnb105,pnb106,pnb99}-*` are back to
  AWAITING REVIEW: `ROIs_candidates.jpg` + `review_neuron.mat` present, no `ROIs.jpg`, no
  `labels.mat`, no `labels_provenance.txt`, no `merge_log_*.mat`. `review_assigned.txt`
  (Taylor, 10-06 08:11-08:15) untouched. All review inputs are the 10-06 versions again.
- `agent/model/vCA1/classifier.joblib` = `classifier_prepilot_2026-09-24.joblib`
  (161 sessions, T 0.02, md5 affdfaab98ed6341b87cc5cea223dd85, mtime 2026-09-24 08:30:53,
  copied with copy2). No vCA1 `labels.mat` is newer than it, so the watcher's retrain check
  stays quiet; confirmed quiet through 10:15. The 163-session model written at 08:47 exists
  only in the undo session's scratchpad
  (`C:\Users\julia\AppData\Local\Temp\claude\c--code-CNMF-E-LEGACY-BIANE-CLAUDE\b2627b02-24a2-4218-91e8-0deae06f046a\scratchpad\model_backup\`);
  it is reproducible by retraining and may be gone when that session ends.
- BLA and DG_AL models untouched (BLA joblib 09-30, DG_AL 09-04).
- Watchers: BLA and vCA1 alive (heartbeat every 60 s), DG_AL log frozen since 10-05 23:42.

## 2. Timeline of the morning

| time | actor | what |
|---|---|---|
| 08:45:13-08:45:37 | operator, `ingest_returns.py` with no session argument | see section 3 |
| 08:46:55 | vCA1 watcher | routine `animal_params.json` refresh (its git diff predates today) |
| 08:47:09 | vCA1 watcher | retrain -> 163 sessions (161 + 082626-pnb102 + 082626-pnb99) |
| 09:09-09:27 | the OTHER Claude session | edits to ingest_returns.py, watcher.py, review_prep.py, CNMFe_final_save.m, new rereview.py, session_quality_report.py, count_final_neurons.py, tests, skills, docs |
| 09:23:54 | the other session | `labels_provenance.txt` backfilled (reviewer Julian) into 56 older sessions |
| 09:28:27 | the other session | `<area>/.quality/sessions.csv` + `summary.md` |
| 10:08-10:09 | the other session | training bundles pushed for trainee Jeremy (`.training/Jeremy/pushed.csv`) |
| 10:11:00-10:11:02 | this session | the undo (section 4) |

The undo touched only the four sessions and the vCA1 joblib (post-undo ChangeTime scan).

## 3. What the 08:45 ingest actually wrote (NTFS ChangeTime, not mtimes)

copy2 preserves the reviewer's mtimes, so mtimes say nothing about arrival time. The scan
used the NTFS ChangeTime (set on every write, not preservable):
`agent/eval/cohort5_2026-09/scan_changetime.py ROOT "YYYY-mm-dd HH:MM" "YYYY-mm-dd HH:MM" OUT`
(read-only; ~55k entries under D:\Julian_CNMFe in about a minute).

- 112 sessions got `labels_provenance.txt` rewritten with `ingested: 2026-10-12`, because
  `write_provenance` runs for every return that carries labels.mat even when every file was
  size-skipped. 108 of them received nothing else; their previous `ingested:` dates are LOST
  and were not restored (reviewer/source lines unchanged).
- 4 sessions received files:

| session (082626) | inbox path | labels keep/delete | verdict |
|---|---|---|---|
| pnb102 | inbox/Taylor/vCA1/6odorDualDiffRew/ | 44 / 171 of 215 | valid review of the 10-06 set |
| pnb99 | inbox/Taylor/vCA1/6odorDualDiffRew/ | 29 / 52 of 81 | valid review of the 10-06 set |
| pnb105 | inbox/Alisia/<session> (area+task levels dropped) | 10 / 102 of 112 | STALE: est85 set, features hold 266 |
| pnb106 | inbox/Alisia/<session> (area+task levels dropped) | 7 / 109 of 116 | STALE: est85 set, features hold 228 |

  Each created labels.mat, ROIs.jpg, merge_log_*.mat, labels_provenance.txt and overwrote
  neuron.mat, Coor.mat, spatial_footprints.mat, A/C/C_df/C_raw/S.txt and the kept-cell PNGs
  in `<session>_neurons/` (CNMFe_final_save.m writes them at the end of a review). The two
  stale bundles also overwrote review_neuron.mat, review_report.pdf, review_summary.txt
  (09-25 versions, byte-identical to `.prev_est85_2026-10-05/`) and Ybg_weights.mat (09-24).
  The trainer skipped pnb105/106 with "unexpected size mismatch (features=266, labels=112)".

## 4. The undo (agent/eval/cohort5_2026-09/undo_ingest_2026-10-12.py)

Dry run first, then `--go --model-backup <dir>`. It deleted the created files (each one had to
have an NTFS creation date of 2026-10-12, else refused), restored spatial_footprints.mat for
all four and review_summary.txt for pnb105/106 from `<session>/.rerun_mincorr070_2026-10-06/`,
restored review_neuron.mat / review_report.pdf / Ybg_weights.mat for pnb105/106 by copying
FROM the server outbox bundles (10-06 versions; server read only), and rolled the model back.
It is idempotent and refuses to run when the deployed model is not the 08:47 one.

Left as they are (user decision): neuron.mat, Coor.mat, A/C/C_df/C_raw/S.txt and 44/10/7/29
PNGs per session now hold the reviewers' curated output. No pre-ingest copy exists anywhere;
the next ingested review overwrites them again anyway.

## 5. Hazards and open items for the ingest work

1. The four returns are still in the inbox (user: they stay). The sessions now have no
   labels.mat, so a bare `ingest_returns.py` run takes the FIRST-REVIEW path and writes them
   in place again. The stale pnb105/106 returns would land their 112/116-row labels.mat on
   266/228-candidate sessions. The BUNDLE_INPUTS guard added at 09:26 protects
   review_neuron.mat and friends, but not labels.mat. Suggestion: apply `expected_n_review`
   to the first-review path too and SKIP (or park) a return whose labels.mat length does not
   match; the 08:47 trainer WARN is the symptom of exactly this.
2. Reviewer attribution: USER DECISION 2026-10-12 -- the reviewer of record for a session is
   the inbox folder it was returned to and ingested from, regardless of who it was assigned
   to. This is what ingest already does (`src.relative_to(inbox).parts[0]`, ingest_returns.py
   ~line 441); `review_assigned.txt` is routing only and `review_meta.txt`'s
   `reviewer_os_user` is informational. Do NOT add a warning or check that compares the inbox
   reviewer against the assignee or the OS user. Consequence to keep in mind: the duplicate
   guard, the per-reviewer quality table and the training key all key on that folder name, so
   a return dropped in the wrong person's folder is that person's review until the operator
   edits the provenance line. Worth a one-line [explicit] entry in CLAUDE.md section 4; this
   session did not edit CLAUDE.md because the other session has it open (git shows it
   modified).
3. `write_provenance` rewrites `ingested:` on no-copy ingests (108 timestamps lost today).
   Suggestion: write it only when something was copied or the file is absent, or append a
   history line instead of replacing.
4. Operator action, not ours: move or remove `inbox/Alisia/<pnb105>` and `<pnb106>` on the
   server; ask for a fresh review of the 10-06 bundles for pnb105/pnb106 (both still in
   `outbox/Taylor/vCA1/6odorDualDiffRew/`, verified present and dated 10-06).

## 6. Pre-ingest snapshots: what to build

Goal (user, 2026-10-12): stop losing the headless outputs at the first ingest. Today's
incident showed what disappears with no copy anywhere.

What a first ingest overwrites, and whether a copy survives elsewhere:

| file | overwritten by a review | survives elsewhere? |
|---|---|---|
| neuron.mat (all N candidates: A, C, S, options) | yes, becomes the kept set | partly: review_neuron.mat = neuron.mat minus auto-rejected candidates (curator `_write_review_mat`); at threshold 0 that is everything, at T > 0 the auto-rejected candidates' footprints and traces survive NOWHERE after ingest (only their feature rows in candidate_features.npz) |
| spatial_footprints.mat (full set) | yes, kept set | no (the 10-06 `.rerun_mincorr070_*` copies were a one-off) |
| Coor.mat, A/C/C_df/C_raw/S.txt, Cnn.txt | yes | no, but derivable from neuron.mat in MATLAB |
| `<session>_neurons/neuron_*.png` | kept cells re-rendered | no, derivable |
| review_neuron.mat, Cn/pnr/Ybg*, report, summary, run_final_review.m, ROIs_candidates.jpg | no longer (BUNDLE_INPUTS guard, 09:26) | n/a |

Measured on the untouched day-1 pilot `082626-pnb104` (176 candidates): neuron.mat 46 MB,
A.txt 93 MB, C/C_df/C_raw/S.txt 52 MB together, spatial_footprints.mat 3.3 MB, Coor.mat
0.2 MB, PNG folder 8 MB -> about 0.2 GB for the full set, about 50 MB for neuron.mat +
spatial_footprints.mat only. Day-2 sessions roughly double. The session folder is ~4.7 GB
(tif + .mat video). D: had 7.1 TB free at 10:15.

Proposed behaviour (decide the file set with the user; full set recommended, it is 4% of
the session):
- In the first-review path of `copy_session` (no labels.mat locally), before the first file
  is written: for every target that exists and will be overwritten, copy2 it to
  `<session>/.pre_ingest/<relpath>` (dot folder: skipped by iter_sessions, the watcher,
  the trainer, push, training and rereview scanners, same mechanism as `.rerun_*`,
  `.prev_est85_*`, `.rereview`).
- Write once: if `.pre_ingest/` already exists, never overwrite it (the first snapshot is
  the pristine headless state); print that it was kept. `--force` must not override this.
- Report in the per-session block and the summary table (files, MB), and support --dry-run.
- Copy-only: no delete, move, rename; grep the file for those verbs before calling it done
  (CLAUDE.md section 1 applies to every exchange script). ASCII only.
- Test against a temp directory first (CLAUDE.md section 2); `agent/test_ingest_guard.py`
  and `agent/test_ingest_inputs_guard.py` show the pattern.
- Optional: a tiny `PRE_INGEST.txt` in the folder with the ingest timestamp and the list,
  so a later reader knows what the folder is.

Known consumers that must keep ignoring dot folders: ingest `iter_sessions` and
`copy_session` (already skip dot parts), watcher, train_classifier*, push_review_bundle,
push_training_bundle, rereview.py, session_quality_report.py, scan tools. Worth a grep for
`startswith(".")` in any new scanner.
