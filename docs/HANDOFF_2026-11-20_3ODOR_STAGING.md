# HANDOFF 2026-11-20: the 3odor sessions of the six cohort-5 animals -- which to run, at what seed, and the labelled dim run

Written 2026-11-20 for a fresh session and for Julian. Nothing here is staged or
running. The session that wrote this changed the deployed vCA1 model and the seeding
overrides the same day (section 1); the 3odor work itself is undecided (section 3).

Your job, once Julian says go: settle which of the 71 3odor sessions are run, confirm
the seed for each, and run the labelled dim experiment if he approves it.

## STATUS 2026-11-21 (read this first; it overrides "open" in sections 3, 5 and 6)

Julian decided on 2026-11-21 (record: `agent/eval/cohort5_2026-09/LOG.md`, entry of
that date):

1. **Not run**: 092926-pnb105-430um, 092926-pnb109-133um-001, 093026-pnb109-90um (the
   three redone days). 68 of the 71 are run. 092926-pnb102 is run: its field was
   imaged again on 10/9 and 10/15, but at later behavioural stages.
2. **Labelled dim run approved** on the three sessions of section 5, at min_corr 0.575,
   every candidate reviewed.
3. **The other dim sessions** (pnb99 x6, seed 0.60) are reviewed at threshold 0:
   `recurate_sessions_vCA1.py --threshold 0 <session>` before the push.
4. **Staged so far: only the three labelled-run sessions** (11-21 11:51-11:53, with
   `agent/eval/cohort5_2026-09/stage_3odor_2026-11-21.py`, which refuses the three
   not-run sessions). The remaining 65 (59 at 0.70, 6 at 0.60) wait for his order.

5. **The labelled run is done and ready to push (11-21 15:29), not pushed.** All three
   ran at min_corr 0.575 (about 68 min each, no error) and were re-curated at threshold
   0; verified from each `candidate_features.npz`:

   | Session | Candidates = review set | 0.02 would have auto-rejected |
   |---|---|---|
   | 092926-pnb102-460um (FOVA) | 164 | 68 |
   | 092926-pnb99-518um (FOVA) | 183 | 67 |
   | 100126-pnb99-493um (FOVC) | 104 | 37 |

   The watcher's 0.02 curation of each (with scores) is kept in
   `agent/eval/cohort5_2026-09/dimrun_2026-11-21/`; with the labels it gives the false
   auto-reject rate of 0.02 on dim fields. After the labels are ingested: that rate,
   and kept-rate by ceiling band with `analysis_2026-11-20/seeding/c8_c9_batch.py`.

Still open: who reviews the three (the push is the operator's; whole-FOV dealing means
that reviewer also gets the rest of pnb99 FOVA, pnb99 FOVC and pnb102 FOVA), the order
of the remaining 65, who reviews which FOV, and red guides (question 5 of section 6:
the retro project has no results for any 3odor day yet; the push hook would build each
guide from scratch, single images are on the server for these three, hook not run).
`param_overrides.json` now has 3 sessions at 0.575, 9 at 0.60 (the 3 not-run ones among
them, inert) and 59 at 0.70.

One caution on section 4's redo column: the redos repeat the behavioural day at a
different depth (430 -> 373 um for pnb105; 133 -> 440 and 90 -> 310 um for pnb109), so
they are not the same cells. Julian knew this when he decided.

## 0. Ground rules specific to this work

- The server is read-only (CLAUDE.md section 1). The 71 tifs are already local:
  `D:\2p data\3odor\AVG5x-TSeries-<session>.tif` (71 files, counted 11-20).
- Nothing is staged: `D:\Julian_CNMFe\vCA1\3odor\` holds 24 older sessions (pnb88,
  pnb98) and none of pnb99/102/104/105/106/109 (checked 11-20). Staging a tif there
  starts a headless run on the next watcher poll. Stage on purpose, a few at a time.
  The staging helper used for the 6odor batch is
  `agent/eval/cohort5_2026-09/stage_cohort5_28_2026-10-28.py`; read it before reuse.
- One heavy job at a time (commit charge, memory note reference_machine). A headless
  MATLAB run is about 75 GB; a busy watcher counts as one.
- Review work is dealt in whole FOVs (memory note project_fov_reviewer_assignment).
- A threshold-0 review is manual: `recurate_sessions_vCA1.py --threshold 0 <session>`
  on explicit paths BEFORE the push (the watcher curates at the joblib threshold).
- Every number below was measured on 2026-11-20 unless marked. Re-read the joblib,
  `param_overrides.json` and the imaging sheet before relying on any of them.

## 1. State on 2026-11-20 (each line checked that day)

- **vCA1 model**: `agent/model/vCA1/classifier.joblib`, trained 17:49, 211 sessions
  (86 reviewed + 125 bootstrap), `reject_threshold` 0.02. Earlier models of the day
  are in `agent/model/vCA1/history/` (the trainer now snapshots every model it
  replaces).
- **Threshold 0.02** is the wrapper constant (`train_classifier_vCA1.py`
  `_VALIDATED_THRESHOLD`), the joblib value and `gate_decision.json` `chosen_T`.
  Basis: `agent/eval/cohort5_2026-09/analysis_2026-11-20/threshold/`. Dim sessions
  were not part of that test.
- **Duplicate deletes** (a reviewed delete that is a second copy of a kept cell) now
  train at weight 0 in reviewed sessions that have `.pre_ingest` footprints (587 rows
  in 33 vCA1 sessions). Cache: `<session>/review_duplicates.json`.
- **Seeding** (`agent/config/param_overrides.json`): all six animals at 0.70,
  pnb105 included again. Session-level 0.60 for the 11 3odor sessions of section 4
  (and for 090526-pnb99, already run). An entry does nothing until its session is
  staged.
- **6odor batch**: 19 of 28 labelled. 9 still with Taylor (pnb104 x4, pnb106 x4,
  090826-pnb102). 090826-pnb104 and 090826-pnb102 are the "too much motion" sessions;
  Julian is re-registering them and Taylor reviews those two last
  (`docs/HANDOFF_2026-11-20_MOTION_CORRECTION.md`).
- All of the above is uncommitted on branch `training-program`.

## 2. Decisions Julian took on 2026-11-20

1. vCA1 auto-reject threshold 0.02 for sessions curated from now on.
2. min_corr 0.70 for bright sessions, 0.60 for dim ones, set per session by hand.
   No automatic bright/dim detector.
3. pnb105 back to the animal-level 0.70.
4. min_pnr +2 dropped. Red-soma exemption not built.
5. Duplicate deletes ignored in training, as for bootstrap sessions.

## 3. What is open, with Julian's two constraints

Open: which 3odor sessions are run at all, and whether the labelled dim run of
section 5 goes ahead. His constraints, verbatim:

- "We should make sure the dim 3odor sessions come from different FOVs (we can look at
  the imaging sheet in Analysis)."
- "if an FOV was revisited after (like with PNB105 and PNB109 on 10/2/26), then the
  previous dimmer sessions don't really need to be run."

The sheet is `D:\Analysis\3odor\2p imaging parameters 4.0 (retro) - 3odor.csv` (export
of 2026-11-20 06:00). FOV letters are per animal and per task (a 3odor "FOVB" is not
the 6odor "FOVB").

## 4. The 11 dim sessions (pk50 below 0.70), with FOV and redo status

pk50 = Cn of the 50th-strongest Cn peak of the full-movie seeding image; noise_rel =
relative image noise. Both from `agent/eval/3odor_intake_2026-11/redteam/c6_sessions.tsv`
(11-11). Day, FOV and note are the sheet's. "Redo" is my reading of the sheet notes,
not something Julian has confirmed line by line.

| Session | pk50 | noise_rel | Sheet: day, FOV, note | Redo on a later day? |
|---|---|---|---|---|
| 093026-pnb99-312um | 0.584 | 0.309 | D5, FOVB, "Still much dimmer" | No. FOVB was imaged only this once |
| 101626-pnb99-366um | 0.599 | 0.354 | D16 WSR d3, FOVA, "looks hazy and dim" | No (last session) |
| 092926-pnb105-430um | 0.609 | 0.318 | D4, FOVB, "super hazy and barely any cells" | YES: 100226-pnb105-373um, "cleaned lens with acetone and redoing from FOVA (0929)", pk50 0.866 |
| 092926-pnb109-133um-001 | 0.635 | 0.091 | D4, FOVA, "The lens is incredibly damaged" | YES: 100126-pnb109-440um, "We cleaned the lens! ... new real FOVA", pk50 0.894 |
| 092926-pnb102-460um | 0.648 | 0.251 | D4, FOVA, "Cells also much dimmer" | No |
| 100126-pnb99-493um | 0.657 | 0.205 | D6, new FOVC, "still dim" | No |
| 093026-pnb109-90um | 0.669 | 0.065 | D5, "try to find FOVB", lens still damaged | YES: 100226-pnb109-310um, "FOVB. redoing because past was really bad", pk50 0.752 |
| 100826-pnb99-492um | 0.685 | 0.196 | D11 rev d3, FOVC | No |
| 101426-pnb99-574um | 0.687 | 0.198 | D14 WSR d1, FOVD | No |
| 101526-pnb99-473um | 0.693 | 0.200 | D15 WSR d2, FOVC | No |
| 101326-pnb99-560um | 0.699 | 0.192 | D13 rev d5, FOVD | No |

Reading:
- **Three of the 11 have a redo** (092926-pnb105, 092926-pnb109, 093026-pnb109) and by
  Julian's second constraint need not be run. pnb105 and pnb109 were given extra days
  after the lens cleaning; their 10/2 sessions are labelled "redoing".
- **The other four animals have no redo.** For pnb99, pnb102, pnb104 and pnb106 the
  10/2 session is the next behavioural stage ("wsInitRev"), so their 9/29-10/1
  recordings are the only imaging of those days.
- **Eight dim sessions remain**: pnb99 x7 (FOVA 101626; FOVB 093026; FOVC 100126,
  100826, 101526; FOVD 101326, 101426) and pnb102 x1 (FOVA 092926). pnb99 stays dim
  after the cleaning; no other animal does.
- The two pnb109 sessions are low on pk50 because few cells are visible, not because
  the image is noisy (noise_rel 0.09 and 0.07). pk50 falls with cell count as well as
  with brightness (11-11 red team, C6), which is why there is no automatic detector.

Sessions between 0.70 and 0.80 (start at 0.70; watch the candidate count, the LOW
flag is N < 100): 100926-pnb99 (0.707, FOVA), 100226-pnb99 (0.736, FOVA),
092926-pnb99 (0.747, FOVA, noted dim), 100226-pnb109 (0.752, FOVB redo),
100126-pnb105 (0.755, new FOVC), 093026-pnb106 (0.783), 100226-pnb102 (0.787),
100826-pnb105 (0.790), 092326-pnb104 (0.791), 092326-pnb99 (0.791),
101426-pnb109 (0.792), 100826-pnb109 (0.796). The other 48 sessions are at 0.80 or
above.

## 5. The labelled dim run (proposed 2026-11-11, not approved)

Source: `agent/eval/3odor_intake_2026-11/redteam/REPORT.md` sections 6-7.

- **What**: run 2-3 dim sessions once at min_corr 0.575 (noise floor 0.55 + 0.025) and
  have every candidate reviewed (threshold 0). Each candidate has a ceiling (the
  highest seed at which it would still be found), so one review tells what 0.60, 0.65
  and 0.70 would each have kept and what review they would have spared.
- **Why it still matters after the 0.60 decision**: (a) whether 0.60 leaves cells
  behind in dim fields (on 090526-pnb99, the one labelled dim session, 2 of the 6
  reviewed candidates with ceiling 0.60-0.65 were kept); (b) where the bright/dim cut
  belongs (labels so far: 0.60 gained 6 cells at pk50 0.643, 1 at 0.713, 2 at 0.759).
- **Sessions proposed on 11-11, checked against the sheet on 11-20**:

  | Session | Animal, FOV | pk50 | Redo exists? |
  |---|---|---|---|
  | 092926-pnb99-518um | pnb99 FOVA (sheet: depth probably 418, mislabelled) | 0.747 | No |
  | 092926-pnb102-460um | pnb102 FOVA | 0.648 | No |
  | 100126-pnb99-493um | pnb99 FOVC | 0.657 | No |

  Three different fields, none superseded, so they meet both constraints. 092926-pnb99
  is the dim day of a field with labelled bright 6odor sessions, and it sits inside the
  undecided 0.70-0.80 band. If a third animal is wanted there is none: only pnb99 and
  pnb102 have dim sessions without a redo. Further distinct pnb99 fields are FOVB
  (093026, the dimmest at 0.584) and FOVD (101326, 101426).
- **What it needs**: Julian's go-ahead; session-level 0.575 in `param_overrides.json`
  for the chosen sessions (two of the three currently say 0.60, 092926-pnb99 resolves
  to 0.70); staging those tifs; one headless run each; threshold-0 re-curation before
  the push; a full review. First-pass seeds in the 10 dim-noted sessions on 11-11:
  median 53 at 0.70, 128 at 0.60, 171 at 0.575 (candidate counts grow more than seeds).
- **Analysis afterwards**: `agent/eval/cohort5_2026-09/analysis_2026-11-20/seeding/c8_c9_batch.py`
  already computes kept-rate by ceiling band for any labelled session.

## 6. Ask Julian before acting

1. Skip the three superseded sessions (section 4)? Any other pre-cleaning session he
   considers redone?
2. Approve the labelled dim run, and with which sessions?
3. Should the remaining dim sessions be reviewed at threshold 0 as well? The 0.02
   threshold was not tested on dim fields, and CLAUDE.md treats a new regime as a
   threshold-0 first pass. This is my suggestion, not his decision.
4. Order of staging and who reviews which FOV.
5. Whether the 3odor sessions get red guides (retro project; the push builds them).

## 7. Numbers to re-derive, not trust

- `n_sessions` and `reject_threshold`: read the joblib.
- Seeds: resolve each session through `agent/params.py` (animal level, then session).
- pk50: the table is from 11-11; the 11-11 report notes the sheet export it used was
  older than Julian's online fixes. The export read here is from 11-20 06:00.
- The redo column: re-read the sheet notes for the session in question.

## 8. Files

- Record of 2026-11-20: `agent/eval/cohort5_2026-09/LOG.md` (entries of that date),
  outputs under `agent/eval/cohort5_2026-09/analysis_2026-11-20/`
  (`threshold/`, `seeding/`, `duplicates/`, `red_somata/`, `reviewer_check_yawen.*`).
- 11-11 seeding work: `agent/eval/3odor_intake_2026-11/` (README, `redteam/REPORT.md`),
  `docs/HANDOFF_2026-11-11_SEEDING_THRESHOLD_TEST.md`.
- Earlier batch handoff: `docs/HANDOFF_2026-09-23_NEW_BATCH.md`.
