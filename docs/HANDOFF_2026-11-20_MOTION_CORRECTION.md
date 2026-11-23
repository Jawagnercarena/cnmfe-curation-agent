# HANDOFF 2026-11-20: motion correction -- full-session check, a rule for process_2p, re-run of 090826-pnb104 and 090826-pnb102

Written 2026-11-20 for a fresh session. The session that wrote this diagnosed the
problem on raw SAMPLES (about 8% of each session) and changed nothing in production.
Your job is the three follow-ups the user asked for the same day, in this order:

- **A. Full-session check.** Re-register the two flagged sessions from the raw data,
  whole session, and measure whether the sample result holds.
- **B. A standing rule for the upstream pipeline** ("a set processing rule that we
  can implement into process_2p.bat").
- **C. Re-run the two sessions through ACORN** with the new movies ("Rerunning the
  two sessions seems smart").

The user approved the direction, not specific file changes. Section 7 lists what to
ask before acting.

## STATUS 2026-11-21 19:00 (second session, continued): job B INSTALLED; job C DONE and verified; push = user's action; 2026-11-22 01:25: 3odor 093026-pnb102 + 100926-pnb102 RE-MADE in D:\2p data\3odor (see below)

Read this block first; the 11-20 block below is the measurement record.

**Job C DONE (verified 18:55-19:05).** Background run 15:07:55-18:54:45, log
`agent/logs/rerun_cohort5_mc_new_movie.log`, "0 session(s) skipped or failed", no
ERROR / TIMEOUT / Traceback. Parameters as on 10-29 apart from min_pnr (pnb104 6.4, was
6.6; pnb102 7.1, was 7.0); gSig 9 / gSiz 36 / min_corr 0.70 (override) / bd 10. Headless
106.2 min (pnb104) and 112.2 min (pnb102). In each folder 22 top-level entries were
rewritten after 15:07:55; only the new tif (11-20) and `review_assigned.txt` (Taylor,
10-30 12:03) were not; no labels.mat / ROIs.jpg / labels_provenance.txt; archive intact
(24 entries, newest 15:06/15:07; archived tif hashes match the README). The vCA1 watcher
only read the two folders for its gSig/gSiz pass (15:27, 16:57, 18:53); it ran nothing
on them.

| session | Oct candidates | new candidates | auto-rejected (stored) | to review (stored) | merge pairs Oct -> new |
|---|---|---|---|---|---|
| 090826-pnb104 | 276 | 228 | 92 @0.01 -> 91 @0.02 | 184 -> 137 | 58 -> 3 |
| 090826-pnb102 | 333 | 346 | 100 @0.01 -> 114 @0.02 | 233 -> 232 | 29 -> 40 (1 motion flag, neuron 241) |

Same model, same threshold (current vCA1 model, 211 sessions, T 0.02, applied to both
stored feature matrices; October's matrix carries the October first-pass neighbour
column, so its row is approximate): pnb104 Oct 181 to review vs new 137; pnb102 Oct 228
vs new 232. Rescoring the new matrices reproduces the stored 91 / 114 exactly. Score
>= 0.5: pnb104 39 -> 46, pnb102 51 -> 43. Script:
scratchpad `rescore_oct_vs_new.py` (not kept). Whether the new candidates are better is
for Taylor's review to show; nothing here measures that.

Push dry run 19:05, both sessions to Taylor: clean, 8 items each (3.57 GB / 4.62 GB);
"red guide: would be built at push (drawn for 276/92, the live set is 228/91)" and
"(333/100 vs 346/114)". Watch the real push output: it must say `copy red_guide.png ...
replaces a different copy`. If it says "pushing without a guide", the 11-02
`red_guide.png` still in Taylor's outbox folder is STALE and must be removed by the
operator (server delete = human). Tell Taylor to discard any copy of the 10-29 bundles.

**Job B INSTALLED 2026-11-21 15:33 on the user's "Yes, install the rule draft."**
`D:\2p data\scripts`: `mc_rule.py` added, `process_2p.py` and `run_suite2p.py` replaced
by the draft (SHA-256 equal to `scripts_proposed\`; the other four files unchanged).
The replaced versions are in `D:\2p data\scripts\backup_2026-11-21_before_mc_rule\`
(equal to `scripts_original\`, with a README saying how to go back). `ops_user.npy`
untouched (last written 2026-07-22), so the suite2p GUI on its own still uses
maxregshift 0.1; the override lives in `run_suite2p.REG_OVERRIDES` and applies to
`run_process_2p.bat`, `process_2p.py` and `run_suite2p.py` whenever no `--ops` file is
given. It changes no existing movie: the 71 3odor movies were made 11-08..11-11 with
the old settings. Post-install test (15:31-15:35), installed scripts run in the suite2p
env on two SSD scratch copies of 8 raw trials of 090826-pnb104 (copied read-only from
the server): (1) default run: reference recentred (y +12, x +52) from 83 frames, QC
row bad 0.33% / at limit 0.0%, shift record written (4499 shifts = 4499 raw frames,
settings maxregshift 0.2 + force_refImg), TSeries folder deleted only after the AVG
tif verified (899 frames); (2) same run with QC_MAX_BAD forced to -1: flagged, TSeries
folder KEPT, record and QC row written. Both AVG tifs byte-identical (SHA-256). Scratch
copies deleted afterwards.

**User's answers 2026-11-21** (section 7): the new movies "do look better" in Fiji;
5% bad frames stays the QC line; archiving the old files and re-running both sessions
while the watchers are up is approved; the reviewers get the new versions ("see if it
works"); 090726-pnb104 and 091026-pnb104 are NOT redone (already curated). The job-B
install was a separate plain yes (above).

**Job C, done so far** (script `agent/eval/motion_correction_2026-11/rerun_new_movie.py`,
`prepare` and `run`, both dry run unless `--go`; scratch-replica test 42 of 42 checks):
- 15:06-15:07 `prepare --go` on both sessions. In each session folder
  `.prev_maxregshift01_2026-11-21/` now holds the old tif and old movie `.mat` (renamed
  into it), the `<session>_neurons/` PNG folder (renamed), a checked COPY of the 20 other
  output files, and `ARCHIVE_README.txt` with the hashes. The originals of the outputs
  were left in place on purpose: the watcher keys on `neuron.mat`, `review_report.pdf`
  and `Ybg_weights.mat` and would start its own job if one went missing; the new run
  overwrites them. `review_assigned.txt` untouched (Taylor, 10-30 12:03).
- The old and the new tif have the SAME size (pnb104 4,419,748,108 bytes, pnb102
  5,512,102,156), so only a hash tells them apart. SHA-256, first 16 hex digits:
  pnb104 old f848a41a08ee7930 / new 7dac6088a1f975ed; pnb102 old 049569eddce86c2d /
  new 72f21489934d12d1.
- 15:07:55 `run --go` started as one background process, pnb104 first, then pnb102
  (log `agent/logs/rerun_cohort5_mc_new_movie.log`). Per session: movie `.mat` rebuilt
  from the new tif through `movie_mat.ensure` (pnb104: 104 s, 32 of 32 sampled frames
  equal to the new tif), then `watcher.process_session` as in `rerun_queue.py`.
  Curation at the deployed model's threshold, which is 0.02 since 11-20 17:49 (the
  10-29 run of these two sessions was curated at 0.01); `param_overrides.json` gives
  both animals min_corr 0.70 as before. Expect about 2 h per session (114 and 120 min
  on 10-29). It occupies the second MATLAB slot; the vCA1 watcher was on the last of
  three 3odor sessions when it started.
- If the process is gone and a session did not finish: the folder then holds a mix of
  old and new outputs. Check the log, then run `rerun_new_movie.py run <session> --go`
  again (it reuses the rebuilt `.mat`; `prepare` must NOT be run again).

**Push (not done, the user's action):** `/push-review-bundle` for both sessions once
the runs are verified. Server state read 15:10: Taylor's outbox holds the 10-29 bundles
(9 items each, with the 11-02 `red_guide.png`); no return for either session in any
inbox. The push replaces every file that is not byte-identical. The new movie's field
sits (+14, +47) px (pnb104) and (+4, +36) px (pnb102) from where it was in the old
movie; the retro hook handles that (`red_guide.py`: a movie newer than the guide makes
it stale, and a movie newer than the cached mean image forces a re-fetch and a new
registration). Read in the code, not run.

**3odor question:** the 71 movies in `D:\2p data\3odor` were written 11-08 to 11-11
with the old settings; the rule would not touch them unless they are re-registered
from raw. Raw exists on the server for all 71
(`X:\Jeremy\2p data\bla-pnb 3odor\<MMDDYY>\TSeries-...`, 2,311 GiB in all, median
31 GiB). Their delivered-movie scores: `logs/3odor_delivered_scores.csv` in the record
folder (`score_delivered_dir.py`; known-answer row first: old pnb104 9/8 = 16.61% /
29.27%, as on 11-20). Result, 71 movies: median 0.02% bad; 7 above 1%, 4 above the 5%
line: 100826-pnb106-258um 25.4% (bad = trials ~84-113, the last 30% of the session;
sheet: "matlab crashed on trial 25. no recording afterwards"), 100926-pnb102-452um 7.7%
(first 30% of the session), 092926-pnb105-430um 5.3% (one of the three NOT-run hazy
sessions), 093026-pnb102-600um 5.2% (first tenth, trials ~2-9). Also 092926-pnb99 2.9%,
spread evenly (dim session, already run in ACORN today). In all four checked
(`motion_metrics.py`) the bad frames show no lateral offset (median 0.2-0.3 px, p90 <=
0.6 px); but undoing a rigid shift also recovered 0% of the bad frames of the old pnb104
9/8 movie, so that test does NOT show whether re-registration would help. The one
difference: 10% of pnb104's bad frames sit at the metric's 40 px edge, none in the 3odor
ones (suggestive only).

**3odor raw-sample test DONE 2026-11-21 21:00-21:25 (user's "Yes, run the raw-sample
test on those three").** Script `raw_sample_test.py` (record folder; suite2p env;
server read only): 7-8 cycles from inside the bad stretch + 4 clean cycles, registered
with the OLD settings and with the INSTALLED rule (`mc_rule.recentred_reference` +
`REG_OVERRIDES`, imported from `D:\2p data\scripts`), all three movies scored against a
template from the clean cycles. Summaries `logs/raw_sample_3odor/<TSeries>.json`, log
`run.log` there (first attempt of two sessions failed on `reg_tif` needing a save path;
fixed, re-run). Old settings reproduce the delivered movie in every case (pixel r
0.9994-1.0000, same bad share). Bad-stretch cycles, bad % old -> new (degraded),
raw frames at the +/-51 px limit with the old settings:
- 093026-pnb102-600um: 74.05 -> 0.14 (82.8 -> 3.9), at limit 31.2% (clean cycles
  2.0%); every bad cycle 44-97% -> 0-1%. Reference moved (+6, -32). FIXED by the rule.
- 100926-pnb102-452um: 45.3 -> 0.71 (54.8 -> 10.1), at limit 21.3% (clean 1.9%);
  bad cycles 31-70% -> 0-4%. Reference moved (+2, -39). FIXED by the rule.
- 100826-pnb106-258um: 100 -> 100 under both, at limit 0.2%; none of the 953 still-bad
  frames recovers at ANY x-y shift; mean intensity unchanged. NOT a registration
  problem: `cycle_trace.py` on the delivered movie (template cycles 30-60) shows the
  field drifting slowly through the WHOLE session -- r/quiet 0.55-0.68 in cycles 1-12,
  ~1.0 in cycles 43-55, 0.82 by cycle 70, 0.59 by 76, bad frames from cycle 77
  (9%), 80 (60%), 82 on (97-100%), 0.07-0.10 by cycles 108-120. `late_stretch_check.py`:
  early and late templates correlate 0.20. Sheet note "matlab crashed on trial 25. no
  recording afterwards" may make most of the session unusable anyway (user's knowledge).
  Raw folder is named 258um but its files, XML and SingleImages say 158um.
OPEN, user's decision: 100826-pnb106: truncate, run as is, or drop. Not tested:
092926-pnb105 (not run), and the three movies at 1-3% (092926-pnb99 2.9% already run,
101626-pnb106 1.5%, 101526-pnb102 1.0%).

**093026-pnb102 + 100926-pnb102 RE-MADE 2026-11-22 00:54-01:25 (user's "Yes, re-make
the two pnb102 sessions").** Raw TSeries copied server -> scratchpad `remake_3odor\`
(robocopy, top level only, 123 files / 31.3 GB each, names + sizes checked) and run
through the INSTALLED `process_2p.py` (ops_user + override, `--keep-tseries`;
registration of both side by side with `--skip-average`, 1381 / 1392 s, then the Fiji
step + QC one at a time with `--skip-suite2p`). Logs `logs/remake_3odor/`. Reference
recentred by (+6, -32) / (+2, -39), as in the sample test. QC rows (same
`mc_rule.movie_quality` that scored the delivered movies; installed mc_rule.py is
byte-identical to the draft): bad / degraded 5.23 / 6.39 -> 0.01 / 0.36 (093026) and
7.71 / 11.22 -> 0.09 / 3.01 (100926), at limit 0.0 / 0.0, unflagged. `remake_check.py`:
frame counts 12764 / 12768 = delivered = XML // 5; rigid shifts identical to the sample
test at all 5851 / 6384 sampled raw frames; sampled averaged frames r >= 0.9998.
Raw frames shifted > 51 px (wrapped strip): 0.13% / 0.48% (100926 has real runs at
about (-55, -30)); max shift 98 / 101 px. `swap_in_remade_movies.py` (tested on dummy
files first; dry run, then `--go`) put them in `D:\2p data\3odor`: old tifs MOVED to
`D:\2p data\3odor\.prev_maxregshift01_2026-11-22\` (README there has both SHA-256s;
old and new have the SAME byte size), shift records in `D:\2p data\3odor\mc_record\`,
QC rows in a new `D:\2p data\3odor\mc_qc.csv`. Verified by hash after the swap. The
stager (`stage_3odor_2026-11-21.py`, non-recursive) dry-runs both at min_corr 0.7;
NOT staged. Scratch TSeries copies deleted (250 GB); no other copy of the old movies
existed (server, D:\Julian_CNMFe checked).

## STATUS 2026-11-20 19:10 (second session): A done, B drafted and tested, C not started

Nothing in production was changed: `D:\2p data\scripts` is byte-identical to its
state at 16:45 (SHA-256 checked), `ops_user.npy` untouched, the two ACORN session
folders untouched, nothing committed. Everything below is measured on the full
sessions unless marked.

**Job A result** (bad / degraded = share of AVG5x frames below 0.5 x / 0.75 x the
movie's quiet level; N = all averaged frames of the session):

| Session (frames) | delivered | new: maxregshift 0.2 + recentred reference |
|---|---|---|
| 090826-pnb104 (16860) | 16.6% / 29.3% | 0.6% / 8.0% (single-pass recipe of section 2: 0.4% / 8.2%) |
| 090826-pnb102 (21027) | 10.2% / 16.7% | 4.3% / 10.3% |
| 082726-pnb104, control (16861) | 0.9% / 2.2% | 0.0% / 0.6% |

- Raw frames within 1 px of the rigid search limit: control with UNCHANGED settings
  2.88% (limit 51 px); new settings (limit 102 px) 0.00% (pnb104 9/8), 0.06% (pnb102
  9/8), 0.00% (control).
- Control with unchanged settings: the new AVG5x tif is byte-identical to the
  delivered one (same SHA-256, 4.4 GB). The pipeline is deterministic and today's
  `ops_user.npy` is what made that movie. The "pixel r 0.85-0.88" limit of section 1
  was an artefact of the sample comparison, not a different pipeline.
- pnb102 9/8 keeps 4.3% bad frames, mostly 10-12 s into the trial in a subset of
  trials (figure). In those frames suite2p's shift is arbitrary (spikes to +/-100 px).
- The 8-bit scale differs as predicted: pnb104 9/8 centre-crop mean 57.7 (delivered)
  vs 72.5 (new, single-pass movie), 400 sampled frames; pixels at 255: 0.004% / 0.012%.

**The recentring recipe of section 2 is biased; use the iterated one.** The
phase-correlation peak falls with the size of the shift (r = -0.64 on the 300
reference frames of pnb104 9/8), so "median offset of the better-correlated half"
under-estimates how far suite2p's reference is from the field's usual position:
(+13, +24) px from the better half, (+14, +45) from all 300 frames. Iterated recipe
(`build_session_ops.py --recipe iter`): median of all frames but the worst-correlated
10%, rebuild there, repeat until within 2 px. pnb104 9/8: total move (+14, +47), 67
frames in the reference instead of 16, median shift of the full run (+1, +2) instead
of (+2, +25), raw frames shifted by more than 51 px 1.50% instead of 3.18%. pnb102
9/8 and the control: the two recipes agree within 1 px. The 40 extra bad frames of
the iterated run on pnb104 (102 vs 62 of 16860) are borderline frames, r about 0.5 x
quiet in both runs.

**Clean-session test of the wider search** (section 3 item 1; 12-trial samples,
`fetch_sample.py` + `reg_experiment2.py`, logs in `logs/<session>.log`): 090826-pnb105,
-pnb99, -pnb109 (vCA1) and 06072023-DG7A (DG AL) give identical scores for production,
0.2 and 0.2 + recentred (0.0% bad). 081126-bla36 (BLA): production 1.5% of raw frames
at the limit and 0.3% bad, new settings 0.0% and 0.0%. N = 5 sessions, samples only.

**Job B draft** (NOT applied): `D:\2p data\mc_rerun_2026-11\scripts_proposed\`
(`mc_rule.py` new, 362 lines; 50 changed lines in `process_2p.py`, 35 in
`run_suite2p.py`); diff in
`D:\2p data\mc_rerun_2026-11\process_2p_rule.diff` and
`agent/eval/motion_correction_2026-11/job_b_process_2p_rule.diff`; pristine copies in
`scripts_original\`. Tested on an 8-trial scratch session: movie byte-identical to
the production script + ops file; clean session -> QC row, record, folder deleted
after `verify_avg_tif()`; forced QC failure -> folder kept. The rule's reference
images are `np.array_equal` to the ones used for the three full runs, and its QC
reproduces the scores above (44-64 s per session; record 1.6-1.8 MB; a full `ops.npy`
is 305-374 MB). With the 5% / 1% limits none of the three new runs is flagged; the
control under unchanged settings is (2.9% at the limit).

**Job C: not started.** Waiting for the user to look at the movies and to answer
section 7. Movies, shift records and scores:
`D:\2p data\mc_rerun_2026-11\movies\new\` (the two sessions, iterated reference),
`movies\single_pass\` (pnb104 9/8, section-2 recipe), `movies\ctrl_*`, `scores\<run>\`
(`ops.npy`, metrics, json), `fig\` (delivered vs new). Use the `movies\new\` tifs.

**Facts that correct or add to this file:**
- Run on the SSD, not on D:. D: is one SATA hard disk: the same 8-trial conversion
  took 494 s on D: (while a copy was running) and 5.6 s on C:. A full session on the
  SSD, two or three runs side by side: 37 min (84,303 frames) to 48 min (105,137);
  copy from the server 90-131 s per session.
- suite2p halves uint16 input (`io/tiff.py:131`); a forced `refImg` must be built
  from halved frames because `norm_frames` clips every frame to the reference's
  1st-99th percentile. `build_session_ops.py` and `mc_rule.py` do this; the sample
  scripts of the first session did not (internally consistent, so their results stand).
- `data_raw.bin` is not raw: suite2p 0.14.6 writes the rigid shift back into it
  (checked 60 of 60 frames per run). Irrelevant for the movies; it matters for anyone
  who wants to re-register from that file.
- Fiji loads the whole registered movie (and a trimmed duplicate): never run two
  averaging steps at once. Use `--skip-average` / `--skip-suite2p` to stagger.
- The LG utility was closed before this session started (50 of 431 GiB committed at
  16:44); section 5 is history.
- Scratch raw copies were deleted after scoring (the server has the originals).

## 0. Ground rules specific to this work

- The server (`X:`, CLAUDE.md section 1) is read-only. The raw data are copied FROM it
  with a copy-only tool. Never use robocopy `/MOV`, `/MOVE`, `/MIR` or `/PURGE` with
  the server on either side.
- The upstream scripts live in `D:\2p data\scripts` (`run_process_2p.bat` ->
  `process_2p.py` -> `run_suite2p.py`, `avg_project.py`). The user called the place
  "Analysis/2p data"; `D:\Analysis\2p data` does not exist (checked 11-20). The
  folder is NOT under git (no `.git` in it or its parent): there is no undo. Copy
  the originals aside and show the user the diff before replacing anything there.
- `process_2p.py` deletes the WHOLE TSeries folder (raw tifs, xml, suite2p output)
  once the averaged tif verifies. Every run you make uses `--keep-tseries`, on a
  scratch task directory, never on `D:\2p data\6odorDualDiffRew` or `D:\2p data\3odor`.
- Do not overwrite `C:\Users\julia\.suite2p\ops\ops_user.npy` (the suite2p GUI's saved
  defaults, read by every production run). Write alternative ops files elsewhere and
  pass `--ops`.
- Do not touch the two ACORN session folders until the user has looked at the new
  movies. Nothing in them is deleted or overwritten; old files are archived.
- suite2p (0.14.6) imports only in `C:\Users\julia\.conda\envs\suite2p\python.exe`.
- Check the machine's commit charge before every heavy job (section 5). One heavy
  job at a time; a headless MATLAB run is about 75 GB and a busy watcher counts.

## 1. What was found on 2026-11-20 (measured unless marked)

**Pipeline.** `process_2p.py` runs suite2p registration on the raw Prairie View
per-trial ome.tifs, Fiji averages every 5 frames and converts to 8-bit
(`AVG5x-<session>.tif`), then the local TSeries folder is deleted, so suite2p's
`ops.npy` (xoff / yoff / corrXY) is gone for every delivered movie. Raw data remain
on the server: `X:\Jeremy\2p data\6odorDualDiffRew\<MMDDYY>\<TSeries>\`.

**Settings in force** (`ops_user.npy`, modified 2026-07-22, re-read 11-20): nonrigid
True, block_size [32, 64], maxregshiftNR 7, **maxregshift 0.1 (+/-51 px of 512)**,
smooth_sigma 1.15, smooth_sigma_time 0, snr_thresh 1.2, nimg_init 300, batch_size
400, two_step_registration 0, keep_movie_raw True, reg_tif True, roidetect False,
force_refImg False.

**The two sessions** (vCA1, 6odorDualDiffRew, zoom 3, 0.385 um/px, 30.2 Hz raw):

| | 090826-pnb104-425um | 090826-pnb102-497um |
|---|---|---|
| Raw on the server (`...\090826\TSeries-<name>-3z-000`) | 150 ome.tif, 44.3 GB | 187 ome.tif, 55.2 GB |
| Delivered AVG5x movie | 16860 frames | 21027 frames |
| Imaging sheet | "Too much motion, JSB confirmed" | "Too much motion" |

**Delivered-movie score.** Metric = share of AVG5x frames whose correlation with the
session's own clean template is below 0.5 x ("bad") or 0.75 x ("degraded") the
session's quiet level (90th percentile). 17 movies scored: all 13 pnb102 / pnb104
sessions and the four other animals imaged on 9/8.

| Movie | bad | degraded |
|---|---|---|
| 090826-pnb104 | 16.6% | 29.3% |
| 090826-pnb102 | 10.2% | 16.7% |
| 090726-pnb104 | 4.6% | 11.5% |
| 091026-pnb104 | 3.8% | 9.9% |
| the other 13 | <= 1.0% | <= 6.0% |

**Cause.** Production settings were re-run on 12-13 raw trials per session with the
same 300 reference frames suite2p takes from the whole session:

- Raw frames pinned at the +/-51 px search limit: 30.3% (pnb104 9/8), 6.4% (pnb102
  9/8); on two good days 3.6% (082726-pnb104) and 8.1% (090726-pnb102).
- suite2p's reference image sits far from where the field usually is: median rigid
  offset (y, x) = (+14, +43), (+8, +33), (+5, +27), (+38, +2) px in those four.
- On 9/8 the resting field position also drifted about 25-30 px across the session
  in both animals. suite2p shifts with `np.roll`, so every large shift wraps a strip
  of image to the opposite edge; the delivered pnb104 9/8 movie shows that strip.

**The tested change** (same raw trials; bad / degraded on the 5-frame averages):

| Session (N averaged frames) | production | maxregshift 0.2 | 0.2 + recentred reference | 0.3 + smooth_sigma 2 + recentred |
|---|---|---|---|---|
| pnb104 9/8 (1449) | 17.2% / 32.8% | 0.9% / 11.7% | 1.2% / 11.8% | 0.8% / 11.5% |
| pnb102 9/8 (1340) | 12.8% / 21.7% | 6.0% / 14.1% | 7.8% / 14.1% | 5.1% / 13.4% |
| 082726-pnb104 (1339) | 0.8% / 2.3% | 0.0% / 0.3% | not run | not run |
| 090726-pnb102 (1341) | 1.3% / 6.9% | 0.0% / 4.5% | 0.0% / 4.6% | not run |

- maxregshift 0.3 gives the same as 0.2. Rigid-only, 128-px blocks, 1-frame temporal
  smoothing and a second pass were no better or worse (logs in the record folder).
- Recentring the reference does not change the bad rate; it cuts the raw frames
  shifted by more than 51 px (the wrapped strip) from 45.4% to 4.0% (pnb104 9/8),
  16.8% to 4.0% (pnb102 9/8) and 6.8% to 0.2% (090726-pnb102).
- Not repairable by x-y registration: 2.8% (pnb104 9/8) and 7.3% (pnb102 9/8) of raw
  frames match the template at no shift at all (0.0% on the two good days). They
  come as multi-second episodes after reward in some trials (focal-plane motion).
  Expect pnb102 9/8 to keep about 6% bad frames whatever the settings.

**Limits.**

- Samples only: 13 of 150 trials (pnb104) and 12 of 187 (pnb102).
- The reproduction is pixel-exact only for pnb102 9/8 (pixel r median 0.992 against
  the delivered movie). For the other three sessions the frame POSITIONS agree with
  the delivered movie (zero displacement in 97.7%, 100% and 99.8% of 447-483 sampled
  averaged frames) but pixel r is 0.85-0.88; a 1-2 raw-frame grouping offset was
  ruled out, the cause was not found. Who produced each delivered movie, and on
  which machine, was not verified.
- The "bad" / "degraded" thresholds are the author's definition. Their ranking agrees
  with the reviewers on N = 2 flagged sessions.

## 2. Job A -- full-session check

Answer given to the user: yes, do it. It is the same run that produces the movies
for job C, so the only extra cost is scoring it.

1. Check section 5 (commit charge) and that no MATLAB run is active.
2. Make a scratch task directory on D: (6,545 GB free on 11-20), for example
   `D:\2p data\mc_rerun_2026-11\`, and copy the two TSeries folders into it (the
   ome.tifs and the .xml; `References\` is not needed). Verify file counts and bytes
   against the table above. A run needs roughly 4x the raw size (raw copy, two
   suite2p binaries because keep_movie_raw is on, reg_tif) -- estimated, not measured.
3. Build one ops file per session, saved in the scratch task directory:
   `ops_user.npy` + `maxregshift` 0.2 (+ `smooth_sigma` 2.0 if you test that arm) +
   the recentred reference as `refImg` with `force_refImg` True. `run_s2p.py:110`
   uses `ops["refImg"]` only when `force_refImg` is set. This path was exercised at
   function level (`register.compute_reference_and_register_frames(refImg=...)`),
   NOT yet through `run_s2p`; test it on a few trials first.
   Recentred reference, as in `reg_experiment2.py` lines 121-139: suite2p's own
   reference from its 300 frames (`np.linspace(0, n_frames, 301, dtype=int)[:-1]`
   over the natsorted files, `register.py:396`) -> rigid-register those 300 frames
   to it -> median offset of the better-correlated half -> average the frames within
   5 px of that offset (10 px if fewer than 15), each rolled onto it.
   `fetch_ref_frames.py` reads exactly those 300 frames.
4. Run each session on its own (one ops file each):
   `C:\Users\julia\.conda\envs\suite2p\python.exe "D:\2p data\scripts\process_2p.py" --task-dir "D:\2p data\mc_rerun_2026-11" --only pnb104 --ops <file> --keep-tseries`
   (`--dry-run` first). The sample runs took 120-160 s per 7,000 raw frames for
   registration alone; a full session is 84,000-105,000 frames.
5. Score the result:
   - `motion_metrics.py <new AVG5x tif> --out <dir>`, then `summarize.py`; compare
     with the delivered scores above.
   - From the kept `suite2p\plane0\ops.npy`: share of frames within 1 px of the
     search limit, share beyond 51 px, median offset, corrXY.
   - Give the user both movies to open side by side in Fiji. Their eyes and the
     reviewers' are the real test; the metric is a proxy.
6. Recommended control: one good day through the same two runs (unchanged ops, and
   the new ops), for example `082726-pnb104-416um-3z-000` (same field as pnb104 9/8).
   The unchanged-ops run settles whether these settings reproduce a delivered movie
   pixel for pixel; the new-ops run shows no harm at full length.

Expected from the samples: pnb104 9/8 near 1% bad, pnb102 9/8 near 6%. If the full
session is clearly worse than that, stop and report before job C.

## 3. Job B -- the rule for process_2p

`run_process_2p.bat` only activates the env and forwards its arguments; the rule
belongs in `process_2p.py` / `run_suite2p.py` (`session_ops()` currently overrides
data_path, save_path0, fast_disk, fs, look_one_level_down) or in `ops_user.npy`.
Proposal to validate in job A and then show the user as a diff:

1. **Search range: maxregshift 0.2.** Evidence is four sessions from two animals in
   one area. Before making it the default for everything, run the sample test
   (`reg_experiment2.py`) on two or three sessions whose current movie is clean
   (pnb99 / pnb105 / pnb106 / pnb109 on 9/8 scored 0.0% bad) and, if raw data can be
   found, one BLA and one DG_AL session: a wider search could pick a wrong peak on a
   low-signal frame, and that has not been tested outside pnb102 / pnb104.
2. **Recentre the reference** when the 300 reference frames sit more than a few px
   from suite2p's reference (27-50 px in all four sessions tested). This is what
   removes the wrapped edge strip.
3. **Keep the shift record.** Before the delete step, save xoff, yoff, corrXY,
   refImg, meanImg and the settings used next to the averaged tif. `ops.npy` itself
   also holds the non-rigid arrays; its size was not checked.
4. **QC line and gate.** After averaging: share of frames at the search limit,
   reference offset, and the `motion_metrics.py` bad / degraded shares; one line per
   session into a CSV. A session over the threshold keeps its TSeries folder and is
   marked for a human. Starting thresholds, from 17 movies with 2 flagged (thin):
   flag above 5% bad (reviewers accepted up to 4.6%, rejected 10.2% and 16.6%), or
   when more than 1% of frames reach the limit. The gate must not weaken
   `verify_avg_tif()`.
5. Not worth adding: temporal smoothing, rigid-only, larger blocks, two-step.

Where the setting lives is the user's call: `ops_user.npy` is shared with the
suite2p GUI (the `run_suite2p.py` docstring says the script deliberately follows
it); an override in `session_ops()` is visible in the code and printed per run.

Test any edit per CLAUDE.md section 2: scratch task directory, a copied session,
`--dry-run`, then `--keep-tseries`. `motion_metrics.py` lives in this repo; the 2p
scripts do not import from it, so the gate needs its own copy of the metric.

## 4. Job C -- re-run the two sessions through ACORN

State of both session folders under `D:\Julian_CNMFe\vCA1\6odorDualDiffRew\`
(listed 11-20): delivered tif from 2026-09-19, headless outputs and review bundle
from 2026-10-29, `review_assigned.txt` from 10-30 (Taylor), no `labels.mat`, no
`ROIs.jpg`, and the movie `<session>.mat` is present. Both watchers were alive and
idle at 15:10 with no MATLAB running.

Traps, each read in the code on 11-20:

- **The old `.mat` movie would be reused.** `run_cnmfe.py:175-177`: the tif is
  converted to `.mat` once; "on subsequent calls ... the .mat already exists and
  conversion is skipped". Putting a new tif in the folder is not enough: the old
  tif AND the old `.mat` have to be moved into an archive folder first (a rename
  inside the session folder; ask the user), then the `.mat` is rebuilt from the new
  tif (`rebuild_movie_mat.m` refuses to overwrite).
- **`agent/eval/cohort5_2026-09/rerun_queue.py`** is the precedent for re-running a
  cohort-5 session outside the watcher (`--sessions <name> --archive <folder>
  --threshold model --go`; history in `agent/eval/cohort5_2026-09/LOG.md`, entries of
  10-05, 10-18 and 10-28). It copies the old candidate set into a dot folder and
  calls `watcher.process_session`. It was written for a re-run on the SAME movie:
  its `check()` requires the `.mat` to exist and its archive list does not include
  the tif or the movie. Read it before using it.
- The new 8-bit movie is scaled by Fiji from its own first frame, so intensities
  are not on exactly the old scale; the pre-pass estimates will move. Animal-level
  overrides are in `agent/config/param_overrides.json`; read the curation threshold
  from `agent/model/vCA1/classifier.joblib`, not from any note.
- The candidate set changes, so the red guide must be rebuilt: the push does that
  through the retro hook (CLAUDE.md section 4). The 10-30 bundles for both sessions
  are still in Taylor's outbox on the server; the push is the user's action
  (`/push-review-bundle`), and anything stale there is the operator's to remove.
- pnb102 9/8 will still carry about 6% bad frames. Excluding frames from CNMF-E is
  not built; tell the user before they promise the reviewers a clean session.

## 5. Machine state on 2026-11-20

**Commit charge.** At 15:32 the system had 485.9 GB committed of a 525.8 GB limit
(decimal GB; Task Manager shows the same state as 452 / 490 GB; 255 GiB RAM plus a
system-managed page file), while all processes
together held 27.6 GB of private memory and most of the RAM was free. Parallel
analysis jobs died with allocation errors earlier in the day because of it.

Cause (measured): **LG Dual Controller 2.89**, the LG monitor utility that shares a
keyboard and mouse between PCs (`C:\Program Files (x86)\LG Electronics\Dual
Controller\bin\`, started at sign-in by the HKLM `WOW6432Node\...\CurrentVersion\Run`
value `DualControl`). Its three helper processes (`Dual_Controller_Drop`,
`Dual_Controller_OpacityDlg`, `DualControlFileTransferSession`) each held about
704,000 Section handles and gained 1.92 per second. Over a 10-minute sample the
commit that no process owns grew by 1.13 MB/s (4.1 GB/h), 589 KB per new handle in
one process; 702,000 x 589 KB = 414 GB of the 457 GB that no process owned. The
section sizes themselves could not be read because the processes run elevated. At
that rate the limit is about 10 hours away from 15:30; whether Windows would enlarge
the page file first was not verified.

It is not needed for a second monitor: its own help text (`bin\Help\English.js`)
calls it "a program that can be used to share keyboard and mouse connected to one PC
with another PC". It was installed on 2026-10-14 and had no connection to another PC
open (a TCP listener only).

**RESOLVED at 16:44: the user closed the utility.** Committed memory fell from 463.7
GiB (16:39) to 49.5 GiB (Task Manager units), system-wide handles from 2.33 million
to 225,000, and Windows shrank the commit limit from 489.7 to 431.2 GiB. The startup
entry is still there, so the leak starts again at the next sign-in unless the user
has disabled it since; run the check below first.

Also at 16:39-16:44: three MATLAB processes that neither the watchers nor the
writing session started were active (one since 15:44); find out whose they are
before starting anything.

History of the attempt, for the record: at 16:39 it was still running (about 712,000
handles per helper, 26 GiB of headroom). The user approved
closing it, but `Stop-Process` from a non-elevated shell was denied (all four LG
processes are elevated), and the user then asked what the utility is before closing
it. Only the user can exit it: tray icon > Exit, or Task Manager > Details >
`DualController.exe` > End process tree.

Before any heavy job, check:

    Get-CimInstance Win32_PerfRawData_PerfOS_Memory | Select-Object CommittedBytes, CommitLimit
    Get-Process Dual* | Select-Object Name, Id, HandleCount, StartTime

A healthy reading on 11-20 was about 50 GiB committed with no `Dual*` process
listed. The utility starts again at every sign-in
and leaks about 100 GB a day until its startup entry is disabled or it is updated or
removed (the user's decision; the entry is in HKLM and needs an elevated shell).

**Watchers.** BLA and vCA1 watchers were restarted by the user around 13:20-13:36 and
were heartbeating and idle at 15:10 (`agent/logs/watcher.log`,
`watcher_vCA1.log`); no MATLAB was running. Their command lines are not readable from
a non-elevated shell; detect them by the log heartbeat.

## 6. Where things are

- Record (untracked, 8 MB): `agent/eval/motion_correction_2026-11/` -- 13 scripts,
  `logs/` (every table row above except the two noted below), `fig/`, `xml_frames/`.
  - `motion_metrics.py <tif> --out <dir> [--chunk 64] [--limit N]`: the QC metric
    (`selftest.py`: known synthetic shifts recovered within 0.1 px).
  - `run_batch.py`, `summarize.py`: batch scoring, table and per-session figure.
  - `xml_info.py`, `fetch_raw_trials.py`, `fetch_ref_frames.py`: read-only readers
    of the server raw data (frames per trial, chosen trials, the 300 reference frames).
  - `reg_experiment2.py <session> --variants ... [--recenter ...] [--two-step ...]
    [--validate]`: suite2p variants on a raw sample (suite2p env).
  - `fixable.py`: can any x-y shift restore each raw frame. `validate_repro.py`:
    reproduced against delivered frames.
  - The scripts expect a scratch layout next to themselves (`raw/`, `reg/`, `reg2/`,
    `out/`, `out_xml/`); they were written in a scratchpad and copied here.
  - Not in `logs/`: the production line for pnb104 9/8 and the "no shift restores
    it" shares were recomputed on 11-20 from the saved arrays in the scratchpad.
- Bulky data, in the writing session's scratchpad (may be cleared at any time; all
  of it can be regenerated from the server):
  `C:\Users\julia\AppData\Local\Temp\claude\45dc093469cd1656\6110185a-8d68-4acf-9756-21c41a1390e3\scratchpad\mc\`
  -- `raw\` (22 GB: 12-13 raw trials and the reference frames per session), `reg2\`
  (19 GB: `prod.tif`, `mrs20.tif` ... sample movies for Fiji), `out\` (delivered-movie
  scores).
- Memory: `project_motion_correction_suite2p_2026-11-20.md`.

## 7. Ask the user before

1. Editing anything in `D:\2p data\scripts`, and whether the new setting goes into
   `ops_user.npy` or into the script.
2. Whether maxregshift 0.2 becomes the default for every animal and area, or only
   after the extra clean-session tests in section 3.
3. Moving the old tif, `.mat` and outputs of the two sessions into an archive folder,
   and starting a headless run while the watchers are up.
4. What to tell the reviewers about pnb102 9/8 (about 6% of frames stay bad).
5. Whether the other weak pnb104 days (090726: 4.6% bad, 091026: 3.8%) should be
   redone too; nobody flagged them.
