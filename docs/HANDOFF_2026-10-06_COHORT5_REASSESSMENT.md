# HANDOFF 2026-10-06: cohort-5 hyperparameter reassessment (vCA1, retro-tdTomato)

Written 2026-10-06 08:40 for the session that picks this up once Taylor returns the
12 re-run pilot sessions. Every number here was read from a file or a log on the date
stated; the batch record with sources is `agent/eval/cohort5_2026-09/LOG.md`
(sections "2026-10-05" and "2026-10-06").

## 0. Purpose

Wait for the 12 pilot sessions to come back and be ingested, then reassess how the
pipeline chooses its CNMFe hyperparameters for this prep -- min_corr (Cn threshold),
min_pnr, gSig and gSiz -- against human labels, and let the result set the
parameters, the curation threshold and the review plan for the other 40 cohort-5
sessions. Nothing about the 40 should be decided before the analyses in section 4.

## 1. State on 2026-10-06 08:30

- Animals: pnb99, pnb102, pnb104, pnb105, pnb106, pnb109 (six, never in training),
  area vCA1, same prep as pnb97/98. 52 sessions: 12 pilots (082626 odors-only day,
  082726 day 1; 082826 for pnb99) + 40 not yet staged (28 in `D:\2p data\6odorDualDiffRew`,
  12 in `D:\2p data\valence`; skip the 450-frame stub `AVG5x-TSeries-082726-pnb99-440um-3z-000.tif`).
- The 12 pilots were re-run on 2026-10-05/06 at **min_corr 0.70** (user decision; the
  estimator had given 0.78-0.85), min_pnr as estimated (5.7-8.4), gSig 9 / gSiz 36,
  curated at **threshold 0** (all candidates to the reviewer), and pushed to Taylor on
  2026-10-06 08:11-08:15 (markers verified; server bundles byte-identical to local).
  Candidates went from 874 to 2,237 (table in section 6). Taylor's earlier partial
  review (31 kept on 082726-pnb109) is lost; she was told to discard the old bundles.
- The 0.70 lives in `agent/config/param_overrides.json` (new manual file, animal level,
  read last by `agent/params.py:estimate_all_params`; never rewritten by the pipeline).
  Do NOT put overrides in `animal_params.json`: `extract_params.m` rebuilds it from scratch
  after every completed session (it did so at 08:13 today; cohort entries intact).
- vCA1 and BLA watchers were restarted ~08:13 (heartbeats resumed; the vCA1 one now runs
  the override-aware params.py). The DG_AL watcher log is frozen since 2026-10-05 23:42
  (process alive): press Esc in its console or restart it.
- The vCA1 watcher curates at the deployed model's T = 0.02 (no THRESHOLD_OVERRIDE in
  `watcher_vCA1.py`). The 40 must NOT be curated at 0.02 until section 4.5 says so.
- Deployed vCA1 model (read from `agent/model/vCA1/classifier.joblib` today): xgboost,
  35 features, feature_version 2, reject_threshold 0.02, n_sessions 161, agent_weight 5.0.
  Snapshot that never saw the pilots: `classifier_prepilot_2026-09-24.joblib`.
- The gSig sweep tree `D:\Julian_CNMFe\vCA1\.sweep\<session>\gSig07|gSig11` (145 GB) still
  exists for all 12 pilots. Keep it until section 4.3 is done, then the operator deletes it.
- Uncommitted on branch `training-program`: `agent/params.py` (override hook),
  `agent/config/param_overrides.json`, `agent/eval/cohort5_2026-09/{rerun_queue.py,
  archive_pilot_outputs.py, analysis_2026-10-05/}`, LOG.md, this file. Commit only when asked.

## 2. What was measured so far (why the thresholds are in question)

- Candidate counts do not scale with session length anywhere in the corpus; the
  estimator takes the 85th percentile of each session's own Cn and PNR images, so the
  count reflects how many strong correlation peaks the field has. Cohort-5 fields had
  about half as many Cn peaks above 0.85 as pnb97/98 (median 62-72 vs 116 per session)
  with identical absolute thresholds, hence 30-116 candidates instead of 67-167.
- On the old regime (pnb97/98, 19 sessions, 2,319 candidates, 533 kept) the human
  kept-rate by peak-Cn margin above min_corr was 0 of 65 below 0.03, 5% at 0.03-0.05,
  14% at 0.05-0.10, 33% above 0.10; by PNR margin 0 of 70 within 4 of min_pnr. I.e. the
  old threshold was not cutting into keepable cells THERE. Cohort 5 differs: the mean raw
  images show many visible somata with few strong transient sources.
- The estimator reads the 1000-frame pre-pass images while greedyROI seeds on the
  full-movie Cn/PNR: on 082726-pnb109 the pre-pass gave min_corr 0.81 / min_pnr 6.5 where
  the full-movie 85th percentiles are 0.73 / 9.0. So the effective Cn gate was stricter
  than "85th percentile" and the PNR gate looser.
- greedyROI's 9x9 median filter on Cn.*PNR (256 res) drops eligible blobs with too few
  above-threshold pixels; lowering min_corr rescued already-eligible cells, which is why
  0.70 gave 2.6x the candidates when the first-pass replay predicted +75%.
- Unseeded Cn peaks by band (12 pilots): share passing min_pnr 95-100% above 0.70;
  soma-like contrast on the mean image (null 10%): 13-14% at 0.75-0.85, 20% at 0.85-0.90,
  7% at 0.70-0.75, 3-5% below 0.70; seeded candidates 50%. These are lower bounds (cells
  dim at rest are invisible in a mean image).
- Deployed-model score split, old -> new candidate sets: >= 0.30: 310 -> 517; < 0.02:
  252 -> 960 (rank features are per set, indicative only).
- gSig sweep 2026-09-25 (arms at the estimator's own thresholds, first-pass model
  scores, no labels): confident extras at gSig 7 per live candidates pnb102 3/147,
  pnb104 5/114, pnb105 6/142 (all on day 1), pnb106 0/200, pnb109 1/153, pnb99 0/118;
  gSig 11 added nothing except 3 doubtful objects on pnb105 day 1. Decision then: 9/36.
- Frame rate: AVG5x data are 6.05 Hz (Bruker XML framePeriod 0.0331 s, 562 raw frames per
  18.5-s trial, 150 or 187 trials, cycle imaging); the pipeline assumes 3.75 Hz
  (`CNMFe_Biane_headless.m:33`, `run_cnmfe.py:54`, `features.py:171`). Open, not changed:
  a fix alters the events_per_min feature (contract) and needs a retrain decision.

## 3. Data on disk for each pilot session (`D:\Julian_CNMFe\vCA1\6odorDualDiffRew\<session>\`)

| generation | where | contents |
|---|---|---|
| old run (estimated min_corr, Sept 24-25) | `.prev_est85_2026-10-05/` | candidate_features.npz, spatial_footprints.mat (all old candidates), review_neuron.mat, review_summary.txt, agent_run.log, Cn.mat, pnr.mat, ROIs_candidates.jpg |
| new run (min_corr 0.70, Oct 5-6) | `.rerun_mincorr070_2026-10-06/` | spatial_footprints.mat (ALL new candidates), candidate_features.npz, review_summary.txt, Cn.mat, pnr.mat, ROIs_candidates.jpg |
| new run, live files | session root | the same, plus neuron.mat, review_neuron.mat, review_report.pdf, agent_run.log (both runs appended; the LAST "Final parameters" line is the 0.70 run) |

Ingest will replace neuron.mat, spatial_footprints.mat, A/C txt, ROIs.jpg etc. with
Taylor's FINAL kept set and add labels.mat (N x 1, 1 = kept, in candidate order) and
labels_provenance.txt; candidate_features.npz and review_neuron.mat survive
(`ingest_returns.py` CENTRAL_ONLY). Full-candidate footprints after ingest therefore come
from `.rerun_mincorr070_2026-10-06/spatial_footprints.mat` (Python-readable) or from
review_neuron.mat via MATLAB (`analysis_2026-10-05/export_peaks.m` pattern; scipy cannot
read the Sources2D object). Cn.mat/pnr.mat in the session root are greedyROI's
pre-extraction images at 512 (Cn, resized) and 256 (pnr) resolution.

## 4. Steps

### 4.1 Wait, then ingest

1. Returns appear in `X:\Julian\cnmfe_review\inbox\Taylor\vCA1\6odorDualDiffRew\<session>`.
   Read-only on the server. Check for renames before calling anything new
   (ingest matches by name).
2. `/ingest-returns` (or `python agent/ingest_returns.py --dry-run` first). Expect 12
   sessions, reviewer Taylor, labels.mat length == n_candidates of each
   candidate_features.npz (215, 176, 266, 228, 196, 81, 239, 170, 79, 184, 205, 198).
   A length mismatch here is the recurring silent bug: stop and investigate.
3. Record kept counts per session in LOG.md. Reference: pnb97/98 kept 16-45 per session.

### 4.2 Analysis A: min_corr (Cn)

Question: do keepable cells extend into the 0.70-0.80 band, or did 0.70 only add junk?

- Per candidate: peak Cn inside the footprint core (A > 0.5 max) from
  `.rerun_mincorr070_2026-10-06/spatial_footprints.mat` and the session root Cn.mat
  (`analysis_2026-10-05/cn_margin_full.py` logic; the MATLAB export is not needed when
  the saved footprints are used). Tabulate kept-rate by absolute band 0.70-0.75,
  0.75-0.80, 0.80-0.85, >= 0.85, pooled and per animal, plus kept cells per session
  that have NO counterpart in the old set (cosine similarity of footprints < 0.45
  against `.prev_est85_2026-10-05/spatial_footprints.mat`; the rule the sweep used,
  `sweep_gsig.py:SPATIAL_MATCH_THRESHOLD`) = the net gain attributable to 0.70.
- Proposed rule (user decides): if the 0.70-0.75 band's kept-rate is at least a third
  of the 0.80-0.85 band's and the net gain is >= ~5 kept cells per session, keep 0.70
  for the 40 (and consider a two-session test at 0.65). If the 0.70-0.80 bands are
  below ~5% kept and add fewer than ~5 kept cells per session, go back to 0.75 or 0.80
  for the 40 (roughly 40% less review load). Report the per-animal picture: pnb105
  day 2 (dim) and pnb99 day 1 (motion border, bd 19) behaved differently in the replay.

### 4.3 Analysis B: min_pnr

- Same join with the pnr image (256 res, kron x2): kept-rate by PNR margin above the
  session's min_pnr (0-1, 1-2, 2-4, 4-8, >= 8). On pnb97/98 nothing kept sat within 4.
- Also tabulate, for the 12 sessions, the pre-pass estimate (first "Estimated from
  Cn/pnr images" line in agent_run.log) against the full-movie 85th percentile
  (recompute from the saved pnr.mat): the gap tells whether min_pnr should be estimated
  on the full movie.
- Proposed rule: if kept cells never appear within ~2 units of min_pnr, PNR is not
  limiting; leave it estimated. If they pile up at the threshold, lower it (fixed 5.0,
  or the full-movie-based estimate) and re-test on two sessions before applying.

### 4.4 Analysis C: gSig / gSiz

Two independent checks, both per animal (zoom differs: pnb106 4z, pnb105 3.5z,
others 2.5-3z):

1. Kept-cell size: equivalent radius sqrt(area/pi) of the FINAL kept footprints
   (session root spatial_footprints.mat after ingest, 512 res, A > 0.5 max). Compare
   with the gSig 9 / gSiz 36 assumption (full-res units in this pipeline; CNMF-E works
   at ssub 2, i.e. 4.5 / 18 at 256 res). A median kept radius clearly above ~11 or
   below ~7 px per animal argues for 11/44 or 7/28 for that animal.
2. Recall by arm: match Taylor's kept footprints against the candidate sets of
   `.sweep/<session>/gSig07` and `gSig11` and against the archived old 9/36 set
   (`.prev_est85_2026-10-05/`), cosine >= 0.45. All three ran at the estimator's
   thresholds, so compare the arms with the OLD 9/36 set, not with the 0.70 set.
   An arm that recalls >= 10% more of the kept cells on an animal is a reason for a
   per-animal gSig override (param_overrides.json supports gSig/gSiz per animal).
   Caveat: changing gSig for an animal changes footprints and features, so the 12
   pilots of that animal would need another re-run for consistency with the 40.

### 4.5 Analysis D: the estimator rule and the classifier

- Estimator: if A shows real cells spread across 0.70-0.85 with no natural gap, the
  fixed per-animal value stays and "0.70 is the vCA1 default" means every new vCA1
  animal gets a line in param_overrides.json. If a natural cutoff appears, consider
  (i) a fixed vCA1 default in the estimator, (ii) computing the pre-pass Cn/PNR on all
  frames (removes the 0.81-vs-0.73 discrepancy), or (iii) a noise-anchored threshold.
  Any of these is a code change: test on the 12 before touching the watcher.
- Classifier, in this order:
  1. `python agent/eval/cohort5_2026-09/pilot_far.py --model agent/model/vCA1/classifier_prepilot_2026-09-24.joblib --area vCA1 --sessions <12 dirs> --out agent/eval/cohort5_2026-09/pilot_far_2026-10.json`
     = out-of-sample false auto-reject of the pre-pilot model on the NEW regime at
     0.01-0.04 (every candidate labelled). Expect it to be worse than the 0.44%
     measured on pnb97/98-like folds; it is the number that says whether 0.02 was
     ever safe here.
  2. `/retrain-area` vCA1 (train_classifier_vCA1.py; agent weight 5.0; the 12 sessions
     add ~2,200 labelled candidates to an agent pool of ~4,200). Read AUC and the
     false auto-reject rate under ANIMAL grouping from the printed tables, apply the
     1% rule (CLAUDE.md), and only then choose T for the 40. If no T passes, the 40
     stay at threshold 0.

### 4.6 Decide for the 40, then stage

1. Write the outcome of A-D into `param_overrides.json` (animal level; per-session
   block where a session needs its own value) and into LOG.md with the numbers.
2. Curation threshold: if the 40 are to go out at threshold 0, add
   `curator.THRESHOLD_OVERRIDE = 0.0` to `agent/watcher_vCA1.py` (one line, same as
   `watcher_DG_AL.py:36`) and restart the vCA1 watcher while it is idle; otherwise
   recurate each finished batch (`python agent/recurate_sessions_vCA1.py <dirs> --threshold 0`)
   before its push. Never rely on the model's 0.02 on this regime without 4.5.
3. Stage the 40 as loose tifs into `D:\Julian_CNMFe\vCA1\6odorDualDiffRew\` (28) and
   `...\vCA1\Valence\` (12); the watcher runs them in name order, one at a time,
   65-120 min each (2-3 days). Two headless MATLAB runs is the machine's limit; the
   watcher counts as one.
4. Push in waves: `python agent/push_review_bundle.py --all --area vCA1 --animal pnb99,pnb102,pnb104,pnb105,pnb106,pnb109 --assignee Taylor --dry-run`, then without
   `--dry-run` (no `--force`: the 12 pilots are already marked out). Verify markers.

## 5. Hazards and housekeeping

- Never write on the server; the operator deletes stale bundles and the `.sweep` tree.
- `analysis_2026-10-05/*.py` carry scratch-folder paths in HERE/OUT/CSV constants; edit
  them before re-use. The MATLAB exports need `cnmfe_setup.m` on the path (done inside).
- Do not start a second watcher; detect a frozen one by its log heartbeat.
- A gSig change for an animal invalidates that animal's pilot features (4.4 caveat).
- The frame-rate fix (section 2) is a separate decision; if taken, derive Fs per file
  from the XML (framePeriod x frame counts) and plan a retrain.
- Memory files: `project_batch_cohort5_2026-09`, `project_seeding_percentile`,
  `project_frame_rate_avg5x` (in the Claude memory folder) mirror this state.

## 6. Reference numbers (2026-10-06)

| session | old N (est. min_corr) | new N (0.70) | score >= 0.30 old -> new | score < 0.02 old -> new |
|---|---|---|---|---|
| 082626-pnb102 | 71 | 215 | 31 -> 60 | 10 -> 78 |
| 082626-pnb104 | 64 | 176 | 34 -> 51 | 11 -> 67 |
| 082626-pnb105 | 112 | 266 | 31 -> 53 | 32 -> 113 |
| 082626-pnb106 | 116 | 228 | 20 -> 38 | 56 -> 114 |
| 082626-pnb109 | 83 | 196 | 28 -> 47 | 23 -> 90 |
| 082626-pnb99 | 43 | 81 | 13 -> 24 | 8 -> 21 |
| 082726-pnb102 | 76 | 239 | 31 -> 51 | 22 -> 109 |
| 082726-pnb104 | 50 | 170 | 20 -> 30 | 11 -> 79 |
| 082726-pnb105 | 30 | 79 | 18 -> 28 | 6 -> 28 |
| 082726-pnb106 | 84 | 184 | 15 -> 24 | 46 -> 113 |
| 082726-pnb109 | 70 | 205 | 28 -> 46 | 17 -> 82 |
| 082826-pnb99 | 75 | 198 | 41 -> 65 | 10 -> 66 |
| total | 874 | 2237 | 310 -> 517 | 252 -> 960 |

Estimated min_corr / min_pnr per session (first run, pre-pass): 0.85/7.3, 0.82/6.5,
0.83/8.4, 0.79/6.9, 0.81/6.6, 0.79/5.9, 0.82/6.4, 0.84/6.2, 0.81/5.7, 0.78/6.2,
0.81/6.5, 0.84/6.4 (same order as the table); bd 10 except 082626-pnb99 (19) and
082726-pnb105 (14).
