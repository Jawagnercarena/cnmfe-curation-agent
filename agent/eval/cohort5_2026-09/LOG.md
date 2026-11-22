# Cohort-5 vCA1 batch log (six new retro-tdTomato animals, 52 sessions)

Record of the batch planned in `C:\Users\julia\.claude\plans\i-have-this-handoff-fluffy-twilight.md`
(validated handoff `docs/HANDOFF_2026-09-23_NEW_BATCH.md`). Every number here was
read from a file or a log at the time stated; nothing is carried from memory.

## Batch

- Source: `D:\2p data\6odorDualDiffRew` (41 tifs) and `D:\2p data\valence` (12), all
  AVG5x, 512x512 8-bit, 10k-21k frames; 206.5 GB. Same files exist on the NAS
  under `X:\Jeremy\for CNMFe\`.
- Animals (none in training before this batch): pnb99, pnb102, pnb104, pnb105,
  pnb106, pnb109. Area vCA1, same retro-tdTomato prep as pnb97/98 (user, 2026-09-23).
- Excluded: `AVG5x-TSeries-082726-pnb99-440um-3z-000.tif` (450 frames, aborted).
  Processed: 52. Reviewer: Taylor (all 52).
- gSig/gSiz start at 9/36 (the unknown-animal fallback, same as pnb88/97/98);
  a sweep on the 12 pilot sessions decides the final per-animal value.
- Zoom (Nz token x10): pnb106 4z, pnb105 mostly 35z, others 25z-3z.

## Decisions (user)

- 2026-09-23: classifier = deployed XGBoost; CNN blend not deployed for this
  batch (vCA1: ranking gain only, worse for a held-out animal at its admissible T).
- 2026-09-24: parallel pilot. 12 pilot sessions (082626 + 082726 for all six
  animals, 082826 for pnb99) curated at threshold 0; the other 40 at T = 0.02;
  all pushed to Taylor without waiting for returns. Basis: 09-19 animal-held-out
  folds (results_0919, 8 seeds, threads 8), per held-out animal:

  | held-out animal | kept | T=0.01 | T=0.02 | T=0.03 | T=0.04 |
  |---|---|---|---|---|---|
  | pnb97 | 206 | 0.30% (worst 2) | 1.03% (3) | 1.76% (5) | 2.49% (7) |
  | pnb98 | 618 | 0.12% (1) | 0.32% (3) | 0.42% (3) | 0.55% (5) |
  | pnb88 | 110 | 0 | 0 | 0 | 0 |
  | pooled | 934 | 0.15% | 0.44% | 0.67% | 0.91% |
  | junk caught | | 25.1% | 34.7% | 40.4% | 44.7% |

## Phase 0 (2026-09-23 22:10 to 2026-09-24)

- Watchers: all three alive throughout (heartbeats 22:10); nothing restarted.
- BLA retrain (`train_classifier.py --prospective-only --model xgboost`, 22:12):
  179 -> 181 sessions (the two bla36 3odor sessions whose labels predated the
  10:51 joblib). Joblib: xgboost, 35 cols, T 0.04, w 4.0. One-seed agent-pool
  line: AUC reviewed 0.912 (12629 candidates, 2663 kept), false auto-reject
  0.71% (19 of 2663) at 0.04, junk 42.3% of 11992; the 10:52 run read 0.57%
  (15 of 2643), 41.9%. Snapshot `agent/model/BLA/classifier_pre_2026-09-23.joblib`.
- Fork repair (operator, 2026-09-24): Aneesh's return of
  `vCA1/3odor/AVG5x-TSeries-071526-pnb98-471um-3z-000` had been ingested as
  `...07152026...` (labels, no features). Six duplicated files md5-identical,
  Cn arrays identical, labels 123 = npz N (66 keep, 57 delete). Reviewer files
  merged into the original (ROIs.jpg first), fork deleted, folder and inner
  .tif/.mat/_neurons renamed to the reviewer's name; agent_run.log annotated.
  Trainer dry run lists `[AG] 3odor/AVG5x-TSeries-07152026-pnb98-471um-3z-000:
  66 keep, 57 delete (123 total)`.
- Duplicate `AVG8x-TSeries-06222022-99-303um-35z-000` (identical under 4CS and
  AA, trained twice): AA copy parked to `vCA1/.excluded/AA/` with
  EXCLUDED_REASON.txt (operator). Trainer now lists it once ([BS] 4CS).
- Stray `D:\Julian_CNMFe\TSeries-040723-DG6E-295um-245um-2z-000A` (67 files
  md5-identical to the DG_AL session, Cn arrays identical): deleted (operator).
- Pool after repairs: 161 sessions (51 agent, 110 bootstrap).
- `agent/sweep_gsig.py` gained `--area {DG_AL,vCA1,BLA}` (default DG_AL
  unchanged); vCA1 compare-only smoke test resolved `vCA1\.sweep\` and the
  live-gSig label. ASCII-clean.
- `agent/eval/cohort5_2026-09/pilot_far.py` written (read-only scorer for the
  threshold-0 pilot sessions); functional test on three in-training pnb97
  threshold-0 sessions with the live joblib ran end to end (in-sample, so
  0 misses; not a measurement).
- Fresh 8-seed control on the repaired pool: `results_0924/` (see below).

## results_0924 (2026-09-24 08:27, `bakeoff_tabular.py --area vCA1 --grouping session,animal --contenders xgb_deployed --threads 8`)

Pool snapshot 161 labeled sessions; 43 CV sessions (8 agent sessions with < 5 reals train only), 4267 rows, 1042 kept cells, 49772 bootstrap rows (6539 weight 0); agent weight 5.0; 8 seeds; threads 8 (ACORN env). 09-19 had 39 CV sessions / 934 kept.

| grouping | AUC full | AUC reviewed | T=0.01 far mean/worst, junk | T=0.02 | T=0.03 | T=0.04 | rule_T (grid 0.03-0.10) |
|---|---|---|---|---|---|---|---|
| session (5 folds over 43) | 0.9225 | 0.9212 | 0.25 / 0.38, 25.9% | 0.47 / 0.67, 36.1% | 0.82 / 1.15, 42.1% | 1.20 / 1.63, 46.2% | **None** (0.03 fails the 1.0% worst-seed cap) |
| animal (3 fixed folds) | 0.9159 | 0.9145 | 0.10 / 0.19, 24.9% | 0.44 / 0.67, 35.0% | 0.72 / 0.96, 40.9% | 1.10 / 1.34, 45.4% | 0.03 |

Per animal (session-grouped, 8 seeds, misses mean (worst) / kept): pnb97 206 kept: 0.01 1.38 (2), 0.02 2.25 (3), 0.03 2.88 (4), 0.04 4.62 (6); pnb98 726 kept: 0.01 1.25 (2), 0.02 2.62 (4), 0.03 5.62 (9), 0.04 7.88 (13); pnb88 110 kept: 0 at every T. Animal-grouped: pnb97 0.02 1.27% / 0.03 1.82%; pnb98 0.02 0.28% / 0.03 0.52%.

The 0.01 and 0.02 rows are computed from the saved out-of-fold scores (the JSON threshold_table starts at 0.03). Rows and the rule's verdict are in `agent/eval/algo_bakeoff_2026-09/results_0924/` (`control_t8.log`, `report_vCA1_t8.md`, `vCA1_{session,animal}_xgb_deployed_t8.{json,npz}`).

## vCA1 threshold flip 0.04 -> 0.02 (user decision 2026-09-24 08:30)

- The plan's gate ("0.03 only if admissible under both groupings") failed: 0.03 is admissible under animal grouping only. The user chose 0.02 (holds the 1% worst-seed posture under both groupings with margin; same T as the new animals; headroom for drift). 0.03 was the alternative offered.
- Source: `agent/train_classifier_vCA1.py` `_BOOTSTRAP_THRESHOLD = _VALIDATED_THRESHOLD = 0.02` (comment block rewritten with results_0924). `agent/eval/vca1_v2_2026-08/gate_decision.json`: `chosen_T` 0.02 and a `user_decision_T_2026-09` block; the 08-26 `user_decision_T` block kept as history.
- Snapshot before: `agent/model/vCA1/classifier_pre_2026-09-24.joblib` (= the 10:28 09-23 joblib, T 0.04).
- Retrain `train_classifier_vCA1.py --prospective-only --model xgboost` (no --threshold), 08:30:53: `[vCA1] 51 agent sessions -> using threshold 0.02`, `Sessions: 161 total (51 agent, 110 bootstrap)`, `Overriding reject_threshold: 0.04 -> 0.02`, agent-pool AUC reviewed 0.920 (4210 candidates, 1042 kept), operating point at 0.02: false auto-reject 0.58% (6 of 1042), junk 36.6% of 3225 (one seed, unpinned threads). Joblib verified: xgboost, fv 2, 35 cols, n_sessions 161, T 0.02, w 5.0, n_excluded_ambiguous 6539.
- `verify_vca1.py`: sections 1-2 ok except the known `n_sessions = 161 (pinned manifest 134)` FAIL; section 3 (corpus identity against the 08-26 swap_report.json) now crashes with FileNotFoundError on the fork session's OLD path (`...071526-pnb98...\candidate_features.npz`), a consequence of adopting the reviewer's name; section 4 did not run. Manual check of section 4 below.
- Pre-pilot reference model for the false-AR measurement: `agent/model/vCA1/classifier_prepilot_2026-09-24.joblib`, md5 affdfaab98ed6341b87cc5cea223dd85 (identical to the deployed joblib at 08:30:53).

## Phase 1: pilot sessions (staged by the operator 2026-09-24 08:36, picked up 08:38:52, name order)

| # | session | frames | gSig/gSiz | min_corr / min_pnr | candidates | auto-rejected @0.02 | run time | clamp |
|---|---|---|---|---|---|---|---|---|
| 1 | 082626-pnb102-479um-3z | 10134 | 9/36 (fallback) | 0.85 / 7.3 | 71 | 10 (61 to review) | 08:38-09:34 (56 min) | none |
| 2 | 082626-pnb104-404um-3z | 10134 | 9/36 (fallback) | 0.82 / 6.5 | 64 | 11 (53 to review) | 09:34-10:38 (63 min, sweep arm concurrent) | none |
| 3 | 082626-pnb105-397um-35z | 10134 | 9/36 (fallback) | 0.83 / 8.4 | 112 | 32 (80 to review) | 10:38-11:45 (67 min, sweep arm concurrent) | none |
| 4 | 082626-pnb106-205um-4z | 10134 | 9/36 (fallback) | 0.79 / 6.9 | 116 | 56 (60 to review) | 11:45-12:48 (63 min, sweep arm concurrent) | none |
| 5 | 082626-pnb109-398um-26z | 10116 | 9/36 (fallback) | 0.81 / 6.6 | 83 | 23 (60 to review) | 12:48-13:53 (65 min, sweep arm concurrent) | none |
| 6 | 082626-pnb99-399um-3z | 10134 | 9/36 (fallback) | 0.79 / 5.9, bd 19 (motion border) | **43** (below the 67-167 band) | 8 (35 to review) | 13:53-14:51 (58 min, sweep arm concurrent) | none |
| 7 | 082726-pnb102-497um-3z | 16882 | 9/36 (fallback) | 0.82 / 6.4 | 76 | 22 (54 to review) | 14:51-16:31 (100 min, sweep arm concurrent; parpool idle-timeout reconnect at 15:46) | none |
| 8 | 082726-pnb104-416um-3z | 16861 | 9/36 (fallback) | 0.84 / 6.2 | **50** (below the band) | 11 (39 to review) | 16:31-18:07 (96 min, sweep arm concurrent) | none |
| 9 | 082726-pnb105-391um-35z | 16860 | 9/36 (fallback) | 0.81 / 5.7, bd 14 (motion border) | **30** (far below the band; day 1 gave 112) | 6 (24 to review) | 18:07-19:41 (94 min, sweep arm concurrent) | none |

| 10 | 082726-pnb106-201um-4z | 16860 | 9/36 (fallback) | 0.78 / 6.2 | 84 | 46 (38 to review) -- 55%, again the highest share | 19:41-21:19 (98 min, sweep arm concurrent) | none |
| 11 | 082726-pnb109-405um-25z | 16860 | 9/36 (fallback) | 0.81 / 6.5 | 70 | 17 (53 to review) | 21:19-22:57 (98 min, sweep arm concurrent) | none |
| 12 | 082826-pnb99-436um-3z-000-000 | 16890 | 9/36 (fallback) | 0.84 / 6.4 | 75 | 10 (65 to review) | 22:57-00:38 (100 min, sweep arm concurrent) | none |

Recurate at threshold 0 (2026-09-25 00:42-01:00, `recurate_sessions_vCA1.py <12 explicit dirs> --threshold 0`, logs `agent/logs/recurate_cohort5_pilot_{A,B}.log`): every session `Threshold override: 0.000 (model's calibrated value was 0.020)`, `New auto-rejected: 0`, rescued 10/11/32/56/23/8 (day 1) and 22/11/6/46/17/10 (day 2); all 12 npz verified `auto_rejected` empty, review sets = N (874 candidates). animal_params.json refreshed by the watcher at 00:40 with the six animals at 9/36 (flag ok, 2 sessions each). Push dry run (`--all --area vCA1 --animal <six> --assignee Taylor --dry-run`): exactly 12 sessions, 34,263 MB, no warnings; the real push waits for the operator and the gSig decision.

Pilot complete 2026-09-25 00:38 (16.0 h for 12 sessions with a sweep arm running alongside for all but the first). Totals: 874 candidates, 252 auto-rejected at 0.02 (28.8%), 622 to review at 0.02; at threshold 0 Taylor sees all 874. Auto-reject share by animal (both days): pnb102 22%, pnb104 19%, pnb105 27%, pnb106 51%, pnb109 26%, pnb99 15%. All 12 packages complete, all at 9/36, no clamp warnings; queue lists 12 awaiting.

Session 9 diagnosis (read-only, 19:45): the day-2 recording is intrinsically weaker, not a run fault. Cn 90th percentile 0.72 vs 0.87 on day 1, fraction of pixels with Cn > 0.8 4.2% vs 19.9%, pnr 90th percentile 8.3 vs 12.6 (max similar, 44 vs 43); the estimator adapted (min_corr 0.81 / min_pnr 5.7 vs 0.83 / 8.4) and CNMF-E initialised in 40 s vs 80 s. pnb104's two days have near-identical Cn/pnr statistics and gave 64 and 50, so the pipeline is consistent; pnb105 day 2 is a dim or motion-affected (bd 14) field. Nothing changed.

Sweep (arms gSig 7 and 11, gSiz 4x, reference = the live 9/36 run): logs `agent/logs/sweep_cohort5_<session>.log` (smoke) / `agent/logs/sweep_<session>.log` (queue) and `agent/logs/sweep_gsig.log`; outputs `D:\Julian_CNMFe\vCA1\.sweep\<session>\`. Queue: `agent/eval/cohort5_2026-09/sweep_queue.py` (detached, one arm at a time, waits for each reference run).

| session | arm | min_corr / min_pnr (arm's own estimate) | candidates | arm time | missed by the 9/36 reference (cosine < 0.45) |
|---|---|---|---|---|---|
| 082626-pnb102 | 7/28 | 0.75 / 6.3 | 92 (live 9/36: 71) | 56 min | 11 unmatched (81 matched) |
| 082626-pnb102 | 11/44 | 0.90 / 8.3 | 63 | 77 min (concurrent with the watcher) | 3 unmatched (60 matched) |
| 082626-pnb104 | 7/28 | 0.72 / 5.8 | 73 (live 9/36: 64) | 51 min | 8 unmatched (65 matched): first-pass median 0.10, 5 of 8 >= 0.02, one cell-like (0.38), four marginal (0.10-0.15), three junk-like |
| 082626-pnb104 | 11/44 | 0.88 / 7.4 | 60 | 74 min | 6 unmatched (54 matched): first-pass median 0.02, one marginal (0.18), five junk-like; circularity 0.08 |
| 082626-pnb105 (35z) | 7/28 | 0.74 / 7.2 | 161 (live 9/36: 112, 32 auto-rejected) | 58 min | 32 unmatched (129 matched): first-pass 6 >= 0.30 (0.81, 0.76, 0.74, 0.46, 0.44, 0.39), 11 in [0.02, 0.30), 15 < 0.02; median area 388 px (matched 537), circularity 0.19, peak SNR 17, skewness 0.9 |
| 082626-pnb105 (35z) | 11/44 | 0.88 / 9.4 | 101 | 83 min | 23 unmatched (78 matched): 3 >= 0.30 (0.72, 0.43, 0.37), 12 marginal, 8 junk-like; circularity 0.10 (large, less circular objects) |
| 082626-pnb106 (4z) | 7/28 | 0.69 / 6.1 | 130 (live 9/36: 116, 56 auto-rejected) | 57 min | 21 unmatched (109 matched): none >= 0.30 (max 0.24), 8 marginal, 13 junk-like; tiny (median area 268 px), peak SNR 13, skewness 0.6 -- fragments, not cells. Note the whole animal scores low: matched-candidate first-pass median 0.08, live kept 0.24 (other animals 0.36-0.51) |
| 082626-pnb106 (4z) | 11/44 | 0.86 / 7.6 | 75 | 80 min | 6 unmatched (69 matched), all junk-like (max 0.03) |
| 082626-pnb109 (26z) | 7/28 | 0.71 / 5.9 | 98 (live 9/36: 83, 23 auto-rejected) | 54 min | 8 unmatched (90 matched): none >= 0.30 (max 0.10), 3 marginal, 5 junk-like. Animal scores moderately low overall (live kept first-pass median 0.19) |
| 082626-pnb109 (26z) | 11/44 | 0.87 / 7.2 | 59 | 78 min | 1 unmatched (58 matched), junk-like (0.00, area 1322 px, circularity 0.06) |
| 082626-pnb99 (3z) | 7/28 | 0.65 / 5.5, bd 19 | 57 (live 9/36: 43, 8 auto-rejected) | 51 min | 12 unmatched (45 matched): none >= 0.30 (max 0.11), 10 marginal (0.02-0.11), 2 junk-like; circularity 0.13 |
| 082626-pnb99 (3z) | 11/44 | 0.86 / 6.4 | 51 | 70 min | 14 unmatched (37 matched): 3 marginal (max 0.05), 11 junk-like; large low-circularity blobs (area 1021 px, circularity 0.07) |

| 082726-pnb102 (3z, day 2) | 7/28 | 0.71 / 5.9 | 103 (live 9/36: 76) | 85 min | 16 unmatched (87 matched): 1 >= 0.30 (0.40), 4 marginal, 11 junk-like |
| 082726-pnb102 (3z, day 2) | 11/44 | 0.87 / 7.0 | 74 | 104 min | 12 unmatched (62 matched): none >= 0.30 (max 0.27), 4 marginal, 8 junk-like; area 1064 px, circularity 0.11 |

| 082726-pnb104 (3z, day 2, 50 live) | 7/28 | 0.72 / 5.6 | 86 (live 9/36: 50) | 70 min | 28 unmatched (58 matched): **4 >= 0.30 (0.51, 0.48, 0.44, 0.34)**, 11 marginal, 13 junk-like; circularity 0.23 and skewness 1.27 at the matched medians |
| 082726-pnb104 (3z, day 2) | 11/44 | 0.87 / 6.9 | 51 | 101 min | 10 unmatched (41 matched): none >= 0.30 (max 0.13), 3 marginal, 7 junk-like |

| 082726-pnb105 (35z, day 2, dim, 30 live) | 7/28 | 0.66 / 5.0 | 42 (live 9/36: 30) | 69 min | 9 unmatched (33 matched): none >= 0.30 (max 0.04), 2 marginal, 7 junk-like; peak SNR 47 and skewness 7.4 = spiky artifacts, not cells |
| 082726-pnb105 (35z, day 2) | 11/44 | 0.84 / 6.6 | 29 | 97 min | 1 unmatched (28 matched), junk-like (0.01) |

| 082726-pnb106 (4z, day 2, 84 live) | 7/28 | 0.66 / 5.8 | 122 (live 9/36: 84) | 72 min | 35 unmatched (87 matched): none >= 0.30 (max 0.13), 4 marginal, 31 junk-like; small (405 px), peak SNR 14, skewness 0.5. The whole session scores very low with the first-pass model (live median 0.008, 37% >= 0.02; day 1 was 0.24) |
| 082726-pnb106 (4z, day 2) | 11/44 | 0.85 / 6.7 | 65 | 106 min | 12 unmatched (53 matched), all junk-like (max 0.01) |

| 082726-pnb109 (25z, day 2, 70 live) | 7/28 | 0.71 / 6.0 | 95 (live 9/36: 70) | 70 min | 19 unmatched (76 matched): 1 >= 0.30 (0.66), 11 marginal, 7 junk-like; circularity 0.28 (matched 0.37) |
| 082726-pnb109 (25z, day 2) | 11/44 | 0.86 / 6.9 | 62 | 102 min | 4 unmatched (58 matched): none >= 0.30 (max 0.10) |

| 082826-pnb99 (3z, day 2, 75 live) | 7/28 | 0.70 / 5.9 | 89 (live 9/36: 75) | 69 min | 6 unmatched (83 matched): none >= 0.30 (max 0.29), 6 marginal |
| 082826-pnb99 (3z, day 2) | 11/44 | 0.86 / 6.9 | 61 | 103 min | 0 unmatched (61 matched) |

Sweep queue complete 2026-09-25 16:23 (24 arms over 12 sessions, 09-24 09:35 to 09-25 16:23; the `.sweep` tree holds 145 GB of tif copies and arm outputs, operator's to delete). No arm hit a clamp.

**Final tally, confident extras (first-pass >= 0.30) / live candidates, both days:**

| animal (zoom) | gSig 7 | gSig 11 |
|---|---|---|
| pnb102 (3.0) | 3 / 147 | 0 |
| pnb104 (3.0) | 5 / 114 | 0 |
| pnb105 (3.5) | 6 / 142 (all on the dense day-1 field) | 3 (large low-circularity objects, day 1) |
| pnb106 (4.0) | 0 / 200 | 0 |
| pnb109 (2.6) | 1 / 153 | 0 |
| pnb99 (3.0) | 0 / 118 | 0 |

**Decision (user, 2026-09-25 evening): gSig/gSiz 9/36 for all six animals; all 12 pilot sessions go to Taylor at threshold 0 (wave 1, operator-run push).**

Reading: no quality cliff on any animal. gSig 7 recovers 0-5 confident cells per 100 live candidates on the zoom-3/3.5 animals at the cost of 20-70% more candidates (mostly junk); gSig 11 recovers nothing except three doubtful objects on pnb105 day 1. Recommendation to the user: keep 9/36 for all six (also keeps the cohort on the pnb97/98 footing the T=0.02 estimate assumes).

Both-days tally of confident extras (first-pass >= 0.30) at gSig 7 / live candidates: pnb102 3 / 147, pnb104 5 / 114, pnb105 6 / 142 (all six on the dense day-1 field; none on the dim day-2 recording), pnb106 0 / 116 (day 2 pending), pnb109 0 / 83 (pending), pnb99 0 / 43 (pending). At gSig 11 the only confident extras anywhere are pnb105 day 1's three large low-circularity objects.

Day-1 sweep summary (one session per animal, confident = first-pass >= 0.30 among the arm's unmatched candidates): pnb102 gSig7 2 of 11 / gSig11 0 of 3; pnb104 1 of 8 / 0 of 6; **pnb105 6 of 32 / 3 of 23**; pnb106 0 of 21 / 0 of 6; pnb109 0 of 8 / 0 of 1; pnb99 0 of 12 / 0 of 14. Only pnb105 (zoom 3.5) shows cells the 9/36 run misses in numbers that matter (about 6 of its 80 kept candidates at gSig 7); gSig 11 never adds a convincing cell for the zoom-3 animals. Day-2 sweeps follow in the queue.

Smoke sweep complete 11:49 (`missed_by_reference.pdf` in the session's .sweep folder). Extras characterised with `sweep_extras.py` (13 base features on the arm dir + the joblib's 13-col first-pass model; `extras_082626-pnb102_gSig07.json`): the 11 gSig-7 extras have first-pass median 0.088, 8 of 11 >= 0.02, two clearly cell-like (0.54, 0.34), three junk-like (< 0.02); their circularity (0.19), peak SNR (25) and skewness (1.7) sit at the matched candidates' medians (0.22 / 23 / 1.9). Reference groups: live kept median 0.48 (98% >= 0.02), live auto-rejected 0.015 (20%). The 3 gSig-11 extras are junk-like (0.04, 0.001, 0.001; circularity <= 0.07, skewness <= 0.7), and gSig 11 finds fewer candidates overall (63 vs 71). Queue launched 11:50 (PID 82688, `agent/logs/sweep_queue.log`, per-session `agent/logs/sweep_<session>.log`, `.done` markers) for the other 11 pilot sessions in name order, one arm at a time, waiting for each reference run.

## 2026-10-05: low candidate counts, seeding decision (min_corr 0.75), frame rate

User question: pilot counts (30-116) looked low for this prep (pnb97/98 reviewers started with
67-167, median 125); Taylor reviewed 082726-pnb109 (31 kept of 70, result lost). Scripts and raw
outputs for every number below: `analysis_2026-10-05/` (run from a scratch folder; paths inside
point there).

Measured, read-only:
- Day-2 pilot tifs are 16,860-16,890 frames, identical to every pnb97/98 6odor session; all frames
  loaded; every bundle held all candidates (threshold 0). Candidate counts do not scale with length
  anywhere in the corpus (pnb98 3odor at 4.8k/12.8k/17k frames: 98 / 88-157 / 102-109).
- pnb97/98 6odor review sets (19 sessions, 2,319 candidates, 533 kept): candidate peak Cn minus
  the session's min_corr -> kept-rate 0 of 65 below 0.03, 7 of 155 (5%) in 0.03-0.05, 14% in
  0.05-0.10, 33% above 0.10; PNR margin: 0 of 70 kept within 4 of min_pnr (`cn_margin_full.txt`).
- Local maxima of the saved pre-extraction Cn (cell-scale window, 20-px border excluded), per-session
  medians ref / pilot d1 / pilot d2: Cn>0.85 116 / 72 / 62; Cn>0.90 74 / 44 / 36; seeded share of
  the >0.90 maxima 78% / 91% / 83%; unseeded maxima above the session's own min_corr 44 / 32 / 36
  (`peaks.txt`). Absolute thresholds were the same as pnb97/98 (min_corr 0.78-0.85 vs 0.80-0.87).
  Mean raw frames show many somata (the user's density impression is right); PNR images show few
  strong transient sources. Raw means 42-90 vs 29-71 (8-bit), p99/p50 2.0-3.2 vs 2.2-6.1.
- Imaging settings (Bruker XML, read-only): cohort 5 Pockels 461-481 / green PMT 590-660 vs pnb98
  441/570 and pnb97 621/620.
- The estimator reads the 1000-frame pre-pass Cn while greedyROI seeds on the full-movie Cn, whose
  85th percentile is lower (082726-pnb109: 0.81 pre-pass vs 0.73 full-run; 062626-pnb97: 0.83 vs
  0.80), so the effective seeding percentile was above 85.

Frame rate (Bruker XML + Arduino logs): framePeriod 0.03308 s (30.2 Hz raw), one Sequence per trial,
562 raw frames per 18.5-s imaging window (5.5 pre + 2 odor + 2 trace + 9 post), cycle imaging (no ITI
frames). 150 trials -> 84,303 raw -> 16,860 AVG5x frames = 6.05 Hz; the 090826 sessions are
within-session extinction (behavior files `_WSE`) with 187 trials -> 105,137 raw -> 21,027 frames, same
rate. The pipeline assumes 3.75 Hz (`CNMFe_Biane_headless.m:33`, `run_cnmfe.py:54`,
`features.py:171`). NOT changed: it alters events_per_min (feature contract). Derivation without a
constant: Fs = (1/framePeriod) x (XML frame count / tif frame count); XML at
`X:/Jeremy/2p data/<task>/<MMDDYY>/<TSeries>/<TSeries>.xml`.

Decision (user, 2026-10-05): seed the six cohort-5 animals at min_corr 0.75 instead of the estimate,
re-run the 12 pilot sessions, re-send them to Taylor, re-evaluate the threshold with the returned
labels. Accepted cost: more junk per session (the gSig-7 arms at min_corr 0.65-0.75 added ~205
candidates of which 46% scored >= 0.02 and 7% >= 0.30).

Implementation: new manual file `agent/config/param_overrides.json` (min_corr 0.75 for pnb99/102/
104/105/106/109 at animal level, so the 40 unstaged sessions inherit it), read last by
`agent/params.py:estimate_all_params` (animal level, then a per-session block). The documented
`session_overrides` block in animal_params.json is not safe: `extract_params.m` rebuilds that file
after every completed session. Verified on live session folders (read-only): override applied /
not applied for pnb97 / session-level wins / unknown key warned / missing file = no override;
folders untouched.

Re-run mechanics: `archive_pilot_outputs.py` (dry run by default) moves each pilot session's outputs
into `<session>/.prev_est85_2026-10-05/` (neuron.mat last) so the watcher retries the session
(tif inside, no neuron.mat). The converted .mat movie and review_assigned.txt stay. The watcher
curates at T 0.02 -> recurate at threshold 0 before the push. Operator: delete the 12 stale outbox
bundles on the server, then `push_review_bundle.py --all --area vCA1 --animal <six> --assignee
Taylor --force`. OPEN: curation threshold for the 40 (recommend 0 for the whole cohort; the T=0.02
safety estimate was measured on 85th-percentile candidate sets).

User follow-up (will 0.75 matter, what about PNR, what is a liberal-but-reasonable value), measured on the 12 pilots (`analysis_2026-10-05/peak_bands.*`, `seed_sim.*`):
- Unseeded Cn maxima by band (pooled n, share passing the session's min_pnr near the peak, share 'soma-like' = mean-image contrast above the 90th percentile of random interior points, null = 10%): 0.60-0.65 349 / 76% / 3%; 0.65-0.70 302 / 91% / 5%; 0.70-0.75 289 / 95% / 7%; 0.75-0.80 291 / 99% / 13%; 0.80-0.85 295 / 100% / 14%; 0.85-0.90 141 / 100% / 20%; >=0.90 85 / 100% / 13%. Seeded peaks (646 candidates): 50% soma-like, median PNR 17. PNR is not the gate above 0.70; leave min_pnr at the estimate.
- First-pass seed replay of greedyROI (Cn.*PNR, 9x9 median, 19x19 max, bd) on the saved pre-extraction images: median seeds at est / 0.80 / 0.75 / 0.70 / 0.65 = pilot d1 132 / 157 / 208 / 236 / 259, pilot d2 123 / 132 / 185 / 210 / 255, ref 171 / 215 / 246 / 250 / 251 (proxy: replay exceeds the actual candidate count by ~1.4-1.8x because extraction rejects seeds). Expected gain at 0.75 ~ +50% candidates (median ratio 1.50-1.58), at 0.70 ~ +75%. Exceptions: 082726-pnb105 (dim, 88 -> 90 at 0.75, 121 at 0.70) and 082626-pnb99 (bd 19, 79 -> 85 at 0.75, 139 at 0.70) gain little at 0.75.
- Reading: 0.75 is the liberal-but-reasonable value (extra peaks still above the random-null soma rate); 0.70 is the floor (7% soma-like, below the 10% null); below 0.70 the added peaks are texture. Mean-image soma-likeness undercounts dim-at-rest cells, so these are lower bounds on the real-rate.

## 2026-10-05 23:40: DECISION min_corr 0.70, 12 pilots re-running

User (2026-10-05 evening): Cn >= 0.70 for these sessions, provisional default from now on, reassess once the 12 come back; send them out immediately. `agent/config/param_overrides.json` now carries min_corr 0.70 for pnb99/102/104/105/106/109 (min_pnr stays estimated). Re-run path: `rerun_queue.py` (same folder), two detached runners started 23:39:53 (list A, PID 87724) and 23:40:23 (list B, PID 61328), each six sessions (three day-1 + three day-2), curator THRESHOLD_OVERRIDE 0.0 so the packages come out at threshold 0 directly; logs `agent/logs/rerun_cohort5_{A,B}.log` (+ .stdout/.stderr). Each runner calls `watcher.process_session` in place; the old candidate set is COPIED to `<session>/.prev_est85_2026-10-05/` first (8 files); nothing deleted or moved; review_assigned.txt untouched; the watcher ignores the sessions because neuron.mat/review_report.pdf stay until overwritten. `archive_pilot_outputs.py` (move-based, for the watcher path) was NOT used. Expected ~8 h per list with two MATLAB runs concurrent (~100 min per day-2 session, ~65 per day-1). Do not stage the 40 tifs while both runners are active (third MATLAB). After the runs: verify 12 x (threshold 0, counts), operator deletes the 12 stale outbox bundles, push `--all --area vCA1 --animal pnb99,pnb102,pnb104,pnb105,pnb106,pnb109 --assignee Taylor --force`, verify markers. Before the 40 are staged: restart the vCA1 watcher (old params.py in memory) and decide its curation threshold (recommend 0 for the cohort).

## 2026-10-06 08:08: re-run complete, 12/12 at min_corr 0.70, threshold 0

Runner A finished 07:44, runner B 08:07 (8.1 / 8.5 h); day-1 sessions 56-65 min, day-2 95-120 min with two MATLAB runs concurrent; stderr logs empty; every session: new neuron.mat, 'Final parameters' min_corr 0.7 (min_pnr as estimated), review_summary 'Auto-rejected: 0 (threshold: 0.00)', awaiting = all candidates; old outputs copied to `<session>/.prev_est85_2026-10-05/` (8 files each).

| session | old N | new N | score<0.02 old->new | score>=0.30 old->new |
|---|---|---|---|---|
| 082626-pnb102 | 71 | 215 | 10->78 | 31->60 |
| 082626-pnb104 | 64 | 176 | 11->67 | 34->51 |
| 082626-pnb105 | 112 | 266 | 32->113 | 31->53 |
| 082626-pnb106 | 116 | 228 | 56->114 | 20->38 |
| 082626-pnb109 | 83 | 196 | 23->90 | 28->47 |
| 082626-pnb99 | 43 | 81 | 8->21 | 13->24 |
| 082726-pnb102 | 76 | 239 | 22->109 | 31->51 |
| 082726-pnb104 | 50 | 170 | 11->79 | 20->30 |
| 082726-pnb105 | 30 | 79 | 6->28 | 18->28 |
| 082726-pnb106 | 84 | 184 | 46->113 | 15->24 |
| 082726-pnb109 | 70 | 205 | 17->82 | 28->46 |
| 082826-pnb99 | 75 | 198 | 10->66 | 41->65 |
| total | 874 | 2237 | 252->960 | 310->517 |

Scores = deployed vCA1 classifier.joblib on each set's own 35-col features (curator.score_neurons; rank columns are per set, so old/new are not strictly comparable). 2.6x the candidates; +207 at >= 0.30 (+67%), +708 below 0.02 (likely junk, all go to Taylor at threshold 0). The first-pass seed replay (+75%) under-predicted the gain: later greedy passes and the median-filter rescue add more.

Watcher note: all three watcher logs (vCA1, BLA, DG_AL) stopped at 23:42-23:43 on 2026-10-05, ~3 min after the runners started, while the python processes stay alive -> frozen consoles (QuickEdit pattern) or a hang; REVIEW_QUEUE.md is stale since then. The vCA1 watcher needs a restart anyway before the 40 are staged (it holds the pre-override params.py). Operator next: delete the 12 stale outbox bundles, push `--all --area vCA1 --animal <six> --assignee Taylor --force` (dry run below), tell Taylor to discard the old downloads.

2026-10-06 08:15-08:40: operator deleted the 12 stale bundles and pushed the re-run pilots (markers 08:11-08:15, server bundles verified byte-identical, threshold 0); vCA1 + BLA watchers restarted ~08:13 (animal_params.json rebuilt, cohort entries intact, overrides file untouched); DG_AL watcher still frozen. Full 0.70 candidate sets saved per session in `.rerun_mincorr070_2026-10-06/` (6 files) because ingest replaces spatial_footprints.mat with the final kept set. Handoff for the reassessment (Cn, PNR, gSig/gSiz, estimator rule, classifier; decision rules for the 40): `docs/HANDOFF_2026-10-06_COHORT5_REASSESSMENT.md`.


## 2026-10-12 08:45: whole-inbox ingest, UNDONE at 10:11

The user ran `ingest_returns.py` without a session argument at 08:45:13-08:45:37 and asked for it to be
undone (returns stay in the inbox; some sessions are to be re-reviewed). What it wrote, measured with NTFS
ChangeTime (copy2 preserves source mtimes, so mtimes say nothing about arrival; scan script was
session-scratch only):

- 112 sessions got `labels_provenance.txt` rewritten with `ingested: 2026-10-12` because `write_provenance`
  runs on every ingest that carries labels.mat, even when every file was size-skipped. 108 of them (BLA
  2tones/3odor/6odor, DG_AL, older vCA1) received nothing else; their previous `ingested:` timestamps are
  lost and were NOT restored (reviewer/source lines unchanged). BLA and DG_AL classifiers untouched.
- 4 sessions received files, all `vCA1/6odorDualDiffRew/AVG5x-TSeries-082626-*`:

| session | from | labels (keep/delete) | verdict |
|---|---|---|---|
| pnb102 | inbox/Taylor/vCA1/6odorDualDiffRew | 44 / 171 of 215 | valid review of the 10-06 set |
| pnb99 | inbox/Taylor/vCA1/6odorDualDiffRew | 29 / 52 of 81 | valid review of the 10-06 set |
| pnb105 | inbox/Alisia/<session> (area+task dropped) | 10 / 102 of 112 | STALE: review of the est85 set (features 266) |
| pnb106 | inbox/Alisia/<session> (area+task dropped) | 7 / 109 of 116 | STALE: review of the est85 set (features 228) |

  Each created labels.mat, ROIs.jpg, merge_log_*.mat, labels_provenance.txt and overwrote neuron.mat,
  Coor.mat, spatial_footprints.mat, A/C/C_df/C_raw/S.txt and the kept-cell PNGs (CNMFe_final_save.m
  lines 767-794). The two stale bundles also overwrote review_neuron.mat, review_report.pdf,
  review_summary.txt (09-25 versions, identical to `.prev_est85_2026-10-05/`) and Ybg_weights.mat (09-24).
- vCA1 watcher retrained at 08:47:09 on 163 sessions (161 + pnb102 + pnb99; pnb105/106 skipped with a
  size-mismatch WARN). md5 of that model 12b7e4041f7aadf155a80cac695a324d.

Undo (`undo_ingest_2026-10-12.py --go`, 10:11:00-10:11:02; dry run first): deleted the created files
(each verified to have NTFS creation date 2026-10-12); restored spatial_footprints.mat (all four) and
review_summary.txt (pnb105/106) from `<session>/.rerun_mincorr070_2026-10-06/`; restored review_neuron.mat,
review_report.pdf, Ybg_weights.mat for pnb105/106 by copying FROM the server outbox bundles (10-06
versions, server read-only); pnb102/pnb99's review files were unchanged and left alone. classifier.joblib
rolled back to `classifier_prepilot_2026-09-24.joblib` (161 sessions, T 0.02, md5
affdfaab98ed6341b87cc5cea223dd85 = the model deployed since 09-24; byte-identity with the overwritten file
cannot be proven). Today's 163-session model is kept only in the Claude session scratchpad. No vCA1
labels.mat is newer than the restored model, and the watcher log shows no retrain afterwards.

All four sessions are back to awaiting-review (ROIs_candidates.jpg + review_neuron.mat, no ROIs.jpg);
review_assigned.txt untouched. NOT restorable (no pre-ingest copy anywhere): neuron.mat, Coor.mat,
A/C/C_df/C_raw/S.txt and 44/10/7/29 PNGs per session now hold the reviewers' curated output. Decision:
leave them; the next ingested review overwrites them again anyway.

Open: the four returns are still in the inbox, so a bare `ingest_returns.py` run would re-import them (use
explicit paths). pnb105/pnb106 need a fresh review of the 10-06 bundles. Two ingest flaws surfaced:
provenance is rewritten even when nothing is copied, and the ingest keeps no copy of the local files it
overwrites.

## 2026-10-16 15:50: pre-ingest audit of Taylor's cohort-5 returns (read-only, nothing ingested)

Scripts: Claude scratchpad audit_returns.py / audit_returns2.py / cn_maxima_plaus.py (server read-only).

NINE returns sit in inbox/Taylor/vCA1/6odorDualDiffRew (user expected 8; the ninth,
082726-pnb105, landed 2026-10-16 12:40-12:43). Missing: 082626-pnb105, 082626-pnb106, 082726-pnb106.
Every one of the nine is a review of the 2026-10-06 min_corr-0.70 bundle, three ways:
  - labels.mat length == 0.70 candidate count (215, 239, 79, 196, 205, 176, 170, 81, 198); none equals
    the est85 count.
  - bundle inputs in the return (review_summary.txt, pnr.mat md5; Cn/Ybg/report/review_neuron.mat
    sizes + first-8MB md5) identical to the local 0.70 run and to the outbox bundle.
  - every returned kept footprint matches a labels==1 candidate of the local 0.70 set
    (cosine >= 0.68, none < 0.45); 3-12 kept cells per session have NO counterpart in the est85 set.
Kept / reviewed: pnb102 44/215, 40/239; pnb104 32/176, 31/170; pnb105 (082726) 24/79; pnb109 49/196,
57/205; pnb99 29/81, 49/198. Total 355 / 1559.
The 79-candidate session (082726-pnb105) is a sparse FOV, not a wrong bundle: Cn local maxima > 0.70 /
0.80 / 0.85 = 40 / 29 / 22 vs 55-86 / 36-70 / 29-55 in the other eleven; est85 run had 30 there.
No review_meta.txt in any return (Taylor's MATLAB checkout predates the 10-12 change).

Ingest dry run (whole inbox): would ingest the nine; SKIPs the two stale Alisia est85 returns
(082626-pnb105 112 vs 266, 082626-pnb106 116 vs 228) by the labels-length guard, plus an Alisia
3odor pnb88 return with the area level missing; everything else "unchanged". The four 082626
sessions from the 10-12 incident still carry reviewer neuron.mat / A,C,S.txt / Coor.mat in the root
(pnb105/106: Alisia's stale est85 output); extract_params.m reads only gSig/gSiz from neuron.mat, so
animal_params.json is unaffected, and Taylor's eventual returns overwrite them.

## 2026-10-17: pnb105 "condensed cells, are we missing many?" (read-only measurements)

Folder `analysis_2026-10-17/` (scripts, 3 figures, RESULTS.txt). Findings for 082726-pnb105 (79 cand, 24 kept):
- not a packing problem: kept-cell NN distance median 22.2 px, candidate NN 19.2 px (the SPARSEST set of the
  12); the day-1 field 082626-pnb105 is the dense one (266 cand, candidate NN 10.5 px, not yet reviewed).
- seeding is saturated at 0.70: 62 Cn maxima > 0.70, 53 inside a candidate, the 9 outside are all tiny blobs
  (< 150 px) that the seeding median filter removes; no cell-sized Cn >= 0.75 blob is unseeded.
- gSig is not the lever: the gSig-7 arm (min_corr 0.69) recovers 21 of the 24 kept cells and adds 2 objects;
  gSig 11 recovers 20, adds 0.
- same field as day 1 (mean-image corr 0.80 at zero shift), zero residual motion both days, same laser/PMT/
  zoom per the imaging sheet, but Cn p90 0.72 vs 0.87 and PNR p90 8.3 vs 12.6: a weaker recording day.
- visible somata without any candidate: 25 (vs 15-22 in three reference sessions), at Cn 0.60-0.70 (14) or
  < 0.60 (9); only min_corr < 0.70 could reach them, and it would do the same in every session.
- the dense central patch (33 candidates in 160x160 px) is over-segmented (overlapping fragments, deleted),
  not under-seeded.
Assessment given to the user: no parameter change for the day-2 session; the day-1 dense field is the one to
examine (Analysis C) once Taylor returns it; FOV A recurs in the 40 (9/7-9/10 sessions).

## 2026-10-18: pnb105 day 0 vs day 1 -- CORRECTION and cause (read-only)

Correction to the 10-17 entry: the two pnb105 fields are the same FOV but displaced 73 px (~24 um) in y
(high-pass alignment; the zero-shift claim came from the vignetting band). With the shift applied, all 24
of Taylor's kept day-1 cells have a day-0 counterpart, and 145 of the 246 day-0 candidates inside the day-1
field have NO day-1 candidate: they were faint-but-active cells on day 0 (Cn ~0.90, 22% soma-like) and sit
at day-1 Cn 0.50-0.70 (median 0.60). Bright cells also dropped (0.96 -> 0.85). The day-1 recording is
uniformly weaker (PNR p90 6-8 in every 1000-frame block vs 10-15 on day 0), no drift, no motion.
Acquisition (XML): same laser/PMT/zoom; z 5-10 um apart (motor -397/piezo 41.8 vs -391.9/46.9); bitDepth 12
on 082726 for ALL five animals (13 everywhere else) -> raw counts halved rig-wide that day (ratios 0.52-0.58;
pnb99 0826 vs 0828 1.16), pnb105's field a further ~20% dimmer (p50 ratio 0.42, 27% zero pixels). No ADC
clipping. The other four animals kept their candidate counts on the 12-bit day. uint8 AVG5x tifs are scaled
per session (not comparable across days).
Numbers + scripts + 2 aligned figures: analysis_2026-10-17/RESULTS.txt items 7-10.
Open (user's call): session override min_corr 0.60 for 082726-pnb105 (seeds ~70 known day-0 cells at low
SNR) vs accept 24; acquisition-side check of the 12-bit setting and the z offset; the 85th-pct estimator
moves the wrong way on weak sessions (reassessment input).

## 2026-10-18 13:13-15:02: 082726-pnb105 RE-RUN at min_corr 0.60 (user decision; one-session test)

- `agent/config/param_overrides.json`: session-level entry for 082726-pnb105 = min_corr 0.60 (animal stays 0.70).
- `rerun_queue.py` gained `--archive <name>` and refuses a session whose archive folder exists.
- Pre-run copies: `.prev_mincorr070_2026-10-18/` (8 files of the 0.70 run) and
  `.review_mincorr070_taylor_2026-10-16/` (Taylor's 0.70 review outputs, 35 files, copied from the inbox;
  README inside). The inbox return itself is untouched; ingest will SKIP it (79 labels vs 169 live).
- Run: detached runner PID 5476, log `agent/logs/rerun_cohort5_custom.log`; pre-pass estimate 0.81/5.7/bd 14,
  animal override 0.70, session override 0.60 -> final {gSig 9, gSiz 36, min_corr 0.6, min_pnr 5.7, bd 14};
  headless done 15:00 (169 candidates), curator threshold 0 (auto-rejected 0/169, 13 merge pairs),
  review_neuron.mat 169, Ybg_weights cached; 108.9 min with the vCA1 watcher running alongside.
- Verified (analysis_2026-10-17/verify_060.py): all outputs 15:00-15:02; Cn at candidate peak new set
  >= 0.85: 57 | 0.75-0.85: 55 | 0.70-0.75: 20 | 0.60-0.70: 32 | < 0.60: 5 (old 0.70 set 35/37/5/2/0);
  77 of 79 old candidates have a counterpart (cos >= 0.45); Taylor's 24 kept: 24 of 24 (min cosine 0.85);
  53 new candidates without a 0.70 counterpart (20 at Cn >= 0.70, 28 at 0.60-0.70, 5 below).
- 15:03 STAGED (one-time server rewrite authorised by the user 2026-10-18): the stale 0.70 outbox bundle was
  RENAMED aside to `outbox/Taylor/vCA1/6odorDualDiffRew/.stale_mincorr070_2026-10-06_<session>/` (8 files,
  verified against the 10-06 copies first; nothing deleted) and the 0.60 bundle pushed with
  `push_review_bundle.py <session> --assignee Taylor` (3585 MB, 7 files + launcher). Server bundle verified
  against the local session (sizes + md5 of the first 4 MB): all OK. Local marker rewritten 15:03.
  Operator items: tell Taylor the new 169-candidate bundle replaces her 10-16 review of this session; the
  dot-prefixed stale folder can be deleted; her 0.70 return stays in the inbox (ingest skips it by length).
  Whole-inbox ingest of the other eight returns is safe to run now.

## 2026-10-28: all 12 pilot returns in, counts audited, ingest script ready (nothing ingested yet)

Audit (analysis_2026-10-17/.. scratch audit_12_sheet.py, read-only): every pilot session has a return in
inbox/Taylor whose labels.mat length equals the live review set; kept 44/40/32/31/24/34/10/11/49/57/29/49
(= 410 of 2327 candidates; 082726-pnb105 = the 0.60 re-run, 34 kept of 169, returned 10-20). Imaging sheet
(D:/Analysis/6odor dual DR/2p imaging parameters 4.0 (retro) - 4_6odor DR.csv, on-disk copy dated 10-27 07:00)
agrees on 9 rows; the three pnb105/pnb106 rows carry the final count at the end of the cell (24, 10, 11),
user is putting it in front in the online sheet -- the CSV is a manual Google Sheets export (no script pulls
it; fov_sheets.json / plan_fov_assignment.py / build_6odorDR_tree.py all read the local file).
Provenance note: the 082626-pnb105 review (24) was done by Julian with Yawen but sits in inbox/Taylor ->
reviewer of record will be Taylor unless labels_provenance.txt is edited after ingest.
Hazard found: a whole-inbox ingest walks inbox/Alisia first; her 082726-pnb106 return has an unreadable
labels.mat (255 bytes, 10-19) that passes the length guard (unreadable -> None -> no comparison), would be
copied in, and Taylor's real review would then be PARKED as a re-review. Her 082626 pnb105/pnb106 returns
are the known est85 ones (112/116 labels; guard rejects them). Unrelated inbox items: pnb88 3odor return with
the area level missing, two Alisia pnb98 re-reviews (052026, 052626), Alisia's 052726-pnb98 return.
Script `agent/eval/cohort5_2026-09/ingest_pilots_2026-10-28.py` (dry-run default, --go acts): (1) OPERATOR-RUN
server step: renames the three broken Alisia returns into inbox/Alisia/.stale_cohort5_2026-10-28/<session>/
(dot path = skipped by ingest; rename only, nothing deleted; moves only returns measured unreadable or
length-mismatched); (2) runs ingest_returns.py once per session with the explicit Taylor path; (3) checks the
landed labels.mat against the live set. Dry run 2026-10-28 06:40: 3 would move, 12 of 12 consistent.
Next after --go: watcher retrains vCA1; Analysis A (kept-rate by Cn band, + the 0.60 session) and the
pilot_far.py false-auto-reject check decide min_corr and the curation threshold for the 29 unstaged
6odorDualDiffRew tifs (41 on the server for the six animals minus the 12 local, incl. 082726-pnb99-440um).

## 2026-10-28 06:42-07:10: 12 pilots INGESTED; Analysis A and the auto-reject checks (analysis_2026-10-28/)

- Ingest (user's one-time server permission): `ingest_pilots_2026-10-28.py --go` renamed the three broken
  Alisia returns into inbox/Alisia/.stale_cohort5_2026-10-28/ and ingested the 12 Taylor returns by explicit
  path; 12 of 12 labels.mat lengths == live sets (log agent/logs/ingest_pilots_2026-10-28.log). Provenance
  names Taylor for all 12 (082626-pnb105 was reviewed by Julian + Yawen; edit by hand if wanted).
- Watcher retrain fired 06:42:30 on the FIRST FIVE landed sessions only (model 06:42:39, "Sessions used: 170")
  and will not fire again: _needs_classifier_update compares labels.mat mtimes (copy2 keeps the reviewers'
  10-06..10-27 dates) with the model's. 7 of the 12 are NOT in the deployed model -> manual /retrain-area vCA1
  (or train_classifier_vCA1.py --prospective-only --model xgboost) before the 29 are curated.
- Analysis A (kept_by_cn_band.py, Cn peak in the footprint core, 2327 candidates / 410 kept):
  pooled kept-rate 0.60-0.70: 0/33 | 0.70-0.75: 1/74 (1%) | 0.75-0.80: 10/221 (5%) | 0.80-0.85: 35/407 (9%)
  | >= 0.85: 364/1588 (23%). Kept cells with Cn peak < 0.75: 1 of 410; < 0.80: 11. Kept cells with no est85
  counterpart (cos < 0.45): 53 total, median 3.5/session, median Cn 0.80-0.86 -> the 0.70 seed's gain came
  from high-Cn cells the est85 run dropped (median-filter effect), not from the 0.70-0.80 bands.
  The 0.60 session (082726-pnb105, 34 kept of 169): 0.60-0.70 band 0/25 kept. Handoff rule: 0.70-0.75 rate is
  0.16 of the 0.80-0.85 rate (< 0.33) and the band gain is < 5/session -> the bands below 0.80 add almost
  nothing, but a 0.75/0.80 seed would lose 1/11 kept cells plus whatever the median filter drops (est85 lost
  53). Reading: keep 0.70 for the 29 (margin below the kept cells' Cn), drop 0.60.
- pilot_far.py, prepilot model (161 sessions, never saw these animals), 410 kept: T 0.005/0.01: 5 misses
  (1.22%), junk 37.9%; 0.02: 13 (3.17%), junk 52.6%; 0.03: 20 (4.88%); 0.04: 26 (6.34%). Per animal at
  0.02: pnb102 1.2, pnb104 6.4, pnb105 5.2, pnb106 0, pnb109 4.7, pnb99 0 %.
- cohort5_within_animal_oof.py (rt_lib, 177-session pool, sessions held out / animals seen, 3 seeds),
  cohort-5 pooled 410 kept: T 0.005: 0.00% (junk 17%); 0.01: 0.00% (26%); 0.02: 0.49% mean / 0.73% worst
  seed (37%); 0.03: 1.22% (44%); 0.04: 1.87% (50%). Per animal at 0.02: pnb109 1.57% (worst 2.83%),
  pnb104 0.53%, others 0. AUC cohort-5 rows 0.923 (rest 0.922). Rest of pool at 0.02: 0.24%.
- Decision pending (user): curation threshold for the 29 -- 0.01 (0 misses measured, junk 26%) or 0.02 (meets
  the 1% rule pooled, 0.49%, junk 37%; pnb109 alone 1.6%). Seeding: 0.70 via the existing animal-level
  overrides. Then retrain, stage the 29 tifs, push in waves.

## 2026-10-28 07:30: DECISION threshold 0.01 for the remaining cohort-5 sessions; retrained; 28 tifs staged

- User decision: curation threshold 0.01 for the 28 remaining 6odorDualDiffRew sessions of the six animals
  (29 unstaged tifs on the server minus the 450-frame stub 082726-pnb99-440um, PrairieView crash per the
  sheet); reassess when they return. Seeding stays 0.70 (animal-level overrides), 0.60 dropped.
- No watcher pause and no multiple retrains were needed: the watcher was idle (last tif 10-18, polls only),
  the trainer takes every labelled session at once, and the curator re-reads the model file per session.
- Implementation (so the value survives the watcher's own auto-retrains, which pass no --threshold):
  `train_classifier_vCA1.py` _VALIDATED_THRESHOLD 0.02 -> 0.01 with the policy history in the comment block;
  `agent/eval/vca1_v2_2026-08/gate_decision.json` chosen_T 0.02 -> 0.01 plus a dated user_decision_T_2026-10
  block (same pattern as the 2026-09 decision) so verify_vca1.py stays consistent. Both files were already
  modified-uncommitted from the 09-24 decision.
- Retrain (manual, `train_classifier_vCA1.py --prospective-only --model xgboost`, log
  agent/logs/retrain_vCA1_2026-10-28.log): Sessions used 177 (65 agent incl. the 12 pilots), threshold 0.01,
  agent-pool OOF false auto-reject 0.13% (2 of 1522 kept), junk auto-rejected 22.1%, AUC reviewed 0.920.
  Deployed joblib verified: xgboost, n_sessions 177, feature_version 2, 35 features, reject_threshold 0.01,
  agent_weight 5.0; snapshot agent/model/vCA1/classifier_cohort5_T001_2026-10-28.joblib.
  verify_vca1.py: all threshold/contract checks ok; two pre-existing failures unrelated to today (n_sessions
  vs a manifest pinned at the 134-session 08-26 corpus; corpus-identity hash on 3odor/071526-pnb98, a folder
  renamed in the 09-24 fork repair).
- Staging: `stage_cohort5_28_2026-10-28.py --go` (copy-only, .part then rename, server read-only; log
  agent/logs/stage_cohort5_28_2026-10-28.log): 28 tifs, 130 GB, into vCA1/6odorDualDiffRew/ as loose files
  for the watcher (120-s stability check). The watcher processes them one at a time (~1.5-2 h each);
  pushes to Taylor in waves as they complete (push_review_bundle.py --all --area vCA1 --animal <six>
  --assignee Taylor; no --force).

## 2026-10-28 07:45: Analyses B (min_pnr) and C (gSig/gSiz) scored on Taylor's labels (pnr_gsig_with_labels.py)

- B, PNR gate (full-movie PNR peak in the footprint core minus the session's pre-pass min_pnr, gates 5.7-8.4):
  kept cells within 1 unit of the gate 0 of 410, within 2 units 0; kept-rate by margin 0-1: 0/8, 1-2: 0/47,
  2-4: 4/300 (1%), 4-8: 61/758 (8%), >= 8: 345/1209 (29%). Kept PNR median 20.4, 5th pct 11.9; absolute
  bands < 6: 0/4, 6-8: 0/32, 8-10: 3/222, 10-15: 80/884, 15-20: 114/533, 20-30: 166/499, >= 30: 47/153.
  The PNR gate is not binding; lowering it buys nothing, raising it to 10 would cost 3 kept of 410. No
  change: min_pnr stays estimated per session for the 28.
- C, gSig arms (2026-09-25 sweep, gSiz = 4 x gSig, each at its own estimated seed 0.65-0.90) vs the live
  9/36 sets, cosine >= 0.45: recall of kept cells with live Cn >= 0.85 (above every arm's seed) gSig 7
  353/364 (97.0%), gSig 11 329/364 (90.4%); of all kept 370/410 (90.2%) and 338/410 (82.4%); arm candidates
  with no live counterpart 46 (gSig 7) and 36 (gSig 11) over 12 sessions, never labelled (Sept first-pass
  scoring called most junk, 6 confident on pnb105 day 1). Kept-cell core radius at the processing resolution
  4.1-5.4 px per animal (diameter 8-11 px) against gSig 9 / gSiz 36. Neither arm beats 9/36 on recall; the
  untested direction is a smaller kernel (gSig 5-6, closer to the measured radius), not run. No change:
  9/36 stays for the 28; a gSig 5 arm on 2-3 sessions is the only open gSig question.
- Running state: watcher started 090526-pnb102 at 06:58 (first of the 28); staging copy in progress.

## 2026-10-28 08:05: candidate-count flag for the 28; what the auto-reject does with over-seeded junk

- `check_candidate_counts.py` (read-only): full N per cohort-5 session from candidate_features.npz, review
  set, resolved seeds, Cn/PNR p90; flags LOW < 100, HIGH > 300; exit 1 when a new session is flagged.
  Background loop (10-min cadence) reports the first flagged new session or all 28 done. Pilots at 0.70: 169-266
  except 082626-pnb99 (81, odor-only day 0) and 082726-pnb105 (79; 169 at 0.60).
- Measured on the 0.60 session (25 band candidates 0.60-0.70, all deleted by Taylor): the prepilot model
  (never saw the animal) scores 60% of them below 0.01 and 72% below 0.02; of all 135 junk in that set 47% /
  64%; kept cells 3% / 6%. So at the deployed 0.01 the auto-reject removes about half of the junk a lower seed
  adds and the rest goes to the reviewer; the band itself yielded 0 kept cells (0/33 pooled over 12 sessions).
  Other costs of over-seeding: the extraction changes (0.60 run: 20 new objects above 0.70, 2 old candidates
  lost, 13 merge pairs, +15% runtime; dense patches over-segment), and the 13 within-session rank features
  shift, so the thresholds calibrated on 0.70 sets do not transfer exactly. No harm to kept cells was measured
  (24/24 recovered). Posture for the 28: seed 0.70, flag N < 100 as a recording-quality signal, decide per
  session rather than lowering the seed.

## 2026-10-28 10:15: first flag -- 090526-pnb105-520um-32z-000 (N = 44)

- Watcher timing: 090526-pnb102 06:58 -> 08:37 (244 candidates, 68 auto-rejected at 0.01, Cn p90 0.78,
  PNR p90 9.5, ok); 090526-pnb105 08:37 -> 10:01 (82 min); 090526-pnb106 started 10:01. About 1.5 h per
  session, serial, so the 28 finish around the morning of 2026-10-30.
- 090526-pnb105: 16,863 frames (full length); estimator min_corr 0.72 -> animal override 0.70, min_pnr 5.5
  (lowest of the cohort), bd 10; 44 candidates, 7 auto-rejected at 0.01, 37 to review. Deployed-model scores:
  7 >= 0.5, 9 in 0.2-0.5, 21 in 0.01-0.2, 7 < 0.01. Cn p50/p85/p90/p95 0.53/0.60/0.63/0.69, PNR p90 6.8:
  weaker than 082726-pnb105 (0.53/0.63/0.68/0.75, PNR p90 8.3), which gave 79 at 0.70 and 169 at 0.60 with
  0 kept cells from the added band. ROIs_candidates.jpg shows a sparse field, candidates in the upper-left half.
- Imaging sheet (export of 10-27 07:00): 9/5/26 pnb105 z 520 um, zoom 3.2, laser 471/80, PMT 680 (its FOVA
  days 620-630), "d8 ... 22 trials pre-training. new FOV (FOVC) ... lots of issues with face recording ...
  had to image continuous". FOVC appears on 9/5 only (9/6 FOVB 472 um, 9/7 onwards FOVA 353-404 um): a
  single-day field with no CellReg partner. All five 9/5 sessions are new fields (pnb99 FOVB 330 um, pnb102
  FOVC 627 um, pnb105 FOVC, pnb106 FOVB 180 um, pnb109 FOVC 504 um); pnb102's came out fine (244). pnb106's
  pre-pass image has Cn p90 0.68 and PNR p90 6.3, so it may also come in low.
- Seed-yield proxy (`analysis_2026-10-28/seed_yield_estimate.py`: Cn local maxima passing the PNR gate; it
  saturates near 80-90 on good fields, so only the ratio across thresholds is informative): 090526-pnb105
  38 at 0.70, 57 at 0.65, 67 at 0.60 -- a 0.60 seed would roughly double N, as it did on 082726-pnb105
  (40 -> 57 maxima, 79 -> 169 candidates, 0 extra kept).
- Recommendation given: keep 0.70 and send the 37 as they are; the one 0.60 test on a weak pnb105 field added
  candidates and no kept cells. Alternatives offered: a one-session 0.60 override (second MATLAB job, ~1.5 h,
  expected ~90 candidates, mostly junk) or retiring the session as a single-day FOV. User's call. Flag loop
  re-armed with this session excluded.

## 2026-10-28 12:50: DECISION -- pnb105 at min_corr 0.60 for every session; 090526-pnb105 re-running

- User: "keep 0.60 on all 105 videos". `agent/config/param_overrides.json`: pnb105 animal-level min_corr
  0.70 -> 0.60 (the 082726 session entry folded into it; the other five animals stay at 0.70).
  `params.py:estimate_all_params` reads the file on every call, so the running watcher applies it to the five
  pnb105 tifs still queued (090626-472um FOVB, 090726-404um, 090826-385um, 090926-394um, 091026-353um FOVA)
  without a restart; verified read-only on the flagged session ("Animal override (pnb105): min_corr=0.6").
  The reviewed 082626-pnb105 (0.70, 266 candidates, 24 kept) is not re-run.
- `rerun_queue.py` gained `--threshold` (default 0 as before; `model` = the deployed joblib's
  reject_threshold, curator.THRESHOLD_OVERRIDE None) and reports the threshold actually used from
  review_summary.txt; compiled, grepped for delete/move verbs (none; only a log-handler removal).
- 12:47 started: `rerun_queue.py --sessions AVG5x-TSeries-090526-pnb105-520um-32z-000 --archive
  .prev_mincorr070_2026-10-28 --threshold model --go` (log agent/logs/rerun_cohort5_custom.log; the 0.70
  outputs copied into `<session>/.prev_mincorr070_2026-10-28/`, 8 files). Second MATLAB job next to the
  watcher (machine limit two). Nothing of this session was pushed, so there is no server bundle to rename.
- Watcher meanwhile: 090526-pnb106 done, 135 candidates (46 auto-rejected at 0.01, 89 to review; Cn p90
  0.73, PNR p90 8.5, ok); 090526-pnb109 running.
- For the reassessment: the 0.01 threshold and the within-session rank features were calibrated on 0.70
  sets; pnb105's 0.60 sets carry more junk and shifted ranks (10-28 08:05). Measure the false auto-reject on
  the pnb105 sessions separately when the labels return.

## 2026-10-28 13:40: PMT question; video-statistics check for visibly missed cells

- Correction from the user: "image continuous" / "continuous face imaging" in the imaging sheet Notes is
  the FACE camera; the 2p recording is always trial-triggered. The continuous-imaging hypothesis offered at
  12:55 for the higher auto-reject fraction is withdrawn (memory: feedback_imaging_sheet_notes).
- Bruker XML, all 41 cohort-5 sessions (`analysis_2026-10-28/xml_all_2026-10-28.txt`, server read-only):
  bitDepth 13 on every 09/05-09/10 session (12-bit was 082726 only). 09/05 vs the animal's pilot days:
  pnb105 PMT 680 vs 620, laser 470 vs 460, z 520 vs 391-397 um, zoom 3.2 vs 3.5; pnb106 PMT 640 vs
  610/570, z 180 vs 201-205; pnb109 PMT 640 vs 640/590, laser 480 vs 450-460, z 504 vs 398-405; pnb102
  PMT 610 vs 630/600, z 627 vs 479-497, zoom 2.8; pnb99 PMT 700 vs 610-660, z 330 vs 399-440. The PMT was
  raised on pnb105/106/99, not on pnb102 (normal count). On 090526-pnb105 the raised gain shows in the raw
  levels (p50 68, p99 150: as bright as any pilot) and in the temporal noise (15.7, the highest; pilots
  10.5-13.6): the gain raised the counts, not the signal. Cn and PNR are gain-insensitive ratios; the deficit
  is the field (125 um deeper than FOVA; 1.6% of pixels with Cn > 0.8 vs 3.1-18.7% on the pilots).
- `video_stats_unseeded.py` (+ .txt, figs/): per session raw levels (300 spread frames), temporal noise
  (200 mid-session frames), Cn/PNR percentiles, soma-like blobs in the mean raw image with no candidate
  support (binned by local Cn; "strong" = contrast above the 99th percentile of random interior points),
  and Cn maxima > 0.60 / > 0.70 outside every candidate (PNR gate, Cn blob >= 150 px). Pilots scored on
  their full 0.70 sets (.rerun_mincorr070_2026-10-06; after ingest the live spatial_footprints.mat holds
  only the kept cells -- first run of the script was wrong for that reason). Reference on the 12 pilots:
  uncovered blobs with Cn >= 0.70 0-10 per session (strong 0-3); Cn maxima > 0.70 outside all candidates
  1-16 (median 5.5). New sessions: 090526-pnb102 6 (strong 0) / 5; 090526-pnb106 13 (2) / 8; 090526-pnb109
  8 (1) / 2; 090526-pnb105 at 0.70: 5 (1) / 0. Nothing above the 0.70 seed is visibly missed on any of
  them. 090526-pnb105 has 31 visible blobs at Cn 0.60-0.70 and 42 Cn maxima > 0.60 outside candidates
  (pilots 5-50, median 33): anatomically visible, weakly correlated; the 0.60 re-run seeds that band (the
  same band gave Taylor 0/33 kept on the pilots). Saturation 0.00-0.04% everywhere.
- Figures: figs/fig_090526-pnb105-520um-32z-000_unseeded.png (candidates cluster in the upper-left; the
  right half of the field is dark, no structure in Cn), fig_090526-pnb106-180um-4z-000_unseeded.png (dense
  shallow field split by a vessel, clump-like candidates, motion-border objects at the right edge),
  fig_090526-pnb102-627um-28z-000_unseeded.png, fig_082726-pnb105-391um-35z-000_unseeded.png (reference).
- 090526-pnb109 finished in the watcher: 247 candidates, ok. pnb105 re-run at 13:05 in the background
  estimate (expected to finish ~14:15).

## 2026-10-28 14:40: 090526-pnb105 re-run at min_corr 0.60 finished (168 candidates, 111 to review)

- rerun_queue.py --threshold model, 112.6 min (12:47-14:40). Params: estimated 0.72/5.5 -> animal override
  min_corr 0.60, gSig 9/36. 168 candidates (0.70 run: 44); curator at the deployed 0.01 auto-rejected 57 ->
  111 to review (0.70 run: 7 -> 37). The 0.70 outputs are archived by copy in
  <session>/.prev_mincorr070_2026-10-28/ (8 files). The live Cn.mat (512) is identical to the archived one.
- analysis_2026-10-28/rerun_060_breakdown.py (-> rerun_060_breakdown.txt): 42 of the old 44 recovered
  (support Jaccard >= 0.5, median 0.78); their scores moved little (median +0.016; one review-set cell is now
  auto-rejected at 0.009, one rejected one now passes); the >= 0.5 count went 7 -> 14, mostly the within-
  session rank features (more weak candidates in the session push the old cells' ranks up). The ~125 added
  candidates: 51 auto-rejected, 62 at 0.01-0.2, 11 at 0.2-0.5, 3 >= 0.5. Their Cn peak (3x3-mean 512 Cn
  over the footprint core, the Analysis A definition) has median 0.72, 84 >= 0.70 and 39 >= 0.80, while the
  same proxy puts all 44 of the 0.70 run at >= 0.72. So the gain is NOT only the 0.60-0.70 band (41
  candidates, 19 of them in the review set): the lower seed also changed what the greedy initialisation
  picked up in the better part of the field. Mechanism NOT verified (seeding uses the 256-res ssub-2 Cn,
  which is not kept after the full run).
- Expected yield, pilot kept-rates applied per band (ESTIMATE, not measured): 0.60-0.70 -> 0 (0/33 on the
  pilots), 0.70-0.80 -> ~2 of 45, >= 0.80 -> 4-9 of 39; roughly 5-10 extra kept cells for 74 extra review
  candidates, and only if the added high-Cn candidates behave like the pilots' (doubtful, since the 0.70
  run did not seed them).
- video_stats_unseeded.py on the live 0.60 set (-> video_stats_unseeded_pnb105_060.txt, figs_060/):
  uncovered blobs with Cn >= 0.70 5 -> 1, 0.60-0.70 31 -> 12; Cn maxima > 0.60 outside all candidates
  42 -> 9 (> 0.70 stays 0). The right half of the field is still dark. The new footprints in the bright
  centre are clumpy/filamentous, and there are motion-border objects along the top and bottom edges.
- Watcher: 090526-pnb99 (PMT 700, 330 um) running since 13:17; 23 loose tifs remain (the other five pnb105
  sessions will seed at 0.60). Nothing pushed. The 0.01-vs-0 curation question for the 28 is still open.

## 2026-10-28 15:00: second flag -- 090526-pnb99-330um-3z-000 (N = 30), assessment only, no change made

- Watcher output 14:58: 30 candidates at 0.70/5.2 (gSig 9/36), 1 auto-rejected at 0.01, 29 to review. Scores
  of the 30: 11 >= 0.5, 7 in 0.2-0.5, 11 in 0.01-0.2, 1 < 0.01; peak_snr median 15.9, cn_correlation median
  0.84 -> the candidates themselves are clean, compact somata clustered in the bright centre of the field
  (ROIs_candidates.jpg) plus 3-4 motion-border objects on the right edge.
- Field: Cn p50/p90/p95 0.51/0.58/0.64 (lowest p90 of the cohort), 1.1% of pixels with Cn > 0.8 (pilots
  3.1-18.7%), PNR p90 6.3. Raw p1/p50/p99 39/81/169 (brightest of the cohort), noise 14.7 (second highest),
  saturation 0.00%. XML: bitDepth 13, PMT 700 (pnb99 pilot days 610-660; the highest gain in the cohort),
  laser 470.75, z 330 um (pilot FOVA 399-440 um). Imaging sheet 9/5/26: "d8 ... 25 trials pre-training. new
  FOV (FOVB)"; 9/6: FOVA at 459 um; 9/7: "No good FOV left"; 9/8-9/9: FOVA at 463 um. So FOVB is a
  single-day field, as 090526-pnb105 FOVC was, and the remaining three pnb99 sessions are FOVA (pilot
  counts there: 81 at 399 um, 198 at 436 um).
- video_stats_unseeded.py (-> video_stats_unseeded_pnb99.txt, figs/fig_090526-pnb99-330um-3z-000_unseeded.png):
  130 soma-like blobs in the mean image (pilots 85-118), only 22 covered; uncovered: 8 with Cn >= 0.70
  (3 strong; pilots 0-10 / 0-3), 25 at 0.60-0.70, 75 at Cn < 0.60. Cn maxima outside all candidates:
  15 > 0.60, 1 > 0.70 (pilots 1-16). The figure shows the point: dozens of bright, visible somata in the
  centre of the field with no Cn signal -- anatomically rich, functionally quiet. Nothing above the 0.70
  seed is visibly missed; the visible-but-uncovered cells are the Cn < 0.60 ones.
- seed_yield_estimate.py: Cn maxima passing the PNR gate at 0.55/0.60/0.65/0.70/0.80 = 79/42/28/20/12,
  i.e. a 0.60 seed would roughly double the seeds (pnb105's proxy ratio was 1.8x and its actual gain
  3.8x), mostly in the band that gave 0/33 kept on the pilots.
- Per-session choices brought to the user (none taken): (a) leave at 0.70 and push the 29 as they are
  (the recommendation: clean candidates, the extra visible somata are inactive at Cn < 0.60, single-day
  FOV with no cross-day counterpart); (b) session-level 0.60 re-run (~1.9 h, the second MATLAB slot; NOT
  animal-level, because the other pnb99 sessions are FOVA at 459-463 um); (c) exclude the session.
- Flag loop restarted at 15:00 excluding both flagged fields; 22 tifs remain (watcher is on the next one).

## 2026-10-29 04:15: first HIGH flag -- 090726-pnb105-404um-35z-000 (N = 372 at the animal-level 0.60), no change made

- Watcher 04:08: estimated min_corr 0.80 -> animal override 0.60, min_pnr 6.4; 372 candidates, 126 auto-rejected
  at 0.01 (34%), 246 to review. Cn p90 0.80, PNR p90 10.6: a GOOD field. Sheet 9/7/26: "d10 ... 20 trials
  pre-training. FOVA"; XML z 404 um, PMT 630, laser 471, bitDepth 13. FOVA is the pilots' field (082626 at
  397 um: 266 candidates at 0.70, Taylor kept 24; 082726 at 391 um: 79 at 0.70 / 169 at 0.60, Taylor kept 34
  of the 169). The remaining three pnb105 sessions (090826-385um, 090926-394um, 091026-353um) are FOVA too
  per the sheet, so at the animal-level 0.60 they will look like this one.
- band_scores.py (new, generic per-set band/score table): 090726 at 0.60 = 305 candidates with core Cn >= 0.80
  (85 auto-rejected / 78 score >= 0.5), 49 at 0.70-0.80, only 18 below 0.70. So the 0.60 seed did not mainly
  add band candidates here; it over-seeds a dense field (ROIs_candidates.jpg: overlapping contours across the
  upper half, motion-border objects along the bottom and sides). Sibling sets for scale: 090626-pnb105 FOVB
  at 0.60 229 (71 rejected, 45 >= 0.5); 082626 FOVA at 0.70 266 (42 >= 0.5); 082726 FOVA 79 at 0.70 (29 >=
  0.5) vs 169 at 0.60 (42 >= 0.5).
- MEASURED on 082726-pnb105 (the one FOVA session with both a 0.70 and a reviewed 0.60 set): 32 of Taylor's 34
  kept cells have a counterpart in the 0.70 set (support Jaccard >= 0.5; one at 0.18, one with no overlap).
  The 0.60 seed added 90 candidates and 2 kept cells on that day.
- Per-session choices brought to the user (none taken): (a) leave it, push 246 to review (Taylor reviewed 266
  on this FOV before); (b) re-run 090726 at 0.70 as a session override (~1.9 h, second MATLAB slot; expect
  ~250-270 like 082626); (c) narrow the pnb105 override to the weak fields (session-level 0.60 for 082726 and
  090526, optionally 090626 FOVB) and 0.70 for FOVA, which the watcher would pick up for 090826-pnb105 (starts
  after 090726-pnb109, 090826-pnb102, 090826-pnb104: roughly 09:00-09:30 on 10-29). The animal-level 0.60 was
  decided on "uniformly weak fields"; FOVA on 9/7 is not weak.
- Flag loop restarted excluding the three flagged sessions; 14 tifs remain.

## 2026-10-29 08:00: second HIGH flag -- 090826-pnb102-497um-3z-000 (N = 333 at 0.70), no change made

- Watcher 07:51: estimated min_corr 0.84 -> animal override 0.70, min_pnr 7.0; 333 candidates, 100 auto-rejected
  at 0.01 (30%), 233 to review; 29 merge pairs (090726-pnb102: 15). Cn p90 0.83, 15.7% of pixels with Cn > 0.8,
  PNR p90 13.0: the best field in the cohort. Sheet 9/8/26: "d11. WSE. 18 pre trials. FOVA ... matlab crashed
  after trial 144. restarted ..." -> 21027 frames (the others 16.8-16.9k). XML: laser 481 (9/7: 441, the pnb102
  low), PMT 660 (9/7: 640), z 497, bitDepth 13. FOVA is the pilots' field: 082626 at 479 um 215 candidates /
  Taylor kept 44; 082726 at 497 um 239 / kept 40; 090726 at 494 um 262.
- band_scores.py: 315 of 333 at core Cn >= 0.80, 17 at 0.70-0.80, 1 below; score >= 0.5 only 36 (the other
  pnb102 sets 51-65). Footprint over-splitting (support A > 0.3 max, share of candidates with > 50% of their
  support inside another candidate): 090826 59% (median area 885 px) vs 090726 44% (663 px), 090626 42%
  (622 px), pilots 31-36% (554-597 px); 090726-pnb105 at 0.60 for scale 49% (441 px). So this is the same good
  field, imaged brighter and longer, segmented into larger and more overlapping pieces; the extra ~70
  candidates are fragments, not a different population. video_stats: raw p50 63, noise 12.6, saturation 0.00;
  uncovered blobs Cn >= 0.70 3 (strong 0), Cn maxima > 0.70 outside candidates 5 -> nothing missed.
- Per-session choices brought to the user (none taken): (a) leave at 0.70 and push the 233 (Taylor reviewed
  215 and 239 on this FOV; the fragments are merge/delete work); (b) session-level re-run at a higher seed
  (0.75-0.80) to thin the fragments, NOT recommended -- Analysis A measured the est85 seed cutting real cells;
  (c) nothing else; the bloated footprints are a brightness/gSiz effect outside the standing decisions.
- 090826-pnb104 running since 07:54; 090826-pnb105 (FOVA, still at the animal-level 0.60 unless the override
  is narrowed) is next after it, ~09:30. Flag loop restarted excluding the four flagged sessions; 12 tifs remain.

## 2026-10-29 08:30: USER DECISION -- leave everything as is; state of the batch

- User: "we leave everything as is". So: no re-run of any flagged session (090526-pnb105 stays at its 0.60 set,
  090526-pnb99 at 0.70 / 30, 090726-pnb105 at 0.60 / 372, 090826-pnb102 at 0.70 / 333), the pnb105 animal-level
  0.60 stays for the remaining FOVA sessions, curation stays at the deployed 0.01.
- REVIEW_QUEUE.md is STALE (written 10-28 14:58, lists 5 awaiting): watcher.py writes it only at the end of a
  main-loop pass, and the pass that found 23 ready tifs at 15:01 on 10-28 processes them all before it gets
  there (watcher.py:784-823). It will refresh when the last of the 23 finishes. Real state on disk at 08:25:
  15 cohort-5 sessions complete with review packages and none assigned or pushed, 090826-pnb104 running since
  07:54, 12 tifs loose. Sessions take 1.6-2.0 h each (090826-pnb102 2.0 h, the long movie), so the batch and
  the queue file should complete around 06:00 on 10-30.

## 2026-10-29 12:55: 090826-pnb105 HIGH flag (351 at animal-level 0.60) -- left as is

Flag loop caught AVG5x-TSeries-090826-pnb105-385um-35z-000: 351 candidates, 240 in the review set,
111 auto-rejected at 0.01 (min_corr 0.6 animal override / min_pnr 6.1; Cn p90 0.78, PNR p90 10.6).
Headless run 09:50 -> 12:40 (2.84 h, 21019 frames, 5.51 GB tif).

Context (imaging sheet row PNB105 9/8/26, Bruker XML): FOVA again (same field as the 082726 pilot and
090726-pnb105, which gave 372 at 0.60), day 11 within-session extinction, z 385, laser 491 (vs 471 on
09/07, 461 on 08/27), PMT 630, zoom 3.5.  The sheet's "continuous face imaging" note is the face camera.

band_scores.py (3x3-mean Cn over the footprint core): 267/351 at Cn >= 0.80 (82 of them score >= 0.5),
60 at 0.70-0.80 (0 >= 0.5), 23 at 0.60-0.70 (18 auto-rejected, 0 >= 0.5), 1 below 0.60.  Score
distribution 82 >= 0.5 / 31 at 0.2-0.5 / 127 at 0.01-0.2 / 111 < 0.01, i.e. the same shape as
090726-pnb105 (79 / 25 / 142 / 126).  Over-splitting (support A > 0.3 max): median area 433 px, 49% of
candidates have > 50% of their support inside another candidate, 109 Jaccard >= 0.5 pairs; 090726-pnb105
on the same field: 441 px, 49%, 99 pairs.  Nothing new: it is the FOVA-at-0.60 pattern already assessed
on 10-29 04:15, not a bright/long-recording fragmentation like 090826-pnb102 (885 px, 59%).

Decision: left as is per the user's 08:30 decision (no re-runs, pnb105 stays at 0.60, curation 0.01).
Flag loop restarted excluding the five flagged sessions.

Batch state 12:55: 17 of 28 complete (2727 candidates, 1915 in review sets), 090826-pnb106 running since
12:41 (5.51 GB), 10 tifs loose (090826-pnb109/pnb99 and 090926-pnb104 at ~5.2 GB, the other seven at
~4.2 GB).  Run times so far: median 1.63 h, the three 5.5 GB 09/08 sessions 2.0-2.8 h.  Rough finish:
late morning 2026-10-30.  Nothing pushed.

## 2026-10-30 01:50: 090926-pnb105 HIGH flag (348 at animal-level 0.60) -- left as is

Flag loop caught AVG5x-TSeries-090926-pnb105-394um-35z-000: 348 candidates, 259 in the review set,
89 auto-rejected at 0.01 (min_corr 0.6 animal override / min_pnr 6.4; Cn p90 0.80, PNR p90 11.2).
Headless run 23:50 -> 01:40 (1.83 h).  Imaging sheet PNB105 9/9/26: FOVA, day 12 extinction recall,
z 394, laser 491, PMT 650, zoom 3.5 (XML agrees).

Third FOVA session at 0.60 and the same shape as the other two: 297/348 at Cn >= 0.80 (76 of them
score >= 0.5), 35 at 0.70-0.80, 16 at 0.60-0.70 (8 auto-rejected, 0 >= 0.5), 1 below 0.60; scores
77 >= 0.5 / 45 at 0.2-0.5 / 137 at 0.01-0.2 / 89 < 0.01.  Over-splitting median area 435 px, 52% of
candidates with > 50% of their support inside another, 123 Jaccard >= 0.5 pairs (090726: 441 px / 49% /
99; 090826: 433 px / 49% / 109).  Fewer auto-rejects than the earlier two (89 vs 126 / 111) -- the
review set is the largest of the batch so far at 259.

Decision: left as is per the 10-29 08:30 decision.  Flag loop restarted excluding the six flagged
sessions.  Expect 091026-pnb105 (FOVA, 0.60) to flag HIGH for the same reason.

Batch state 01:50: 23 of 28 complete, 090926-pnb106 running since 01:42, 4 tifs loose (090926-pnb109,
090926-pnb99, 091026-pnb104, 091026-pnb105).  Run times since noon 10-29: the three 5.5 GB 09/08 sessions
2.0-2.7 h, the 4.2 GB ones 1.7-2.1 h.  Rough finish: ~11:00 2026-10-30.  Nothing pushed.

## 2026-10-30 11:10: BATCH COMPLETE -- all 28 processed, curated at 0.01, awaiting push

- Last headless run 091026-pnb105 done 10:34; watcher idle since (polls only); REVIEW_QUEUE.md rewritten
  11:01:57 and lists all 28 under "Awaiting your review", BG cache Ready for every one.
- check_candidate_counts.py --only-new: 28 complete, 0 pending; 6645 candidates, 4857 in review sets, 1788
  auto-rejected at 0.01.  Review load per animal: pnb99 4 sessions / 533, pnb102 5 / 946, pnb104 4 / 716,
  pnb105 6 / 1232, pnb106 4 / 576, pnb109 5 / 854.
- Every session folder has neuron.mat, review_neuron.mat, candidate_features.npz, spatial_footprints.mat,
  Ybg_weights.mat, Cn.mat and one PDF; none has review_assigned.txt or labels.mat (nothing pushed).
- Log scan of the batch's own lines (from line 336423, 10-28 06:58): no MATLAB errors, tracebacks or
  timeouts; 28 curator lines.  The 15 CNMFe_precompute_BG/imresize errors and the bla36 PermissionError
  that a naive grep shows are from much earlier in the log (last at line 119331), not this batch.
- Server check (read-only, X:/Jeremy/for CNMFe/6odorDualDiffRew): 41 tifs for the six animals, 40 processed
  locally (12 pilots + these 28); the only one not processed is the 450-frame stub
  082726-pnb99-440um-3z-000, excluded on purpose.
- Flags 7 in total.  Two new since 01:50, both left as is per the 10-29 08:30 decision:
  * 090926-pnb106 HIGH 315 (0.70; 196 review / 119 auto-rej): FOVA (same field as 090826-pnb106, 250),
    day 12, Cn p90 0.85 = highest of the batch; 287/315 at Cn >= 0.80 but only 31 score >= 0.5; 0.60-0.70
    band 2; over-split 39% / 397 px (090826-pnb106: 36% / 362 px).  A bright dense field, not a seeding
    or fragmentation problem; the review set is ordinary.
  * 091026-pnb105 HIGH 304 (0.60; 218 / 86): fourth FOVA session at 0.60, as predicted; 230 at >= 0.80, 64
    score >= 0.5, 17 in 0.60-0.70 (11 auto-rej, 0 >= 0.5), over-split 53% / 475 px.
- Next (operator, on the user's say only): push the 28 to Taylor -- dry run first:
  push_review_bundle.py --all --area vCA1 --animal pnb99,pnb102,pnb104,pnb105,pnb106,pnb109 --assignee Taylor
  (no --force); measure the false auto-reject rate on the pnb105 0.60 sessions separately when they return.

## 2026-11-02: re-check of 090526-pnb99 (30 candidates, 29 to review) on the user's question -- correct, no change

- Run is complete and normal: 16890 frames (full length), estimated min_corr 0.73 -> animal override 0.70,
  min_pnr 5.2, gSig 9/36, bd 10; no errors; pushed to Taylor 2026-10-30 11:57 (review_assigned.txt).
- Field vs the same animal's other days (Cn.mat / pnr.mat, 10-px border cut): pixels with Cn >= 0.70 = 2.8%
  (FOVA days 090626/090826/090926: 21.1 / 19.2 / 23.0%; pilots 082626 11.8%, 082826 19.3%); Cn p90 0.58 vs
  0.72-0.80; PNR p90 6.3 vs 8.2-11.5.  XML: z 330 um (FOVA 436-463), PMT 700 (other days 610-660).
- Candidates found are as clean as elsewhere: peak_snr median 15.85 (other 09/xx pnb99 days 15.62-16.85).
- Uncovered Cn peaks (3x3 mean, 19-px spacing, 20-px border cut, coverage A > 0.1 max): 11 at >= 0.70; 10 of
  them are 4-51 px regions at 0.71-0.75 (candidate core median 197 px), the 11th is a 5296-px region at the
  bottom-right corner 3 px from a candidate = the motion-border object.  Agrees with the 10-28 count (1 > 0.70
  with a >= 150-px region).  Nothing cell-sized above the seed is missed.
- Pilot reference (Taylor): 082626-pnb99 kept 29 of 81, 082826-pnb99 kept 49 of 198 (both FOVA).

## 2026-11-02 17:02: USER DECISION -- re-run 090526-pnb99 at min_corr 0.60 (session-level), then push again

- User: "Let's do a rerun of that session at 0.6. We can push it out again."
- `agent/config/param_overrides.json`: pnb99 gains a `sessions` entry for AVG5x-TSeries-090526-pnb99-330um-3z-000
  = min_corr 0.60, plus `_decision_2026-11-02`; animal level stays 0.70 (the other pnb99 sessions are FOVA).
  Validated: JSON parses, ASCII only, params.load_param_overrides() returns the session entry; pre-edit copy in
  the session scratchpad.  pnb105 unchanged (animal-level 0.60).
- Pre-run checks: no headless MATLAB running (the three MATLAB.exe processes are the VS Code MATLAB language
  server, R2025b); vCA1 watcher idle (polls only); no file in the session folder open by another process
  (exclusive-open probe, read-only) and none written since 10-30; the re-run reuses the converted movie .mat
  (10-28 precedent: "mapped to RAM" directly, no TIFF conversion).  Server, read-only: Taylor's outbox holds the
  0.70 bundle (8 items, staged 10-30 11:57), no return of this session in any inbox.
- 17:01 started: `rerun_queue.py --sessions AVG5x-TSeries-090526-pnb99-330um-3z-000 --archive
  .prev_mincorr070_2026-11-02 --threshold model --go` (dry run first; log agent/logs/rerun_cohort5_custom.log;
  the 0.70 outputs copied into `<session>/.prev_mincorr070_2026-11-02/`).  Curation at the deployed 0.01, as
  for the rest of the batch.  review_assigned.txt untouched (still Taylor, 10-30 11:57).
- Push plan (server write = operator, or a one-time user authorisation as on 10-18): rename the stale outbox
  folder aside to a dot folder, then `push_review_bundle.py vCA1\6odorDualDiffRew\<session> --assignee Taylor`.
  Renaming first matters: stage_one skips any file whose size matches, so a same-size stale file would survive
  an in-place push.  Tell Taylor the new bundle replaces the 30-candidate one.

## 2026-11-02 18:45: 090526-pnb99 0.60 re-run DONE -- 76 candidates, 58 to review; push is the operator's

- Run 17:01 -> 18:43 (101.5 min); final params {gSig 9, gSiz 36, min_corr 0.6, min_pnr 5.2, bd 10} ("Session
  override (pnb99): min_corr=0.6"); headless 76 candidates; curator at 0.01: 18 auto-rejected, 58 to review,
  2 merge pairs, 0 motion flags.  All outputs dated 18:42-18:43 (neuron.mat, review_neuron.mat, npz,
  footprints, Ybg_weights, PDF, summary, launcher); movie .mat untouched (10-28 13:16); no labels.mat/ROIs.jpg;
  review_assigned.txt untouched (Taylor, 10-30 11:57).  No errors in the run's log lines.
- Old vs new (support A > 0.3 max, Jaccard): all 30 of the 0.70 set recovered (median 0.98, min 0.76); the one
  old auto-reject now passes.  46 added: 18 auto-rejected, 28 to review; scores 1 >= 0.5 / 8 at 0.2-0.5 / 19 at
  0.01-0.2 / 18 < 0.01; core Cn of the added: 16 at 0.60-0.70, 15 at 0.70-0.75, 5 at 0.75-0.80, 10 >= 0.80;
  peak_snr median added 11.1 vs retained 16.1.  band_scores.py: new 76 = 0.60-0.70 16, 0.70-0.80 27, >= 0.80 33.
- Score shift of the retained 30 (>= 0.5: 11 -> 23) is the within-session rank columns: old features with the
  new 13 rank_ columns swapped in give 22/30 >= 0.5, with the new 22 absolute columns swapped in 11/30.  Same
  cells, ranked higher next to weaker neighbours -- not better cells.  Reassessment note: on a 0.60 set the
  scores (and the 0.01 cut) move with the set's composition.
- Cn.mat / pnr.mat: md5 differs from the archive but every value is identical (max abs diff 0) -- MAT header
  only.  So an in-place push is safe here: push_review_bundle skips same-size files, and the three it would skip
  (movie .mat, Cn, pnr) are content-identical; it copies review_neuron.mat, Ybg_weights.mat, PDF, summary and a
  fresh launcher.  Dry run (read-only): 8 items, 3688.8 MB listed (the movie is then skipped as already staged).
- OPERATOR: `push_review_bundle.py vCA1\6odorDualDiffRew\AVG5x-TSeries-090526-pnb99-330um-3z-000 --assignee
  Taylor` (in place; or the 10-18 route: rename the outbox folder aside to a dot folder first, then push --
  recopies the 3.5 GB movie, keeps the old bundle).  Tell Taylor the 58-neuron bundle replaces the 29-neuron
  one; if she already pulled the old one, she re-pulls.  An old-bundle return would be skipped at ingest
  (labels length 29 vs live 58).

## 2026-11-03 16:01: 090526-pnb99 58-neuron bundle PUSHED to Taylor (in place)

- User authorised Claude to run this one push ("You can push this session out").  Pre-checks: local set = the
  11-02 0.60 run (76 / 18 / 58); no return for this session in inbox/Taylor; outbox held the 10-28 29-neuron
  bundle; dry run 8 items, 3688.8 MB.
- push_review_bundle.py (in place, exit 0): skipped as already staged the movie .mat, Cn.mat, pnr.mat; copied
  review_neuron.mat, Ybg_weights.mat, review_report.pdf, review_summary.txt (142.6 MB); fresh run_final_review.m
  16:01:58; local review_assigned.txt now "2026-11-03 16:01, Taylor".
- Verified: md5 local == server for the 4 copied files; outbox Cn/pnr md5 == the archived 0.70 copies (values
  identical to the live 0.60 ones, see 11-02 18:45); server summary reads 76 / 18 / 58.  Server writes were
  copy2 + the launcher only; nothing deleted or renamed.
- Open: tell Taylor the 58-neuron bundle replaces the 29-neuron one (re-pull if she already copied it); the
  retro project's red_guide for this session is stale (30/29) and needs regenerating before its copy script runs.

## 2026-11-20 13:19-13:22: 13 of the 28 returned and INGESTED (operator); read-only audit afterwards

Ingest and retrain were run by the operator; this entry is the audit (scripts + outputs in
`analysis_2026-11-20/`, all read-only on D: and on the server).

- Back (13): pnb99 x4 (090526 at 0.60, 090626, 090826, 090926), pnb109 x5 (090526-090926), pnb102 x4
  (090526, 090626, 090726, 090926). Still with Taylor (15, 2757 candidates to review): pnb104 x4, pnb105 x6,
  pnb106 x4, 090826-pnb102.
- `ingest_survey.txt`: 13/13 labels.mat length == review set (N - auto-rejected), 13/13 byte-identical (md5)
  to the inbox copy, 13/13 `.pre_ingest/` snapshots (taken 13:19-13:21), provenance Taylor, no `.rereview`
  parking. `ingest_returns.py --dry-run` at 13:26: 137 inbox sessions, 0 new, 0 skipped, 0 parked.
  Totals: 2682 candidates, 553 auto-rejected at 0.01, 2129 reviewed, 711 kept, 1418 deleted, 26 motion tags.
  090526-pnb99 carries 58 labels = the 0.60 bundle (not the replaced 29-neuron one).
- `label_alignment.txt`: of 712 final footprints, 695 sit on a KEEP-labelled candidate (IoU >= 0.5), 12 on a
  DELETE-labelled one, 0 on an auto-rejected one, 5 on none; 26 merges logged over the 13 sessions. Labels
  are index-aligned. 090526-pnb102 has n_final 38 vs 37 kept (one final footprint matches no candidate, 4 merges).
- Auto-retrain 13:22:03-13:22:17 (vCA1 watcher): 205 sessions (80 agent + 125 bootstrap), all 13 present with
  the counts above, T 0.01 kept; agent-pool AUC 0.915 (reviewed), OOF false auto-reject at 0.01 = 1 of 2290,
  junk auto-rejected 19.4% of 7433. The 190-session model of 11-06 was overwritten without a snapshot.
- `quality_vs_curating_model.txt`: `classifier_cohort5_T001_2026-10-28.joblib` reproduces the stored
  auto_rejected set exactly on 28/28 sessions, so it is the curating model and is out of sample for these
  labels. Per-session AUC 0.80-0.92 (pooled 0.889; lowest 090526-pnb102 FOVC 0.804). Kept per session:
  pnb99 28 / 54 / 45 / 54, pnb102 37 / 65 / 46 / 48, pnb109 51 / 70 / 70 / 66 / 77 (pilots: 29, 49 / 44, 40 /
  49, 57). Deleted non-motion with score > 0.9: 35 in total, 9 of them in 090626-pnb99.
- Threshold, measured on the 13 with the curating model's scores (losses below 0.01 cannot be seen):
  T 0.02 would have lost 3 of 711 kept (0.42%) and removed 212 more deletes (review load 2129 -> 1914);
  T 0.03: 7 (0.98%), 338; T 0.04: 9 (1.27%), 436; T 0.05: 11 (1.55%), 505. Kept-rate by score band:
  0.01-0.02 3/215, 0.02-0.05 8/301, 0.05-0.10 16/219, 0.10-0.20 32/236, 0.20-0.50 93/348, 0.50-0.80 158/329,
  >= 0.80 401/481.
- 090526-pnb99 (dim FOVB, re-run at 0.60 on 11-02): 35 of the 76 candidates match one of the 30 old 0.70
  candidates (IoU >= 0.5; all 30 old matched), 22 of those kept; of the 41 added by 0.60, 18 were
  auto-rejected, 23 reviewed, 6 KEPT (scores 0.15, 0.20, 0.21, 0.23, 0.23, 0.72; IoU 0.0 with every old
  candidate). First labelled dim session: +6 kept cells for +23 reviewed. One session, one reviewer.
- Not measured here: kept-rate by ceiling band and the min_pnr +2 check on the new labels
  (`redteam/c8_c9_pilots.py`, `c10_minpnr.py` need their session lists widened); pnb105 at 0.60 (none back).
- Operations at 13:33: vCA1 watcher alive (PID 7572, valence python, restarted 13:21:51); BLA watcher log
  silent since 13:19:42 after its 13:18:41 start and no second watcher process visible; DG_AL silent since 10-05.
  (Update 15:40: the BLA watcher was started again at 13:36:21 and is logging.)

## 2026-11-20 15:29: the 6 pnb105 sessions INGESTED from inbox/Yawen (operator); new-reviewer audit, read-only

Ingest by the operator 15:29:04-15:29:43. Reviewer of record Yawen (first production reviews; the bundles had
been pushed to Taylor, `review_assigned.txt` still says Taylor = routing only). Scripts + outputs in
`analysis_2026-11-20/`: `ingest_survey_after_yawen.txt`, `label_alignment_after_yawen.txt`,
`reviewer_check_yawen.py` / `.txt`. Nothing written on D: or on the server.

- Sessions: 090526 (FOVC), 090626 (FOVB), 090726 / 090826 / 090926 / 091026 (FOVA), all seeded at 0.60.
  1772 candidates, 540 auto-rejected at 0.01, 1232 reviewed, 215 kept (10 / 25 / 40 / 46 / 51 / 43), 1017
  deleted, 17 motion tags. Still out: 9 (pnb104 x4, pnb106 x4, 090826-pnb102), all Taylor.
- Files: 6/6 labels length == review set, 6/6 labels.mat md5-identical to the inbox copy, 6/6 `.pre_ingest/`,
  review_meta n_review / n_keep / n_final agree with labels and footprints. `ingest_returns.py --dry-run`
  at 15:42 lists the six as unchanged.
- NOT retrained: `classifier.joblib` is still the 13:22:17 file (205 sessions). The watcher retrains only when
  a labels.mat is NEWER than the joblib (watcher.py `_needs_classifier_update`); these six are dated by their
  review (11-04 .. 11-19), so nothing fires. The vCA1 watcher was restarted 15:29:49 and logged no retrain.
  They will be swept in by the next retrain from any cause.
- Alignment: of 215 final footprints 199 sit on a KEEP candidate (IoU >= 0.5), 10 on a DELETE one, 1 on an
  auto-rejected one (091026 final #42: IoU 0.64 with auto-rejected candidate 98, 0.46 with KEEP candidate 223),
  5 on none. The 10 DELETE cases are second copies of a kept cell (the KEEP candidate has almost the same IoU:
  0.94 vs 0.97, 0.93 vs 0.93, ...). Index-aligned.
- Against the models (A): per-session AUC with the curating model 0.850-0.907 (pooled 0.878; Taylor's 13:
  0.889); with the pre-pilot model (161 sessions, no cohort-5 animal) 0.719-0.885 (pooled 0.851; Taylor's 13:
  0.868). Lowest: 090526 FOVC, 10 kept of 111.
- Keep rate at matched pre-pilot score (B): Yawen keeps far fewer than Taylor did on pnb99/102/109 (logistic
  offset -1.10, odds x 0.33, session-bootstrap CI -1.42 .. -0.83) but the same as Taylor on pnb105's two pilots
  (offset -0.01, CI -0.77 .. +0.49; bands 0.5-0.8: 32% vs 40%, >= 0.8: 61% vs 62%). An animal effect.
- Cause (F, G): duplicates. 282 of Yawen's 1017 deletes (27.7%) have support IoU >= 0.5 with a candidate kept
  in the same session = 1.31 per kept cell (Taylor's 13: 159 of 1418, 0.22 per kept; Taylor's pnb105 pilots
  33 of 377, 0.57). Of Yawen's 68 non-motion deletes scored >= 0.8 by the curating model, 61 are such copies,
  2 near, 5 unique (2.6% of the 192 high-score candidates; Taylor other animals 56 of 753 = 7.4%). With the
  duplicates out of the denominator the bands match Taylor's 13 (0.2-0.5: 49% vs 51%; 0.5-0.8: 68% vs 68%;
  >= 0.8: 96% vs 92%) and the offset is -0.20 (CI -0.63 .. +0.16); vs all 25 Taylor sessions +0.04.
- Same field, other day (C; registration by centroid voting + two cross-correlations, shifts composed over
  reliable edges, residual <= 1.8 px): cells Yawen kept on one FOVA day were kept again on another in 435 of
  536 cases (kept / (kept + deleted) 0.833; 12 directed pairs 1-3 days apart). Taylor's own figure on
  pnb99/102/104/106/109 at 1-5 days: 0.742 (784 of 1165). Across reviewers on pnb105 FOVA (11-13 days apart, 2
  session pairs with a clear vote peak): Taylor -> Yawen 0.778 (42 of 68 kept again), Yawen -> Taylor 0.712;
  Taylor -> Taylor at 9+ days on pnb109: 0.662. With all 8 composed pnb105 pairs: 0.718 / 0.581 vs 0.592.
  Different fields and seeds per row, so read it as "no worse than Taylor against Taylor", not as a ranking.
- Training record (D; `training/Yawen/<area>/<task>/<session>/training_results/progress.csv` on the server,
  never copied to `returns/`, so summarize_training.py does not see it): 9 attempts 10-08 .. 10-20, all coach
  mode, no exam mode. BLA 0.50 -> 0.71 -> 0.57 -> 0.82 -> 0.79 (one aborted attempt left out), errors mostly
  false keeps; vCA1 last three 0.93 / 0.90 / 0.88, false deletes 2/21, 2/22, 5/22, false keeps 4/62, 6/57,
  10/102 (first decision before feedback, contested items unscored).
- review_meta: 5 of 6 on ANALYSIS5-TRPRO with MATLAB 2017b (docs/REVIEW_SETUP.md asks for R2023b), 2.6-7.5 h
  each, 5-9 video passes; 091026 finished on C538T (MATLAB 2023b), resumed_from_stage 7, so its 50 min /
  3 passes cover only the part after the resume. Outputs complete on both.
- Open for the user: (1) retrain or not; the 282 duplicate deletes would enter as agent-weight (5.0) negatives
  (duplicates are zero-weighted for bootstrap sessions only); (2) these six are the pnb105-at-0.60 returns the
  seeding decision was waiting for; (3) `.quality/` table still from 10-12.

## 2026-11-20 16:38: vCA1 RETRAINED on 211 sessions (user request); the trainer now snapshots the old model

- Code: `agent/train_classifier.py` `_snapshot_previous_model()` runs just before the one `joblib.dump`. The
  old `classifier.joblib` is copied to `agent/model/{AREA}/history/classifier_<its mtime>.joblib` (mtime kept,
  copied under `.partial`, byte-compared, then renamed; write-once, a name collision with different bytes
  gets `_2`; nothing prunes). If the copy cannot be verified the trainer raises and does NOT save. Covers the
  watcher auto-retrain too (same script). `--dry-run` never reaches it. `curator.py` writes a model only
  when none exists, so this is the only overwrite path. Test `agent/test_model_snapshot.py`: 24 checks, all
  pass (temp dir). `test_stale_final_review.py` still passes. Skill `/retrain-area` text updated.
  UNCOMMITTED (train_classifier.py already carried uncommitted changes).
- Retrain, skill procedure: both watchers alive (BLA 16:37:40, vCA1 16:36:58), no labels.mat newer than the
  model so no race; dry run then real run 16:38:08-16:38:15 with
  `train_classifier_vCA1.py --prospective-only --model xgboost` (threshold from the wrapper constant).
  Before: 205 sessions, T 0.01, md5 f283b0a0...; snapshot `history/classifier_2026-11-20_132217.joblib` has
  that md5 and loads as the 205-session model. After: xgboost, feature_version 2, 35 features, 211 sessions
  (86 agent + 125 bootstrap), T 0.01, agent weight 5.0. Agent-pool OOF (78 testable sessions, 10259 reviewed
  candidates, 2505 kept): AUC reviewed 0.917 (was 0.915 on 2290 kept), false auto-reject at 0.01 0 of 2505
  (was 1 of 2290), junk auto-rejected 21.4% of 8990 (was 19.4% of 7433).

## 2026-11-20 16:40-16:50: the pending decisions on the 19 new labelled sessions (all read-only)

Outputs under `analysis_2026-11-20/`: `threshold/`, `duplicates/`, `seeding/` (fork), `red_somata/` (fork).

- THRESHOLD (`threshold/threshold_oof.py/.txt`; harness rt_lib, 8 seeds, threads 8, 5 folds; false
  auto-reject = kept cells scored below T / kept, mean / worst seed):
  * 12 threshold-0 pilots (410 kept, every candidate shown = unbiased), session held out, animal seen:
    0.01 0.00% / 0.00%; 0.02 0.21% / 0.49%; 0.03 0.40% / 0.73%; 0.04 0.58% / 0.73%; 0.05 0.88% / 1.46%.
    On 10-28 the same sessions gave 0.49% / 0.73% at 0.02 and 1.22% / 1.46% at 0.03 (3 seeds, 177 sessions).
    Animal held out: 0.01 0.06% / 0.24%; 0.02 0.34% / 0.49%; 0.03 0.58% / 0.98%; 0.04 1.10% / 1.46%.
  * All 46 agent sessions with no auto-reject (1389 kept, unbiased): session 0.02 0.24% / 0.36%, 0.03 0.44% /
    0.58%, 0.04 0.67% / 0.94%, 0.05 0.96% / 1.08%; animal 0.02 0.31% / 0.36%, 0.03 0.50% / 0.65%, 0.04 0.75% /
    0.94%, 0.05 1.05% / 1.30%.
  * 19 sessions curated at 0.01 (926 kept; LOWER BOUND, cells under 0.01 were never shown): session 0.02
    0.16% / 0.32%, 0.03 0.47% / 0.65%, 0.04 0.82% / 0.97%. Yawen's pnb105 alone: 0.02 0.06%, 0.03 0.29%.
  * Worst animal pnb109: pilots 0.71% at 0.02 and 1.06% at 0.03 (session), 1.30% and 2.24% (animal held out).
  * Standing rule (largest T with mean <= 0.85% and worst seed <= 1.0%, both groupings): whole pool 0.04,
    the 46 unbiased sessions 0.04, the 12 pilots 0.03 (animal grouping limits it).
  * Review load: junk auto-rejected on the 46 unbiased sessions 19% at 0.01, 28% at 0.02, 35% at 0.03, 39% at
    0.04; on the 19 already cut at 0.01, a further 9.6% / 16.2% / 21.6% of the reviewed deletes.
- DUPLICATES (`duplicates/duplicates_weight_test.py/.txt`; 33 agent sessions have all-candidate footprints,
  587 duplicate deletes = 13.1% of their reviewed deletes; pnb105 315 for 273 kept, pnb102 113 / 280):
  giving them training weight 0 instead of 5.0, same folds, 8 seeds: AUC on reviewed non-duplicate rows
  0.9239 -> 0.9265 (+0.0012 .. +0.0035, every seed); junk auto-rejected +1.0 to +1.3 points at 0.01-0.04
  (every seed positive); false auto-reject unchanged (within +-4 cells of 2505); median score of the
  duplicates 0.44 -> 0.76. Small, consistent, not built.
- SEEDING (`seeding/c8_c9_batch.txt`, `c10_minpnr_batch.txt`; ceiling = estimate from the 11-11 cache; the
  12 pilots reproduce the 11-11 figures):
  * 23 sessions run at 0.70: 0 of 1059 kept cells have a ceiling below the rule value or below 0.70.
  * pnb105 FOVA at 0.60 (4 sessions, 963 reviewed, 180 kept): 49 reviewed candidates with ceiling 0.60-0.70,
    2 kept; cells only the 0.60 seed could find 0 / 1 / 0 / 1. FOVB 090626: 0 of 16. FOVC 090526: 1 of 36.
    Duplicates per kept cell 1.31 (sessions at 0.70: 0.23-0.34; 082626-pnb105 FOVA at 0.70: 0.79); 216 of
    the 235 FOVA duplicates have ceiling >= 0.90, and where an archived 0.70 run exists 26 of 34 duplicate
    deletes have no partner in it.
  * Measured yield of 0.60 where both runs exist: 090526-pnb99 (pk50 0.643) +6 kept for +28 reviewed;
    090526-pnb105 (0.713) +1 for +74; 082726-pnb105 (0.759) +2 for +99.
  * min_pnr +2 on Taylor's 12 at 0.70: loses 7 of 683 kept (1.0%), removes 91 of 1379 reviewed deletes
    (6.6%) and 137 of 504 auto-rejected (pilots 11-11: 3 of 376, 222 of 1750). +1: 0 of 1059 kept, 99
    reviewed deletes. Dim 090526-pnb99: +2 loses 4 of 28 kept. Duplicates: 0 of 584 removed.
- RED SOMATA (`red_somata/red_soma_fate.py/.txt/_somata.csv`; rule of red_exempt_yield.py: candidate mask
  covers >= 30% of the 7-px disk at a non-bleed red centre; all 28 guides match the live sets):
  * 19 labelled, 380 red somata: 85 kept (22%), 106 deleted (28%), 32 auto-rejected only (8%), 157 no
    candidate (41%); 191 of 380 (50%) reach the reviewer (all 28: 264 of 501, 53%; 11-04 figure 50%).
  * CONTROL: the same rule at transposed positions gives 78 kept / 100 deleted / 19 auto / 183 none, at
    180-degree rotated positions 37 / 82 / 25 / 236. The disk rule barely separates a red soma from an
    arbitrary point of the field, so "50% reach the reviewer" is mostly field coverage. A candidate CENTRED
    within 6 px: 33 kept, 46 deleted, 16 auto, 285 none (transposed 24 / 29 / 11 / 316; rotated 6 / 26 / 8 / 340).
    Orientation of the red positions against the footprints not independently checked.
  * Exemption: 68 auto-rejected red-overlapping candidates on the 19 (+2.0% of 3361 reviewed) on 32 somata,
    curating score median 0.0033. Proxy yield: just above threshold (0.01-0.02) red candidates kept 0 of 31
    (rest 3 of 337); on the pilots 27 red somata had only sub-0.01 candidates and none was kept. About 1 kept
    cell for 68 extra reviews (estimate). Centred rule: 20 candidates on 16 somata.
  * At the same model score a red-overlapping candidate is not kept more often (log-odds -0.03, CI -0.34 ..
    +0.32); a red-centred one is (odds x 1.78), but below 0.05 only 1 of 19 was kept.
- What the labels support (user's decisions, none taken): threshold 0.02 passes the standing rule with margin
  on every unbiased basis and 0.03 passes at the cap; 0.70 confirmed for bright fields; pnb105's animal-level
  0.60 bought about 2 cells on its bright field for 1.3 duplicates per kept cell; min_pnr +2 costs 1.0% of
  kept for 6.6% of reviewed deletes; the red exemption yields about 1 cell; zero-weighting duplicates is a
  small free gain. Not decidable: the dim-session rule (needs the proposed labelled run), pnb104 / pnb106
  (9 sessions out), anything about dim 3odor sessions.

## 2026-11-20 (evening): the user's answers to the decision list -- STATED, nothing applied yet

- Threshold: "it seems that 0.02 should be the new threshold". NOT deployed: `_VALIDATED_THRESHOLD` in
  `train_classifier_vCA1.py` is still 0.01 and so is the 16:38 joblib. Applying = the wrapper constant and
  its comment, `eval/vca1_v2_2026-08/gate_decision.json` (verify_vca1.py checks it), one retrain.
- Seeding: 0.70 for bright sessions, 0.60 for dim ones "make sense"; the user expects automatic bright / dim
  detection to be finicky. `param_overrides.json` unchanged (pnb105 still animal-level 0.60).
  From the 11-11 table (`3odor_intake_2026-11/redteam/c6_sessions.tsv`, pk50 of the full-movie seeding
  image): of the 71 3odor sessions 11 have pk50 < 0.70 (7 of the 10 dim/hazy-noted + 4 pnb99 after
  cleaning), 12 lie in 0.70-0.80, 48 are >= 0.80. pk50 also falls with cell count (-0.084 for half a field).
- min_pnr +2: dropped. Red-soma exemption: not built.
- Duplicates in training: explained to the user, no decision.
- Still open: the labelled dim run of 11-11 (092926-pnb99, 092926-pnb102, 100126-pnb99 once at 0.575,
  reviewed at threshold 0), and what the 9 outstanding sessions add (pnb106 has only 21 kept cells in its
  pilots; 090826-pnb104 and 090826-pnb102 are the two motion sessions slated for a re-run).

## 2026-11-20 17:49: APPLIED on the user's instruction -- threshold 0.02, seeding overrides, duplicates at weight 0

User: "apply the 0.02 as a constant for future runs"; "writing the 11 into param_overrides.json is good and
put 105 back to 0.70"; "ignoring the duplicates is the right approach, it's what we do for bootstrap and we
should do it here too"; the 3odor questions go on a handoff; the user is working on the two motion sessions
and Taylor reviews those two last.

- Threshold: `train_classifier_vCA1.py` `_VALIDATED_THRESHOLD` 0.01 -> 0.02 (comment block rewritten with the
  numbers); `eval/vca1_v2_2026-08/gate_decision.json` `chosen_T` 0.02 + block `user_decision_T_2026-11`.
- Seeding: `agent/config/param_overrides.json`: pnb105 0.60 -> 0.70 (animal level); session-level 0.60 for
  the 11 3odor sessions with pk50 < 0.70 (pnb99 x7, pnb109 x2, pnb102 x1, pnb105 x1). Checked through
  params.py: of the 71 3odor sessions 11 resolve to 0.60 and 60 to 0.70; every entry matches a tif in
  `D:\2p data\3odor`. No session is re-run; entries are inert until staged.
- Duplicates: `train_classifier.py` `_agent_duplicate_mask()` (reviewed delete, not motion-tagged, support IoU
  >= 0.5 with a kept candidate of the session; footprints from `.pre_ingest/spatial_footprints.mat`) gives
  those rows weight 0 in the fit and in the held-out evaluation, and leaves them out of the headline AUC and
  the junk count. Cache `<session>/review_duplicates.json` (keyed on labels.mat and the snapshot; not written
  on --dry-run). Applies to every area through the shared trainer; BLA and DG_AL have no reviewed session
  with `.pre_ingest` footprints today (0 of 91, 0 of 17), so it starts there with their next returns and is
  NOT measured on them. Sessions ingested before 2026-10-12 cannot be flagged (53 of the 86 vCA1 ones).
  The harness (rt_lib) does not know about it. Test `agent/test_agent_duplicates.py`: 19 checks pass;
  `test_model_snapshot.py` 24 pass; `test_stale_final_review.py` 0 failures.
- Retrain (skill procedure; watchers alive, no labels newer than the model; dry run 17:46, real run
  17:48:19-17:49:37): 211 sessions (86 + 125), xgboost, 35 features, T 0.02, agent weight 5.0, 587 duplicate
  deletes at weight 0 (33 of 86 agent sessions checked), 6739 ambiguous bootstrap negatives. Agent-pool OOF,
  one seed: AUC reviewed 0.927 on 9672 candidates / 2505 kept with the duplicates left out (0.910 with them
  counted as deletes, against 0.917 at 16:38 when they were also trained as deletes); at 0.02 false
  auto-reject 4 of 2505 (0.16%), junk auto-rejected 33.9% of 8403. Previous model kept as
  `history/classifier_2026-11-20_163815.joblib` (md5 7272378...). 33 cache files written.
- `eval/vca1_v2_2026-08/verify_vca1.py`: the threshold checks pass (joblib 0.02, wrapper 0.02, gate 0.02);
  it then fails on its pinned 134-session August manifest and stops in "corpus identity" on a session renamed
  since August. Both are stale parts of that script, not touched.
- Handoff written: `docs/HANDOFF_2026-11-20_3ODOR_STAGING.md` (FOV and redo table for the 11 dim sessions
  from `D:\Analysis\3odor\2p imaging parameters 4.0 (retro) - 3odor.csv`, export of 11-20 06:00): 3 of the 11
  have a redo after the lens cleaning (092926-pnb105, 092926-pnb109, 093026-pnb109); 8 remain (pnb99 x7 over
  FOVA/B/C/D, 092926-pnb102 FOVA); the three sessions proposed for the labelled dim run are three different
  fields with no redo. Nothing staged.
- All uncommitted.

## 2026-11-21 11:20-11:53: 3odor -- handoff re-checked, USER DECISIONS, labelled dim run STAGED (3 sessions at 0.575)

Fresh session working from `docs/HANDOFF_2026-11-20_3ODOR_STAGING.md`.

Re-check at 11:20, read-only (script kept only in the session scratchpad):
- 71 tifs in `D:\2p data\3odor` (230.9 GiB), none present under `D:\Julian_CNMFe\vCA1`. All 71 have a row in
  `redteam/c6_sessions.tsv` and in `fov/best.tsv`. pk50 < 0.70: 11; 0.70-0.80: 12; >= 0.80: 48.
- `param_overrides.json` through `params.py`: 12 session-level keys, 11 match a 3odor tif name exactly (the
  12th is the 6odor 090526-pnb99); 11 resolved to 0.60 and 60 to 0.70, and "pk50 < 0.70" == "seed 0.60" on
  all 71.
- vCA1 joblib: `n_sessions` 211, `reject_threshold` 0.02, `feature_version` 2, `n_features` 35.
- Both watchers alive and idle (log lines at 11:20; vCA1 = PID 6648 started 11-20 15:29:49, after the last
  edit of params.py on 10-05, so it holds the current override code, and the file is read per call). No
  loose tif anywhere under vCA1. Commit charge 45.8 of 431.2 GB.
- Sheet `D:\Analysis\3odor\2p imaging parameters 4.0 (retro) - 3odor.csv` was re-exported 11-21 06:00; the
  three redo notes read as in the handoff (10/2 pnb105 "cleaned lens with acetone and redoing from FOVA
  (0929)"; 10/1 pnb109 "We cleaned the lens! Acetone did the trick. new real FOVA"; 10/2 pnb109 "FOVB.
  redoing because past was really bad witthout cleaning").

Two things the handoff did not say, both told to the user before he decided:
- The redos repeat the behavioural day, not the field. Depths 430 -> 373 um (pnb105), 133 -> 440 and 90 ->
  310 um (pnb109); 11-11 registration (`fov/pairs.tsv`, same-field cut 0.354) of each original against its
  redo: 0.274, 0.123, 0.107. The two damaged-lens pnb109 sessions match each other (0.392) and no later
  session (<= 0.19). 092926-pnb105 scores 0.474 against 100926-pnb105 (FOVA) at the same offset as the other
  FOVA sessions, so it is probably a shifted hazy view of FOVA. The 9/29 -> 10/1 pairing for pnb109 rests on
  the FOV letter; only the 10/2 and 10/3 notes say "redoing".
- Read by FOV letter, "revisited after" also fits 092926-pnb102 (FOVA, pk50 0.648): imaged again on 10/9
  (pk50 0.901, registration 0.691) and 10/15 (0.847), but at later stages (rev d4, WSR d2).

USER DECISIONS 2026-11-21 (question dialog of the session):
1. Not run: 092926-pnb105-430um, 092926-pnb109-133um-001, 093026-pnb109-90um. 68 of the 71 are run.
   092926-pnb102 IS run (he chose the three redone days only).
2. Labelled dim run approved on the three proposed sessions: 092926-pnb99-518um (FOVA), 092926-pnb102-460um
   (FOVA), 100126-pnb99-493um (FOVC); one run each at min_corr 0.575, every candidate reviewed.
3. The dim sessions outside the labelled run (pnb99 x6, 0.60) are reviewed at threshold 0.
4. Stage the three labelled-run sessions now and nothing else; the bright sessions wait for his order.
Not asked, still open: order of the remaining 65 (59 at 0.70 + 6 at 0.60), who reviews which FOV.

Done:
- `agent/config/param_overrides.json`: session-level 0.575 for the three (092926-pnb99 was at the animal-level
  0.70, the other two at 0.60); block `_decision_2026-11-21`. The three not-run sessions keep their inert
  0.60 entries. After the edit: 3 at 0.575, 9 at 0.60 (6 to run), 59 at 0.70.
- New `agent/eval/cohort5_2026-09/stage_3odor_2026-11-21.py`: local copy-only stager, explicit session names,
  `.tif.part` then rename after ALL copies pass size + SHA-256, refuses the three not-run sessions, anything
  already present under the area, a leftover `.part`, and (with `--expect-min-corr`) a session whose override
  is not the expected seed. Dry run by default. Scratch-directory test: 11 checks pass (first run had one
  failure caused by the test harness passing `--dest` twice; fixed in the test, not the script).
- Dry run on the real root 11:50 (3 sessions, 10.04 GB, each "min_corr 0.575 (session level)"), then `--go`
  11:51-11:53: 3 tifs copied and SHA-256 verified in 1.8 min; log `agent/logs/stage_3odor_2026-11-21.log`.
  Sources unchanged (71 tifs).

## 2026-11-21 11:55 onward: labelled dim run -- watcher runs and threshold-0 re-curation

All numbers below are read from each session's `agent_run.log`, `candidate_features.npz` and
`review_summary.txt` (measured), unless marked otherwise.

| session | started | Final parameters | headless | N candidates | auto-rejected at 0.02 | re-curated at 0 |
|---|---|---|---|---|---|---|
| 092926-pnb102-460um (FOVA) | 11:55:43 | gSig 9, gSiz 36, min_corr 0.575, min_pnr 5.4, bd 10 | 68.0 min, ready 13:06:18 | 164 | 68 (review set 96) | 13:08, review set 164 |
| 092926-pnb99-518um (FOVA) | 13:06:18 | gSig 9, gSiz 36, min_corr 0.575, min_pnr 5.8, bd 17 | 67.9 min, ready 14:16:36 | 183 | 67 (review set 116) | 15:28, review set 183 |
| 100126-pnb99-493um (FOVC) | 14:16:36 | gSig 9, gSiz 36, min_corr 0.575, min_pnr 5.4, bd 17 | 67.5 min, ready 15:26:28 | 104 | 37 (review set 67) | 15:28, review set 104 |

Total to review at threshold 0: 451 candidates (164 + 183 + 104); the deployed 0.02 would have auto-rejected
172 of them (68 + 67 + 37). Classifier score bands (< 0.02 / 0.02-0.5 / >= 0.5): pnb102 68 / 65 / 31,
092926-pnb99 67 / 78 / 38, 100126-pnb99 37 / 44 / 23. Scores are the model's, not labels.

- Each log shows `Session override (...): min_corr=0.575 (was 0.7)`: the estimate from the pre-pass images
  was 0.70, the animal override 0.70, the session override 0.575.
- 092926-pnb102: no error or timeout line in the log; `Ybg_weights.mat` was cached by the headless run.
- Before each re-curation the watcher's curation is kept in
  `agent/eval/cohort5_2026-09/dimrun_2026-11-21/<session>_curated_by_watcher.npz` (new script
  `snapshot_pre_recurate.py` there: original arrays + scores + threshold + model mtime, write-once, nothing
  written in the session folder; tried against a scratch output folder first). Reason: recurate empties
  `auto_rejected`, and with the threshold-0 labels that set gives the false auto-reject rate of the deployed
  0.02 on dim fields, which has not been measured. For 092926-pnb102 the snapshot check passed: the 68
  indices in the npz are exactly the candidates scoring < 0.02 with the model of 11-20 17:49:37.
- 092926-pnb102 re-curated 13:08:07-13:08:48 with
  `recurate_sessions_vCA1.py <session path> --threshold 0` while the watcher ran the second session (one
  short MATLAB call, 0.2 min). Verified from the files: `auto_rejected` empty, `n_candidates` 164,
  `review_neuron.mat` rewritten 13:08 (MATLAB reported 164 neurons), summary "Awaiting your review: 164".
  Merge pairs 11 -> 25 and motion flags 0 -> 3, because both lists are computed over the review set, which
  now includes the 68 low-scoring candidates. Log `agent/logs/recurate_3odor_dimrun_2026-11-21_pnb102.log`.
- The two pnb99 sessions: snapshot 15:27 (both checks passed: 67 and 37 indices == scores < 0.02), then
  re-curated together 15:27:30-15:28:53. Verified from the files for all three at 15:29: `auto_rejected`
  empty, review set == `n_candidates`, feature matrix identical to the snapshot, no error / timeout / failed
  line in any `agent_run.log`. Merge pairs after re-curation: 25, 27, 4. Log
  `agent/logs/recurate_3odor_dimrun_2026-11-21_pnb99.log`.
- Bundle inputs present in all three folders (review_neuron.mat, report, summary, run_final_review.m, Cn,
  pnr, Ybg_weights, Ybg_mean, movie .mat, tif). `REVIEW_QUEUE.md` lists them Ready with 164 / 183 / 104.
- No `labels.mat`, no `review_assigned.txt`: nothing pushed. The push is the operator's
  (`/push-review-bundle`); the assignee is not decided.
- 16:00 USER: "push the three to Taylor, dry run first". Whole-FOV consequence: Taylor now owns 3odor
  pnb99 FOVA, pnb99 FOVC and pnb102 FOVA. Dry run 16:00:49-16:00:55 (log
  `agent/logs/push_3odor_dimrun_2026-11-21_dryrun.log`), one explicit session per call, exit 0 each: 8 items
  per session (movie .mat, review_neuron, Cn, pnr, Ybg_weights, report, summary + generated launcher), 2880.5
  / 2743.0 / 2841.7 MB = 8465.2 MB to `outbox\Taylor\vCA1\3odor\`; no warning; red guide "would be built at
  push (no red_guide.png)" for all three. Before it: no marker, labels or ROIs.jpg in any of the three;
  none of them in Taylor's outbox (read-only check). Real push handed to the user as operator-run (server
  write); not run by Claude.
- PUSHED by the user 2026-11-21 20:55-20:57 (markers 20:56 / 20:56 / 20:57, to Taylor, by
  push_review_bundle.py). Verified read-only 11-22 00:51: each outbox folder holds 9 files (the 8 of the dry
  run + red_guide.png); every non-movie file is SHA-256 identical to the local copy; movies equal in size
  and mtime, and byte-identical in a full comparison (2704.6 / 2559.9 / 2689.8 MB in 14 / 11 / 11 s; that
  speed is above what D: and the network deliver, so both sides were probably served from this machine's
  file cache, not re-read from the NAS); `REVIEW_QUEUE.md` lists all three out to Taylor.
- Red guides built by the hook at push time (RESULT 0 each, `guide_for_push.log` in each guide folder), each
  matching the live review set (164 / 183 / 104 candidates, nothing auto-rejected) and identical to the
  outbox copy. Red somata on the registered mean: 092926-pnb102 1 (3 usable single images; the seven 497um
  images are another animal's view, flagged stray by the hook), 092926-pnb99 16 (1 usable single image),
  100126-pnb99 21 (3 usable). The pnb102 guide is close to empty.
- Red guide (read-only check, hook NOT run): the retro project has results only for the 6odor days of pnb99
  and pnb102, none for a 3odor day. `guide-for-push` builds from scratch (fetch + inventory + draw), and
  `X:\Jeremy\2p data\bla-pnb 3odor\092926` and `\100126` hold SingleImage folders for all three sessions
  (pnb102-460um x4, pnb99-518um x2, pnb99-493um x4). Whether a guide comes out is not verified.
- Not this session's job, seen while checking the machine: a motion-correction re-run
  (`agent/logs/rerun_cohort5_mc_new_movie.log`, GO 15:07:55, 090826-pnb104 then 090826-pnb102 in
  6odorDualDiffRew) was started elsewhere and overlapped the third headless run from about 15:10 to 15:26
  (two heavy MATLAB jobs; commit charge 87.4 of 431.2 GB at 15:27). The vCA1 watcher is idle since 15:26:28.

## 2026-11-22: 3odor -- two pnb102 movies re-registered; USER DECISION: drop 100826-pnb106

- 093026-pnb102-600um and 100926-pnb102-452um: re-made from raw with the installed motion-correction rule
  and swapped into `D:\2p data\3odor` (old tifs in `.prev_maxregshift01_2026-11-22\` there); bad frames
  5.23% -> 0.01% and 7.71% -> 0.09%. Record: `docs/HANDOFF_2026-11-20_MOTION_CORRECTION.md`. Not staged.
- 100826-pnb106-258um DROPPED (user, 2026-11-22): the raw-sample test showed the field drifting through the
  whole session (bad from cycle ~77, 100% bad there under both registrations), not fixable by registration.
  Added to `NOT_RUN` in `stage_3odor_2026-11-21.py`. With the three redone days (092926-pnb105-430um,
  092926-pnb109-133um-001, 093026-pnb109-90um; dropped 11-21) 4 of 71 are not run; 64 remain to stage
  (58 at 0.70, 6 at 0.60). Its tif stays in `D:\2p data\3odor`; nothing moved or deleted.
