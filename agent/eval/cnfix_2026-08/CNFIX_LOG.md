# Retro `cn_correlation` orientation fix + BLA refresh — 2026-08-26

Follow-up (a) of the bootstrap red team (`agent/eval/bootstrap_redteam_2026-08/redteam_report.md`,
attack #4). All numbers below computed this session; commands in `cnfix_refresh.py`.

## The defect

`train_classifier._retro_label_session` reshaped MATLAB footprint columns row-major
(`A_review.T.reshape(N_review, d1, d2)`), so every footprint image it built was the
transpose of the real one. The 12 shape/trace features are transpose-invariant;
`cn_correlation` (footprint weights vs the Cn image) is not. `features.load_spatial`'s
A.txt fallback had the same reshape plus a square-frame assumption (`side = sqrt(n_pixels)`),
firing on 0 sessions.

## Code fix (`features.fcols_to_images`, regression-tested)

- `agent/features.py`: new `fcols_to_images(A, d1, d2)` (column-major reshape, sparse
  input accepted, pixel-count guard); `load_spatial`'s A.txt fallback now uses it with the
  frame dimensions from `Cn.mat` and refuses to guess a square frame when `Cn.mat` is absent.
- `agent/train_classifier.py:235`: the retro path uses `feat_module.fcols_to_images`.
- `agent/test_orientation.py`: 4/4 pass — rectangular and square synthetic round-trips
  (the old reshape provably transposes), the fallback on a temp non-square session, and a
  real prospective BLA session whose stored `cn_correlation` column is reproduced to < 1e-9
  from the extraction A through the fixed helper.

## Identification (numeric, not by provenance)

`cnfix_refresh.py identify` recomputes column 12 for all 79 BLA agent sessions from the
`.feature_expansion` extraction A under both reshapes and matches against the stored column:
73 sessions match the order-F recomputation, **6 match the order-C recomputation bit-for-bit
(|stored − C| = 0.0)** and none matches neither — the same six the red team found:
`2tones/093025-bla12-660um`, `2tones/093025-bla8-765um`, `4odorDO/02092026-bla12-681um`,
`4odorDO/02092026-bla16-314um`, `4odorDO/02092026-bla8-745um`, `Valence/121225-bla12-652um`
(705 rows / 217 reals, all in the CV pool). vCA1 has none (the red team's `prep_retro_ids.py`).

## Refresh

- `build`: corrected column 12 (order-F images × session `Cn.mat`, via `features.cn_features`)
  and the 13 rank columns recomputed with `features.compute_ranks`; columns 0–11 and 26–34,
  `auto_rejected`, `n_candidates`, `feature_names` byte-identical. Per session the median
  `cn_correlation` moved from ≈0 (0.026 / 0.014 / −0.031 / −0.015 / −0.047 / −0.003) to
  0.554 / 0.569 / 0.640 / 0.473 / 0.685 / 0.653; 88–100% of rows moved by > 0.1. Files under
  `D:\Julian_CNMFe\BLA\.feature_expansion\_cnfix\` (`cnfix_report.json`, sha256 per file).
- `backup`: the 6 live npz + the live joblib into `_cnfix_backup\` (sha-verified manifest).
  BLA's retired `_v1_backup\` was not touched.
- `gate` (8 seeds, StratifiedGroupKFold(5), deployed sqrt/4.0 recipe, the red team's
  independent evaluator, unrestricted xgboost threads; `cnfix_gate.json`):

  | corpus | AUC full | AUC reviewed | FAR@0.04 mean (max) | junk@0.04 | FAR@0.03 (max) | junk@0.03 | AUC on the 6 | rule T |
  |---|---|---|---|---|---|---|---|---|
  | live (transposed col 12) | 0.9283 | 0.9149 | 0.64% (0.96) | 43.2% | 0.46% (0.58) | 38.7% | 0.9103 | 0.04 |
  | corrected | 0.9291 | 0.9159 | 0.73% (1.09) | 43.1% | 0.43% (0.63) | 38.7% | 0.9199 | 0.03 |

  Paired: full +0.0007 (7/8 seeds), reviewed +0.0010, **on the six sessions +0.0096 (8/8)**;
  junk at matched false-AR 43.2% → 41.9%. Reproduces the red team's attack #4.
- **Threshold: the user kept T = 0.04** (0.73% / 1.09% worst seed, 43.1% junk) over the
  rule's literal 0.03 (0.43% / 0.63%, 38.7%): the 0.09-pp breach is one cell in one seed and
  the same breach already existed under animal-level grouping.
- `preflight` PASS (BLA watcher not running, log quiet 8,214 min, live files unchanged
  since build, built files match the report, backup covers the six, joblib unchanged) →
  `swap --freeze`: **6 sessions refreshed** (`cnfix_swap_report.json`).
- Retrain `train_classifier.py --prospective-only --model xgboost --threshold 0.04`
  (`cnfix_retrain.log`): **170 sessions (79 agent / 91 bootstrap), agent weight 4.00x,
  4,494 masked, CV AUC 0.942, companion 13-col model, reject_threshold 0.04**, joblib
  996,311 B (was 994,267 B).
- `verify --threshold 0.04` — **ALL PASS** (`cnfix_verify.log`): joblib contract (35 wide,
  companion at 13, T 0.04, feature_version 2, 170 sessions, 4,494 masked); live files == built
  6/6 with columns 0–11 + 26–34 + `auto_rejected` preserved and column 12 changed 6/6; bla21
  smoke cells at 0.599 / 0.574 in-sample vs T 0.04; joblib newer than every labels.mat.

## Rollback

`cnfix_refresh.py rollback` restores the six files and the pre-refresh joblib byte-exact
(refuses if any of the six labels.mat is newer than the backup). The code fix is not
rolled back by it — there is no reason to restore the transposed behaviour.

**Operator: restart `watcher.py` (BLA) and watch three polls; the first must not retrain.**
