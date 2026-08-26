# Bootstrap label transfer: the pixel-order bug, the fix, and the corpus re-run (2026-08)

Companion material: `agent/eval/bootstrap_matching_2026-08/` (AUDIT.md, REPORT.md,
GATE_B_REPORT.md, scripts, per-run JSON), memory `project_bootstrap_transpose_bug`.

## 1. What was wrong

Bootstrap sessions (pre-agent recordings with a human-curated `neuron.mat`) get training
labels by re-running headless CNMFe and matching the new candidates to the curated
neurons by spatial-footprint cosine similarity (Hungarian 1:1, threshold 0.45).

`bootstrap_preagent._load_candidates` flattened the candidate footprint stack
`(N, d1, d2)` in numpy **C-order** (pixel = row·d2 + col), while `A_final` from MATLAB
`full(neuron.A)` carries **column-major** pixel rows (pixel = row + col·d1). The cosine
therefore compared every candidate against the **transposed image** of each curated
footprint. Matches landed at the mirror position across the image diagonal.

Consequences (measured):
- Corpus recovery looked like 0.55 mean (BLA 59.0%, vCA1 49.9%) and was attributed to
  CNMFe merging ("41–49% fundamentally unrecoverable", March 2026). Both artifacts.
- ~94% of the 4,205 stored bootstrap positives were the wrong cell (mirror-position
  candidates); the true candidates of ~7,900 curated neurons sat in the negative pool.
- The 9 legacy pre-agent sessions' near-chance within-session AUC, the catastrophic
  vCA1 CTA 921/924 cluster, the flat no-valley similarity histogram, and the entire
  weighting scheme (4.0x agent floor, 0.4x bad-session, ambiguous mask) were all
  downstream of this single indexing mismatch.

Proof chain: synthetic asymmetric footprint scores 0.0000 against itself under the
production formula and 1.000000 under a consistent one; the mismatched replica
reproduces stored production scores to the 3rd decimal (bla21: 37/50, top-5
0.944/0.924/0.881/0.865/0.802); consistent ordering recovers 50/50 at median 0.971 with
matches 1.0 px from true position; matched candidates sit 6.8 px from the *mirror* of
the curated centroid vs 130 px from the true one; corpus-wide, no session's best pair
ever reached 0.99 (median 0.852) — impossible for same-movie re-runs with a correct
metric. Regression test: `a1_orientation_audit.py`.

Not affected: retro labeling of agent sessions (`train_classifier.py` retro path uses
MATLAB order on both sides) — which is why the deployed models worked at all.
Secondary bugs found by the same audit: retro feature path transposes footprint images
(`train_classifier.py` `_retro_label_session`) → `cn_correlation` corrupted for
retro-labeled sessions; `features.load_spatial` A.txt fallback has the same C/F mixup
(fires on 0 sessions). Both still open (see §6).

## 2. The fix (commit 4cf53e1)

- `_reorder_Fcols_to_C(A, d1, d2)` reindexes `A_final` onto the candidates' pixel order
  before the cosine (pixel-count guard included). Same fix in `validate_threshold.py`,
  whose `--rematch-only` also no longer deletes the cached `_bootstrap_validate/` dir.
- `bootstrap_match_stats.json` **schema_version 2**: legacy keys unchanged; adds
  `ambiguous_candidate_indices` (Hungarian partner of each unmatched curated neuron),
  `duplicate_candidate_indices` (unassigned candidates above threshold to a matched
  curated neuron — same-cell re-detections), `per_curated_best_similarity`,
  `recovery_by_threshold`, CNMFe params, dims, timestamp, matcher tag.
- `--keep-candidates` persists `bootstrap_candidates.npz` (sparse footprints, traces,
  full similarity matrix) so matching is redoable offline forever after.
- `--sessions-file` for explicit re-runs; diag mode (`out_dir`) keeps session dirs
  strictly read-only; A.txt fallback for MATLAB's >2 GB v7 save stub.
- Trainer (commit de42f68): `_get_bootstrap_ambiguous_mask` masks ambiguous + duplicate
  rows (weight 0) from v2 JSONs; legacy fallback unchanged.

## 3. Validation before rollout (Gate B, 2026-08-21)

Pilot of 12 sessions + 3 permissive-parameter variants, outputs to a dot-prefixed shadow
tree, session dirs verified untouched by before/after snapshots: recovery **43.3% →
98.8%** (348 → 794 of 804 curated), flat across thresholds 0.45–0.60, every catastrophic
session normalized (961-420: 0/17 → 14/17; 921-880-A: 5/69 → 69/69; 962 Valence-B:
282/282). Fresh end-to-end run on bla21 vs human labels: 46 verified-kept / 1 deleted /
3 unknown. Hungarian == argmax for every curated neuron, zero greedy sharing: no merging
problem exists. Permissive min_corr=0.30 recovered +2 neurons on one session only at
2–5x the candidates — not adopted globally.

## 4. Corpus re-run (2026-08-21 → 25)

All 202 sessions (111 vCA1 → 91 BLA), `--keep-candidates`, default params, up to two
parallel MATLAB workers (255 GB box; one BLA run ≈ 78 GB main + pool), **zero failures**.

| Area | sessions | curated | matched before | matched after | sessions < 0.40 recovery | masked (ambig + dup) |
|---|---|---|---|---|---|---|
| vCA1 | 111 | 5,187 | 2,587 (49.9%) | **5,148 (99.2%)** | 29 → 0 | 39 + 6,623 |
| BLA | 91 | 2,741 | 1,618 (59.0%) | **2,737 (99.9%)** | 19 → 0 | 4 + 4,490 |

## 5. Model gates and deployments

**vCA1 (deployed 2026-08-24, commit b553d31)** — same agent test folds:
bootstrap-contribution delta (agent-only → +bootstrap) **−0.014 (historical) → +0.009**;
3-seed fixed-weight sweep showed the dynamic `sqrt` agent weight (7.01x here) doubles
false-AR at the deployed 0.05 vs a fixed 5.0 (2.5% → 1.2%, AUC 0.884±0.007 vs
0.886±0.004) → new area-scoped `AGENT_WEIGHT_OVERRIDE = 5.0` in `config_vCA1.py`;
threshold stays 0.05; expected posture ≈1.2% false-AR / ~44% junk auto-caught.
Comparability caveat: today's vCA1 agent pool is the pnb/tdTomato prep (14 usable test
sessions), not the 2026-06 pool behind the historical 0.881 — only same-pool A/Bs are
quoted.

**BLA** — see §5b (filled from `c3_bla_gate8.log` / `c3_bla_eval.log`).

### 5b. BLA gates (8-seed, paired vs pinned post-Step-4 baseline)

Harness: `c3_bla_gate8.py` = Step 5's `threshold_sweep_v2.py` on the live files
(identical seeds, CV, deployed weight recipe = 4.0x floor binding, model factory);
identical OOF pool (12,677 rows, 2,394 real, 8,590 reviewed junk) so deltas are paired.

| gate | pinned (post-Step-4) | after bootstrap fix | verdict |
|---|---|---|---|
| G1 AUC full pool | 0.9246 +/- 0.0014 | **0.9283 +/- 0.0014** (+0.0037, all 8 seeds >= +0.0031) | PASS |
| G1 AUC reviewed pool | 0.9123 +/- 0.0015 | **0.9149 +/- 0.0016** (+0.0026, all seeds >= +0.0021) | PASS |
| G2 at rule-chosen T | T=0.06: 0.80% FAR (max 0.96), 33.8% junk full, 31.8% reviewed | **T=0.04: 0.64% FAR (max 0.96), 43.2% junk full, 34.5% reviewed** | PASS (Pareto-better) |
| G2 at fixed T=0.06 | 0.80% / 33.8% | 1.11% (max 1.75%) / 49.7% | score shift -> threshold re-derived |
| G3 bootstrap delta (agent-only -> +bootstrap, 5-fold) | -0.014 (2026-03, 13-col) | **0.000** (0.929 vs 0.929) | neutral (was harmful) |
| G4 weight sweep (3 seeds, T=0.06) | floor 4.0 | AUC flat 0.926-0.9285 across w=1..7; FAR 0.6% (w=1) -> 1.4% (w=7) | keep 4.0 floor |
| smoke cells bla21 N22/N25 | b13 OOF 0.064/0.101, auto-rejected at 0.06 | rankv2b_35 OOF 0.046/0.048 | survive at T=0.04 |

G5 (per-animal / early-era LOAO) was not re-run: the red-team harness pins the pre-swap
parallel files; the paired all-seeds-positive G1 is the accepted substitute.

Pre-deploy assumption audit (`c3_assumption_checks.py`, 3 seeds): junk-caught at
matched false-AR is 43-46% for every agent weight 1-7 (weight = score rescaling for
BLA; keep 4.0); duplicates masked ~= labeled-0 (43.2% vs 46.1% junk, within noise),
labeled-1 clearly worse (-0.007 AUC); XGBoost still wins the 3-way CV (0.942 vs
LightGBM 0.941 vs LR 0.919); T=0.045 already fails the worst-seed rule at 8 seeds.

**DEPLOYED 2026-08-26**: retrained on the clean corpus, unchanged 4.0x recipe,
T = 0.04 (module default 0.12 -> 0.04 so watcher auto-retrains preserve it).
Joblib verified: xgboost, T=0.04, agent_weight 4.0, 170 sessions, 35 features,
4,494 masked rows, companion first-pass model present.

## 5c. Global (cross-area) model re-evaluation (2026-08-26, `c5_global_model.py`)

Shared first-13 columns (bit-identical across contracts); target agent sessions as test
folds, target's own bootstrap + deployed weighting in both conditions; Cond B appends all
other areas' rows at a neutral weight. 3 seeds, paired.

| target | own-only AUC | + other areas (w=1.0) | + other areas (w=0.3) | verdict |
|---|---|---|---|---|
| vCA1 (14 test sessions) | 0.8823 | 0.8800 (-0.0023) | 0.8786 (-0.0038) | no benefit; pooled scores also miscalibrate vCA1's T=0.05 (junk 41% -> 30%) |
| BLA (75 test sessions, 13-col) | 0.9164 | 0.9154 (-0.0011, all seeds <0) | 0.9167 (+0.0003) | null; deployed 35-col model is 0.928 anyway |
| DG_AL (9 sessions, 955 rows) | 0.8719 | 0.8828 (+0.0110, min +0.0082) | **0.8863 (+0.0145, min +0.0123)** | **helps** the data-starved area, every seed |

Reading: with clean labels the 2026-06 verdict holds for the mature areas (pooling
adds nothing; each has enough in-distribution data), and the combined-model heuristic
holds for the new area — BLA+vCA1 rows at ~0.3 weight are a useful prior for DG_AL
(+0.011..0.015 AUC). Caveat: a pooled model's score scale differs (junk-caught at a
fixed T collapses), so any DG threshold must be re-derived on the pooled model; DG
currently runs THRESHOLD_OVERRIDE=0 (no auto-reject), so ranking is what matters there.
Not implemented — proposal for the DG trainer.

## 6. Open follow-ups

- Retro `cn_correlation` transpose (train_classifier retro feature path) and the
  `features.load_spatial` A.txt fallback: fix as one gated change; needs a feature
  refresh for retro-labeled sessions.
- `diagnose_model.py` / `sweep_weights.py` replicate the pre-override weight formula;
  update to honor `AGENT_WEIGHT_OVERRIDE` so their absolute numbers match the trainer.
- DG_AL pooled prior (see 5c): append BLA+vCA1 rows at ~0.3 weight in the DG trainer;
  re-derive DG's threshold on the pooled model before enabling auto-reject.
- vCA1 v2 feature contract: `docs/VCA1_V2_BRIEF.md`.
- March-2026 matching-study conclusions (incl. temporal-matching tests, which compared
  mirror-cell pairs) are void; memory updated accordingly.
