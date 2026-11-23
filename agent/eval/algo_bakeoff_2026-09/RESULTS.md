# Algorithm bake-off, 2026-09-06 — results

Question (plan `~/.claude/plans/i-still-need-to-agile-backus.md`): is XGBoost the
right learner for the cell-curation classifier, and would a neural net or another
learner do better? Stage 1 answers it on the existing 35-column contract; Stage 2
answers it for a neural net on the raw footprints and traces.

Every number below comes from `results/*.json` written by the scripts in this
directory; the complete per-contender tables are `results/report_BLA_t8.md` and
`results/report_vCA1_t8.md`. Protocol and the pre-registered decision rule:
`README.md` (written before any contender ran).

## Stage 1 — tabular learners on the same 35 columns: **null, XGBoost stays**

### Harness validation

- Reproduction of the pinned deployed scores (`pareto_2026-08/bla_deployed_oof.npz`,
  75 sessions, 12,677 rows, 8 seeds, unpinned `n_jobs=-1` as the original):
  **bit-exact** — every per-seed score identical, all six summary statistics at
  abs diff 0 (`results/repro_BLA_t0.json`).
- Thread-count effect, same rows, pinned 8 threads vs unpinned 48
  (`results/repro_BLA_t8.json` vs `_t0`): aggregate AUC identical to four decimals
  (0.9291 full / 0.9159 reviewed both ways), but individual scores move by up to
  0.42, 48% of rows move by more than 0.01 and 14% by more than 0.05; false-AR at
  T=0.04 moves 0.726% → 0.715% (worst seed 1.086% → 0.919%) and **the threshold
  rule flips from 0.03 (unpinned) to 0.04 (pinned)**. This reproduces the red
  team's open finding and is why every bake-off number is at 8 pinned threads.
- Pool: the bake-off ran on the live corpus of 2026-09-06 (BLA 176 labeled
  sessions, 81 in the CV pool; vCA1 142 / 24), frozen per area by
  `results/pool_<AREA>.json` so watcher ingests during the runs could not change
  the rows. The BLA CV pool is 6 sessions larger than the pinned red-team pool,
  which is why the deployed control reads 0.9245 here versus 0.9291 there.

### Contenders vs the pinned deployed control (reviewed-pool AUC, 8 seeds unless noted)

BLA, session-grouped — control 0.9245 full / **0.9099** reviewed, false-AR 0.76%
(worst 0.92%) and 41.4% junk at T=0.04, rule_T 0.04:

| contender | AUC reviewed | Δ vs control (seeds +) | junk at matched false-AR |
|---|---|---|---|
| LightGBM, production mirror | 0.9092 | −0.0007 (2/8) | +0.9 pp |
| HistGradientBoosting | 0.9090 | −0.0009 (1/8) | −0.4 pp |
| CatBoost default | 0.9088 | −0.0011 (0/8) | −3.1 pp |
| CatBoost nested-tuned (3 seeds) / winner | 0.9091 / 0.9101 | +0.0006 / ≈0 | +2.3 pp / — |
| XGBoost nested-tuned (3 seeds) | 0.9050 | **−0.0035 (0/3)** | −2.9 pp |
| XGBoost winner config (selection-optimistic) | 0.9110 | +0.0010 (8/8) | +0.4 pp |
| LightGBM nested-tuned / winner | 0.8985 / 0.9053 | −0.0100 / −0.0046 | −5.5 / 0.0 pp |
| ExtraTrees | 0.8998 | −0.0102 (0/8) | +0.8 pp |
| sklearn MLP (the tabular neural net) | 0.9056 | −0.0043 (1/8) | −0.7 pp |
| probability mean of XGB + LGBM + CatBoost | **0.9118** | +0.0019 (8/8) | +0.1 pp |

BLA, animal-grouped (5 folds over 6 animals, test animals' bootstrap sessions
dropped) — control 0.9151 / **0.8989**, false-AR 1.01% (worst 1.15%), rule_T 0.03:
LightGBM +0.0005 (7/8), HistGB +0.0003 (5/8), MLP −0.0002, ExtraTrees −0.0109,
ensemble +0.0018 (8/8) with +1.7 pp junk; XGBoost nested-tuned 0.8973 (−0.0015),
its winner 0.9017 (+0.0028 on 8/8, selection-optimistic, +1.0 pp junk), LightGBM
tuned 0.8845, its winner 0.8944, CatBoost default 0.8958 (−0.0030); **CatBoost
tuned 0.9001 (+0.0013, 3/3) and its winner 0.9010 (+0.0022 on 8/8, +4.4 pp junk)
are the only entries that clear the rule under animal grouping — and both tie the
control under session grouping (+0.0006 and +0.0001), so they fail the
both-groupings requirement.**

vCA1, session-grouped (24 sessions, 2,068 rows, 483 real; 50,370 bootstrap rows)
— control 0.9252 / **0.9227**: ensemble +0.0028 (8/8) with +3.4 pp junk, CatBoost
default +0.0009 (6/8) but −8.1 pp junk, LightGBM −0.0011, HistGB −0.0021, all
tuned variants and the MLP (−0.0058) and ExtraTrees (−0.0132) worse. Animal-grouped
(3 animals, fixed split) — control 0.9101 / **0.9071**: ensemble +0.0011 (7/8),
+3.3 pp; everything else at or below the control.

### Reading

1. **No tabular learner clears the pre-registered rule** (≥ +0.003 reviewed AUC
   on 8/8 seeds AND ≥ +2 pp junk at matched false-AR, under both groupings). The
   closest is the three-booster ensemble: +0.002 to +0.003 AUC everywhere, but
   its junk gain is +0.1 pp on BLA session-grouped. Gradient boosting in any
   flavour lands within ±0.002 of the deployed XGBoost; the tabular neural net
   trails by 0.004 to 0.015.
2. **Tuning hurts under honest (nested) evaluation.** Thirty configurations
   searched with early stopping on an inner session split picked 13 different
   winners in 15 folds on BLA, each beating the deployed shape on its inner split
   by only +0.001 to +0.004, and the resulting outer-fold scores were 0.0035
   (XGBoost) to 0.010 (LightGBM) *below* the untuned deployed configuration. The
   single most-selected configuration, re-run fixed, ties the control (+0.001,
   with selection optimism). The deployed 300 trees / depth 4 / lr 0.05 sits on
   the optimum's plateau; there is nothing to gain from hyperparameters.
3. **The score scale is part of the model.** CatBoost and the tuned LightGBM
   winners reach the same AUC with a different probability scale (false-AR at
   T=0.04 of 2.7–3.3% for CatBoost), so a learner swap would also mean a
   threshold re-derivation and a new bootstrap-calibration check; nothing here
   earned that.
4. **Operator flag, vCA1.** On the current 24-session pool the deployed T=0.04
   runs 1.16% mean / 1.45% worst-seed false-AR, and no T in [0.03, 0.10]
   satisfies the 1% rule (0.03 gives 0.98% / 1.24%). The 8 sessions added since
   the 2026-08-26 gate are the threshold-0 pnb97/pnb98 sessions whose reviewers
   saw every candidate, i.e. the hard dim reals, consistent with the 2.08%
   observed on the two-session holdout. New animals of this prep should keep the
   threshold-0 first pass; the vCA1 threshold wants re-deriving on this pool
   with threads pinned (`results/vCA1_session_xgb_deployed_t8.json` has the table).
5. **DG_AL own-model baseline** (17 sessions, 2,025 rows, 1,028 real, 13 columns):
   0.9002 session-grouped, 0.8638 under 2-animal leave-one-out
   (`results/DG_AL_*_xgb_deployed_t8.json`) — the reference for Stage 2's transfer number.

Conclusion for the question asked: on the existing features, XGBoost as
configured is the right learner. The remaining headroom is not in the algorithm.

## Stage 2 — raw-data CNN probe: IN PROGRESS

Corpus (`prep_raw_corpus.py`, `D:\Julian_CNMFe\<AREA>\.feature_expansion\_raw_corpus\`):
3-channel 32×32 footprint patches at a gSig-normalised scale (own footprint, Cn
z-score, per-pixel max over the other candidates) plus the raw trace resampled
to the AVG5x rate. After backfilling the 31 agent sessions that had no
extraction (`extract_backfill.m`, all 31 succeeded), coverage is BLA 45,669 /
vCA1 52,544 / DG_AL 2,025 rows — 100,238 of 102,358 labeled candidates; the
2,120 missing are auto-rejected candidates whose footprints were never kept.
Orientation was verified before training: for labeled reals, Cn under the
footprint mask beats Cn under the transposed mask for 90% (BLA bootstrap) /
92% (BLA agent) / 89% / 90% (vCA1) / 99% (DG_AL) of cells, and the contact
sheet (`results/contact_sheet_BLA_first36.png`) shows contours on Cn blobs for
both pixel-order sources.

Model (`train_cnn.py`): 2-D CNN on the patch + 1-D dilated CNN on 2048-frame
trace crops, 164k parameters, weighted BCE with the corpus weights and
scale_pos_weight semantics, early stopping on an inner session split, folds
identical to the tabular control (same y and groups → same
StratifiedGroupKFold splits), 8 seeds, both groupings, BLA and vCA1.
Comparison (`stack_eval.py`): CNN alone on the raw-data rows, and the
probability / rank blends with the control on all rows, against the plan's gate.
Transfer (`transfer_dg.py`): a BLA+vCA1-trained CNN scoring DG_AL cold.

### Transfer to DG_AL (done, `results/DG_AL_transfer_cnn.json`)

A CNN trained on every BLA + vCA1 row (98,213 rows, 3 seeds, inner-val AUC
0.94–0.95 within its own areas) scores DG_AL's 2,025 candidates cold at
**AUC 0.793** (DG6D 0.792, DG6E 0.792), against DG_AL's own 13-column model at
0.900 session-grouped and 0.864 under 2-animal leave-one-out. As a ranking prior
blended (rank mean) with the own model it costs −0.014 session-grouped and adds
+0.004 animal-grouped. The pooled tabular prior gave +0.079 under the same
leave-one-out (red team #18). **The gSig-scaled raw representation does not
transfer across areas; the pooled tabular prior remains the route for the
data-starved area.**

### CNN vs deployed XGBoost on BLA and vCA1

The rows that have raw data are exactly the reviewed rows (auto-rejected
candidates are the only ones without), so "CNN alone" is measured on the
reviewed pool and the blends on the full pool like any tabular contender.

**BLA, session-grouped, 8 seeds (`results/stage2_BLA_t8.md`):**

| comparison | reviewed AUC (control → new) | Δ (min; seeds +) | FAR at matched junk | junk at matched FAR |
|---|---|---|---|---|
| (a) CNN alone, 11,877 reviewed rows | 0.9099 → 0.8944 | −0.0155 (0/8) | 1.03% vs 0.76% | 28.6% vs 32.3% (−3.6 pp) |
| (b) blend = mean(control, CNN), all 13,766 rows | 0.9099 → **0.9144** | **+0.0045** (+0.0021; 8/8) | **0.54% vs 0.76%** | **45.8% vs 41.4% (+4.4 pp)** |
| (c) rank blend | 0.9099 → 0.9143 | +0.0044 (+0.0007; 8/8) | 0.48% vs 0.76% | 47.3% vs 41.4% (+5.9 pp) |

Gate for the blend (≥ +0.005 AUC on 8/8, ≥ +3 pp junk, rule_T with worst-seed
FAR ≤ 1%): AUC **fails by 0.0005**, junk passes, rule passes (rule_T 0.09 at
0.95% worst) → **FAIL by the letter of the pre-registered rule; the operating
point gain is nonetheless real and larger than the one that justified the
2026-08 feature expansion (+2–3 pp).** Kill criterion (CNN alone < control − 0.02):
not triggered.

Per animal (reviewed AUC, control → CNN alone → blend): bla12 0.921 → 0.902 → 0.924;
bla16 0.911 → 0.871 → 0.904; **bla21 0.852 → 0.900 → 0.898**; bla36 0.902 → 0.887 → 0.906;
bla37 0.922 → 0.899 → 0.921; bla8 0.909 → 0.895 → 0.914. The raw representation is
strongly complementary on bla21, the retro-era animal the tabular model has never
handled (+0.046 in blend), neutral on bla37, and costs 0.007 on bla16.

Exploratory (post-hoc on the same scores, so optimistic for any weight but the
pre-specified 0.5): the gain is not a knife edge. Blend weights 0.2–0.6 on the
CNN all give +0.003 to +0.005 reviewed AUC on 8/8 seeds and +3.2 to +5.6 pp junk
at matched false-AR (0.3 and 0.4 read +0.0052); 0.7 and above fall away. On the
seed-mean score, 18 reals sit below the control's T=0.04; the 50/50 blend leaves
6 of them there (12 rescued, none newly lost). Note the blend is more
conservative at the same nominal T (fewer junk below 0.04 too), which is why the
matched-operating-point columns, not the raw FAR@T, are the comparison.

**vCA1, session-grouped, 8 seeds (`results/stage2_vCA1_t8.md`):**

| comparison | reviewed AUC (control → new) | Δ (min; seeds +) | FAR at matched junk | junk at matched FAR |
|---|---|---|---|---|
| (a) CNN alone, 2,011 reviewed rows | 0.9227 → 0.9205 | −0.0022 (4/8) | 1.14% vs 1.16% | 44.7% vs 44.4% (+0.4 pp) |
| (b) blend, all 2,068 rows | 0.9227 → **0.9319** | **+0.0093** (+0.0049; 8/8) | **0.91% vs 1.16%** | **51.2% vs 46.0% (+5.2 pp)** |
| (c) rank blend | 0.9227 → 0.9332 | +0.0105 (+0.0053; 8/8) | 0.88% vs 1.16% | 49.9% vs 46.0% (+3.9 pp) |

Gate for the blend: AUC pass, junk pass, rule pass → **PASS**. The blend also
restores an admissible threshold on this pool — rule_T 0.07 with worst-seed
false-AR 0.83% — where the deployed control had none (item 4 of Stage 1). Per
animal (control → CNN → blend): pnb88 0.874 → 0.857 → 0.884; pnb97 0.919 →
0.922 → 0.929; pnb98 0.948 → 0.940 → 0.954. Standalone, the CNN ties the
tabular model on vCA1; blended, it is the largest single gain measured on vCA1
since the 35-column contract.

**vCA1, animal-grouped (3-animal leave-one-out, 8 seeds):**

| comparison | reviewed AUC (control → new) | Δ (min; seeds +) | FAR at matched junk | junk at matched FAR |
|---|---|---|---|---|
| (a) CNN alone | 0.9071 → 0.8930 | −0.0142 (2/8) | 2.28% vs 0.91% | 30.2% vs 43.1% (−12.8 pp) |
| (b) blend | 0.9071 → 0.9152 | +0.0081 (−0.0025; **7/8**) | **1.27% vs 0.91%** | **42.7% vs 44.5% (−1.8 pp)** |
| (c) rank blend | 0.9071 → 0.9165 | +0.0093 (−0.0033; 7/8) | 1.19% vs 0.91% | 44.7% vs 44.5% (+0.3 pp) |

Gate: AUC fails (7/8), junk fails, rule passes → **FAIL**. Per animal (control →
CNN → blend): pnb88 0.859 → 0.809 → 0.856; pnb97 0.915 → 0.917 → 0.925; pnb98
0.946 → 0.942 → 0.954. The reading matters more than the verdict: for a held-out
animal the blend still lifts the ranking overall (+0.008) but makes the
*low-score end worse* — exactly the region the auto-reject threshold operates
in. The CNN's scores for a new animal are least reliable where they matter most.

Exploratory asymmetric blends (post-hoc on the same scores): a rescue-only
blend (max of the two scores) is worse than the mean everywhere (BLA session
−0.002 AUC; vCA1 animal −0.001 with junk −3.1 pp), so the CNN's value is not
"raise the dim reals" alone — it re-orders the middle of the ranking. A
geometric mean tracks the arithmetic mean. A lighter CNN weight of 0.3 is the
one variant that is good or neutral in all three finished groupings: BLA
session +0.0052 (8/8) / +5.5 pp, vCA1 session +0.0076 (8/8) / +5.0 pp, vCA1
animal +0.0080 (8/8) / +1.2 pp with FAR at matched junk 1.01% vs 0.91%. That is
a lead for a properly nested stacked model (plan step (c)), not a result: the
weight was chosen on the scores it is evaluated on.

**BLA, animal-grouped (5 folds over 6 animals, test animals' bootstrap sessions dropped, 8 seeds):**

| comparison | reviewed AUC (control → new) | Δ (min; seeds +) | FAR at matched junk | junk at matched FAR |
|---|---|---|---|---|
| (a) CNN alone | 0.8989 → 0.8881 | −0.0108 (1/8) | 1.13% vs 1.01% | 30.2% vs 31.5% (−1.3 pp) |
| (b) blend | 0.8989 → **0.9067** | **+0.0078** (+0.0010; 8/8) | **0.65% vs 1.01%** | **46.8% vs 40.6% (+6.2 pp)** |
| (c) rank blend | 0.8989 → 0.9069 | +0.0080 (+0.0003; 8/8) | 0.66% vs 1.01% | 47.3% vs 40.6% (+6.7 pp) |

Gate: AUC pass, junk pass, rule pass (rule_T 0.09 at 0.95% worst) → **PASS**.
Per animal (control → CNN → blend): bla12 0.910 → 0.907 → 0.919; bla16 0.911 →
0.875 → 0.903; **bla21 0.857 → 0.913 → 0.904**; bla36 0.896 → 0.887 → 0.901;
bla37 0.915 → 0.902 → 0.918; bla8 0.905 → 0.904 → 0.915. Held-out-animal
generalisation on BLA is where the blend is strongest.

### Stage 2 verdict

| area / grouping | blend Δ reviewed AUC (seeds +) | junk at matched FAR | gate |
|---|---|---|---|
| BLA session | +0.0045 (8/8) | +4.4 pp | AUC short by 0.0005 → fail |
| BLA animal | +0.0078 (8/8) | +6.2 pp | **pass** |
| vCA1 session | +0.0093 (8/8) | +5.2 pp | **pass** |
| vCA1 animal | +0.0081 (7/8) | −1.8 pp | fail (operating point worse) |

**By the pre-registered rule (BLA under both groupings, vCA1 not worse) the
probe does not pass**, and it should not be deployed as a 50/50 blend. It is
also not a null. Every cell shows a seed-consistent ranking gain, three of four
show a +4 to +6 pp gain at the operating point — larger than the +2–3 pp that
justified the 2026-08 feature expansion — the retro-era animal bla21 gains
about +0.05 in every view, and on the vCA1 session pool the blend restores an
admissible threshold where the deployed model has none. The failure is specific:
for a held-out vCA1 animal the CNN's low-end scores are unreliable, and a fixed
50/50 mix lets that leak into the auto-reject region. The kill criterion never
triggered (CNN alone within 0.011–0.016 of the control everywhere).

What this says about the original question: the raw footprints and traces do
carry signal the 35 columns miss (the tabular bake-off found none in the
algorithm; the CNN finds some in the representation), the gain concentrates on
the hardest animals and eras, and it is an ensemble gain — standalone, a small
CNN does not beat the engineered features.

Recommended next step (not started; the plan gates it on (b) passing, which it
did in two of four cells): the stacked variant (c) — the CNN's out-of-fold score
as a 36th column into XGBoost with nested folds — which learns the mix instead
of assuming 0.5, lets the trees discount the CNN where the other features say
its score is unreliable (the vCA1-animal failure mode), and keeps the score
scale the threshold machinery is calibrated on. Its cost is the nested CNN
training (roughly 3× tonight's GPU time) plus, if it passes, the Stage 3
deployment brief: torch in the production env, in-line patch/trace extraction
in the curator, a 36-column contract through the atomic-swap path, and a
threshold re-derivation with threads pinned. The 14 backfilled BLA/vCA1
sessions and the tuned-vs-untuned CNN itself (one architecture, one learning
rate, no search) are further headroom this probe did not use.

Files: `results/stage2_{BLA,vCA1}_t8.{md,json}` (full tables and per-animal),
`results/<AREA>_<grouping>_cnn_t8.{npz,json}` (OOF scores, per-fold curves),
`results/DG_AL_transfer_cnn.{npz,json}`, corpora and manifests under
`D:\Julian_CNMFe\<AREA>\.feature_expansion\_raw_corpus\`, extractions for the
31 backfilled sessions under `D:\Julian_CNMFe\<AREA>\.feature_expansion\`.
Nothing in production was touched: no session dir, joblib, label file or
watcher; the `valence` env is unchanged (all of this ran in `ACORN`).

## Round 2 — CNN training-regime and input variants (2026-09-07, in progress)

**Why a round 2.** The user asked whether the Stage 2 shortfall was the CNN's
size or quality. `cnn_seed_variance.py` (results/cnn_seed_variance.txt) answered
from the round-1 fold logs: the standalone gap is mostly variance, not
capacity. Per-seed reviewed-AUC spread is 0.005–0.014 for the CNN against
0.001–0.003 for XGBoost, and averaging the 8 seeds' OOF (a legitimate bagging
ensemble: each seed scores a row from a model that never saw its session or
animal) lifts the CNN alone by +0.011 (BLA, both groupings), +0.012 (vCA1
session) and +0.022 (vCA1 animal) while XGBoost gains +0.002; the averaged CNN
alone beats averaged XGBoost on vCA1 (0.932 vs 0.925 session, 0.915 vs 0.909
animal) and trails by 0.007 on BLA. Two training-regime defects: the inner
early-stopping split is only 75–516 reviewed rows on vCA1 session and 20–394
on vCA1 animal (val−test gap +0.012/+0.031 vs +0.003 on BLA), and 5–8 of 40
folds ran to the 25-epoch cap. Otherwise training is sane (val AUC 0.87 after
one epoch, best at median epoch 10, mild overfit after). On inputs: the CNN
sees none of the 35 columns; 18 of them are functions of what it sees, 17 are
not (13 within-session ranks, nb_corr_max/any, motion_correlation, the flag).
The plan, gate and amendments are in README.md ("Round 2"); everything lives
under `results/r2/`.

**Corpora (`prep_raw_corpus2.py`).** `<AREA>_raw_corpus_v2.h5` reproduces the
round-1 corpus bit-for-bit (images, traces, metadata, session order; checked)
and adds per row the max-correlated neighbour within 60 px of centroid
distance (`nb_row`): 45,445 of 45,669 BLA rows and 52,406 of 52,544 vCA1 rows
have one, median raw-trace correlation 0.47 / 0.48. `_v2_p48` holds 48×48
patches. A background-signal channel was dropped: `Ybg_mean.mat` exists for 0
of 202 bootstrap sessions (the majority of training rows) and 112 of 116
agent sessions.

**Two negative results on the round-1 scores, before any CNN retraining:**

| combiner (8 seeds, reviewed-AUC delta vs control) | BLA session | BLA animal | vCA1 session | vCA1 animal |
|---|---|---|---|---|
| fixed 50/50 probability blend (Stage 2) | +0.0045 (8/8) | +0.0078 (8/8) | +0.0093 (8/8) | +0.0081 (7/8) |
| 36th/37th-column XGBoost stack (`stack_xgb.py`; CNN score + within-session rank, NaN on bootstrap rows) | +0.0034 (7/8) | +0.0032 (7/8) | +0.0040 (8/8) | −0.0009 (3/8) |
| per-fold logistic combiner of the two logits (`blend_fit.py`) | +0.0034 (7/8) | +0.0033 (6/8) | +0.0077 (8/8) | −0.0177 (1/8) |

The stack fails the gate in every cell (junk at matched false-AR +2.9 / +2.2 /
−3.1 / −1.3 pp); the 35-column re-run through the same code path reproduces
the pinned control to four decimals, so this is not a harness artefact. A
level-0 leak can only inflate a stack, so the pre-registered leak measurement
(`stack_leak.py`) is moot and was not run. Likely cause, not proven: the CNN
column is missing on the 70–96% bootstrap rows, so its splits are learned on
agent rows only and "missing" doubles as a bootstrap flag. The logistic
combiner learns CNN weights of 0.40–0.54, i.e. close to 0.5, yet loses to the
fixed blend everywhere (−0.001 to −0.026), worst on vCA1 animal: combining in
logit space lets an overconfident CNN tail dominate, which is exactly the
held-out-animal failure mode; probability averaging caps the CNN's pull at
±0.5. The fixed 0.5 probability blend is therefore the pre-declared combiner
for confirmation.

**Screening (3 seeds 42/1/7, paired per seed; `results/r2/screen.md`).**
CNN alone = reviewed AUC on the rows both CNNs scored, delta vs the round-1
CNN (min over seeds; seeds positive); blend = 50/50 with the control, delta vs
the control and junk at the control's matched false-AR. Round-1 reference on
these seeds: BLA session CNN alone 0.8979, blend +0.0054 / +4.3 pp; vCA1
animal CNN alone 0.9027, blend +0.0117 / −2.5 pp.

| variant | BLA session: CNN alone Δ | BLA session: blend Δ / junk pp | vCA1 animal: CNN alone Δ | vCA1 animal: blend Δ / junk pp | rule |
|---|---|---|---|---|---|
| `valfix` (bootstrap sessions in the inner val, cap 40) | −0.0022 (−0.0103; 2/3) | +0.0057 / +4.0 | +0.0115 (+0.0035; 3/3) | +0.0172 / +0.3 | not carried |
| `fixed12` (12 epochs, cosine, no early stopping) | +0.0035 (+0.0023; 3/3) | +0.0067 / +5.9 | +0.0104 (−0.0030; 2/3) | +0.0143 / +2.2 | **carried** |
| `ens3` (3 members per fold) | +0.0036 (+0.0013; 3/3) | +0.0064 / +6.1 | +0.0129 (+0.0042; 3/3) | +0.0169 / +3.1 | **carried** |
| `nbtrace` (neighbour trace channel) | +0.0075 (+0.0049; 3/3) | +0.0101 / +5.6 | −0.0111 (−0.0258; 0/3) | +0.0120 / +1.2 | not carried |
| `patch48` | +0.0026 (+0.0020; 3/3) | +0.0065 / +3.9 | −0.0182 (−0.0302; 0/3) | +0.0061 / −6.4 | not carried |
| `fixed12+nbtrace` (pre-declared combination, README 19:10) | **+0.0104** (+0.0086; 3/3) | +0.0101 / +5.3 | **+0.0123** (+0.0007; 3/3) | +0.0162 / +1.1 | **carried; replaces `fixed12`** |

Reading. (1) The two early-stopping fixes buy their gain where the round-1
inner validation was 27–264 rows (held-out vCA1 animal, +0.010 to +0.012) and
are neutral-to-positive on BLA; the fixed schedule is also the cheapest
regime (no validation pass; 12 epochs) and every session trains. (2) The
neighbour trace is the largest single BLA gain (+0.0075, worst seed +0.0049)
but under the round-1 regime it loses 0.011 on the held-out vCA1 animal; on
the fixed schedule the combination keeps the BLA gain (+0.0104) and turns the
vCA1-animal cell positive (+0.0123, 3/3). That combination is a deviation from
the single-variant rule for `nbtrace`, pre-declared before it was run. (3) The
48-px patch never clears the rule. (4) Three within-fold members recover about
a third of the cross-seed averaging gain on BLA (+0.0036 of +0.011) and all of
it on vCA1 animal (+0.0129 of +0.022), and give the best operating-point gains
of any single change (+6.1 / +3.1 pp junk). Cost: the fixed schedule ran a
screening cell in 31–46 min, `ens3` in 116–162 min (GPU shared between two
queues for most of the afternoon).

**Confirmation configuration: `fixed12+nbtrace+ens3`** at 8 seeds on all four
cells, judged by the unchanged Stage 2 gate with the fixed 50/50 blend. Member
0 (`fixed12+nbtrace`, launched 21:02) and members 1–2 (launched 22:55, saved
per member and averaged by `assemble_ens.py`) run overnight; `confirm_eval.py`
writes `results/r2/confirm_<variant>.md`.

### Round 2 confirmation — 8 seeds, all four cells (2026-09-08)

**Member 0 alone (`fixed12+nbtrace`: 12-epoch cosine schedule, every session
in training, two-channel trace = own + max-correlated neighbour within 60 px;
single model per fold; `results/r2/confirm_fixed12+nbtrace.md`).** Combiner:
the pre-declared fixed 50/50 probability blend with the deployed control.

| cell | CNN alone Δ vs control (min; +) | blend Δ reviewed AUC (min; +) | junk at matched false-AR | FAR at matched junk | rule_T (worst-seed FAR) | gate |
|---|---|---|---|---|---|---|
| BLA session | −0.0006 (−0.0033; 2/8) | **+0.0098** (+0.0083; 8/8) | 46.8% vs 41.4% (**+5.4 pp**) | 0.51% vs 0.76% | 0.09 (0.95%) | **PASS** |
| BLA animal held out | +0.0046 (+0.0018; 8/8) | **+0.0140** (+0.0123; 8/8) | 48.6% vs 40.6% (**+8.0 pp**) | 0.54% vs 1.01% | 0.08 (0.88%) | **PASS** |
| vCA1 session | +0.0065 (+0.0010; 8/8) | **+0.0119** (+0.0086; 8/8) | 49.7% vs 46.0% (+3.7 pp) | 0.98% vs 1.16% | 0.04 (0.83%) | PASS (not worse) |
| vCA1 animal held out | +0.0080 (+0.0033; 8/8) | **+0.0167** (+0.0141; 8/8) | 47.4% vs 44.5% (+3.0 pp) | 0.78% vs 0.91% | 0.04 (0.83%) | PASS (not worse) |

**The pre-registered Stage 2 gate is met** (BLA under both groupings: AUC,
junk and threshold rule; vCA1 better, not merely "not worse", under both).
For comparison the round-1 blend on the same seeds was +0.0045 / +0.0078 /
+0.0093 / +0.0081 with junk +4.4 / +6.2 / +5.2 / −1.8 pp. The CNN alone,
which trailed the tabular model by 0.011–0.016 in round 1, now ties it on BLA
session (−0.0006) and beats it in the other three cells (+0.005 to +0.008).

Restricted to the five seeds never used for screening (13, 100, 2024, 31337,
9; `results/r2/clean_seeds_fixed12+nbtrace.md`), so free of variant-selection
optimism: blend Δ +0.0097 (5/5) / +0.0139 (5/5) / +0.0113 (5/5) / +0.0170
(5/5), junk +5.4 / +8.3 / +2.8 / +4.1 pp. The screening cells' numbers are
therefore not inflated by selection.

Per animal (reviewed AUC, control → CNN alone → round-1 blend → new blend),
BLA session-grouped: bla12 0.921 → 0.918 → 0.924 → 0.930; bla16 0.911 →
0.883 → 0.904 → 0.908; **bla21 0.852 → 0.918 → 0.898 → 0.906**; bla36 0.902 →
0.901 → 0.906 → 0.911; bla37 0.922 → 0.914 → 0.921 → 0.926; bla8 0.909 →
0.916 → 0.914 → 0.923. bla16, the one BLA animal the round-1 blend hurt
(−0.007), is now within 0.003 of the control; every other animal improves.
vCA1 animal-grouped: pnb88 0.859 → 0.819 → 0.856 → 0.861; pnb97 0.915 →
0.927 → 0.925 → 0.929; pnb98 0.946 → 0.949 → 0.954 → 0.955 — the round-1
failure mode (a held-out vCA1 animal made worse at the low-score end) is
gone: FAR at matched junk 0.78% vs the control's 0.91%.

**Ensemble (`fixed12+nbtrace+ens3`, the pre-declared configuration):** members
1–2 running; assembled with `assemble_ens.py` and evaluated with
`confirm_eval.py --variant fixed12+nbtrace+ens3` when done (section below).

**Ensemble — the pre-declared configuration `fixed12+nbtrace+ens3`** (3
members per fold, member 0 = the run above, members 1–2 trained separately
and averaged by `assemble_ens.py`; `results/r2/confirm_fixed12+nbtrace+ens3.md`):

| cell | CNN alone Δ vs control (min; +) | blend Δ reviewed AUC (min; +) | junk at matched false-AR | FAR at matched junk | rule_T (worst-seed FAR) | gate |
|---|---|---|---|---|---|---|
| BLA session | +0.0005 (−0.0009; 6/8) | **+0.0101** (+0.0093; 8/8) | 46.9% vs 41.4% (**+5.5 pp**) | 0.48% vs 0.76% | 0.09 (0.84%) | **PASS** |
| BLA animal held out | +0.0061 (+0.0040; 8/8) | **+0.0144** (+0.0131; 8/8) | 48.4% vs 40.6% (**+7.7 pp**) | 0.51% vs 1.01% | 0.08 (0.88%) | **PASS** |
| vCA1 session | +0.0075 (+0.0039; 8/8) | **+0.0119** (+0.0101; 8/8) | 49.8% vs 46.0% (+3.8 pp) | 0.88% vs 1.16% | 0.05 (0.83%) | PASS |
| vCA1 animal held out | +0.0115 (+0.0092; 8/8) | **+0.0174** (+0.0154; 8/8) | 46.8% vs 44.5% (+2.3 pp) | 0.88% vs 0.91% | 0.05 (0.83%) | PASS |

Clean seeds only (13, 100, 2024, 31337, 9; `results/r2/clean_seeds_fixed12+nbtrace+ens3.md`):
blend Δ +0.0100 / +0.0143 / +0.0115 / +0.0178, all 5/5; junk +5.5 / +8.3 /
+3.4 / +2.1 pp. Per animal (control → CNN → round-1 blend → blend): BLA
session bla12 0.921 → 0.918 → 0.924 → 0.929; bla16 0.911 → 0.886 → 0.904 →
0.909; bla21 0.852 → 0.919 → 0.898 → 0.906; bla36 0.902 → 0.903 → 0.906 →
0.911; bla37 0.922 → 0.915 → 0.921 → 0.926; bla8 0.909 → 0.918 → 0.914 →
0.923. vCA1 animal held out: pnb88 0.859 → 0.819 → 0.856 → 0.861; pnb97 0.915
→ 0.928 → 0.925 → 0.929; pnb98 0.946 → 0.951 → 0.954 → 0.956.

### Round 2 verdict

**The pre-registered gate is met** — by the pre-declared ensemble
configuration and, separately, by its single-model member alone. Relative to
the single model the ensemble adds +0.0001 to +0.0007 reviewed AUC and raises
the worst seed in every cell, at three times the training cost (≈ 5 min per
area per member on the GPU when trained on the full pool; see the brief).

What changed since round 1, and why it worked: the round-1 shortfall was
variance and a noisy epoch selection, not capacity. Removing early stopping
(12 epochs, cosine decay, every session trains) fixed the held-out-vCA1-animal
failure, whose inner validation had been 20–394 rows; the neighbour-trace
channel supplied the raw form of the two neighbour-correlation columns and is
the largest BLA gain; three members recover part of the seed variance. The
36th-column stack and a learned logit combiner both lose to the fixed
probability blend, so the deployable combiner is the simplest one. Not
pursued: larger patches (never cleared the rule), a background-signal channel
(no data on bootstrap sessions), more capacity (no evidence it binds).

Measured prize at the operating point, BLA: at the deployed false auto-reject
rate the blend catches 46.9% of junk instead of 41.4% (session-grouped) and
48.4% instead of 40.6% (held-out animal); at the deployed junk rate the false
auto-reject rate falls from 0.76% to 0.48% and from 1.01% to 0.51%. vCA1: the
blend restores an admissible threshold (rule_T 0.05, worst-seed 0.83%) on the
pool where the deployed control had none.

Caveats that survive: the label ceiling is still unmeasured (Stage 0 not
run); vCA1 has three animals; scores are cuDNN-nondeterministic across
retrains (report seeds, not bits); the neighbour channel and the gSig-scaled
patch require the same preprocessing at curation time as in
`prep_raw_corpus2.py`. Deployment is a separate decision:
`docs/CNN_BLEND_STAGE3_BRIEF.md`. Nothing has been committed; production is
untouched; GPU idle.

### Pooled BLA + vCA1 CNN (2026-09-08 20:30; pre-registered in README "Pooled BLA + vCA1 CNN")

`train_cnn_pooled.py`: per target cell, the confirmed single-model
configuration (`fixed12+nbtrace`) with every row of the other area's corpus
added to training at its native weights (×1.0); folds, rows, seeds and the
target area's weights identical to the per-area run; 3 seeds (42, 1, 7),
paired with the per-area run at the same seeds (`results/r2/pooled1_fixed12+nbtrace.md`).

| cell | CNN alone, per-area → pooled (Δ; min; seeds +) | blend Δ vs control, per-area → pooled | junk at matched FAR, per-area → pooled | FAR at matched junk |
|---|---|---|---|---|
| BLA session | 0.9083 → 0.9114 (+0.0032; +0.0017; 3/3) | +0.0101 → +0.0109 | +5.3 → +6.0 pp | 0.55 → 0.51% |
| BLA animal held out | 0.9027 → 0.9074 (+0.0047; +0.0034; 3/3) | +0.0140 → +0.0156 | +7.4 → +9.5 pp | 0.50 → 0.48% |
| vCA1 session | 0.9295 → 0.9246 (**−0.0048**; −0.0081; 0/3) | +0.0128 → +0.0117 | +5.2 → **+0.8 pp** | 0.90 → 1.10% |
| vCA1 animal held out | 0.9150 → 0.9214 (+0.0064; +0.0025; 3/3) | +0.0162 → +0.0200 | +1.1 → −0.8 pp | 0.90 → 1.04% |

Reading, against the pre-registered rule (pooled CNN alone ≥ per-area in
both of an area's cells and the blend not worse at the operating point):
**BLA qualifies** — the vCA1 rows help BLA in both groupings, on every seed,
mostly through bla16 (0.881 → 0.897 session-grouped) and bla37, and the
operating point improves as well (+6.0 / +9.5 pp junk). **vCA1 does not** —
session-grouped the pooled model is worse on all three seeds and the
operating-point gain nearly vanishes (+5.2 → +0.8 pp), driven by pnb88, the
old-prep animal (0.861 → 0.833); under leave-one-animal-out the ranking
improves (+0.0064) but the low-score end is worse (junk +1.1 → −0.8 pp, FAR
at matched junk 0.90 → 1.04%). The 45k BLA rows outnumber vCA1's own agent
rows and pull the model toward BLA's prep. This is the opposite of the
tabular precedent (pooling null for BLA, +0.028 for vCA1 animal), which is
consistent with the CNN's inputs being scale-free across sessions and preps
(gSig-scaled patches, rate-normalised traces) while the tabular ranks are
scale-free only within a session.

**Decision candidate, not yet confirmed:** BLA pooled (BLA + vCA1 rows), vCA1
per-area. A BLA-pooled adoption needs the 8-seed confirmation with the
3-member ensemble before it replaces the per-area BLA model in the brief; it
is a +0.003 to +0.005 refinement on top of a configuration that already meets
the gate, so it is optional and should not delay Stage 3.

## vCA1 refresh on the 2026-09-19 pool (pre-registered in README "vCA1 refresh"; outputs in `results_0919/`)

**Pool.** 16 new labeled vCA1 agent sessions since 09-06 (15 pnb98, threshold-0,
fully human-labelled; 1 pnb88 with 2 kept cells, below the 5-cell CV floor):
158 sessions, CV pool 24 → 39 sessions, 2,068 → 3,726 rows, 483 → 934 kept
cells (pnb88 110, pnb97 206, pnb98 618). Still three animals. All 16 sessions
MATLAB-backfilled (16 OK); corpus `vCA1_raw_corpus_v3.h5` 54,221 rows, join OK.
BLA gained one session and was not re-run.

**Tabular (8 seeds, threads pinned).** Deployed control reviewed AUC 0.9206
session-grouped / 0.9125 animal-grouped (09-06: 0.9227 / 0.9071). An
admissible threshold now exists: **rule_T = 0.03** (false auto-reject 0.74%
mean / 0.96% worst seed session-grouped; 0.67% / 0.86% animal-grouped; junk
40.9% / 40.4%). The deployed **T = 0.04 still fails** the 1%-worst-seed rule
(1.06% / 1.28%; junk 45.2%). Per animal at 0.04: pnb97 2.3% of kept cells
lost, pnb98 0.8%, pnb88 0%. pnb98's AUC fell 0.948 → 0.922 as its sessions
went 7 → 22: the earlier figure was a small-sample high. Stage 1's null
holds: lgbm −0.0011, histgb −0.0010, ens_mean +0.0001, mlp −0.0040.

**CNN single model `fixed12+nbtrace`, 8 seeds, 50/50 blend vs control**
(`results_0919/r2/confirm_fixed12+nbtrace.md`; nothing re-chosen on the new pool):

| cell | CNN alone Δ (min; +) | blend Δ reviewed AUC (min; +) | junk at matched FAR | FAR at matched junk | rule_T: control → blend (junk there) |
|---|---|---|---|---|---|
| session | +0.0059 (+0.0014; 8/8) | **+0.0102** (+0.0082; 8/8) | 46.6% vs 45.2% (+1.5 pp) | 0.96% vs 1.06% | 0.03 (40.9%) → 0.06 (41.9%) |
| animal held out | +0.0043 (+0.0013; 8/8) | **+0.0119** (+0.0104; 8/8) | 43.9% vs 44.7% (−0.9 pp) | 1.00% vs 0.91% | 0.03 (40.4%) → 0.06 (42.3%) |

Pre-registered vCA1 criterion (blend Δ AUC ≥ 0, junk ≥ −1 pp): **met in both
cells**. Against the BLA-strength criteria: AUC met (≥ +0.005 on 8/8 in both),
**operating-point gain not met** (+1.5 / −0.9 pp vs +3 pp). On the 09-06 pool
the same single model read +0.0119 / +0.0167 AUC and +3.7 / +3.0 pp junk, so
**the ranking gain held up with twice the data but the operating-point gain
shrank to about +1 to +2 pp** (at each model's own admissible threshold: +1.0
pp session, +1.9 pp animal). The earlier operating-point figures rested on
483 kept cells, i.e. on roughly five false auto-rejects. Per animal (control →
CNN → blend), session-grouped: pnb88 0.871 → 0.852 → 0.878; pnb97 0.920 →
0.928 → 0.931; pnb98 0.922 → 0.931 → 0.933; held out: pnb88 0.865 → 0.835 →
0.866; pnb97 0.919 → 0.925 → 0.929; pnb98 0.912 → 0.925 → 0.926. The CNN alone
is still unreliable for the old-prep animal pnb88 when it is held out, and
that is where the low-score end suffers.

**Pooled BLA+vCA1 CNN, vCA1 cells, 3 seeds** (`results_0919/r2/pooled1_fixed12+nbtrace.md`):
session CNN alone 0.9267 → 0.9266 (−0.0001, 2/3), blend junk +2.7 → +5.0 pp,
FAR at matched junk 0.89 → 0.93%; animal 0.9165 → 0.9223 (+0.0058, 3/3),
blend junk −1.2 → −2.3 pp, FAR at matched junk 0.96 → 1.03%; pnb88 worse in
both (0.848 → 0.835, 0.837 → 0.814). With twice the vCA1 data the
session-grouped penalty seen on 09-08 (−0.0048) is gone — pooling is now
neutral there — but the rule (pooled ≥ per-area in both cells AND blend not
worse at the operating point) is still not met because the held-out-animal
operating point is worse. **vCA1 stays per-area.**

Ensemble members 1–2 on the new pool: running; appended below when done.

**Ensemble `fixed12+nbtrace+ens3` on the 09-19 pool (8 seeds;
`results_0919/r2/confirm_fixed12+nbtrace+ens3.md`), done 2026-09-19 20:26:**

| cell | CNN alone Δ (min; +) | blend Δ reviewed AUC (min; +) | junk at matched FAR | FAR at matched junk | blend rule_T (junk there) |
|---|---|---|---|---|---|
| session | +0.0070 (+0.0042; 8/8) | **+0.0106** (+0.0091; 8/8) | 47.1% vs 45.2% (+1.9 pp) | 1.00% vs 1.06% | 0.06 (41.8%; control 0.03 → 40.9%) |
| animal held out | +0.0057 (+0.0021; 8/8) | **+0.0124** (+0.0106; 8/8) | 44.3% vs 44.7% (−0.4 pp) | 0.95% vs 0.91% | 0.05 (38.9%; control 0.03 → 40.4%) |

Same reading as the single model: vCA1 criterion met, AUC gain consistent,
operating-point gain small (+1.9 / −0.4 pp). The ensemble adds +0.0004 /
+0.0005 AUC over the single model and lifts the worst seed. Note the
held-out-animal threshold: under the 0.85%-mean clause the blend's admissible
T is 0.05, where it catches 38.9% of junk against the control's 40.4% at its
own admissible 0.03 — for a new vCA1 animal the blend is a ranking
improvement, not an auto-reject improvement. Per animal (control → CNN →
blend), held out: pnb88 0.865 → 0.831 → 0.866; pnb97 0.919 → 0.927 → 0.930;
pnb98 0.912 → 0.927 → 0.926. GPU idle; nothing running.
