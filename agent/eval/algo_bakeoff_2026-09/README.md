# algo_bakeoff_2026-09 — is XGBoost the right learner for the 35-column contract?

Stage 1 of the plan approved 2026-09-06 (`~/.claude/plans/i-still-need-to-agile-backus.md`).
Question: on the SAME 35 tabular columns, with the SAME rows, weights, folds and
seeds as the deployed model, does any other tabular learner beat the deployed
XGBoost configuration (300 trees, depth 4, lr 0.05, subsample/colsample 0.8,
never tuned)?

Everything here is read-only with respect to the corpus: no session dir, npz,
joblib or label file is written. Outputs land in `results/`.

## Pre-registered decision rule (written before any contender ran)

A tabular change is adopted only if, against the **thread-pinned deployed
control** (`xgb_deployed`, 8 threads), under **both** session grouping and
animal grouping, on BLA:

1. reviewed-pool AUC delta >= +0.003 with 8/8 seeds positive, **and**
2. junk caught at the control's matched false-AR (T = 0.04) >= +2 percentage points.

vCA1 must not get worse under the same two measures. Anything short of that is
a null: recorded here, and the "was XGBoost configured right" question is closed.

Tuned contenders (`*_tuned`) are evaluated by NESTED selection: each outer fold
picks its configuration on an inner session-grouped split of the outer-train
rows only; the outer test fold never informs the choice. Their 3-seed numbers are
the honest ones. The single winning configuration is then re-run as a fixed
contender for 8 seeds (`*_winner`); that number carries mild selection optimism
and is reported as such.

## Protocol (mirrors `agent/eval/bootstrap_redteam_2026-08/rt_lib.py`)

- Pool: every labeled session under `D:\Julian_CNMFe\<AREA>`; CV pool = agent
  sessions with >= 5 reals (full candidate set, auto-rejected rows as label 0);
  bootstrap sessions always in train with their native weights (0.4 for bad
  recovery, 0 for ambiguous/duplicate rows kept in the fit call as in the
  harness); per-fold agent weight = override (vCA1 5.0) or max(sqrt(n_bs/n_ag), 4.0).
- Folds: StratifiedGroupKFold(5, shuffle, random_state = seed), seeds
  [42, 1, 7, 13, 100, 2024, 31337, 9]; model seed 42 unless the split is fixed
  (n_folds == n_groups), in which case the model seed varies per replicate.
- Animal grouping: folds over animals, and the test animals' own bootstrap
  sessions are dropped from training (the a08/a18 convention; NOTE
  `pareto_2026-08` kept them, so its animal-grouped numbers are not this protocol).
- StandardScaler fit on the training block for every contender (the harness
  convention; a no-op for trees, needed by the MLP).
- Class balance: xgboost/lightgbm/catboost via scale_pos_weight = weighted
  neg/pos over w > 0; sklearn learners by multiplying positive sample weights by
  the same ratio.
- Threads pinned: `--threads 8` sets OMP/MKL/OPENBLAS_NUM_THREADS before numpy
  is imported and n_jobs/thread_count on every learner; the count is written
  into every result file. `--threads 0` = unpinned (n_jobs = -1), used only for
  the reproduction check against `pareto_2026-08/bla_deployed_oof.npz`.
- Pool snapshot: the first run per area writes `results/pool_<AREA>.json`
  (session list + labels.mat sha256); later runs refuse to proceed if a session
  is missing or its labels changed, so a watcher ingest mid-bake-off cannot
  silently change the rows under a contender.

## Contenders

| name | learner | notes |
|---|---|---|
| `xgb_deployed` | `rt_lib.make_xgb` verbatim | the control; run once unpinned (repro) and once pinned (reference) |
| `xgb_tuned` | XGBoost, nested random search (30 configs, early stopping) | depth 3–8, lr 0.02–0.10, min_child_weight, subsample, colsample, reg_lambda, gamma |
| `xgb_winner` | the config `xgb_tuned` selected most often | fixed, 8 seeds |
| `lgbm_deployed` | LightGBM mirror of `train_classifier._make_clf("lightgbm")` | |
| `lgbm_tuned` / `lgbm_winner` | same search protocol | |
| `cat_default` | CatBoost defaults (1000 it, lr 0.05, depth 6) | ACORN env only (python at `C:/Users/julia/.conda/envs/ACORN/python.exe`; created 2026-09-06 with the numeric stack pinned to the valence versions, catboost 1.2.10, torch 2.11.0+cu128) |
| `cat_tuned` / `cat_winner` | 10-config nested search | ACORN env only |
| `histgb` | sklearn HistGradientBoosting (300 it, lr 0.05, depth 4) | no new dependency |
| `extratrees` | sklearn ExtraTrees (500 trees, min_samples_leaf 5) | control |
| `mlp` | sklearn MLP (128, 64), early stopping | the "neural net on the 35 features" control |
| `ens_mean` | probability mean of xgb_deployed + lgbm_deployed (+ cat_default) | computed in `--report` from saved OOFs |

## Metrics per contender, area, grouping

reviewed-pool AUC and full AUC (mean, sd, per seed); paired delta vs the pinned
control; `rule_T` (largest T in [0.03, 0.10] with mean false-AR <= 0.85% and
worst-seed <= 1.0%, gate junk >= 30%); `matched_operating_point` vs the control at
T = 0.04 (false-AR at matched junk, junk at matched false-AR); runtime.

## Usage

```
python bakeoff_tabular.py --area BLA --repro --threads 0          # reproduce pareto's OOF
python bakeoff_tabular.py --area BLA --grouping session,animal --contenders xgb_deployed,lgbm_deployed,histgb,extratrees,mlp
python bakeoff_tabular.py --area BLA --grouping session,animal --contenders xgb_tuned,lgbm_tuned --seeds 3
python bakeoff_tabular.py --area BLA --grouping session,animal --winner-of xgb_tuned,lgbm_tuned
python bakeoff_tabular.py --area BLA --report
```

Results and the bake-off table: see `RESULTS.md` (written after the runs).

## Round 2 (2026-09-07): CNN training-regime and input variants — pre-registered before any run

**Why.** The Stage 2 verdict (RESULTS.md): the 50/50 blend gain is real but the
gate was not met. `cnn_seed_variance.py` (results/cnn_seed_variance.txt) then
showed (1) the standalone shortfall is mostly *variance* — averaging the 8 seeds'
OOF lifts the CNN alone by +0.011 (BLA) / +0.012 (vCA1 session) / +0.022 (vCA1
animal) while XGBoost gains +0.002 — and (2) the inner early-stopping split is
20–516 reviewed rows on vCA1 (val−test gap +0.012/+0.031 vs +0.003 on BLA).
The CNN sees none of the 35 columns; 17 of them are not reconstructible from
its inputs (13 within-session ranks, nb_corr_max/any, motion_correlation, flag).

**Screening protocol.** 3 seeds (42, 1, 7) on two cells: BLA session-grouped
(the near miss) and vCA1 animal-grouped (the failure). Folds, rows, weights and
model seeds identical to Stage 2 (`train_cnn2.py` mirrors `train_cnn.py`), so
every OOF pairs 1:1 with the round-1 CNN at the same seeds and with the control.
One change per variant:

| variant | change vs round 1 |
|---|---|
| `valfix` | inner early-stopping split also holds out 1/8 of the fold's bootstrap **sessions** (they leave that fold's training block), so the AUC that picks the epoch is computed on thousands of rows; epoch cap 25 → 40 |
| `fixed12` | no early stopping: 12 epochs, cosine LR decay, every session trains (12 ≈ round-1 median best epoch 10 plus margin) |
| `ens3` | 3 members per fold (different init seeds, identical data), probability mean — does within-fold ensembling capture the cross-seed averaging gain? |
| `nbtrace` | 2-channel trace input: own + the trace of the max-correlated neighbour within 60 px centroid distance (`features.NB_DIST`), zeros when none — the raw form of nb_corr_max/any; corpus `v2` |
| `patch48` | 48×48 footprint patches (a gSig-9 window is 54 px; 32 px downsamples it 1.7×); corpus `v2_p48` |
| dropped | background-signal channel: `Ybg_mean.mat` exists for 0 of 202 bootstrap sessions (the majority of training rows) and 112 of 116 agent sessions |

**Screening metric.** CNN-alone reviewed AUC, paired per seed against the round-1
CNN on the same rows. Secondary: the 50/50 blend vs the control and the stacked
model vs the control at the matched operating point.
**Selection rule.** A variant is carried forward if its CNN-alone reviewed AUC
is ≥ the round-1 CNN's on **both** cells (3-seed mean) and ≥ +0.003 on at least
one. Carried-forward regime and input changes are combined into one
configuration; ensembling stacks on top of whatever wins.
**Confirmation.** The combined configuration at 8 seeds on all four cells,
judged by the **unchanged** Stage 2 gate (BLA under both groupings: reviewed AUC
≥ +0.005 on 8/8 seeds, junk at matched false-AR ≥ +3 pp, `rule_T` under animal
grouping with worst-seed false-AR ≤ 1%; vCA1 not worse), applied to the stacked
model and to the blend. Caveat recorded now: BLA session and vCA1 animal are
screening cells, so their confirmation numbers are partly in-sample with respect
to variant selection; the clean evidence is BLA animal, vCA1 session, and the 5
unseen seeds. Kill: no variant beats round 1 on either cell → write up, stop.

**Stacked model (plan step (c), `stack_xgb.py`).** The CNN's OOF score and its
within-session percentile rank as columns 36–37 of the deployed XGBoost config,
same folds and bootstrap block as the control. Bootstrap rows carry NaN in both
columns (the CNN trained on all of them, so no OOF score exists; xgboost routes
missing values natively). The level-0 scores are out-of-fold with respect to the
outer split only, not nested: the CNN that scored a training row was trained on
the outer test fold's rows. The size of that optimism is measured once (BLA
session, 1 seed, CNNs trained with pairs of folds held out) before any stacked
number is used for a decision.

**Budget** (measured 2026-09-07 on the idle GPU: 6.8 s per epoch of 37k rows,
identical per-sample cost at batch 256/512/1024 = compute-bound; cuDNN autotune
and concurrent processes buy nothing): ≈ 50 GPU-min per single-model screening
variant, ≈ 150 for `ens3`; confirmation ≈ 5.5 GPU-h per ensemble member.
Outputs under `results/r2/`.

**Amendment 2026-09-07 14:55 (before any round-2 CNN result was read).** The
36th-column stack on the round-1 scores (`results/r2/stack_r1.md`) is *worse*
than the fixed 50/50 blend in all four cells (BLA session +0.0034 vs +0.0045,
BLA animal +0.0032 vs +0.0078, vCA1 session +0.0040 vs +0.0093, vCA1 animal
−0.0009 vs +0.0081 reviewed AUC) and fails the gate everywhere; the 35-column
re-run through the same code path reproduces the pinned control to 4 decimals.
Since a level-0 leak can only inflate a stack, the leak measurement is moot for
this negative result and is not run (`stack_leak.py` stays available should the
stack be revisited). Likely cause, not proven: the CNN column is NaN on the
70–96% bootstrap rows, so its splits are learned on agent rows only and
"missing" doubles as a bootstrap flag. Added instead, pre-registered here:
`blend_fit.py` — a 3-parameter logistic combiner of logit(control OOF) and
logit(CNN OOF) fitted per outer fold on the training folds' agent rows
(nested by construction; too small to exploit the level-0 leak), reported
against the fixed 50/50 blend. The confirmation gate applies to whichever of
{fixed 0.5 blend, fitted combiner} is pre-declared after screening: the fitted
combiner if it beats the fixed blend on both screening cells, else the fixed
blend.

**Amendment 2026-09-07 15:00 (before any screening result was read).** The
Stage 2 gate says "vCA1 not worse" without a metric. Fixed now, for
`confirm_eval.py`: under both vCA1 groupings the blend's reviewed-AUC delta vs
the control must be ≥ 0 and its junk at matched false-AR ≥ −1 pp. The fitted
logistic combiner (`results/r2/blendfit_r1.md`) lost to the fixed 50/50 blend
in all four cells on the round-1 scores (BLA session −0.0010, BLA animal
−0.0045, vCA1 session −0.0016, vCA1 animal −0.0258 reviewed AUC vs the fixed
blend), so the fixed 0.5 probability blend is the pre-declared combiner for
confirmation.

**Amendment 2026-09-07 19:10 (after the four single-variant screens, before
`ens3` and before any combination result).** Screening (`results/r2/screen.md`,
3 seeds, paired CNN-alone reviewed AUC): `fixed12` carried (+0.0035 BLA
session 3/3, +0.0104 vCA1 animal 2/3); not carried by the rule: `valfix`
(−0.0022 BLA), `nbtrace` (+0.0075 BLA 3/3 but −0.0111 vCA1 animal 0/3),
`patch48` (+0.0026 BLA, −0.0182 vCA1 animal). `nbtrace` is nevertheless the
largest BLA gain measured, and its vCA1-animal loss was measured under the
round-1 regime, whose inner validation in that cell is 27–264 rows.
Pre-declared now: ONE combination screen, `fixed12+nbtrace`, 3 seeds, both
cells. It replaces `fixed12` as the confirmation configuration only if its
CNN-alone reviewed AUC is ≥ `fixed12`'s on BOTH cells (paired 3-seed mean);
otherwise the confirmation configuration is `fixed12`, plus `ens3` if `ens3`
is carried by the rule on its own. Confirmation gate unchanged. This is a
deviation from the single-variant rule for `nbtrace` and is labelled as such
in the write-up.

## Pooled BLA + vCA1 CNN (pre-registered 2026-09-08 09:40, before any run)

**Question (user's).** Is one CNN trained on both areas better for each area
than the per-area CNN? Tabular precedent (`agent/eval/global35_2026-08`,
8 seeds, other area appended at native weights): BLA session −0.0010, BLA
animal +0.0020, vCA1 session −0.0012, vCA1 animal **+0.0283** reviewed AUC —
null except for the 3-animal vCA1 leave-one-animal-out, where the extra data
helps. **Design.** `train_cnn_pooled.py`: per target cell the folds, rows,
seeds and target-area weights are exactly `train_cnn2.py`'s; every row of the
other area's corpus (agent + bootstrap, its own native weights, ×1.0) joins
the training block; the target fold is scored by the pooled model; the other
area is never in a test fold. Configuration `fixed12+nbtrace` (single model,
the confirmed member 0), **3 seeds** (42, 1, 7), all four cells, compared with
the per-area member-0 run at the same seeds (`pooled_eval.py`).
**Reading rule.** Pooling is adopted for an area only if the pooled CNN alone
beats the per-area CNN alone on reviewed AUC in both of that area's cells
(3-seed mean, paired) and the blend is not worse at the operating point;
anything else = keep per-area models (the deployment default). A +0.003 or
larger 3-seed result would be confirmed at 8 seeds before adoption.

## vCA1 refresh (pre-registered 2026-09-19, before any run)

**Why.** Since the 2026-09-06 snapshot the vCA1 pool grew from 142 to 158
labeled sessions: 16 new agent sessions, 15 of them pnb98 (tdTomato prep,
curated at threshold 0, every candidate human-labelled) and one pnb88; BLA
gained one session (not re-run: 1 of 177). Still three vCA1 animals.
**What is re-run, vCA1 only**, all under `results_0919/` (env
`BAKEOFF_RESULTS=results_0919`; the 09-06 results are untouched) on a fresh
pool snapshot and corpus `vCA1_raw_corpus_v3.h5` (`BAKEOFF_CORPUS_TAG_vCA1=v3`):
1. the pinned-thread deployed control and the fast tabular contenders
   (lgbm_deployed, histgb, mlp), both groupings, 8 seeds — answers the
   operator flag (is T=0.04 inside the 1% worst-seed rule now?) and whether
   Stage 1's null still holds;
2. the confirmed CNN configuration `fixed12+nbtrace+ens3`, 8 seeds, both
   groupings, judged by the same criteria as `confirm_eval.py` (vCA1: blend
   reviewed-AUC delta ≥ 0 and junk at matched false-AR ≥ −1 pp; reported
   against the BLA-strength criteria too: ≥ +0.005 on 8/8, ≥ +3 pp);
3. the pooled BLA+vCA1 CNN for the vCA1 cells (`fixed12+nbtrace`, 3 seeds,
   BLA corpus v2 unchanged), same reading rule as before.
No configuration choice is revisited; nothing is tuned on the new pool.
Expectation recorded now: more pnb98 data should shrink the per-seed variance,
raise the control under session grouping, and leave leave-one-animal-out
dominated by pnb88.
