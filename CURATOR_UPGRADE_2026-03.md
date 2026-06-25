# CNMFe Neuron Curator: Pre- vs Post-Bootstrap Upgrade Summary
*March 2026*

---

## Background

The curator pipeline automatically scores each CNMFe candidate neuron and either
auto-rejects obvious garbage or passes it to a human reviewer in MATLAB. The goal
is to reduce review burden without silently discarding real neurons.

The critical metric is the **false auto-reject rate**: the fraction of genuine neurons
discarded without ever being shown to a human reviewer. This is the worst outcome —
lost data you will never know existed. The secondary metric is **garbage auto-rejected**:
bad candidates removed automatically, saving review time.

---

## What Changed

### 1. Training data: human-reviewed sessions only → human-reviewed + bootstrap sessions

**Before:** Logistic Regression (LR) trained exclusively on **23 manually reviewed
sessions** (594 confirmed neurons, 3,894 garbage = 4,488 total candidates). These are
gold-standard labels — every candidate was inspected and decided by a human.

**After:** XGBoost trained on **105 sessions total**:

| Data source | Sessions | Candidates | Confirmed neurons | Label quality |
|---|---|---|---|---|
| Human-reviewed (agent) | 23 | 4,488 | 594 | Gold standard |
| Bootstrap (auto-labeled) | 82 | 29,561 | 1,485 | Noisy (see below) |
| **Total (active)** | **105** | **~33,100** | **2,079** | |

The 82 **bootstrap sessions** are recordings where CNMFe ran but no full human review
occurred. Rather than review them manually, we labeled candidates automatically by
re-running CNMFe headless (without expert guidance) on each session's raw .tif file and
then comparing the resulting candidate footprints to the curated neuron footprints from
the original expert-guided run on the same file. If a candidate's spatial footprint
overlaps more than 45% with a confirmed neuron's footprint (measured as cosine similarity
of pixel weights), it is labeled "keep."

**Why ~40% of real neurons cannot be recovered this way:**

The root cause is CNMFe merging during the fresh headless run. During initialization on
400–500 candidates, CNMFe aggressively merges spatially overlapping components. When a
real neuron gets merged with an adjacent noise component or neighboring neuron, the
resulting footprint becomes a blurred hybrid. The cosine similarity between this merged
footprint and the original clean curated footprint drops to 0.15–0.30 — below the 0.45
threshold — even though it is the same physical neuron in the same recording. These
neurons are lost as clean individual components in the headless run. This is a fundamental
limitation of running CNMFe without expert parameter tuning, not a fixable matching
problem.

We validated this extensively. Temporal matching (spike trains, raw calcium traces) was
tested as an alternative but failed: CNMFe's deconvolved traces are not stable identifiers
across independent runs because the entire matrix factorization converges to a different
solution each time. Only the spatial footprints (A matrix), anchored to physical cell
morphology, are consistent across runs. Spatial-only matching at threshold 0.45 is the
correct and final approach.

The ~40% unrecovered neurons become false-negative contamination in the bootstrap training
labels (labeled as "garbage" when they are actually real neurons). We handle this through
three mitigations: (1) human-reviewed sessions are up-weighted 2.57× to counteract
bootstrap volume; (2) the 18 worst-contaminated sessions (recovery < 40%) are
down-weighted 0.4×; (3) the 1,050 specific candidates identified as likely
false-negative contamination are excluded from training entirely.

The 82 bootstrap sessions span: 46 × 3odor (56%), 11 × CTA, 12 × WSE, 8 × Valence,
3 × Block_Valence, 2 × 4odorDO — covering animals at later experimental timepoints
with lower signal-to-noise, fewer active cells, and less stable recordings than the
human-reviewed sessions.

---

### 2. Model: Logistic Regression → XGBoost

**What is XGBoost?** XGBoost (eXtreme Gradient Boosting) is a decision-tree ensemble.
Rather than fitting a single linear boundary through the feature space (as logistic
regression does), it builds hundreds of small decision trees sequentially, each one
correcting errors of the previous. This lets it capture non-linear interactions between
features — for example, a neuron that looks borderline on SNR alone but is clearly real
given its combination of circularity, skewness, and spatial compactness. It is the
standard high-performance method for tabular/structured data in machine learning.

XGBoost was selected via 3-way grouped cross-validation against LR and LightGBM. XGBoost
and LightGBM tied on performance; XGBoost was chosen.

**Discrimination accuracy**, measured by AUC (Area Under the ROC Curve — 1.0 = perfect
separation, 0.5 = random) using 5-fold cross-validation on held-out human-reviewed
sessions. Only human-reviewed sessions are used as test folds because bootstrap labels
are too noisy to evaluate against:

| Model | AUC on human-reviewed test folds |
|---|---|
| Logistic Regression | 0.772 ± 0.042 |
| **XGBoost** | **0.843 ± 0.040** |

XGBoost improves AUC by **+0.071**, approximately 1.7 standard deviations. Separately
validated on the original 12 sessions used to build the first LR model: AUC 0.888
(XGBoost) vs 0.862 (LR, +0.026).

**Did bootstrap improve model accuracy?** Adding bootstrap data to XGBoost slightly lowers
raw AUC by 0.015 on human-reviewed test folds. This is expected: bootstrap positives look
different from reviewed positives (lower SNR, less circular footprints), which slightly
shifts the model's decision boundary. However, bootstrap training is *required* for a
different and more important reason: **probability calibration**. XGBoost trained on
human-reviewed sessions only assigns unreliably low probability scores to borderline
candidates — it cannot be safely thresholded at any value without unacceptable false
auto-rejects. Adding 82 bootstrap sessions, with their wider diversity of positive and
negative examples, stabilizes the probability scale so that a meaningful threshold can be
set.

---

### 3. Auto-reject threshold

The model outputs a **score between 0 and 1**: its estimated probability that a candidate
is a genuine neuron. The **threshold** is the cutoff below which a candidate is
automatically rejected without human review. A candidate scoring 0.08, for example, means
the model estimates an 8% chance it is a real neuron — almost certainly garbage. A
candidate scoring 0.65 is uncertain enough that a human should decide.

Threshold values are not directly comparable between LR and XGBoost because the two
models calibrate their probability scales differently. We derived the correct XGBoost
threshold via a 5-fold out-of-fold sweep on human-reviewed sessions, computing the false
auto-reject and garbage-caught rates at every possible threshold value:

| Threshold | False auto-reject rate | Garbage auto-rejected | Notes |
|---|---|---|---|
| **LR @ 0.10 (old baseline)** | 1.1% | 4.1% | |
| XGB @ 0.10 | 0.8% | 5.9% | Strictly better than LR on both metrics |
| **XGB @ 0.11 (deployed)** | **0.8%** | **7.1%** | **Chosen — see below** |
| XGB @ 0.12 | 1.1% | 8.0% | First threshold that costs false-AR |
| XGB @ 0.15 | 1.8% | 12.2% | 3× garbage, small false-AR increase |
| XGB @ 0.19 | 3.2% | 16.7% | Efficiency drops sharply — do not exceed |

**Why 0.11?** The step from threshold 0.10 to 0.11 catches more garbage at zero additional
false auto-reject cost: no real neuron moves below the threshold in this range (the
efficiency — extra garbage caught per unit increase in false-AR — is infinite). At 0.12,
false-AR increases for the first time. Given the priority of minimizing data loss, we stop
at 0.11. This threshold should be re-evaluated as more sessions are reviewed and added to
training.

---

## Net Result

| Metric | Old (LR @ 0.10) | New (XGB @ 0.11) |
|---|---|---|
| Model AUC (human-reviewed test folds) | 0.772 | **0.843** |
| False auto-reject rate | 1.1% | **0.8%** |
| Garbage auto-rejected | 4.1% | **7.1%** |
| Real neurons silently lost / session | ~0.22 | **~0.16** |
| Garbage removed automatically / session | ~5.7 | **~9.9** |

The new model simultaneously loses fewer real neurons **and** removes more garbage
automatically — no trade-off. The small increase in human review burden (~10–15 more
candidates per session versus the old LR) reflects that the model is appropriately more
conservative about borderline cases when false-AR is the priority. As more sessions are
reviewed and added to training, that burden will decrease.

---

## What Is Next

- 4 sessions currently queued for MATLAB review with updated packages
- After each MATLAB review, the watcher auto-retrains on the expanded dataset
- Revisit threshold analysis after ~10 more human-reviewed sessions are added — the
  efficient threshold will naturally rise as the model sharpens
