# Feature-expansion gate report — Steps 0–1 complete (2026-08-18)

Recommendation up front: **PROCEED to Step 2 planning**, with the feature list
re-ranked by autopsy evidence (temporal event-shape features first), and with
**per-session rank augmentation carried forward as a zero-extraction candidate**
that already meets the operating-point bar on its own. The deploy-side
calibration lever (per-session thresholds) is a clean negative — drop it.

All numbers below were computed this session on the pinned pool
(`pool_manifest.json`: 170 sessions = 79 agent + 91 bootstrap; no drift during
the analysis). Scripts and raw outputs live beside this file.

## 1. Baseline re-pin (corrected vs legacy harness)

| harness | AUC (8 seeds) | false-AR@0.12 | junk@0.12 | reals [0.05,0.12) | reals <0.05 | masked rows |
|---|---|---|---|---|---|---|
| corrected (real ambiguous mask) | **0.9099 ± 0.0017** | 0.83% ± 0.09 | 27.8% ± 0.3 | 19.6 | 0.2 | **1123** |
| legacy (mask all-False bug) | 0.9095 ± 0.0014 | 0.83% ± 0.12 | 27.9% ± 0.4 | 19.4 | 0.5 | 0 |

The harness bug was real but **immaterial** (ΔAUC +0.0004, inside seed noise).
The corrected mask zeroes exactly 1123 bootstrap rows — matching the deployed
joblib's `n_excluded_ambiguous=1123`, so the harness now provably reproduces
deployed weighting. **Baseline B = 0.9099 ± 0.0017; 0.83% / 27.8% @0.12.**
The handoff §1 numbers stand.

## 2. Step 0 — review_neuron.mat unlock: **PASS (exact, all eras)**

| session | era | animal/task | expected N (cand−autorej) | N in review_neuron.mat | A, C_raw |
|---|---|---|---|---|---|
| 2tones/093025-bla16-319um | 2026-03-09 | bla16/2tones | 67 (99−32) | **67 = exact** | populated, 0 zero-rows |
| 2tones/100925-bla21-266um | 2026-05-05 | bla21/2tones | 80 (83−3) | **80 = exact** | populated, 0 zero-rows |
| 3odor/072326-bla36-670um | 2026-08-14 | bla36/3odor | 126 (163−37) | **126 = exact** | populated, 0 zero-rows |

All three sessions (3 animals, 2 tasks, autorej>0 so the count discriminates)
hold the full pre-review candidate set — no full-copy fallbacks, no non-finite
values. With the 8 motion sessions proven in July, that is 11/11 across
Oct-2025 → Aug-2026. Additionally, all 38 eligible agent sessions scanned have
the file on disk. **Candidate-level A/C_raw backfill is available for the whole
agent pool; the July "forward-only" constraint is void for agent sessions.**
(Bootstrap sessions still have no candidate data — as scoped.)

## 3. Error autopsy — 16 false-AR cells, all 16 inspected in their review PDFs

Set definition: real-labeled candidates with 8-seed-mean OOF < 0.12.
**16 of 2394 reals (0.67%), spread over 15 sessions** — no session or animal
concentration (bla12 4, bla37 4, bla16 3, bla21 2, bla8 2, bla36 1). Zero are
motion-tagged. 14/16 are stable across seeds (≥6/8 below 0.12).

**The drift narrative is inverted**: established-era reals run 0.89% false-AR
vs 0.38% for recent (post-2026-07-30) — the pressure sits in the older corpus,
not the incoming batches. Mechanism confirmed by the marquee finding:

- **Model-regression class (2 cells, the two worst errors).** bla21
  2tones/093025 Neurons 22 & 25: textbook transient traces, and the
  curation-time model printed **score 0.97 / 1.00** on their PDF pages — the
  current model scores them **0.06 / 0.10**. As the pool filled with recent
  bla36/37 3odor/6odor sessions, the model moved against early-era cells
  (their session-relative feature profile — e.g. max_weight percentile 2–4 —
  no longer matches what "real" looks like in the newer data).
- **Atypical-temporal-pattern class (~5 cells).** Clearly real by trace, but
  with event structure the 13 collapsed features misread: fast-spiking
  (101625-bla12 N25, 071626-bla37 N148), state-switch onset (070226-bla37
  N134, active only after frame ~4800), late-burst (071526-bla36 N60),
  clean-but-sparse (052726-bla37 N167, ~6 events in 17k frames). SHAP confirms:
  skewness (−1.03), circularity (−0.46), max_weight (−0.30) do the dragging.
- **Genuinely marginal class (~9 cells).** Noisy/drifty traces where the keep
  was plausibly video-informed (e.g. 100125-bla16 N64, 100925-bla8 N127 — a
  merge-candidate with no clean transients, arguably a mislabel). These looked
  marginal to the curation-time model too (scores 0.13–0.21 then).

**The missing feature named by the evidence: per-event shape/stereotypy** —
"does this trace contain repeated, stereotyped GCaMP-like transients?" — which
separates the atypical-temporal class AND the regression class from junk, and
is computable from the now-unlocked review_neuron.mat C_raw. This was already
Step 2's top-ranked candidate; the autopsy promotes it decisively.

## 4. Zero-cost experiments (8-seed OOF, paired per seed vs B)

| variant | AUC | ΔAUC | false-AR @ matched junk (27.8%) | junk @ matched false-AR (0.83%) |
|---|---|---|---|---|
| baseline B (13) | 0.9099 ± 0.0017 | — | 0.83% | 28.6% |
| **rank_aug_26** (13 + within-session pct ranks) | **0.9135 ± 0.0017** | **+0.0036** | **0.53% ± 0.12** | **34.8% ± 1.9** |
| robustz_aug_26 | 0.9123 ± 0.0018 | +0.0024 | 0.56% | 33.7% |
| sess_agg_19 (13 + session median/IQR context) | 0.9090 ± 0.0020 | −0.0009 | **0.43% ± 0.08** | **37.3% ± 2.3** |
| rank_replace_13 | 0.8941 | −0.0158 | 0.79% | 29.7% |

- **Session-relative context is the missing ingredient** — exactly what the
  regression-class autopsy predicts. Augmenting (never replacing) the raw 13
  with within-session ranks clears the zero-cost gate (+0.0036 > +0.003) and
  the operating point improves dramatically: −36% false-AR at matched junk, or
  +6.2 pts junk at matched false-AR.
- sess_agg is AUC-flat but improves the low-score operating region even more
  (0.43% / 37.3%) — worth combining with rank_aug in Step 2's evaluation.
- These variants are deployable with **zero new extraction** (computed from the
  npz at curation time, backfillable for bootstrap too) — but deployment still
  requires the Step 3 gates (leave-one-animal-out, 26-col contract swap).

LOFO ablation (3-seed): no redundant features — every drop ≤ +0.0006; most
load-bearing are motion_correlation (−0.0065), skewness (−0.0047),
cn_correlation (−0.0040). The 13 stay.

## 5. Session-threshold calibration rules — **clean negative, drop this lever**

All shrink-side rules (min(0.12, Pq), bottom-r%, robust-z caps) only trade junk
for false-AR at ~20:1 against; all expand-side rules (clip(Pq, lo, hi)) blow
past 1% false-AR (1.46–1.63%) with 13–14% worst-session false-AR for junk gains
a plain global threshold bump matches. No session-relative *threshold* beats
fixed 0.12 at any acceptable operating point. Ranking, not calibration, is the
binding constraint — consistent with the rank-*feature* result: session context
helps inside the model, not at the cutoff.

## Gate decision (per handoff §6, re-anchored to B)

- Step 0: **PASS** (exact, all eras) → the data unlock for temporal features is real.
- Best zero-cost ΔAUC **+0.0036 ≥ +0.003** AND the autopsy names a concrete
  missing signal per error class → **PROCEED to Step 2 planning.**
- CALIBRATE outcome: **NO** — per-session thresholds are a validated dead end.
- Baseline delta vs handoff < 0.003 → handoff §1 numbers stand unchanged.

**Recommended Step 2 shape (for the regroup):**
1. Carry rank_aug_26 (and rank+sess_agg combos) into the Step 2 evaluation as
   feature set candidates — they need no extraction at all.
2. Build the review_neuron.mat extraction (extend `agent/extract_cand_traces.m`
   to save A + C_raw over the 79 agent sessions) and compute the top-ranked
   trace features: event-shape consistency / transient stereotypy first,
   kinetics plausibility second, neighborhood corroboration third.
3. Step 3 gates unchanged: ≥ +0.005 AUC over B *or* operating-point win at
   equal AUC, leave-one-animal-out (bla36/37 held out), boundary-band shrink,
   agent-only and agent+bootstrap variants.
4. The 2 regression-class cells suggest one extra Step 3 check: per-era
   false-AR (early-2026 sessions held out) so a new feature set demonstrably
   fixes, not worsens, the early-era regression.

## Verification notes

- Pool manifest asserted unchanged before every script; no drift occurred.
- No writes outside the scratchpad; session dirs untouched (MATLAB run was
  load-only); no retrains issued; stash untouched; watchers left running.
- Deployed joblib (xgboost @0.12, n_sessions 170) untouched and re-verified.
- Raw outputs: repin_output.txt, baseline_repin.json, baseline_oof.npz,
  autopsy_false_ar.csv, far_pages/*.png (16 rendered PDF pages),
  exp_rank_norm.json, exp_feature_attrib.json, exp_session_threshold.json,
  step0_matlab_report.txt, step0_eligible_sessions.txt.
