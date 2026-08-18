# Feature expansion Step 2 — results (2026-08-18)

**Outcome: the trace-feature hypothesis is CONFIRMED. Every promotion gate
passes.** Recommended candidate: `rankv2_35` = the deployed 13 + within-session
percentile ranks + 8 candidate-level trace/footprint features + a
v2-present flag. Deployment (Step 4 atomic swap) is NOT started — it needs its
own plan and sign-off.

## What was built

- **Extraction**: `.feature_expansion\` (dot-prefixed, scanner-invisible) under
  `D:\Julian_CNMFe\BLA` now holds C_raw + sparse A + Cn for the review set of
  **all 79 labeled agent sessions (79/79 OK, 0 failures)** — pulled from
  `review_neuron.mat` per the Step 0 unlock. Scripts: scratchpad
  `extract_step2.m`, session list from the pinned pool manifest.
- **8 v2 features per reviewed candidate** (`compute_v2_features.py` /
  `compute_v2b_features.py`): ev_rate, ev_snr, **ev_template_corr** (event
  stereotypy — the feature the autopsy named), ev_asym (decay/rise),
  ev_frac_plausible, nb_corr_max (corr with a high-confidence neighbor ≤60 px),
  nb_corr_any, ring_contrast (footprint vs surround on Cn). All computable at
  curation time.
- Two detector variants: **v2** (July's 2.5σ coincidence detector — fires on
  noise) and **v2b** (noise σ from the differenced trace, 3.5σ on a 3-frame
  smoothed trace, peak-z ≥ 5 qualification). v2b passes the mechanism check:
  the bla21 regression cells score 96th/85th pct stereotypy with event counts
  matching the visible transients; the drifty marginal cell drops to 14th pct.

## Integrity checks that shaped the numbers

1. **Neighbor-score leak found and fixed**: nb_corr_max originally selected
   high-conf neighbors by the deployed model's in-sample scores (label-
   informed). Recomputed with grouped-OOF scores (the honest offline analog of
   live scores; deployed scores only for the 4 sessions outside the OOF pool).
   Effect on results: negligible (the win was never the leak).
2. **Pool**: reviewed candidates only for agent sessions (v2 exists only for
   the review set; every real is reviewed) — a harder pool, so b13 reads
   0.8921 here vs 0.9099 on the full pool. All variants compared on the
   identical pool, ranks computed over the full candidate set (deploy-
   realistic) then subset. Bootstrap rows carry v2=0 + present=0.
3. Pool manifest asserted unchanged throughout; no session data modified; no
   retrains; watchers untouched.

## Results (8 seeds, StratifiedGroupKFold, deployed weights)

**Mixed arm (agent + bootstrap = deployment-realistic):**

| variant | AUC | ΔAUC | false-AR @ matched junk | junk @ matched false-AR |
|---|---|---|---|---|
| b13 (reference) | 0.8921 ± 0.0027 | — | 0.86% | 15.4% |
| rank26 | 0.8974 ± 0.0031 | +0.0053 | 0.37% | 25.0% |
| v2_22 | 0.9107 ± 0.0014 | +0.0186 | 0.17% | 32.2% |
| **rankv2_35** | **0.9137 ± 0.0022** | **+0.0216** | **0.11%** | **34.9%** |
| v2b_22 | 0.9088 ± 0.0014 | +0.0168 | 0.18% | 30.4% |
| rankv2b_35 | 0.9127 ± 0.0023 | +0.0206 | 0.12% | 32.3% |

Agent-only arm reproduces the ordering (rankv2_35 = 0.9160 ± 0.0021), so the
gain does not depend on the bootstrap rows.

**Promotion gates (deterministic holdouts, mixed arm):**

| variant | bla12 | bla16 | bla21 | bla36 | bla37 | bla8 | early-era (16 sessions) |
|---|---|---|---|---|---|---|---|
| b13 | 0.903 | 0.893 | 0.830 | 0.887 | 0.891 | 0.885 | 0.844 |
| rank26 | 0.900 | 0.888 | 0.831 | 0.883 | 0.905 | 0.882 | 0.845 |
| **rankv2_35** | 0.914 | 0.893 | **0.876** | 0.916 | 0.915 | 0.909 | **0.874** |
| rankv2b_35 | 0.914 | 0.909 | 0.850 | 0.916 | 0.914 | 0.911 | 0.876 |

- **≥ +0.005 AUC:** passed 4× over (+0.0216).
- **Operating point:** false-AR at matched junk 0.86% → 0.11% (8×); junk at
  matched false-AR 15.4% → 34.9% (2.3×).
- **Leave-one-animal-out:** improves or holds on ALL six animals — including
  bla21 0.830 → 0.876, the regression animal, and it is portable in exactly
  the way every motion feature was not.
- **Early-era holdout:** 0.844 → 0.874 — the autopsy's regression class
  demonstrably improves, not worsens.
- Caveat stated plainly: the fixed-window count of reals in [0.05, 0.12) is
  LARGER under v2 (score distributions recalibrate); the matched-operating-
  point numbers above are the calibration-fair comparison, and the deploy
  threshold must be re-derived from the new model's sweep (already required by
  the Step 4 protocol).
- rank26 alone largely fails LOAO (per-animal ~flat) — the pooled rank gain is
  partly within-animal; ranks earn their keep only combined with v2.

**v2 vs v2b:** near-tie (0.9137 vs 0.9127). v2's noisy event features still
help XGB, but only v2b measures what it claims (mechanism check) and v2b wins
bla16 (+0.016) while v2 wins bla21 (+0.026, n=2 sessions). Recommendation:
carry BOTH event-feature sets into the Step 4 candidate (or re-run the pair
with 8-seed significance before choosing) — decide at the deploy review.

## What Step 4 (not started) requires

Atomic swap per the handoff §4, now concretely: extend `features.py` with the
v2 computation (traces/A/Cn are all in curator memory at package time) + ranks;
update the two lockstep re-implementations (`train_classifier.py` retro path,
`bootstrap_preagent.py`); regenerate every `candidate_features.npz` (agent
sessions from `.feature_expansion\` — already extracted; bootstrap = 13 + ranks
+ zeros + flag); retrain `--prospective-only --model xgboost`; re-sweep the
threshold; restart watchers; coordinate vCA1/DG_AL (shared `features.py`).
Backfill note: historical agent npz get v2 for reviewed rows only
(auto-rejected rows = 0 + flag), production computes v2 for the full candidate
set — the flag column makes this explicit to the model.

## Files

Scratchpad (session 6d3eba36…): `extract_step2.m`, `extract_step2_log.txt`,
`compute_v2_features.py`, `compute_v2b_features.py`, `step2_v2_features.npz`,
`step2_v2b_features.npz`, `step2_eval.py`, `step2_eval.json`,
`step2_eval_output2.txt`, `harness.py`. Extraction data:
`D:\Julian_CNMFe\BLA\.feature_expansion\*.mat` (79 files, reusable for the
Step 4 backfill).
