# Step 4 deploy log (feature-expansion, 35-column BLA contract)

Branch `feature-expansion-step4` off `dg-al-cold-start` (2ae3950).
Plan approved 2026-08-20. All numbers below computed fresh this deploy.

## Step 1 — provenance guard (commit 7297860)
`recurate_sessions.py` now refuses any session with labels.mat (explicit-path
mode could previously regenerate review_neuron.mat post-review). Inherited by
the vCA1/DG_AL wrappers.

## Step 2 — features.py v2 + lockstep copies (commit d04747d)
- `parity_check.py` phase 1: shipping `compute_v2b_features` vs pinned
  `step2_v2b_features.npz` on **79/79 sessions: max|diff| = 0** (gate was
  allclose rtol 1e-6). The deployed joblib (mtime 2026-08-18 01:10) predates
  the pin (03:47) — no retrain since, so the 4 non-OOF sessions' hiconf
  source is exactly the pin's.
- phase 2: `load_spatial` geometry == extraction pixel-vector convention on 2
  pending sessions (order-F rel diff ≤ 7.3e-06 = A.txt text round-off;
  order-C rejected at 1.0).
- Lockstep copies: trainer retro path refuses under v2; bootstrap_preagent
  assembles 13+ranks+zeros+flag0.

## Step 3 — curator two-pass + trainer companion model, dormant (commit 3f86d7f)
- Unit checks: arity guard (12 rejected / 13 accepted vs live scaler);
  vCA1+DG_AL joblibs load unchanged, `_load_first_pass` → None; two-pass
  construction (first 13 bit-identical, flag=1, end-to-end arity error when
  scoring 13 cols vs a 35 model); one-class fallback fits + scores at 35.
- Behavioral invariance: worktree A/B vs `dg-al-cold-start` with the real
  deployed joblibs — vCA1 (N=15) and DG_AL (N=127) extraction + scoring
  **bit-identical** (fm, overlap, scores, thr, names, model_type).

## Step 4 — re-pin + backfill (this commit)
- `repin_manifest.py`: live pool = **170 labeled sessions (79 agent / 91
  bootstrap), ZERO drift vs the Step 2 pin** (0 new / 0 gone / 0 changed).
  No new extractions needed. Step 2 / red-team numbers remain directly
  comparable.
- 12 pending sessions (all bla36; 8 3odor, 3 CTA, 1 Valence), all with
  candidate files intact.
- `backfill_v2.py`: **182 `candidate_features_v2.npz` written (79 labeled +
  91 bootstrap + 12 pending), 47,886 rows.** Hard checks per session, all
  passed: v1 width 13 in / 35 out; row count unchanged; first 13 columns
  bit-identical to v1; ranks identical on recompute; flag/zero patterns
  (labeled: flag=1 exactly on reviewed rows, zeros+flag=0 on auto-rejected;
  bootstrap: all flag=0+zeros; pending: all flag=1, auto_rejected verbatim);
  and the written v2b of all **79 labeled sessions re-verified against the
  pinned values**.
- Hiconf sources: labeled = pinned grouped-OOF (deployed scores for the 4
  non-OOF sessions, matching the pin); pending = current deployed 13-col
  model (production-realistic; red-team hole #4 bounded-imprecision option).
- `record_preswap_scores.py`: old-model scores for all 182 sessions →
  `preswap_scores.npz` (not committed; regenerable from the v1 backup +
  old joblib). Deployed old model: xgboost @ 0.12, n_sessions=170.
- **Smoke-test cells confirmed** in 2tones/AVG5x-TSeries-093025-bla21
  (98 candidates, all reviewed, 50 reals): candidate idx 21 ("Neuron 22")
  pinned-baseline OOF **0.064**, idx 24 ("Neuron 25") **0.101** — the
  autopsy's 0.06/0.10 regression cells. (Deployed in-sample scores are 0.53+
  because the model trained on this session; the smoke test therefore checks
  the OOF analog primarily and the deployed in-sample score secondarily.)

## Step 5 — threshold sweep + reference eval on the parallel files (this commit)

`threshold_sweep_v2.py` (8 seeds × 5-fold grouped OOF × {b13, rankv2b_35},
threshold_robustness.py methodology, arm-b corpus from the parallel v2 files;
OOF pool 12,677 rows / 2,394 real / 75 agent sessions, bootstrap always in
train):

- **b13 full-pool AUC 0.9099 ± 0.0017** — exact reproduction of the pinned
  baseline; reviewed 0.8927 ± 0.0020 (red team: 0.8921).
- **rankv2b_35 reviewed AUC 0.9123 ± 0.0015** — exactly the red team's C10
  arm-b figure; full-pool 0.9246 ± 0.0014.
- **Paired delta (reviewed) +0.0197, min +0.0176, positive on all 8 seeds**
  → reference gate (≥ +0.015, all seeds positive) PASSES.
- Threshold table (false-AR / junk-full / junk-reviewed, mean over 8 seeds):
  0.05 → 0.60% / 28.7 / 27.8;  **0.06 → 0.80% / 33.8 / 31.8 (CHOSEN)**;
  0.07 → 1.06% / 38.2 / 35.2;  0.12 → 2.18% (confirms the red team's "0.12
  would be 2.2%").  Gate at T=0.06 (far ≤ 1%, junk ≥ 30%): **PASS**.
  Matches red-team C9 to the decimal.

### Smoke-cell finding (STOP-AND-REPORT before the freeze)

The bla21 autopsy cells do NOT clear the new threshold in the
deploy-realistic OOF: Neuron 22 **0.064 → 0.065** (a hair above T=0.06),
Neuron 25 **0.101 → 0.052** (below T). Verified pre-existing in the red
team's own pinned arm-b OOF (c10_oof_v2b.npz: 0.065 / 0.052 — my pipeline
reproduces it exactly), i.e. this was a property of the approved result that
no one had checked at the per-cell model-score level; the brief's "well above
the new threshold" expectation was untested. Context:

- Under the CURRENT deployment (b13 @ 0.12) both cells' OOF analogs
  (0.064 / 0.101) are below the deployed threshold — the status quo already
  auto-rejects both. The new deploy rescues Neuron 22 (barely) and still
  loses Neuron 25: strictly better on these cells, not the full rescue the
  autopsy narrative implied.
- Under v2 (rejected by red-team C8 for the bla16 tie + noise-proxy
  mechanism): Neuron 25 OOF 0.097, Neuron 22 0.062 — v2 rescues neither at
  T≈0.06 either (0.097 > 0.06 — one cell better, same structure).
- The literal Step 7.6 test (deployed model in-sample, retrain-identical
  simulation `simulate_deployed_smoke.py`): 0.792 / 0.608 — would "pass",
  but in-sample scores are inflated for training rows; the OOF number is the
  honest deploy-realistic proxy for how such cells fare in FUTURE sessions.

Deploy paused for a user decision per the stop-and-report rule.

## Freeze window (Steps 6–8)
(pending user decision; tooling ready: `swap_v2.py`
backup/rehearse/swap/rollback with sha256 verification at every move,
`verify_deploy.py` for the post-retrain gate. Dry-run candidates not out for
review: CTA 080626/080926/081126, Valence 070326. The 8 3odor pending
sessions are out with Taylor since 2026-08-03 — untouched.)
