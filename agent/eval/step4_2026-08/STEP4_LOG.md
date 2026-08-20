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

## Step 5 — threshold sweep + reference eval on the parallel files
(pending)

## Freeze window (Steps 6–8)
(pending)
