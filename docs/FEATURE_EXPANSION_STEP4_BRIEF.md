# Step 4 execution brief: deploy the 35-column BLA feature contract

_Written 2026-08-18 for a fresh session on this machine. Prerequisites: the
Step 2 result (docs/FEATURE_EXPANSION_STEP2_2026-08-18.md) survived a
full adversarial review (agent/eval/redteam_2026-08/redteam_report.md) —
deploy-worthy with four changes, all baked into this brief. Read both before
writing any code. This is the pipeline's riskiest operation; work in plan mode
first and get the plan approved._

## Decisions already made (do not relitigate)

1. **Ship rankv2b_35**: the deployed 13 (bit-identical, same order) + 13
   within-session percentile ranks + 8 v2b trace/footprint features + a
   v2_present flag = 35 columns. v2b detector, not v2 (red-team C8: v2b
   improves every gate cell at every seed; v2 ties on bla16 and its event
   features are noise proxies). Reference implementation:
   `agent/eval/step2_2026-08/compute_v2b_features.py` (+ its base
   `compute_v2_features.py`); pinned expected values:
   `D:\Julian_CNMFe\BLA\.feature_expansion\_pinned\step2_v2b_features.npz`.
2. **Training option (b)** (red-team C10): auto-rejected agent rows STAY in
   training as label 0 with v2=0 + flag=0 (their traces are unrecoverable).
   Bootstrap rows: v2=0 + flag=0. Reviewed agent rows and ALL rows of newly
   curated sessions: real v2 values + flag=1. The flag↔label correlation in
   history was measured inert; excluding those rows degrades auto-reject
   competence (60% vs 84% of auto-rejected rows below threshold).
3. **Threshold is re-derived, never carried.** Expect ≈0.06 (red-team C9: the
   new model at 0.12 would run 2.2% false-AR). Derive from the ACTUAL
   retrained model with the 8-seed sweep methodology
   (`agent/eval/threshold_robustness.py`), pick the sub-1% false-AR posture,
   report the 0.03–0.10 table before deploying the number.
4. **Neighbor features stay, via two-pass scoring** (red-team's faithful fix
   for the chicken-and-egg): nb_corr_max needs high-confidence neighbor scores
   before the 35-col model can run. At retrain time, train a companion
   13-column model on the same corpus and store it IN THE SAME joblib (e.g.
   `first_pass_scaler`/`first_pass_clf`); curation becomes: pass 1 = 13-col
   scores → compute v2b (incl. nb features at score ≥ 0.5) + ranks → pass 2 =
   35-col final scores. One joblib = the swap stays atomic. (Fallback if this
   proves ugly: dropping both nb features costs −0.004 AUC — but that is a
   user decision, not yours.)

## Hard invariants (verify each; the red-team outline's per-step checks)

- **vCA1 and DG_AL must be completely unaffected.** `features.py` is shared;
  their deployed models are 13-column. The 35-col contract must be
  area-scoped (per-area feature version selected by the area config/wrapper —
  design it, but the invariant is: a vCA1/DG_AL curation run before and after
  this deploy produces byte-identical behavior). Their watchers keep running;
  only the BLA watcher stops for the swap window.
- **First 13 columns of every regenerated npz bit-identical to v1** (compare
  before overwriting; the ranks/v2b/flag are appended, never interleaved).
- **The two lockstep row-assembly copies** must produce the same 35-col BLA
  contract or be explicitly guarded: `train_classifier.py` retro path
  (~:230-239) and `bootstrap_preagent.py` (~:318-327). The watcher pins
  `--prospective-only` (retro path skipped), but a manual run without it must
  not silently emit 13-col rows into a 35-col corpus.
- **Backfill parity**: the shipping `features.py` v2b code, run against
  `D:\Julian_CNMFe\BLA\.feature_expansion\*.mat`, must reproduce the pinned
  `step2_v2b_features.npz` values (allclose, rtol 1e-6) before any npz is
  written. This proves the deployed code computes what was evaluated.
- **Curator scores positionally with no arity guard** (curator.py:159; the
  joblib stores no feature names). A half-swapped state = silently wrong
  scores. Hence: BLA watcher STOPPED before touching anything, swap fully
  (features.py + all npz + joblib together), restart after verification.
  While stopped, no `labels.mat` may be ingested (check the exchange is idle).
- Also check the curator's one-class IsolationForest fallback path still
  works at 35 columns (it trains on the session's own candidates).

## Execution outline (red-team's, with the decisions folded in)

0. **Plan mode first.** Read the red-team report §Step-4-outline, this brief,
   `docs/FEATURE_EXPANSION_STEP2_2026-08-18.md` §"What Step 4 requires", and
   the current `features.py` / `curator.py` / `train_classifier.py` /
   `watcher.py`. Produce the concrete plan, get it approved.
1. **Guard the provenance hole first** (independent of the swap; red-team's
   standing recommendation): `recurate_sessions.py` explicit-path mode must
   refuse to regenerate `review_neuron.mat` when `labels.mat` exists.
   One small commit.
2. **Extend features.py (area-scoped)** with ranks + v2b + flag; update the
   two lockstep copies; unit-check against the pinned values (backfill
   parity invariant).
3. **Backfill in parallel files**: build every BLA session's 35-col matrix as
   `candidate_features_v2.npz` alongside the v1 file (agent sessions from
   `.feature_expansion\` extractions; bootstrap = 13 + ranks + zeros + flag 0).
   Nothing deployed reads these yet. Verify counts, first-13 identity, spot
   checks. Back up every v1 npz (e.g. to `.feature_expansion\_v1_backup\`)
   and the current joblib before the swap moment.
4. **Freeze**: confirm exchange idle, stop the BLA watcher (only after any
   in-flight conversion finishes — never mid-conversion).
5. **Atomic swap**: rename v2 npz into place, retrain
   `--prospective-only --model xgboost` (companion 13-col first-pass model
   trained and stored in the same joblib), verify joblib
   (model_type=xgboost, scaler width 35, first-pass present).
6. **Threshold re-derivation** (decision 3). Set it, retrain/verify the
   stored `reject_threshold`.
7. **Verification before restart**: reference 8-seed sweep on the new corpus
   reproduces the Step 2 result; curator dry-run on a held/pending session
   end-to-end (two-pass scoring, PDF, review_neuron.mat); smoke test = the
   bla21 autopsy cells (2tones/093025 Neurons 22/25) now score well above
   threshold.
8. **Restart the BLA watcher**; watch one full cycle; confirm no retrain loop
   (joblib newer than all labels.mat) and no curation crash.
9. **Rollback path (keep ready, test the restore once on a copy)**: restore
   v1 npz backups + old joblib + git revert of features.py/curator.py; the
   watcher restart completes the rollback.

## Rules that always apply

Never `--model auto`. Never `refresh_features.py --write-forward` (predates
this design). Don't touch `git stash@{0}`. Kheirbek server read-only. Branch
from `dg-al-cold-start` (it carries the harness fixes and docs); never commit
to main; commit checkpoints per step. Numbers must be computed fresh — the
pool has grown since 2026-08-18, so re-pin the manifest and expect the
absolute numbers to move slightly; the invariants and deltas are what must
hold. If any invariant fails, stop and report rather than working around it.
