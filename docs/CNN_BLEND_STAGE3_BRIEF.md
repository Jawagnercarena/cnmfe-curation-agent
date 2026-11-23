# Stage 3 brief — deploying the raw-data CNN blend (decision document, 2026-09-08)

Status: **not started; user's decision.** Evidence:
`agent/eval/algo_bakeoff_2026-09/RESULTS.md` (Stage 1, Stage 2, Round 2) and
the pre-registration in `agent/eval/algo_bakeoff_2026-09/README.md`.

## What passed

Configuration `fixed12+nbtrace+ens3`: a two-branch CNN (2-D conv on a
3-channel 32×32 gSig-scaled footprint patch = own footprint / Cn z-score /
max over other candidates; 1-D dilated conv on a 2-channel trace = own trace
and the max-correlated neighbour within 60 px, both asinh(robust-z) at the
AVG5x rate), 164k parameters, trained 12 epochs with cosine decay on every
session (no early stopping), 3 members averaged, **blended 50/50 in
probability with the deployed XGBoost score**. Eight seeds, four cells,
against the deployed control on the same rows and folds:

| cell | blend Δ reviewed AUC | junk at matched false-AR | FAR at matched junk | rule_T (worst seed) |
|---|---|---|---|---|
| BLA session | +0.0101 (8/8) | 46.9% vs 41.4% | 0.48% vs 0.76% | 0.09 (0.84%) |
| BLA animal held out | +0.0144 (8/8) | 48.4% vs 40.6% | 0.51% vs 1.01% | 0.08 (0.88%) |
| vCA1 session | +0.0119 (8/8) | 49.8% vs 46.0% | 0.88% vs 1.16% | 0.05 (0.83%) |
| vCA1 animal held out | +0.0174 (8/8) | 46.8% vs 44.5% | 0.88% vs 0.91% | 0.05 (0.83%) |

The single model (member 0 alone) passes the same gate: +0.0098 / +0.0140 /
+0.0119 / +0.0167. DG_AL is out of scope (cross-area transfer was negative;
the pooled tabular prior remains its route).

## Decisions for the user

1. **Go / no-go.** The gain is about twice what justified the August feature
   expansion, but it adds a GPU dependency and a second model to the
   production path.
2. **Single model or 3-member ensemble.** Ensemble: +0.0001 to +0.0007 more
   AUC, better worst seed, three times the training time. Start single is
   defensible; the code path is identical (members are averaged).
3. **Area scope.** BLA and vCA1 both pass. vCA1 has the larger relative gain
   and the blend restores an admissible threshold there (the deployed T=0.04
   currently runs 1.16% mean / 1.45% worst-seed false auto-reject on the
   24-session pool, with no admissible T for the tabular model alone).
4. **Threshold posture.** The blend is more conservative at the same nominal
   T, so T must be re-derived on the blend's OOF with threads pinned and the
   1%-worst-seed rule under **animal** grouping (BLA: 0.08; vCA1: 0.05). Do
   not carry the current T over.
5. **Retrain cadence.** XGBoost retrains on every reviewer return (watcher).
   Retraining three CNN members on a full area pool is ≈ 5 minutes on the
   GPU; attaching it to the same retrain is feasible, but a fixed cadence
   (e.g. with each ingest batch, operator-triggered) is simpler to reason
   about and keeps scores stable between batches.

## What deployment needs (all verified against the current tree)

- **Scoring path.** `agent/curator.py:206` whitelists `model_type` in
  `("logistic_regression", "lr", "xgboost", "lightgbm")` and the `else`
  branch does not raise; a new type (e.g. `xgb_cnn_blend`) needs its own
  branch and the `else` must raise. The blend needs, per session, the same
  inputs `features.py` already loads (footprints, `C_raw`, `Cn`) plus gSig
  (session `agent_run.log`, as `prep_raw_corpus.gsig_of` reads it) and the
  AVGNx factor from the session name (`prep_raw_corpus.avg_factor`). The
  neighbour is chosen among **all** candidates of the session at curation
  time; in the evaluation corpus, agent sessions' neighbours were chosen among
  reviewed candidates only (auto-rejected ones had no raw data) — a small,
  documented train/deploy asymmetry; the bootstrap sessions (most training
  rows) used all candidates.
- **Model bundle.** The deployed joblib carries `scaler`, `clf`, `model_type`
  (`curator._save_model`); the CNN state dicts (3 × 164k float32 ≈ 2 MB) can
  live in the same joblib or a sidecar `.pt` next to it, area-scoped under
  `agent/model/<AREA>/`, with the blend weight (0.5) and the preprocessing
  constants (patch 32, half = max(8, round(3·gSig)), CROP 2048, NB_DIST 60,
  BASE_AVG 5) stored alongside so a future change cannot silently drift.
- **Threshold.** `agent/train_classifier.py:1002` `_THRESHOLD_BY_MODEL`
  needs an entry for the new type; `curator.THRESHOLD_OVERRIDE`
  (`curator.py:52`, applied at `:633`) keeps working for threshold-0 areas
  and preps.
- **Environment.** torch 2.11.0+cu128 is installed only in `ACORN`
  (`C:/Users/julia/.conda/envs/ACORN/python.exe`); `valence` runs the
  watchers. Least-invasive option: the curator calls the ACORN interpreter
  in a subprocess for the CNN score (inputs written to a temp npz, scores
  read back), leaving `valence` untouched. Alternative: install torch into
  `valence` (a 2.5 GB dependency in the watcher environment; not recommended
  without a rehearsal).
- **Training.** A trainer that builds the corpus rows for the area pool
  (the `prep_raw_corpus2.py` code path, ≈ 6 minutes per area), trains the
  members with the harness weights (agent weight = override or
  max(sqrt(n_bs/n_ag), 4.0); bootstrap weights as stored; positives ×
  scale_pos_weight), and writes the bundle. Re-derive T from an 8-seed OOF
  of the blend with threads pinned (the round-2 scripts already do this).
- **Companion first-pass model.** Unchanged: the 35-column contract still
  needs the 13-column first-pass model for the neighbour high-confidence
  mask. The CNN score is not an input to the tabular features.
- **Swap mechanics.** Follow the August pattern (`agent/eval/step4_2026-08/`:
  `swap_v2.py`, `verify_deploy.py`, `simulate_deployed_smoke.py`,
  `record_preswap_scores.py`): record pre-swap scores on the labeled pool,
  stop the area's watcher for the swap (first poll after restart must not
  retrain), rehearse on a copy, atomic swap, keep the previous bundle for
  rollback, dry-run one session, watch the first real cycle.

## Verification before the swap

1. Parity: the production preprocessing reproduces the evaluation corpus
   rows for a sample of sessions (patches and traces equal within float16).
2. In-sample sanity: the deployed members, trained on the full pool, score
   the labeled pool with AUC ≥ the 8-seed OOF (they have seen the rows) and
   the auto-reject set at the new T contains no real cell that the current
   model keeps, unless reviewed.
3. Threshold: rule_T under animal grouping with pinned threads, recorded
   with the seed list and thread count.
4. Runtime: per-session scoring time and GPU memory on a 700-candidate
   session (the largest bootstrap sessions).
5. Rollback drill on the rehearsal copy.

## Risks

- cuDNN nondeterminism: retrains differ at the 3rd decimal per candidate;
  scores are reproducible only to seeds, not bits. Document, and never gate a
  human decision on a bit-exact score.
- GPU availability: the RTX 5060 Ti is otherwise idle; MATLAB runs on CPU. If
  the GPU is absent, the subprocess must fail loudly (the curator falls back
  to the tabular model only by explicit operator choice, never silently).
- Sessions without `Cn.mat` (140 BLA rows in the corpus) get a zero Cn
  channel — already how the evaluation treated them.
- The label ceiling is unmeasured (Stage 0). The blend's gain is measured
  against single-reviewer labels; part of the remaining 0.08 to 1.0 may be
  reviewer noise.

## Cost estimate (measured, 2026-09-07/08)

Corpus build ≈ 6 min per area (CPU); training ≈ 1.5–2 min per member per
model on the full pool (GPU, 12 epochs); scoring a session ≈ seconds.
Engineering: the scoring branch, bundle format, trainer, subprocess bridge,
threshold derivation, swap rehearsal and verification — on the order of two
to three working days, plus one watcher cycle of observation per area.
