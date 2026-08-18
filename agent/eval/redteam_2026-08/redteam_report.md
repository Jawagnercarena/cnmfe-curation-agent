# Red-team report: Step 2 feature-expansion result (rankv2_35)

_2026-08-18, fresh-session adversarial review per
docs/FEATURE_EXPANSION_REDTEAM_BRIEF.md. All numbers below were computed this
session by an independent from-scratch evaluator
(scratchpad redteam/redteam_lib.py) unless explicitly marked "claimed". Pinned
pool only (170 sessions; no live rescan used for any model number)._

## Verdict summary

**The result survives every attack. Deploy-worthy with changes (see the
overall call): ship v2b not v2, train with option (b), re-derive the
threshold (expect ≈0.06, NOT 0.12), and resolve the neighbor-feature
production design.**

| claim | verdict | one-line evidence |
|---|---|---|
| C1 provenance | **CONFIRMED clean** | 79/79 mtime + N + idx checks pass; no post-review write path in repo; extraction bit-identical to review_neuron.mat |
| C2 headline | **CONFIRMED** | my from-scratch eval: b13 0.8921 ± 0.0027 → rankv2_35 0.9137 ± 0.0022 (+0.0216, 8/8 seeds positive, min +0.0203); full-pool baseline 0.9099 reproduced with r=1.000 per-row vs pinned OOF |
| C3 leakage | **CONFIRMED clean** | drop nb_corr_max: 0.9116 (win keeps +0.0195); drop both neighbor feats: 0.9095 (+0.0174, far@mj 0.09%); deployed-score fallback = 4 sessions / 344 rows (3.0%) |
| C4 flag | **CONFIRMED clean** | flag constant 1 on all eval rows by construction; deleting it: −0.0004 AUC (noise) |
| C5 autopsy circularity | **CONFIRMED clean** | eval minus the 16 autopsy cells: delta +0.0198 (vs +0.0216) |
| C6 gate robustness | **CONFIRMED, one nuance** | 5 XGB seeds: rankv2_35 beats b13 on 5 animals + early-era at every seed; bla16 is a tie (mean +0.0007, min −0.0033). rankv2b_35 improves on ALL cells at every seed (min +0.0088) |
| C7 attribution | **CONFIRMED robust** | win distributed: ranks +0.005, ring alone +0.011, neighbors alone +0.016, events alone +0.017, full +0.022 — no fragile carrier |
| C8 detector choice | **RESOLVED: ship v2b** | pooled near-tie (v2 +0.0010, 8/8 seeds); v2b uniformly positive on all gate cells, mechanism honest (marquee cells 94.9/83.7 pct vs inverted 17/21 under v2) |
| C9 operating point | **QUANTIFIED** | new threshold ≈0.06 → false-AR 0.80%, junk ~32-34%; keeping 0.12 would give 2.2% false-AR |
| C10 Step 4 confound | **RESOLVED: option (b)** | arms tie on ranking (Δ≤0.0007); flag confound empirically inert (flip shift ≈ −0.004, importance ≤4%); arm (a) fragile on auto-reject (60% vs 84% <0.12) |

## Details

### C1 — Provenance of review_neuron.mat: CONFIRMED clean (kill-switch passed)

- **mtime audit, all 79 sessions** (redteam/c1_provenance_audit.py):
  review_neuron.mat strictly older than labels.mat on 79/79; minimum gap
  5.0 h, median 264 h. review_neuron mtimes span 2026-03-02 → 2026-08-05,
  i.e. written per-session at curation time, not regenerated in a batch.
- **N-consistency, all 79**: C_raw rows in the extraction ==
  n_candidates − |auto_rejected| == len(labels) on every session; the pinned
  `__idx` arrays equal recomputed review_indices exactly.
- **Write-path grep** (whole repo, .py + .m): writers of review_neuron.mat are
  (1) curator._write_review_mat at curation time; (2) its fallback full-copy
  (would be caught by the N check; N matched everywhere);
  (3) recurate_sessions.py — auto-discovery requires labels.mat absent;
  explicit-path mode could bypass, but the mtime audit shows no session was
  rewritten post-review. train_merger.py is a NotImplementedError stub.
  CNMFe_final_save.m and every viewNeurons*.m variant contain **no** save
  call targeting review_neuron.mat (final_save writes labels/neuron/Cn/
  Coor/spatial_footprints/merge_log/checkpoint only).
- **Extraction fidelity** (MATLAB load-only, 3 sessions across eras:
  093025-bla21, 052726-bla37, 072326-bla36): C_raw and A in
  .feature_expansion mats are bit-identical to review_neuron.mat
  (max|diff| = 0 on all three).

### C2 — Independent recomputation: CONFIRMED (kill-switch passed)

From-scratch evaluator (own loader, own label reconstruction, own bootstrap
weights/ambiguous mask from bootstrap_match_stats.json, own CV/weights/
metrics; no harness.py / diagnose_model / train_classifier imports):

- Pool loads 170 sessions (79 agent / 91 bootstrap); my independently derived
  ambiguous mask zeroes **1123** bootstrap rows = deployed joblib's
  n_excluded_ambiguous.
- **Full-pool b13**: AUC 0.9099 ± 0.0017, n=12677, real=2394 — matches the
  re-pin. Per-row seed-mean OOF vs pinned baseline_oof.npz: **r = 1.00000,
  max|diff| = 0.00000** (my pipeline reproduces theirs exactly, so the
  comparison below is apples-to-apples).
- **Reviewed pool (n=10984, 2394 real)**:
  - b13 0.8921 ± 0.0027, far@matched-junk 0.86% (claimed 0.8921 / 0.86%)
  - rank26 0.8974 ± 0.0031 (claimed 0.8974)
  - rankv2_35 **0.9137 ± 0.0022**, far@mj **0.11%**, junk@matched-far
    **34.9%** (claimed 0.9137 / 0.11% / 34.9%)
  - Delta +0.0216; paired per-seed deltas all positive, min +0.0203.

### C3 — Residual leakage: CONFIRMED clean

- repin_baseline.py verified: OOF is grouped by session
  (StratifiedGroupKFold, predictions only on held-out sessions), so
  OOF-derived neighbor scores never see the session's own labels.
- compute_v2_features.py inputs audited end to end: .feature_expansion mats
  (pre-decision, C1), candidate_features.npz (curation-time features),
  baseline_oof.npz (grouped OOF), deployed joblib scores **only** for the 4
  agent sessions outside the OOF pool (3odor 022426/031226/031326-bla8,
  072326-bla36) = 344 reviewed rows = 3.0% of the pool. The ev_* and
  ring_contrast features are pure functions of pre-decision C_raw/A/Cn.
- Ablation bound: without nb_corr_max AUC 0.9116 ± 0.0021 (still +0.0195);
  without both neighbor features 0.9095 ± 0.0022 (+0.0174) and false-AR at
  matched junk **improves** to 0.09%. The win does not depend on any
  model-score-derived feature.

### C4 — v2_present flag: CONFIRMED clean

Structural: flag = 1 on every agent row, 0 on bootstrap rows (step2_eval
build_records), and the eval pool is agent-only ⇒ constant 1 across all eval
rows; it cannot rank within the pool. Empirical: deleting the column moves
AUC −0.0004 (inside seed noise), far@mj 0.13% vs 0.11%.

### C5 — Autopsy circularity: CONFIRMED clean

Dropping the 16 autopsy false-AR cells from the eval pool (they remain in
training folds): b13 0.8975 → rankv2_35 0.9172, delta **+0.0198** vs +0.0216
with them. The 16 inspected cells contribute ~0.002 of the win; the effect is
corpus-wide. Both detector variants were computed and reported (v2 and v2b);
gate passage does not depend on choosing between them (both clear every gate
— see C6/C8).

### C6 — Gate robustness under seed spread: CONFIRMED, with one nuance

Step 2's gates were single fits at XGB random_state=42 (the only stochastic
element in a deterministic holdout). Re-run at 5 random_states
(42/7/2024/1/31337):

| cell | b13 | rankv2_35 | paired Δ (min…mean) | rankv2b_35 paired Δ (min…mean) |
|---|---|---|---|---|
| bla12 (12 sess) | 0.9024 | 0.9145 ± 0.0003 | +0.0100…+0.0121 | +0.0088…+0.0105 |
| bla16 (9 sess) | 0.8955 | 0.8962 ± 0.0022 | **−0.0033…+0.0007 (tie)** | +0.0090…+0.0140 |
| bla21 (4 sess) | 0.8300 | 0.8782 ± 0.0021 | +0.0460…+0.0482 | +0.0208…+0.0259 |
| bla36 (20 sess) | 0.8867 | 0.9153 ± 0.0014 | +0.0270…+0.0286 | +0.0276…+0.0288 |
| bla37 (20 sess) | 0.8937 | 0.9154 ± 0.0007 | +0.0196…+0.0218 | +0.0189…+0.0203 |
| bla8 (14 sess) | 0.8838 | 0.9086 ± 0.0009 | +0.0229…+0.0248 | +0.0256…+0.0278 |
| early era (16 sess) | 0.8457 | 0.8752 ± 0.0024 | +0.0242…+0.0295 | +0.0276…+0.0301 |

- The claimed "improves or holds on all six" survives: 5 animals + early era
  improve at **every** seed; bla16 holds (a statistical tie, not an
  improvement — the doc's table cell "0.893 → 0.893" was accurate).
- **New evidence favoring v2b**: rankv2b_35 improves on **all seven cells at
  every seed** (min paired delta +0.0088), including bla16. v2's weakness is
  exactly bla16; v2b's weakest cell (bla21 +0.0259) is still decisively
  positive. Note the LOAO bla21 figure quoted in the doc (0.830 → 0.876) is
  reproduced (0.8782 ± 0.0021 across seeds).

### C7 — Attribution: CONFIRMED robust — the win is distributed

Group ablations, 8-seed OOF, all vs b13 (0.8921):

| feature set | AUC | Δ vs b13 |
|---|---|---|
| ranks only (rank26) | 0.8974 ± 0.0031 | +0.0053 |
| ranks + ring only | 0.9035 ± 0.0024 | +0.0114 |
| ranks + neighbors only | 0.9082 ± 0.0026 | +0.0162 |
| ranks + events only | 0.9094 ± 0.0031 | +0.0173 |
| ranks + events + ring (no nb) | 0.9095 ± 0.0022 | +0.0175 |
| 13 + v2 + flag, no ranks (v2_22) | 0.9107 ± 0.0014 | +0.0186 |
| full rankv2_35 | 0.9137 ± 0.0022 | +0.0216 |

No single fragile feature carries the result: every group contributes, the
groups are partly redundant (any one of events/neighbors/ring recovers most
of the win), and the deltas add sub-linearly. This is the signature of a real
underlying signal (candidate-level trace/footprint quality) measured several
ways, not of an artifact in one column. Recommended shipping columns: see the
overall call.

### C8 — Detector choice: v2 vs v2b is a real trade-off; recommendation: v2b

- Paired per-seed (8 seeds): rankv2_35 − rankv2b_35 = **+0.0010 ± 0.0003**
  (v2 wins 8/8 seeds — consistent but operationally nil; far@mj 0.11% vs
  0.12%).
- C6 gates: **v2b improves on all 7 holdout cells at every seed**; v2 ties on
  bla16. v2 is stronger on bla21 (+0.048 vs +0.026), v2b on bla16 (+0.014 vs
  +0.001). Early-era: v2b +0.0301 ≥ v2 +0.0295.
- Mechanism (independently reproduced): bla21 N22/N25 regression cells score
  94.9 / 83.7 pct within-session stereotypy under v2b (claimed 96/85) with
  GCaMP-plausible event counts (2.4/2.7 per 1k frames); under v2's 2.5σ
  detector the same cells invert to 17.3 / 21.4 pct because its "events" are
  noise firings (9.9/12.0 per 1k frames). The drifty bla16 N64 marginal reads
  12.5 pct under v2b (claimed 14th) but 77.3 under v2.
- Single-feature AUC smell test: max 0.73 across both sets — nothing
  implausibly separable; v2's ev_rate is *inversely* predictive (0.447),
  confirming it operates as a noise-level proxy rather than an event counter.

**Recommendation: ship the v2b detector (rankv2b_35).** The pooled-AUC edge
of v2 (+0.001) is bought with features that work for an unintended reason
(noise proxies) and that tie-at-noise on a whole animal (bla16) — the exact
per-animal-fragility failure mode the motion-feature work taught us to
reject. v2b is uniformly positive on every gate cell at every seed, measures
what it claims, and wins the early-era class this investigation exists to
fix. Carrying BOTH sets (the step2 doc's fallback suggestion) buys +0.001
pooled AUC for 5 extra contract columns and reintroduces the noise-proxy
fragility; not recommended.

### C9 — Operating point: the deploy threshold must move to ≈0.06

Full threshold sweeps (diagnose_model §3 style, 8-seed mean ± sd) from the
C10 deploy-realistic OOF (arm b = auto-rejected rows in training; see C10):

- **0.12 is not a valid threshold for the new model.** At T=0.12 the new
  model's false-AR is **2.20%** (v2) / 2.20% (v2b ≈ same) — the score
  distribution recalibrates hard toward 0. Carrying 0.12 forward would
  ~2.6× the false-AR.
- The equivalent sub-0.85% operating point lands at **T = 0.06**:
  - v2 arm b @0.06: false-AR 0.83% ± 0.08, junk-caught 33.6% (reviewed) /
    35.1% (full pool, autorej rows approximated at v2=0)
  - **v2b arm b @0.06: false-AR 0.80% ± 0.12, junk-caught 31.8% (reviewed) /
    33.8% (full pool)**
  - Reference, current model b13: @0.12 false-AR 0.86%, junk 14.7%
    (reviewed) / 27.8% (full pool, from the C2 full-pool run).
- So at equal false-AR the new representation roughly **doubles junk caught
  on the reviewed (hard-junk) stratum** and adds ~6-7 pts on the full pool.
  The step2 doc's matched-op numbers are honest; the absolute threshold is
  simply a new number.
- The "reals in [0.05,0.12) band grows" caveat resolves benignly: the band
  was anchored to the old threshold; at the new T=0.06 the false-AR is back
  at ~0.8%. Threshold re-derivation at Step 4 (already mandated) is the fix;
  expect it to land at 0.05-0.07.
- Caveat, stated plainly: full-pool junk-caught uses v2=0 for auto-rejected
  rows (their traces were never saved — the C10 premise). Production
  supplies real v2 for all candidates; real junk v2 plausibly scores junkier
  than zeros, so these full-pool junk numbers are likely floors, but that is
  assumed, not measured.

### C10 — Step 4 training-data confound: option (b), keep the flag

Empirical test, both feature sets (v2 and v2b), 8 seeds, identical folds,
eval always on the same reviewed rows:

| arm | v2 AUC | v2b AUC | autorej rows <0.12 (OOF) |
|---|---|---|---|
| (a) exclude autorej from training | 0.9132 ± 0.0014 | 0.9126 ± 0.0015 | 59.2% / 60.2% |
| (b) include autorej, v2=0+flag=0 | 0.9130 ± 0.0013 | 0.9123 ± 0.0015 | 83.6% / 83.6% |
| (b) without the flag column | 0.9133 ± 0.0014 | 0.9126 ± 0.0012 | 83.8% / 83.9% |

- The **flag↔label correlation exists exactly as feared** (agent train rows:
  1693 flag-0 rows, 100% junk; 0 flag-0 reals) — but the model **does not
  anchor on it**: flipping flag 0→1 on auto-rejected rows moves scores by
  median −0.004 (v2) / −0.014 (v2b), p95 ≈ +0.000; flag gain-importance
  1.6-4.0%. The v2=0 zeros make the flag redundant, and arm b-noflag
  reproduces arm b exactly. The feared production inversion is empirically
  inert.
- Ranking skill on reviewed rows is **identical across arms** (Δ ≤ 0.0007).
- The decisive difference is **auto-reject competence**: arm (a) never sees
  easy junk in training and scores only ~60% of auto-rejected rows below
  0.12 (median 0.107, out-of-distribution behavior); arm (b) scores ~84%
  below 0.12. Since scoring the full candidate set is the curator's FIRST
  production act, arm (a) is fragile exactly where it matters. (Same v2=0
  approximation caveat as C9 — but arm (a)'s OOD-ness is structural: rows
  shaped like "flag=1, v2=0, junky 13" never occur in its training.)
- **Recommendation: option (b)** — include auto-rejected rows with v2=0 +
  flag=0, i.e. exactly what train_classifier.py's label reconstruction
  already does naturally (auto-rejected → label 0). No trainer code path
  changes; the negative class keeps its current composition so the threshold
  sweep stays comparable; keep the flag column (harmless, honest marker, and
  it covers bootstrap rows which genuinely lack candidate data). Keep zeros
  (evaluated) rather than switching to XGB-native NaN missing (unevaluated).

## Check coverage (which checks ran on all sessions vs samples)

Everything statistical runs on the **complete pinned pool — no sampling
anywhere**: every reviewed row of every eval-pool agent session (10,984 rows
/ 75 sessions; 12,677 / 75 for full-pool runs) and all 91 bootstrap sessions
in every training fold. Per-animal eval representation: bla12 12 sessions /
1857 rows / 541 reals; bla16 9 / 840 / 169; **bla21 4 / 307 / 106**; bla36
19 / 3221 / 466; bla37 20 / 3038 / 783; bla8 11 / 1721 / 329. bla21 is
additionally a dedicated LOAO holdout cell (all its sessions held out
together, 5 seeds) and its two marquee regression cells were individually
mechanism-checked.

C1 provenance checks on **all 79 sessions**: mtime ordering, N-consistency
(extraction rows == n_candidates − auto_rejected == labels length), and
pinned-idx == recomputed review_indices. The one initially-sampled check —
MATLAB bit-identity of the extraction against review_neuron.mat itself — was
3/79 (all eras, incl. the bla21 marquee session), then **extended to all 79**
(c1_matlab_full79.m): **79 pass, 0 fail — every session's extracted C_raw and
A are bit-identical to its review_neuron.mat** (max|diff| = 0). No sampled
check remains anywhere in this review. Note: 7 sessions have
auto_rejected == 0, where the N-check cannot distinguish curator's proper
write from its full-copy fallback — immaterial for validity (with zero
auto-rejects the review set IS the full set, so both paths produce identical
content), and the bit-identity audit covers them directly.

## New holes found outside the named surfaces

1. **Production computation of the neighbor features is under-specified**
   (Step 4 blocker-level detail, not a validity problem). nb_corr_max needs
   "high-confidence neighbor" scores at package time, but post-swap the
   deployed model needs the full 35 columns to score — a chicken-and-egg the
   step2 doc doesn't address. The offline eval selected neighbors using
   13-feature-model scores (grouped OOF), so the faithful production
   implementation is a **first-pass score from a 13-feature model** (or the
   35-col model with nb columns zeroed) → pick high-conf → compute nb → final
   scores. This must be an explicit design decision in Step 4; the fallback
   (drop both nb features) costs only −0.004 AUC (C3) and removes the
   circularity entirely.
2. **recurate_sessions.py explicit-path mode can regenerate
   review_neuron.mat on an already-labeled session** (the no-labels guard
   only applies to auto-discovery). Never exercised historically (C1 mtime
   audit is clean), but it is the one code path that could silently break
   review_neuron.mat's pre-decision property in the future. Recommend a
   one-line guard: refuse if labels.mat exists.
3. **v2's event features work for an unintended reason** (noise proxies:
   ev_rate single-feature AUC 0.447 = inversely predictive; marquee cells
   inverted to 17-21st percentile). Not leakage — but a fragility argument
   folded into the C8 recommendation.
4. The 4 non-OOF sessions' neighbor selection used deployed in-sample scores
   (C3; 344 rows, 3.0%). At Step 4 backfill, regenerate those from the new
   grouped-OOF or accept the bounded imprecision — either is defensible.

## Overall call: DEPLOY-WORTHY WITH CHANGES

The result survives every attack. The headline is real, independently
reproduced to the fourth decimal from raw session files by a from-scratch
evaluator; provenance of the underlying data is clean on all 79 sessions; no
leakage path found (and the win survives removing every score-derived
feature); the gain is distributed across feature groups, robust across
animals, eras, and seeds; the autopsy cells contribute ~0.002 of the +0.0216.

The changes required before Step 4:

1. **Ship the v2b detector (rankv2b_35), not v2** (C6+C8): uniformly
   positive on every gate cell at every seed (v2 ties-at-noise on bla16),
   mechanistically honest, equal early-era gain; costs +0.001 pooled AUC.
2. **Training data = option (b)** (C10): auto-rejected rows stay in training
   as label-0 with v2=0 + flag=0 — the trainer's existing behavior.
3. **Deploy threshold ≈ 0.06, not 0.12** (C9): re-derive from the actual
   retrained model per protocol; expect false-AR ≈ 0.8%, junk-caught ≈ 32-34%
   (roughly double the current 14.7% reviewed-stratum catch at equal
   false-AR). Anyone who deploys at 0.12 ships a 2.2% false-AR regression.
4. **Resolve the nb-feature production design** (new hole #1): first-pass
   13-feature scoring (faithful to eval), or drop the 2 nb features (−0.004).

## Step 4 execution outline (recommended)

Order of operations, with verification at each step:

0. **Freeze**: re-pin the pool manifest; confirm no reviewer returns
   mid-flight; stop the BLA watcher (swap work must not race the
   auto-retrain trigger — handoff rule 3). vCA1/DG_AL: their extractors must
   not silently inherit the new contract (shared features.py) — version the
   extractor per area for the transition, or coordinate a 3-area bump; DG_AL
   is 12 sessions into a cold start, so per-area versioning is the low-risk
   path.
1. **Extractor** (features.py): add within-session pct ranks (13), v2b
   features (8, the shape-qualified 3.5σ detector from
   compute_v2b_features.py), v2_present flag. Contract: 13 + 13 + 8 + 1 =
   35 positional columns; write feature_names. Implement the nb first-pass
   decision (new hole #1). Update the two lockstep re-implementations
   (train_classifier retro path, bootstrap_preagent) in the same commit.
2. **Backfill into parallel files** (never touch candidate_features.npz
   in place): agent sessions from .feature_expansion mats (reviewed rows
   real v2b; auto-rejected rows zeros + flag=0); bootstrap = 13 + ranks +
   zeros + flag=0. Verify per session: row count unchanged; first 13 columns
   bit-identical to v1; ranks deterministic on recompute. The 4 non-OOF
   sessions: regenerate neighbor scores per the nb design decision.
3. **Atomic swap + retrain**: swap npz files, retrain
   `--prospective-only --model xgboost` (never bare --model auto). Verify
   joblib: model_type=xgboost, n_features=35, n_sessions=170 (or current),
   n_excluded_ambiguous=1123.
4. **Threshold re-derivation**: full sweep on the new model; expect
   T≈0.05-0.07 landing at false-AR ≤ 0.85%. Set reject_threshold in the
   joblib. Gate: false-AR ≤ 1% AND junk-caught ≥ 30% at the chosen T, else
   stop and investigate.
5. **Pre-restart verification**: frozen-pool OOF reproduces ≈ 0.9127 ± 0.002;
   LOAO spot-checks bla16 + bla21 ≥ b13 reference (0.8955 / 0.8300); score
   2-3 pending sessions old-vs-new and eyeball the auto-reject sets in the
   PDFs.
6. **Restart the BLA watcher**; confirm the first auto-retrain keeps
   xgboost@35 features; confirm vCA1/DG_AL watchers still run the v1
   extractor path (and watcher_DG_AL's _TRAIN_SCRIPT redirect is intact).
7. **Rollback path**: keep v1 npz + old joblib untouched until the new model
   has survived one full reviewer-return cycle; rollback = restore both.

Budget honesty: this red-team ran C1-C10 in one session (~2.5 h wall clock);
every number above is from scripts preserved in the session scratchpad
(redteam/*.py) and reproducible against the pinned inputs.
