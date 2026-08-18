# Feature-expansion handoff — can new features lift the BLA classifier past 0.91?

_Written 2026-08-18. All numbers in this doc were measured 2026-08-09 – 2026-08-18
unless dated otherwise; nothing here is carried forward from older notes without
being re-verified._

## 0. The question

BLA classifier skill has plateaued at **OOF AUC ≈ 0.910** while the training pool
grew 61 → 75 agent sessions (2026-08-09 → 2026-08-18). More of the same labels is
tightening the estimate (seed spread ±0.001), not raising skill. The hypothesis:
the ceiling is the **13-feature representation**, and the only material lever left
is new features. This doc scopes how to test that hypothesis without breaking the
running pipeline.

## 1. Current baseline (2026-08-18 — reproduce before starting)

- Deployed: `agent/model/BLA/classifier.joblib` = **xgboost @ reject_threshold 0.12**
  (reverted from 0.13 on 2026-08-18; `n_sessions=170`).
- Pool: **75 agent / 91 bootstrap**; OOF eval pool 12,677 candidates, 2,394 real
  (18.9%), 10,283 garbage.
- **OOF AUC 0.909 ± 0.001** (8 seeds); at 0.12: **0.83% false-AR / 27.9% junk caught**.
- Methodology that produced these (treat as the reference harness):
  5-fold `StratifiedGroupKFold` grouped by session, agent sessions with ≥5 reals
  as the test pool, bootstrap sessions training-only, real deployed weights
  (agent 4.0×, bad-bootstrap 0.4×, ambiguous masked to 0), seeds
  [42, 1, 7, 13, 100, 2024, 31337, 9], `StandardScaler` + XGB
  (300 est / lr .05 / depth 4 / subsample .8 / colsample .8 / spw from weighted
  class balance). Reusable pieces: `agent/diagnose_model.py`
  (`load_all_records`, `make_clf`, `compute_spw`) — it now shares
  `_is_bootstrap_session` with the trainer, do not fork that logic.
- Session-scoped copies of the exact sweep scripts used for these numbers live in
  the 2026-08 session scratchpad
  (`C:\Users\julia\AppData\Local\Temp\claude\c--code-CNMF-E-LEGACY-BIANE-CLAUDE\298a44eb-369c-4491-8cf6-bdccd183ab2e\scratchpad\`:
  `threshold_robustness.py`, `reviewer_quality.py`, `session_quality.py`,
  `counterfactual_reclassify.py`). Copy anything you want to keep into the repo —
  scratchpads are not permanent.

## 2. Evidence the representation is the ceiling (why this handoff exists)

1. **Learning curve is flat at fixed skill.** AUC 0.910 → 0.910 → 0.909 across
   61 → 69 → 75 agent sessions (2026-08-09/-18). Earlier growth did help
   (0.884 → 0.897 over ~45 → 58 sessions), so this is a plateau, not noise.
2. **All residual error is boundary-band, none is confident.** Across every check
   this month: **zero** real-labeled candidates score < 0.05 OOF; the ~20
   false-AR events at 0.12 all sit in 0.05–0.12. The model is uncertain exactly
   where dim-real and junk overlap in feature space — the signature of a
   representation limit, not underfitting.
3. **Even session-local models can't separate with these features.** Per-session
   5-fold CV (LR, session's own labels only; `session_quality.py`, 2026-08-09):
   good modern sessions reach 0.81–0.96 — not 1.0 — with their own labels.
4. **Prior feature hunt hit the same wall.** The 2026-07/08 motion-feature work
   (5 approaches, incl. frame-precise onset coincidence and an onset-locked s_z
   structure channel) concluded on 2026-08-08: leave-one-animal-out 0.74 was a
   **data ceiling — "the deployed 13 already capture the signal"**; s_z was flat
   (+0.003/+0.015) and was NOT deployed.
5. Recurring operational symptom: each returned batch adds a few dim reals near
   the boundary, so fixed-threshold false-AR drifts up while AUC stays flat
   (0.13 walked 0.85% → 0.93% → 1.02%, triggering the 0.12 revert). New features
   that separate dim-real from junk would fix this at the root.

Honest counterweight: point 4 suggests the achievable gain may be small. Set kill
criteria (§6) before investing.

## 3. What exists today (verified 2026-08-18)

- **The 13 features** (`candidate_features.npz` → `feature_names`, extractor
  `agent/features.py`, 287 lines): area, circularity, eccentricity, compactness,
  max_weight, weight_spread, peak_snr, transient_freq, events_per_min,
  baseline_stability, skewness, motion_correlation, cn_correlation.
  Extractor structure: `spatial_features`, `temporal_features`,
  `motion_features`, `cn_features`, `pairwise_overlap`, `extract_all`.
- **A WIP stash exists — do not pop it blindly.** `git stash@{0}`
  "wip: motion features (#1 indicator + #2 coherence infra) — held pending
  motion-label validation". Known issues recorded 2026-07/08: a
  `pop_data_present`/`FORWARD_FEATURE_NAMES` mismatch to fix before any
  `--write-forward`; its #1 motion indicator is cohort-confounded. Treat it as
  reference material, not a starting point.
- `agent/refresh_features.py` exists (feature recompute/extension infra from the
  motion work). Read it before writing new plumbing.
- **`review_neuron.mat` exists on ALL 98 labeled BLA agent sessions** (0
  missing). It is a serialized MATLAB Sources2D object (opaque `s0/s1/s2/arr`
  from Python) containing the candidate set **as sent to review**. This is the
  big deal: the July "candidate traces are GONE, features are forward-only"
  constraint was established before this file was inventoried. If its `A` /
  `C_raw` cover the reviewed candidates, **new features can be backfilled for
  the whole agent pool with a cheap MATLAB extraction per session — no CNMF-E
  re-runs**. NOT yet confirmed from inside MATLAB → that is Step 0.
  - Caveats even if confirmed: auto-rejected candidates (never reviewed) are
    absent — acceptable, every *known* real is in the reviewed set; and
    **bootstrap sessions have no candidate-level data at all** (their
    `_bootstrap/` dirs were deleted after matching).
  - **Partial confirmation already exists** (found 2026-08-18 while cross-checking
    the motion notes): the 2026-07-27 pop-coherence test loaded
    `review_neuron.mat` in MATLAB on the 8 motion sessions via
    `agent/extract_cand_traces.m` (`load` → `rn.neuron.C_raw`, N×T, all 8
    succeeded; candidate *footprints* likewise used by `extract_motion_diag.m`).
    Those 8 span Oct-2025 (bla12/bla16 2tones) → Jul-2026 (bla37), so contents
    are populated across a wide era range. Step 0 shrinks to breadth checks:
    non-motion sessions across eras + candidate-count ≈
    `n_candidates − auto_rejected`.
- **Bootstrap gap is survivable.** Per-session LOO (2026-08-09,
  diagnose_model §1): XGB agent-only averaged **0.889** vs 0.883 with bootstrap —
  bootstrap adds little at current pool size. An agent-only model over richer
  features is a legitimate evaluation path (and deployment fallback), so the
  91 bootstrap sessions do not block this work.
- Raw `.tif`s survive everywhere, so full re-extraction is the expensive
  last-resort backfill (~hours/session; the 2026-08-09 `--refresh-missing-stats`
  run of 9 sessions is the precedent).

## 4. Recommended approach (phased, each phase gated)

**Step 0 — confirm the data unlock (½ day, partly done).** Contents are already
proven populated on 8 sessions (Oct-2025 → Jul-2026) — see §3. Remaining:
MATLAB-load `review_neuron.mat` for 2–3 *non-motion* sessions across eras
(early 2026-03, mid 2026-06, recent 2026-08); confirm candidate counts ≈
`n_candidates − auto_rejected` and that `C_raw`/`A` are populated. The
extraction template already exists: `agent/extract_cand_traces.m` (2026-07-27)
loads `review_neuron.mat` and saves `neuron.C_raw` as a plain `-v7` array —
extend it to also save `A` and take a session list instead of the hardcoded 8.
If the breadth check fails on older sessions, everything temporal is
forward-only again for that era and the scope shrinks accordingly.

**Step 1 — error autopsy + zero-cost experiments (1–2 days, no new data).**
- Characterize the ~20 false-AR cells and the 0.05–0.12 band: which sessions,
  which animals, what do their 13 features look like vs. accepted dim reals?
  Look at them in the session PDFs — would a human keep them from the trace, the
  footprint, or the video? That answer names the missing feature.
- Cheapest experiment first: **per-session rank/quantile normalization of the
  existing 13** (computable from existing npz alone, no extraction). Rationale:
  dim cells in dim sessions may be separable *relatively* even when absolute
  values overlap. Run it through the reference harness; if session-relative
  features move AUC at all, that's a strong signal recording-level context is
  the missing ingredient.
- Also free: XGB feature importances / SHAP on the deployed model, and ablation
  of the existing 13 — know what's already carrying the signal.

**Step 2 — candidate new features, offline only (needs Step 0).** Compute per
candidate from the extracted traces/footprints; write to a **separate parallel
file** (e.g. `candidate_features_v2.npz` with its own `feature_names`), never
touching `candidate_features.npz`. Grounded candidates, in order of promise:
- Event-shape consistency: template-correlate each transient against the cell's
  own mean transient (real cells repeat a stereotyped rise/decay; junk doesn't).
- Rise/decay kinetics of detected events vs. the GCaMP-plausible range.
- Neighborhood corroboration: max lag-0 trace correlation with high-confidence
  (score > 0.5) cells within ~2×gSiz — dim reals often co-fire with the local
  population; junk doesn't.
- Footprint-ring contrast on Cn (is the footprint a local Cn peak or a shoulder
  of a brighter neighbor?).
- Session-context features from Step 1 if they showed signal.

**Step 3 — evaluation gates (same harness, no shortcuts).** A feature set earns
promotion only if ALL of:
- 8-seed OOF AUC gain ≥ **+0.005** over 0.909 on the same pool (spread is
  ±0.001, so +0.005 is unambiguous);
- **false-AR at matched junk-caught** (27.9%) drops, or junk at matched 0.83%
  false-AR rises — the operating point must improve, not just the ranking;
- **leave-one-animal-out** holds (the motion work's lesson: bla37 dominance can
  fake a win; test with each big animal held out);
- the boundary band shrinks: fewer reals in 0.05–0.12.
Evaluate agent-only AND agent+bootstrap-13-features variants (bootstrap rows
can carry NaN/0 + a "v2-features-present" mask column if XGB-native missing
handling is used — decide explicitly, don't let it happen silently).

**Step 4 — deployment protocol (only after Step 3 passes).**
- The npz feature contract is **positional** and shared by BLA + vCA1 + DG_AL
  (`features.py` is common; vCA1/DG_AL wrappers inject their configs). Changing
  `extract_all` changes every area's next extraction. Deploy pattern (from the
  2026-07 motion plan): **atomic swap** — new extractor + regenerated npz for
  all affected sessions + retrained model land together, never mixed.
- Keep v1 npz files untouched until the swap moment; the swap is: regenerate →
  retrain (`--prospective-only --model xgboost`) → verify joblib
  (`model_type=xgboost`, threshold, n_features) → re-run the reference sweep.
- vCA1 (locked at 0.05) and DG_AL (cold-start, THRESHOLD_OVERRIDE=0) must NOT
  silently inherit a changed contract: either bump all three areas in one
  coordinated swap, or version the extractor per area for the transition.
- Re-derive the BLA threshold from the new model's sweep — do not assume 0.12.

## 5. Do-not-break rules (each learned the hard way)

1. **Never retrain with bare `--model auto`** — it flips to lightgbm@0.11 on a
   CV tie (happened 2026-08-09). Always `--model xgboost` for BLA. The watcher
   pins this (`watcher.py` `_retrain_classifier`); manual runs must too.
2. **Don't edit `candidate_features.npz` in place, ever.** The deployed model
   scores new sessions with positional indexing; a half-migrated corpus is
   silently wrong. Parallel v2 files until the atomic swap.
3. **The watcher auto-retrains** whenever any `labels.mat` is newer than the
   model. Don't leave a half-regenerated corpus on disk where a watcher retrain
   can pick it up — do swap work with the watcher stopped, or entirely in
   parallel files.
4. **Don't pop `stash@{0}`** onto a dirty tree; it predates the 2026-08-09
   `_is_bootstrap_session` changes to the same files.
5. Bootstrap-vs-agent classification must go through
   `train_classifier._is_bootstrap_session` (JSON → agent-artifacts → neuron.mat
   fallback). 9 legacy sessions depend on the fallback; a fresh
   "does the JSON exist" check silently reintroduces the 2026-08-09 bug.
6. Kheirbek-lab server is read-only from tooling; all of this work is local
   (`D:\Julian_CNMFe\BLA`).
7. Commit checkpoints; big regenerations should be resumable (the
   `bootstrap_preagent.py` skip-if-done pattern).

## 6. Kill criteria (agree before starting, so sunk cost can't argue)

- Step 0 fails (no candidate traces in `review_neuron.mat`) AND Step 1's free
  experiments show < +0.003: stop; the remaining path (forward-only
  accumulation or 91 bootstrap re-runs) isn't worth it for a maybe-+0.005.
- Step 2/3: best feature set < +0.005 AUC or fails leave-one-animal-out: write
  the negative result into this doc and stop. 0.91 with a stable sub-1%
  operating point is a good place to rest; reviewer throughput matters more.
- Hard budget suggestion: ~2 days for Steps 0–1 before deciding on Step 2.

## 7. Quick repro commands

```powershell
cd C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\agent
# Reference sweep (copy from scratchpad or rebuild per §1 methodology):
C:\ProgramData\anaconda3\envs\valence\python.exe <sweep>.py
# Full diagnostic suite (LOO, pending score dists, Pareto sweep):
C:\ProgramData\anaconda3\envs\valence\python.exe diagnose_model.py
# Retrain (BLA, canonical):
C:\ProgramData\anaconda3\envs\valence\python.exe train_classifier.py --prospective-only --model xgboost
```
