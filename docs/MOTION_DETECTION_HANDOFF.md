# Handoff: catching motion artifacts — findings, the QC flag, and the frame-precise-coincidence R&D

_Status date: 2026-07-28. Companion to memory `project_motion_features`. Supersedes the motion parts of `MOTION_AND_RESUMABLE_REVIEW_HANDOFF.md` (the "collect labels then add coherence/kinetics" plan there is now answered — see below)._

## Why this matters (the ceiling)

Motion deletes are a **large fraction of all deletes** in a session (observed ~15–50%, sometimes ~half; likely undercounted where reviewers don't tag every one). The classifier **cannot catch them** — see below — so its garbage-rejection is *capped*: even a perfect model on the non-motion deletes leaves the motion half for the human. Motion detection is therefore a hard ceiling on the whole auto-curation value proposition, not a nice-to-have.

## What we established (2026-07-27/28) — four approaches, one weak lead

All tested honestly on the 8 motion-tagged BLA sessions (171 `(m)` tags across bla37=129, bla36=21, bla12=19, bla16=2), with **leave-one-animal-out** (holding out bla37, which is 75% of tags, is the decisive test). The task that matters is **Q2: separate motion-deletes from *other* deletes.**

| approach | leave-bla37-out Q2 AUC | verdict |
|---|---|---|
| deployed 13 features (baseline) | 0.616 | — |
| motion-delete **upweight** on existing feats | n/a (inert; motion scores ~0.49, i.e. keep-like) | dead |
| footprint **spatial-stability** (windowed footprint↔video corr) | 0.637 combined (+0.02) | dead-ish |
| **population coherence** (pop_coherence/pop_sync_frac) | 0.611 combined; pop-alone 0.461 (**below chance**) | dead (cohort-confounded) |
| **local-motion-vectors** (LK optical flow in the raw-movie patch) | **0.648 combined (+0.03)** | real but modest — the lead |

**Root cause of the three failures:** CNMF-E fits a clean *static* footprint + smooth trace to the artifact, so in the extracted representation a motion cell *is* a well-formed, real-looking, stably-firing cell. Footprint/trace/population features can't see the motion.

**Why local-motion is different and real:** it measures physical motion **directly from the raw pixels**, bypassing CNMF-E. Motion cells sit in ~1.7× higher-motion locations (`mot_mean` 1.84 vs keeps' 1.10); the model ranks `mot_mean` #3 of 19. But it plateaus at ~0.65 because (a) the signal is "high-motion *neighborhood*", which is weak/overlapping and partly a session-level property, and (b) the **frame-precise** signal — does the transient coincide with a motion *event* — came back **flat** (`mot_trace_corr` didn't separate). The current estimator is a coarse single-region **rigid** Lucas-Kanade solve; it almost certainly misses that.

Key correction to earlier thinking: the per-frame motion reference is **not** unreachable. The upstream global registration shifts are gone, but the movies are only *globally* corrected — **residual local (z / non-rigid) motion survives in the retained `.tif`/`.mat` movies**, and we can measure it ourselves.

## What shipped: the QC flag (advisory, no model risk)

`agent/motion_qc.m` (+ `run_motion_qc.m`) scores each candidate by within-session local-motion severity, flags the top quartile, and writes `motion_qc.mat`, a red-tinted `motion_qc.jpg`, and a ranked report. It **deletes nothing and touches no model / feature contract** — it just points the reviewer's eye at likely-motion cells (attacking the ceiling from the human-efficiency side).

**Measured reliability (self-check vs `(m)` tags):** 1.5–2.0× enrichment in 5/8 sessions (recall 40–52%, precision 25–46%), but **worse-than-chance (0.6×) in 2/8** (bla12-660um, bla37-216um). The inconsistency fits the mechanism: LK catches **lateral x-y** drift but **misses axial z** (a plane-change is a structure/brightness change, not a shift). So today it's a *marginal* aid — usable as a soft hint, not something to rely on. Improving it is exactly the R&D below.

Depends on `motion_vec.mat` from `extract_motion_vectors.m` (computes it on demand if absent; ~1–2 min/session to load the movie + run LK).

## The R&D: frame-precise coincidence + structure-change motion

**Hypothesis.** A motion artifact's *transient* is a motion event: the cell "fires" exactly when its patch physically jumps (x-y drift) or restructures (z plane-change). A real cell brightens **in place** with the patch structurally unchanged. So the discriminating signal is the **frame-precise coincidence** between transient onsets and local motion/structure-change — not the average motion level (which is what `mot_mean` captures and is only weakly separable).

**Why it should beat 0.65.** It targets the causal mechanism and is designed to catch **both** artifact types (lateral drift *and* axial z), whereas the current flag catches only lateral — which is precisely why the flag fails on z-dominant sessions.

### Concrete plan

1. **Better per-frame local-motion signals** (in the footprint patch, from the raw movie):
   - **Subpixel displacement** via phase correlation (FFT-based) instead of the coarse rigid LK — gives a cleaner per-frame `(dx,dy)` and magnitude `|d|(t)`.
   - **Structure-change / decorrelation** `s(t) = 1 − corr(patch(t), resting-patch)` — catches **z** plane-changes that produce little lateral shift. (A crude version, `decorr_active`, already exists in `motion_vec` and ranked #9; make it per-frame and onset-aligned.)
   - Optionally local **non-rigid** flow (e.g., a small Farnebäck/optical-flow field) to capture a neighbor deforming in.

2. **Detect transient ONSETS** from the trace (not just "active frames"): rising edges where `C_raw` crosses baseline + k·σ. The artifact is created *at* the onset, so align there.

3. **Coincidence features** (the new signal):
   - motion/structure elevation in a tight window (±few frames) around each onset vs a matched baseline (shuffled/off-onset frames).
   - fraction of onsets that land on a **motion/structure event** (a peak in `|d|(t)` or `s(t)`).
   - a null via circular-shifting the trace vs the motion series (to get a per-cell significance, not just a magnitude) — this controls for "busy session" confounds that plagued `mot_mean`.

4. **Validate before any deploy.** Backfill on the 8 sessions (all materials are local), score with the existing harness (`test_motion_vectors.py` pattern: existing-13 vs new vs combined, Q2/Q1, **leave-one-animal-out**). Bar to clear: leave-bla37-out Q2 **materially above the current 0.648 plateau** (target ≳0.72) and, unlike `mot_mean`, **consistent across held-out animals**. Also re-run `motion_qc` self-validation — the real success metric is enrichment that's reliable across sessions (no worse-than-chance ones), especially z-dominant sessions.

5. **Get more animals.** Everything here is limited by 3–4 animals with tags. Keep collecting `(m)` spread across animals/tasks; re-run as it grows.

### If it validates — deployment options
- **Enhanced QC flag** (preferred first step): swap the better signal into `motion_qc.m`. No feature-contract swap, no model risk; immediately upgrades the reviewer aid.
- **Classifier feature** (only if the gain is large): the full **atomic feature-contract swap** (rebuild every `candidate_features.npz`, retrain, restart the watcher, per area — see `project_motion_features` / `refresh_features.py`) **plus** adding heavy per-candidate optical-flow-over-the-movie to *live* curation (~minutes/session). The ceiling argument (motion = up to half of deletes) is what would justify that cost; a modest +0.03 would not.
- **Motion-aware objective:** only worth it once a feature genuinely separates motion — then upweight `motion_delete` rows (path b) so the model prioritizes them. Useless before that (an upweight can't manufacture signal the features lack — proven 2026-07-27).

### Scaffold (all untracked, in `agent/`)
`eval_motion.py` (separability gate) · `prototype_motion_upweight.py` (upweight, dead) · `extract_motion_diag.m`/`test_spatial_stability.py` (footprint stability, dead) · `extract_cand_traces.m`/`test_pop_coherence.py` (population, dead) · `extract_motion_vectors.m`/`run_motion_vec.m`/`test_motion_vectors.py` (**local motion — the lead; extend the per-frame signals here**) · `test_motion_combined.py` (stacked best-case) · `motion_qc.m`/`run_motion_qc.m` (**the shipped flag — upgrade its severity signal here**).

_Nothing above changes the deployed model; the QC flag is advisory. The frame-precise work is a prototype-and-validate effort gated on beating 0.65 leave-one-animal-out before any feature-contract change._
