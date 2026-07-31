# Handoff: motion-detector integration + resumable review

Two independent pieces of future work from the 2026-07 motion push.

- **Part 1** is an execution plan for wiring the motion labels into the model — **gated on collecting enough `(m)` labels first.**
- **Part 2** is a design for letting reviewers pause/resume a long review so a forced restart doesn't wipe their work — **no code written yet; this documents the options and trade-offs.**

_Status date: 2026-07-26. Background: memory notes `project_motion_features`, `project_retrain_plan`, `project_bootstrap_matching_study`._

---

## Part 1 — Integrating the motion detector

### 1.0 Where we are today

Reviewers can already tag motion artifacts with **`(m)`** during review (shipped, commit `393977d`). `(m)` deletes the neuron exactly like `(d)` **and** records it as a motion artifact:

- [viewNeurons.m](../ca_source_extraction/@Sources2D/viewNeurons.m) and [viewNeuronsVideo.m](../ca_source_extraction/@Sources2D/viewNeuronsVideo.m) accumulate each tagged footprint in the global `MOTION_DELETE_FP`.
- [CNMFe_final_save.m:427-456](../CNMFe_final_save.m#L427-L456) cosine-matches those footprints (0.60 threshold) back to the review candidates and writes a `motion_delete` vector as an **extra** variable in `labels.mat`, alongside the binary `labels` (keep=1 / delete=0).

**The trained classifier does not use `motion_delete` yet.** [train_classifier.py:287](../agent/train_classifier.py#L287) reads only `labels`. To the model, an `(m)` is identical to a `(d)`. This is deliberate — collect the labels first, then decide how to use them.

**Current label stock** (scan 2026-07-23, `scratchpad/scan_motion.py`):

| Area | Sessions with `(m)` | Total motion tags | Notes |
|------|--------------------|-------------------|-------|
| BLA  | 3 | 54 | cohort-concentrated — 35/54 from one 6odorDualDiffRew cohort (bla36+bla37) |
| vCA1 | 0 | 0 | no reviewer has used `(m)` here yet |

### 1.1 Precondition — enough labels, well spread (the gate)

Do **none** of the steps below until, **per area**:

- **~100+ motion tags across ≥8–10 sessions** (rule of thumb from `project_combined_model_heuristic`), and
- those tags are **spread across animals *and* tasks**, not one cohort. The original motion signal was confounded with cohort; concentrated labels will "validate" a feature that is really just learning the cohort.
- vCA1 needs `(m)` usage to even begin — it has zero today.

Gauge progress with `scan_motion.py`, or better, fold a per-area motion tally into `diagnose_model.py` (the "readout" offered separately) so every retrain prints how close we are.

### 1.2 Step 1 — Build the motion-separability eval (does not exist yet)

This is the **go/no-go gate and the missing tool.** What exists today — `refresh_features.py --validate-motion` — only measures whether the motion *missing-indicator* moves the keep/delete AUC. It does **not** evaluate `motion_delete` as a target class.

Write an eval (new script, e.g. `agent/eval_motion.py`) that:

1. Loads `candidate_features.npz` + `labels.mat` per session; builds `y_motion = motion_delete` (positive = motion), reconstructed onto the reviewed subset the same way [train_classifier.load_prospective_session](../agent/train_classifier.py#L272) reconstructs labels.
2. Runs **grouped-by-session** CV (`StratifiedGroupKFold`, mirror `train_classifier._grouped_cv`), and additionally a **leave-one-cohort-out** split to defeat the confound.
3. Answers two questions:
   - **Q1:** do the features separate motion-deletes from **keeps**? (can we flag them at all)
   - **Q2:** do they separate motion-deletes from **other (non-motion) deletes**? (the harder, more useful question — is "motion" its own signature, or just "bad neuron"?)
4. Measures the lift from the staged features (`motion_data_present`, `pop_coherence`, `pop_sync_frac`, and any kinetics) **specifically on motion separation**, not overall AUC.

**Decision rule:** proceed only if some feature set gives a materially-better-than-chance, **cohort-robust** separation (Q2 matters most). If motion isn't separable even with the new features, stop — more/better labels or a real motion reference (per-frame motion-correction shift vectors, not currently saved) is needed.

### 1.3 Step 2 — Choose how the model consumes motion labels

Driven by 1.2:

- **(a) Feature-only, single model** — if the features separate motion within the ordinary keep/delete classifier, just deploy the forward-only feature schema and let the binary model benefit. Simplest. *Risk:* motion is a minority of deletes, so a generic keep/delete objective under-weights it; the feature can be effectively inert (XGBoost won't split on it) if its utility is diluted — this was the 2026-07-13 realization.
- **(b) Motion-aware weighting** — upweight `motion_delete` rows via `sample_weight` in the existing binary model so it pays more attention to catching them. Cheap; stays one model + one threshold per area.
- **(c) Dedicated motion sub-model** — a second binary classifier (motion vs. everything) after the keep/delete stage, with its own threshold. Most expressive, most maintenance (two models × two areas). Its positives are only the `(m)` subset, so it needs a **lot** of tags. Reserve for when Q2 is strongly positive and motion-positive volume is high.

**Recommendation:** start with **(a)+(b)** — deploy the features forward-only and add a modest motion-delete upweight — before considering (c).

### 1.4 Step 3 — Deploy = the atomic feature-contract swap

The feature matrix is **positional** and the model stores **no** feature names ([train_classifier.py:893-907](../agent/train_classifier.py#L893-L907) dumps scaler/clf/threshold/… but not names). Adding features means rebuilding **every** `candidate_features.npz` + retraining + restarting the watcher as **one atomic operation, per area** (BLA and vCA1 separately):

1. **Stop the watcher** ([watcher.py](../agent/watcher.py) / [watcher_vCA1.py](../agent/watcher_vCA1.py)). It holds `features.py` in memory and **drives loose-`.tif` conversions — never restart it mid-conversion.**
2. **Finalize the feature code.** `git stash@{0}` ("wip: motion features (#1 indicator + #2 coherence infra)") holds the `features.py` + `train_classifier.py` edits. Finalize the schema per the 1.2/1.3 decision (drop coherence/kinetics if they didn't validate).
3. **`python refresh_features.py --write-forward`** — migrates every npz to the new column order (asserts against `features.FORWARD_FEATURE_NAMES`). Historical rows get neutral zeros + `present=0` (**forward-only**; trace features can't be backfilled — the candidate traces are gone, see `project_bootstrap_matching_study`). Run only with the watcher down.
4. **`python train_classifier.py --prospective-only`** (`--dry-run` first). vCA1 uses [train_classifier_vCA1.py](../agent/train_classifier_vCA1.py).
5. **Re-sweep `reject_threshold`** with [diagnose_model.py](../agent/diagnose_model.py) (BLA) / [diagnose_model_vCA1.py](../agent/diagnose_model_vCA1.py) (vCA1). The current thresholds (BLA **0.14**, vCA1 **0.05**) were swept for the 13-column model and **will shift** with new features.
6. **Restart the watcher.** A restarted watcher loads the new `features.py` and **requires a matching new model**, or curation crashes on `scaler.transform` width. Code + npz + model must all move together.

### 1.5 Step 4 — Validate honestly

Validate via a **real** `train_classifier.py --dry-run` / `diagnose_model.py`, **not** the `refresh_features.py` harness — that harness under-reports (0.754 vs 0.874 AUC; 2.1% vs 0.8% false-AR at 0.14) because of how it does CV. Confirm false-AR stays sub-1% **per area** before trusting the new threshold.

### 1.6 Files involved

| File | Role |
|------|------|
| `viewNeurons.m`, `viewNeuronsVideo.m` | capture `(m)` → `MOTION_DELETE_FP` |
| `CNMFe_final_save.m` | matches tags → writes `motion_delete` in `labels.mat` |
| `agent/features.py` | feature defs; staged coherence/kinetics live here (in stash) |
| `agent/refresh_features.py` | `--write-forward` migration; `--validate-motion` (indicator only) |
| `agent/eval_motion.py` | **to be written** — the motion-separability gate (1.2) |
| `agent/train_classifier.py` (+ `_vCA1`) | training; add motion upweight here for (b) |
| `agent/diagnose_model.py` (+ `_vCA1`) | threshold re-sweep |
| `agent/watcher.py` (+ `_vCA1`) | must be stopped/restarted around the swap |

---

## Part 2 — Resumable review (design only — no changes yet)

### 2.0 The problem

A final review is a long interactive MATLAB session — the current BLA queue has a **166-neuron** session awaiting review ([REVIEW_QUEUE.md](d:/Julian_CNMFe/BLA/REVIEW_QUEUE.md)). The reviewer steps through candidates one at a time. **Today nothing is written until the very end** of `CNMFe_final_save.m`: `labels.mat` and the outputs are saved only after *all* steps complete ([CNMFe_final_save.m:456](../CNMFe_final_save.m#L456), [488+](../CNMFe_final_save.m#L488)). If MATLAB dies mid-review (forced OS update/restart, crash, accidental window close), the **entire session's work is lost** and the reviewer restarts `run_final_review.m` from zero. There is no autosave and no "save & quit."

(The recent crash-hardening, commit `f14f5ee`, stops a bad contour from *aborting* a review — but it does nothing for a *killed process*. That's what this feature is for.)

### 2.1 What "progress" actually is — two granularities

The review isn't one loop; it's a pipeline of interactive figures interleaved with expensive, destructive re-estimation passes (`updateTemporal`/`updateSpatial` ~1–2 min each, background reconstruct, merges). So progress lives at two layers:

- **A. Pipeline-step state** — which STEP we've reached (STEP 1 `viewNeurons` → 1b/1c updates/merge → STEP 2 video → STEP 3 merge → STEP 4 update → STEP 4b final loop) and the current, progressively-**mutated** `neuron` object. The coarse "where in the workflow."
- **B. In-figure state** — within a single long `viewNeurons`/`viewNeuronsVideo` pass, the per-neuron decisions made so far. Both files share the *identical* local state ([viewNeurons.m:43-45](../ca_source_extraction/@Sources2D/viewNeurons.m#L43-L45), [viewNeuronsVideo.m:37-39](../ca_source_extraction/@Sources2D/viewNeuronsVideo.m#L37-L39)): `ind_del`, `ind_motion`, `ind_trim`, `Amask`, and the cursor `m` — all applied only at the *end* of the figure ([viewNeurons.m:161-167](../ca_source_extraction/@Sources2D/viewNeurons.m#L161-L167)). The fine "where in the 166 neurons."

**The 166-neuron marathon is a single figure, so layer B is where the real pain is.**

### 2.2 What to persist / what not to

**Persist:**
- `neuron` (the `Sources2D` object) — carries all curation state; tens of MB for a typical session, cheap to save.
- The in-figure struct: `{ind, ind_del, ind_motion, ind_trim, Amask, m}` + which figure/step we're in.
- `MOTION_DELETE_FP` (recoverable from `ind_motion` + `neuron.A`, but simplest to save).
- A **fingerprint** (see 2.4).

**Do *not* persist:**
- `Y`, the raw video (~3.5 GB in RAM). It is reloaded from the session `.mat` on entry ([CNMFe_final_save.m:79-133](../CNMFe_final_save.m#L79-L133)) and the background is reconstructed via the `Ybg_weights.mat` **fast path** (~30–60 s, [lines 138-168](../CNMFe_final_save.m#L138-L168)). **This is why resume is feasible:** a resume re-runs a cheap ~1-min preamble instead of checkpointing gigabytes.

### 2.3 Design options

- **Option C — "save & quit" hotkey (smallest).** A key (e.g. `q`) inside `viewNeurons`/`viewNeuronsVideo` that writes the in-figure state and exits cleanly. Handles **planned** leaves only; does **not** survive a forced restart (no chance to run on-exit code). A stepping stone.
- **Option A — step-level checkpoint (coarse, robust).** After each completed step / update pass in `CNMFe_final_save.m`, save `neuron` + `MOTION_DELETE_FP` + a step marker to `review_checkpoint.mat`. On entry, if a valid checkpoint exists, load `neuron` from it instead of `review_neuron.mat` and jump to the next step. Reuses the existing neuron save/load; minimal new logic. Reviewer loses **at most the current figure**. Does not resume *within* a figure.
- **Option B — in-figure autosave + resume (fine).** Modify both figure functions to (i) **autosave the in-figure state after each keypress** (the vectors are tiny; write `Amask`/`neuron` only when a trim/split actually mutates them), and (ii) on entry, detect a matching checkpoint and restore `ind_*`/`Amask`/`m`, resuming at the saved cursor. **Survives forced restarts** because it saves continuously. Best UX for the marathon figures.

**Correctness tie between A and B:** the in-figure decision vectors are keyed to the exact neuron set `ind` they were taken against. Re-estimation passes reorder / re-estimate / merge / split neurons, so a checkpoint is only valid against the identical `neuron` state. **B must be paired with A** — checkpoint the neuron at the step boundary, and trust in-figure vectors only if the loaded neuron's fingerprint matches.

**Recommendation:** A + B, **phased**. Ship **C or A first** (the safety net), then **B** for the long figures. A alone already removes most of the pain for multi-step sessions; B is what makes a single 166-neuron figure resumable.

### 2.4 Correctness guards (all options)

- **Fingerprint:** store `session_dir` name, `N = size(neuron.A,2)`, a cheap footprint checksum (e.g. `norm`/`sum` of `neuron.A`), and a **schema/version tag**. On resume, refuse to restore on any mismatch — fall back to a fresh start rather than silently applying stale decisions to the wrong neurons.
- **Code-version tag:** if `viewNeurons`/`CNMFe_final_save` change, bump the tag so old checkpoints are ignored (their semantics may have drifted).
- **Delete the checkpoint on successful finalize** (after `labels.mat` + outputs are written), so a completed session never resumes into a stale checkpoint. On entry with a checkpoint present, prompt **"resume or start over?"**.

### 2.5 Interactions / landmines

- **A forced OS restart gives no on-exit hook** — so real crash protection **requires periodic autosave** (Option B), not just a save-on-quit key (Option C).
- **The checkpoint filename must not look like completion** to downstream scanners. Verified-safe signals: `watcher.py` keys on `neuron.mat` / `review_neuron.mat` / `ROIs.jpg` ([watcher.py:200](../agent/watcher.py#L200), [331-333](../agent/watcher.py#L331-L333), [349](../agent/watcher.py#L349)); `ingest_returns.iter_sessions` keys on `labels.mat` / `neuron.mat`; `train_classifier` discovery keys on `candidate_features.npz` / `labels.mat` / `ROIs.jpg`. A name like **`review_checkpoint.mat` collides with none of them.** Do not write anything named those as part of a checkpoint.
- **Single-machine by nature:** reviewers work on their own machines and only push a folder back when done ("_Out for review — don't hand these to another machine_"), so the checkpoint stays local and won't ship mid-review. Keep it in the session folder.
- **Splits/trims:** a split appends neurons to `neuron` beyond the reviewed `ind`; a trim modifies `Amask` (applied at figure end). A mid-figure checkpoint must capture these, or be taken only at safe points.

### 2.6 Files to touch (when implemented)

- `ca_source_extraction/@Sources2D/viewNeurons.m` **and** `viewNeuronsVideo.m` — identical state model; add hotkey/autosave/restore to **both**.
- `CNMFe_final_save.m` — step markers, checkpoint write at step boundaries, resume-on-entry logic, checkpoint deletion on finalize.
- (optional) a short reviewer note in `REVIEW_SETUP.md`.

### 2.7 Rough effort

- **C:** small (a few lines per figure file). **A:** small–moderate (checkpoint I/O + entry branching in `CNMFe_final_save`). **B:** moderate (both figure functions + fingerprinting + resume UX).
- **No Python or model changes** — this is entirely MATLAB, reviewer-side, so it ships to reviewers via `git pull` exactly like the crash fix.

---

_This document is a plan, not a change. Nothing here has been executed: Part 1 waits on the label gate (1.1); Part 2 waits on a decision about which option(s) to build._
