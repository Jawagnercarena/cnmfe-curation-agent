# Red-team brief: attack the Step 2 feature-expansion result

_Written 2026-08-18 for a fresh session on this machine. You were not part of
the work you are reviewing. That is the point._

## Your mandate

A same-day investigation claims the BLA classifier's 13-feature representation
was the ceiling, and that a 35-column variant (**rankv2_35**) clears every
promotion gate: OOF AUC 0.8921 → 0.9137 (+0.0216, 8 seeds) on the
reviewed-only pool, false-AR at matched junk 0.86% → 0.11%, leave-one-animal-
out improved on all six animals, early-era holdout 0.844 → 0.874.

**Assume this result is too good to be true, and find out why.** Your job is
not to corroborate; it is to break it. If you cannot break it after an honest
day, say so — that is the strongest confirmation available. The author already
found and fixed two of their own bugs (a label leak via deployed-model neighbor
scores; a noise-firing event detector), which means the process produces bugs
at a known nonzero rate and a third may exist.

The deployment this gates (Step 4) is the pipeline's riskiest operation
(atomic feature-contract swap across every session npz + retrain + threshold
re-derivation + watcher restarts + three-area coordination). A wrong
green-light is expensive; a wrong red-light wastes a day. Calibrate
accordingly.

## Hard constraints (non-negotiable)

- **Read-only toward all pipeline state**: no retrains, no writes to any
  `candidate_features.npz` / `labels.mat` / model joblib, no watcher stop or
  restart, no `git stash` operations, nothing on the kheirbek1 server. Do not
  run `refresh_features.py --write-forward` under any circumstances.
- Write only inside your own session scratchpad.
- `D:\Julian_CNMFe\BLA` is local disk: read anything; create nothing outside
  your scratchpad.
- Python: `C:\ProgramData\anaconda3\envs\valence\python.exe`. MATLAB R2023b at
  `C:\Program Files\MATLAB\R2023b\bin\matlab.exe` (load-only use).
- The pool has likely DRIFTED since the analysis (reviewer returns land daily).
  All exact-reproduction work must use the pinned inputs (below), not a live
  rescan. If you rescan live, label those numbers as a different pool.

## The evidence you are attacking

- `docs/FEATURE_EXPANSION_HANDOFF.md` — the original scoping doc.
- `docs/FEATURE_EXPANSION_GATE_2026-08-18.md` — Steps 0–1 (baseline re-pin,
  16-cell autopsy, zero-cost experiments).
- `docs/FEATURE_EXPANSION_STEP2_2026-08-18.md` — the Step 2 result under review.
- `agent/eval/step2_2026-08/` — every script that produced the numbers.
- `D:\Julian_CNMFe\BLA\.feature_expansion\` — extracted candidate data
  (79 sessions: C_raw, sparse A, Cn from each session's `review_neuron.mat`).
- `D:\Julian_CNMFe\BLA\.feature_expansion\_pinned\` — the pinned inputs:
  `pool_manifest.json` (the exact 170-session pool), `baseline_oof.npz`
  (8-seed OOF scores of the 13-feature baseline), `step2_v2_features.npz` /
  `step2_v2b_features.npz` (the computed features), `step2_eval.json` /
  `baseline_repin.json` (result numbers), `autopsy_false_ar.csv`,
  `gate_report.md`, `step2_sessions.txt`.

What the features are (plain language): the model's original 13 features are
absolute measurements. The variant adds (a) each feature's **within-session
percentile rank** — its standing among the candidates of its own recording,
because "real" is partly session-relative and the corpus drifted bright; and
(b) **8 per-candidate trace/footprint features** computed from the candidate's
own trace and footprint: event rate, event SNR, event-shape stereotypy (does
the trace repeat one stereotyped transient shape), decay/rise asymmetry,
GCaMP-plausible-envelope fraction, max correlation with a high-confidence
neighbor within 60 px, max correlation with any neighbor, and Cn ring contrast
(is the footprint its own correlation-image peak). Plus a `v2_present` flag
(0 on bootstrap rows, which have no candidate-level data).

## Named attack surfaces, in priority order

**C1 — Provenance of review_neuron.mat (the foundation; kill-switch).**
Every v2 feature derives from `review_neuron.mat`. The claim: it is written at
CURATION time (`curator._write_review_mat`, curator.py:477-519) and contains
the pre-decision review set, so no feature can encode the reviewer's decision.
Attack: audit mtimes of `review_neuron.mat` vs `labels.mat` across all 79
sessions (expect review_neuron older on every one); grep the whole repo
(MATLAB included, esp. `CNMFe_final_save.m` and `viewNeurons*.m`) for any code
path that rewrites `review_neuron.mat` after review; check N in a few files
against `n_candidates − auto_rejected`. **If this fails on any session,
everything downstream is void — stop and report.**

**C2 — Independent recomputation of the headline (do NOT reuse harness.py).**
Write your own evaluation from scratch: reviewed-only pool per
`pool_manifest.json`, 5-fold StratifiedGroupKFold grouped by session, seeds
[42, 1, 7, 13, 100, 2024, 31337, 9], agent sessions with ≥5 reals as eval
pool, bootstrap train-only with deployed weights (agent 4.0×, bad-bootstrap
0.4×, ambiguous masked — use `train_classifier` helpers), XGB 300/0.05/4/
0.8/0.8. Compare b13 vs rankv2_35 built from the pinned feature npz. State
your numbers next to the claimed 0.8921 / 0.9137.

**C3 — Remaining leakage in the 8 features.**
The fixed version selects high-confidence neighbors by grouped-OOF scores
(models never trained on the session in question — verify that in
`repin_baseline.py`). Attack residuals: is there any other path by which a
feature can see labels or post-review artifacts? Check `compute_v2_features.py`
end to end (inputs: `.feature_expansion` mats, `candidate_features.npz`,
`baseline_oof.npz`, deployed joblib for 4 non-pool sessions only). Also test:
drop `nb_corr_max` entirely and re-run — does the win survive without the one
feature that uses model scores at all?

**C4 — The v2_present flag.**
Constant 1 across the entire eval pool (agent rows), 0 only on bootstrap
training rows. Claim: it cannot discriminate within the eval pool. Verify, and
re-run with the flag column deleted — the result should be unchanged; if it
moves materially, understand why before anything else.

**C5 — Analyst degrees of freedom / autopsy circularity.**
The features were designed after inspecting 16 specific false-AR cells.
16 cells cannot move AUC on a 9k-real pool, but verify: exclude those 16 (list
in `autopsy_false_ar.csv`) from the eval pool and re-measure the delta. Also
consider: two detector variants were tried and both reported — is the
selection between them post-hoc? (Both pass gates, so the claim survives
either choice; confirm.)

**C6 — Gate robustness.**
The LOAO and early-era gates are single deterministic fits. Add spread (vary
XGB random_state over ≥3 values) and re-check the bla21 (n=2 sessions) and
bla16 cells especially. Does "improves on all six animals" survive seed noise?

**C7 — Per-feature attribution (unmeasured — the authors admit this).**
Which of the 8 carries the win? Group-ablate: events-only (5), neighbors-only
(2), ring-only (1), ranks-only vs full. If one fragile feature carries
everything, the deploy decision changes. Recommend which columns actually ship.

**C8 — Detector choice.**
v2 (noisy 2.5σ detector) vs v2b (shape-qualified 3.5σ): 0.9137 vs 0.9127 —
near-tie, but v2's event features fail the mechanism check on the marquee
cells while v2b passes. Run the pair at 8 seeds with paired per-seed deltas and
recommend which ships (statistical tie → prefer the mechanistically honest one;
confirm or refute that reasoning).

**C9 — Operating-point reality.**
The fixed-window count of reals in [0.05, 0.12) GROWS under the new models
(score recalibration). Produce the new model's full threshold sweep (false-AR /
junk curve, diagnose_model §3 style) and state the actual recommended
threshold and its operating point — the deploy decision needs the real number,
not the matched-op abstraction.

**C10 — The Step 4 training-data confound (design question, not a bug hunt).**
Historical backfill can never have v2 for auto-rejected candidates (their
traces were never saved). The Step 2 eval dodged this by excluding auto-
rejected rows from the agent pool. At Step 4, training data must either (a)
exclude those rows (changes the negative class), or (b) include them with
v2=0 + flag=0, creating flag↔label correlation in agent history that inverts
in production (all flag=1). Analyze both; recommend one, with evidence.

## Deliverable

One report (your scratchpad, plus tell the user where it is):
1. Verdict per claim C1–C10: CONFIRMED / REFUTED / UNRESOLVED, each with the
   evidence (your numbers, not restatements of the docs).
2. Any NEW hole found outside the named surfaces.
3. Overall: deploy-worthy as claimed / deploy-worthy with changes (name them) /
   not deploy-worthy (why).
4. If deploy-worthy: a concrete Step 4 execution outline (order of operations,
   what can break, verification at each step), including your C10 decision and
   C9 threshold.

Budget: ~1 day. Kill criteria: C1 or C2 fails → stop immediately and report;
do not continue to the later claims on a voided foundation.
