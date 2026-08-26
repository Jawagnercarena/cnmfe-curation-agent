# Red-team brief: attack the bootstrap-label fix and the three model decisions

_Written 2026-08-26 for a fresh session on this machine. You were not part of the
work you are reviewing. That is the point._

## Your mandate

One session (2026-08-20 → 26) claims to have found a pixel-ordering bug that scrambled
every bootstrap training label since the pipeline was built, fixed it, re-labeled all
202 bootstrap sessions at 99.2–99.9% recovery, and redeployed both production
classifiers behind gates:

- **vCA1** (2026-08-24): bootstrap-contribution delta −0.014 → +0.009; agent weight
  changed from the dynamic sqrt formula (7.01x) to a fixed 5.0 via a new config knob;
  threshold kept at 0.05.
- **BLA** (2026-08-26): 8-seed paired AUC +0.0037 full / +0.0026 reviewed vs the pinned
  post-Step-4 baseline; threshold moved 0.06 → 0.04 (0.64% false-AR, 43.2% junk caught
  vs 0.80% / 33.8%); weight recipe unchanged at 4.0x.
- **Global model** (2026-08-26): pooling areas is null for BLA and vCA1 and worth
  +0.011–0.015 AUC for DG_AL (not implemented).

**The same person diagnosed, implemented, evaluated, and deployed all of it.** Assume
every step contains a same-author blind spot and find it. Three things should make you
suspicious before you read a line of code: (1) the recovery number (99.9%) is almost
too clean; (2) the model gains are much smaller than the label change would suggest —
the corpus went from ~250 correct bootstrap positives to ~7,900 and AUC moved +0.004;
(3) several decisions rest on 3-seed evidence with ±0.003 noise. If you cannot break it
after an honest day, say so.

## Hard constraints (non-negotiable)

- **Read-only toward all pipeline state**: no retrains, no writes to any session dir,
  npz, labels.mat, JSON, or model joblib; no watcher stop/restart; nothing on the
  kheirbek1 server; no `git` state changes. Write only inside your own scratchpad or a
  new `agent/eval/bootstrap_redteam_2026-08/` directory.
- Never delete or run anything against the 4 `_bootstrap_validate/` dirs or
  `D:\Julian_CNMFe\.bootstrap_diag\` (the only surviving pre-fix candidate caches).
  `validate_threshold.py --rematch-only` is now safe but do not rely on that.
- MATLAB: load-only. Two headless runs max on this box (255 GB; one BLA run ≈ 78 GB).
- Python: `C:\ProgramData\anaconda3\envs\valence\python.exe`.
- The corpus is LIVE (reviewer returns land; watchers may be restarted). Pin what you
  measure: record session lists + label mtimes with any number you report.

## The evidence you are attacking

- `docs/BOOTSTRAP_MATCHING_BUG_2026-08.md` — the full write-up (bug, fix, corpus,
  gates, global model, open items). Start here.
- `agent/eval/bootstrap_matching_2026-08/` — every script and result:
  `AUDIT.md`, `REPORT.md`, `GATE_B_REPORT.md`; `bmlib.py`, `a1_orientation_audit.py`
  (regression test), `a2_sandbox_requant.py` (+`a2_results.json`), `a3_damage_model.py`
  (+`a3_damage.json/csv`), `b1_run_pilot.py` (+`b1_pilot_log.json`), `b2_rerun_report.py`
  (+`b2_gate_b.json`), `c3_bla_gate8.py` (+`c3_bla_gate8_results.json`, uncommitted
  `c3_bla_gate8_oof.npz` = per-seed OOF vectors), `c3_weight_sweep.py`,
  `c3_assumption_checks.py`, `c5_global_model.py`, and the `*.log` files (uncommitted,
  on disk) for every run quoted in the docs.
- Commits on `feature-expansion-step4`: `4cf53e1` (fix), `de42f68` (trainer masking),
  `0dc0b41` (investigation package), `b553d31` (vCA1 deploy), `6a37ddd` (BLA deploy),
  `21a8749` (global model).
- Pinned baseline you compare against: `agent/eval/step4_2026-08/step5_results.json`
  (8-seed, 12,677-row agent pool, rankv2b_35: full 0.9246, reviewed 0.9123, T=0.06 at
  0.80% FAR / 33.8% junk).
- Data: every bootstrap session now has `bootstrap_match_stats.json` (schema_version 2:
  pairs + `ambiguous_candidate_indices` + `duplicate_candidate_indices` + params + dims
  + `recovery_by_threshold`) and `bootstrap_candidates.npz` (sparse candidate
  footprints, C_raw, full similarity matrix). Ground truth for 4 BLA sessions:
  `D:\Julian_CNMFe\BLA\.feature_expansion\<task>__<session>.mat` (reviewed candidate
  A + labels.mat in the parent).

## Attacks, in priority order

### A. Is the bug real, and is the fix complete?
1. Re-derive the orientation claim from scratch without `bmlib`: load one
   `spatial_footprints.mat` + its `A.txt`, prove which flattening matches MATLAB's
   linearization, then reproduce the stored March scores (bla21: 37/50, top-5
   0.944/0.924/0.881/0.865/0.802) and the fixed 50/50 independently.
2. Audit every place a MATLAB `(pixels, N)` matrix meets a numpy reshape — `AUDIT.md`
   lists 10 sites; find an 11th. Candidates: `curator.py` PDF rendering, `features.py`
   ring/neighbor features that index images, `extract_cand_traces.m`-derived
   `.feature_expansion` extractions (are THOSE A matrices F-order? a2 assumed so),
   `motion_qc`, `review_prep`, anything using `Coor.mat`.
3. The author claims the retro (agent-session) labels are unaffected because both sides
   are MATLAB-ordered. Verify on one agent session by recomputing its labels from
   `review_neuron.mat`/`neuron.mat` and diffing against `labels.mat`.
4. The retro feature path is claimed to transpose footprints, corrupting
   `cn_correlation` for retro-labeled sessions. Quantify: which sessions are retro-
   labeled, and does `cn_correlation` for them differ from a correctly oriented
   recomputation? Does fixing it move the model?

### B. Are the new labels correct — not just "matched"?
5. Sample 30 re-run bootstrap sessions × 5 matched candidates and check them against
   the curated PNGs / `neuron.mat` footprints. Recovery counts pairs above 0.45; you
   are checking that the pairs are the right cells.
6. Duplicates: 4,490 (BLA) / 6,623 (vCA1) rows are masked as "same-cell re-detections".
   Sample them. What fraction are actually distinct real cells (a second neuron
   overlapping a curated one) that would be a valid positive or negative? The a2
   ground-truth transfer left 65–109 per session "unknown".
7. The 10 unrecovered neurons in the pilot and the ~40 corpus-wide: are they dim cells
   (parameter issue) or matching failures? `per_curated_best_similarity` in the JSONs
   tells you where each one landed.

### C. Are the gates honest?
8. **Animal/FOV leakage.** CV groups by session, not animal. Bootstrap sessions from
   the same animal (and sometimes the same FOV/depth on adjacent days) as an agent test
   session sit in the training fold. The pinned baseline shares this structure so the
   *paired* deltas are fair, but the absolute AUCs and the "bootstrap contribution"
   delta may be inflated. Re-run the A/B with animal-level grouping (`-(bla\d+)-` /
   pnb ids / vCA1 animal numbers) and report both.
9. Seeds: the vCA1 deploy decision (w=5.0 vs 7.01x; FAR 1.2% vs 2.5%) and the
   duplicate-handling decision used 3 seeds. Re-run at 8. Does the vCA1 weight
   decision survive?
10. The BLA threshold 0.04 was chosen by the Step-5 rule (mean FAR ≤ 0.85%, worst seed
    ≤ 1.0%) on 8 seeds. Check that the same rule applied to the pinned baseline
    reproduces the historical 0.06 (self-consistency), and that 0.04 holds under
    animal-level grouping and on the reviewed-only stratum.
11. G5 (per-animal / early-era LOAO) was NOT re-run for BLA. Run it: does the fixed
    corpus help or hurt any animal, and the early era? The bootstrap corpus is the
    only source of 2022–2025-era labels, so early-era is where its value should show.
12. The `--eval` bootstrap ROI number (Cond A vs B) and the LOO diagnose number disagree
    in sign for vCA1 (+0.009 vs −0.011). Which protocol is right for the deploy
    decision, and does the answer change the decision?

### D. Why were the gains so small? (the question the author could not fully answer)
Treat each as a hypothesis with a test; report which ones hold.
13. **Label noise was smaller in feature space than in identity space.** The old
    "positives" were mirror-position candidates that CNMFe had detected and that
    resembled a real footprint through the transpose — i.e., mostly cell-shaped blobs.
    Features are position-invariant, so the old positive class may have been *mostly
    cells* even though they were the *wrong* cells. Test: score the OLD labels'
    positives (from the pre-fix JSONs — recover them from git history / the
    `.bootstrap_diag` shadow runs, or from a3's damage model) with the deployed model.
    If most score as cells, the fix mainly repaired the negative pool, which predicts
    exactly what was observed: little AUC movement, real movement in false-AR /
    junk-caught at the operating point.
14. **BLA bootstrap rows are feature-degenerate.** Under the v2 contract, bootstrap rows
    carry zero-filled v2b (8 of 35 columns, the ones that delivered Step 2's +0.02) with
    `v2_present=0`. Clean labels can only teach the 13-col + rank part. Test: compute
    real v2b for bootstrap rows from `bootstrap_candidates.npz` C_raw (the traces are
    from the same run as the labels now) with `features.compute_v2b_features`, set
    `v2_present=1`, and re-run the 8-seed gate. This is the single most likely lever
    for a larger gain — and the riskiest (train/inference trace-regime shift).
15. **Agent data dominates the effective training mass** (52k weighted agent rows vs
    ~29k bootstrap after masking at 4.0x). The per-weight sweep showed junk-at-matched-
    FAR flat from w=1 to 7 — read that as "bootstrap adds little marginal information
    about the agent test distribution". Test: learning-curve the bootstrap pool (0%,
    25%, 50%, 100% of clean bootstrap sessions) at fixed weight; if the curve is flat
    the corpus is redundant with agent data for this test set.
16. **Bootstrap positives are easy.** Pre-agent curation kept cells the old pipeline
    found; the re-run finds them at 0.98 similarity — bright, unambiguous. The model's
    residual errors are dim reals in the 0.02–0.10 score band. Test: compare score
    margins of bootstrap positives vs agent positives; what fraction of bootstrap
    positives fall in the hard band?
17. **The test set is the ceiling.** Reviewer label noise bounds agent-fold AUC. Test:
    estimate reviewer consistency from sessions with two reviewers or from
    `reviewer_quality.py`; if the ceiling is ~0.93, +0.004 is a large share of the
    remaining headroom.

### E. The global model
18. Reproduce `c5_global_model.py` with animal-level grouping and 8 seeds. DG_AL has 9
    sessions from 2 animals (DG6D, DG6E): is the +0.015 real or a 2-animal artifact?
19. The pooled model's score scale collapses junk-caught at a fixed T (DG 45% → 5–10%).
    Is a pooled DG model usable at all without a DG-specific calibration layer?

## Deliverables

`agent/eval/bootstrap_redteam_2026-08/redteam_report.md` with, per attack: what you
ran, the numbers, PASS / FAIL / INCONCLUSIVE, and — for the §D hypotheses — which
explanations of the modest gain are supported. End with a verdict on each deploy
(vCA1 w=5.0 @0.05; BLA @0.04) and a recommendation list ranked by expected AUC/FAR
impact for the follow-up work below.

## Follow-up work this brief gates (do not start these until the report is in)

- **vCA1 v2 (35-col) contract** — `docs/VCA1_V2_BRIEF.md`. The global-model result
  removes the "coordinated two-area swap" contingency (pooling is null for vCA1), so
  vCA1 v2 is a standalone project. Attack #14 is its design question in miniature.
- **DG_AL pooled prior** (BLA+vCA1 rows at ~0.3 weight) — only if attack #18 holds.
- **Retro `cn_correlation` fix + feature refresh** — sized by attack #4.
- ACORN stats refresh (BLA now 0.64% FAR / 43% junk auto-caught / AUC 0.928 full,
  0.915 reviewed; vCA1 ~1.2% / ~44%): after the report, so the numbers survive it.
