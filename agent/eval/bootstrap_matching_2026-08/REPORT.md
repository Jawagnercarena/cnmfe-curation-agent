# Bootstrap matching investigation — Phase A findings (USER GATE A)

Date: 2026-08-20. Scripts: `a1_orientation_audit.py`, `a2_sandbox_requant.py`,
`a3_damage_model.py` (+ `bmlib.py`), all read-only on D:. Results:
`a2_results.json`, `a3_damage.{json,csv}`. Companion: `AUDIT.md`.

## 1. Root cause — a single pixel-ordering bug, not a fundamental limitation

`bootstrap_preagent.py:245` flattens candidate footprints in numpy C-order while
`A_final` (from MATLAB `neuron.mat`) has column-major pixel rows. Every similarity the
bootstrap matcher ever computed compared candidates against **transposed images** of the
curated footprints. Matches therefore land at the mirror position across the image
diagonal (median 6.8 px from mirror vs 130 px from true position on bla21).

Proof chain (all reproducible via a1):
- Synthetic asymmetric footprint: production formula scores it **0.0000** against itself;
  the orientation-consistent formula scores 1.000000.
- bla21-313um sandbox: mismatched replica reproduces the stored pipeline scores to the
  3rd decimal (37/50 matched, top-5 0.944/0.924/0.881/0.865/0.802); consistent ordering
  recovers **50/50 at median sim 0.971**, matches at median **1.0 px** from true position.
- Corpus signature: across all 202 session JSONs the best per-session similarity never
  reaches 0.99 (median 0.852) — impossible for same-movie re-runs with a correct metric.

## 2. What correct matching looks like (a2, all 4 sandbox sessions)

| Session | curated | recovery @0.45 (old → fixed) | true-pair sims (min/med) | stable up to thr |
|---|---|---|---|---|
| bla21-313um | 50 | 37 → **50/50** | 0.657 / 0.971 | 0.65 |
| bla12-639um | 45 | 19* → **45/45** | 0.641 / 0.985 | 0.60 |
| bla12-681um | 63 | 25* → **63/63** | 0.841 / 0.985 | ≥0.70 |
| bla12-652um | 66 | 34* → **66/66** | 0.804 / 0.982 | ≥0.70 |

(*old counts from the mismatched replica on the cached candidates.)

- **Hungarian == argmax for all 224 curated; zero greedy sharing** → no evidence that
  CNMFe merging or 1:1 assignment loses anything. The old "merging / 41–49%
  unrecoverable" story was an artifact of the broken metric. Keep Hungarian 1:1.
- **Human-label validation**: transferring reviewer keep/delete decisions onto the re-run
  candidates (mutual-best cosine >0.6 vs the original reviewed candidate set), the fixed
  matching's positives are **207 verified-kept vs 2 verified-deleted** (15 unknown).
  The OLD positives on the same sessions: of 115, only 29 verified-kept, 17
  verified-deleted, 69 unknown — the current bootstrap positive pool is mostly wrong.
- **Threshold**: 0.45 sits in a wide safe zone (recovery flat 0.30–0.60). Recommend
  keeping 0.45 for continuity; re-derive after the pilot on pre-agent-era sessions.
- **New finding — duplicate negatives**: 75–126 unassigned candidates per session
  (~20–25%) sit above 0.45 to an already-matched curated neuron (same cell re-detected /
  strong overlap). Under current rules they'd be full-weight negatives. The fixed
  pipeline should emit them as `duplicate_candidate_indices` for zero-weighting
  (composition per GT: mostly unknown-to-reviewer, a few verified kept AND a few
  verified deleted — genuinely ambiguous, so masking beats relabeling).

## 3. Corpus damage estimate (a3, all 202 sessions, read-only geometry model)

A stored label can only be accidentally correct where a curated neuron sits near the
image diagonal (mirror ≈ true). Corpus-wide: 7,928 curated neurons, 4,205 old positives,
of which an estimated **3,950 (94%) are wrong cells**; ~**7,673 true positives** are
absent or mislabeled 0. Model sanity: diagonal fraction correlates with old "recovery"
(Spearman 0.203, p=0.004; the remainder is mirror-cell density luck). Priority ranking in
`a3_damage.csv` — worst: vCA1 animal 962 Nov–Dec 2023 (up to 282 curated, damage ~445
rows/session), the vCA1 CTA 921/924 cluster, BLA 3odor Feb-2025.

Implication for training: bootstrap sessions currently contribute ~4x more wrong rows
than right ones in the positive class. The 4.0x agent floor, 0.4x bad-session weight,
0.45 threshold, ambiguous mask, and the 9 legacy sessions' near-chance AUC are all
downstream artifacts. Post-fix, the 202-session corpus (78k candidate rows) should
finally pull its weight.

## 4. Proposed Phase B (needs approval — MATLAB, watchers off)

1. **Fix** `bootstrap_preagent.py` matching + `validate_threshold.py` `_spatial_matrix`
   (one-line orientation conversion each), wire a1's red/green test as a regression test.
2. **Persistence + richer JSON** (default-off flags): keep candidate footprints
   (`bootstrap_candidates.npz`, sparse float16), JSON v2 with legacy keys + params/dims +
   `ambiguous_candidate_indices` + `duplicate_candidate_indices` + per-curated best sims;
   `--redo`/`--sessions-file` to allow re-running bootstrapped sessions.
3. **Pilot re-runs** (serial, outputs to `D:\Julian_CNMFe\_bootstrap_diag\`, session dirs
   untouched): ~12 sessions = catastrophic cluster (vCA1 961-420 0/17 — regen its missing
   Cn via pre-pass; 921-880-A/B; 96-172um AA; bla7 042125/042525), top-damage (962
   12082023 Valence, 962 CTA), 2 median controls, 1 sandbox parent (bla21, to verify the
   fixed driver end-to-end against the cached candidates + human labels). 2–3 sessions
   additionally get a permissive-params variant (min_corr 0.30) to test whether the
   pre-agent era needs the H1 param relaxation on top of the fix.
4. Gate B report: recovery vs a3 predictions, threshold re-check on pre-agent-era data,
   PNG spot-checks, runtime/session → corpus rollout plan (Phase C, after Step 4 lands).

Expected outcome: recovery ≈90%+ on most sessions; any session that stays low post-fix
has a genuine data/param problem the pilot will isolate (H1 params, geometry, missing
artifacts) — those get session-specific handling instead of corpus-wide downweighting.
