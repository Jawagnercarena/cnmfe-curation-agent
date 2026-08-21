# Phase B pilot results — GATE B

Date: 2026-08-21. Pilot: 15 runs (12 sessions + 3 permissive variants), serial MATLAB,
outputs to `D:\Julian_CNMFe\.bootstrap_diag\` only. Aggregation: `b2_rerun_report.py`
(`b2_gate_b.json`); per-run log `b1_pilot_log.json`. One >2GB save failure recovered
offline (`b1b_recover_entry14.py`, see §4).

## 1. Headline: 43.3% → 98.8% recovery

Across the 11 bootstrap pilot sessions: **old matching 348/804 curated neurons (43.3%)
→ fixed matching 794/804 (98.8%)**, with recovery flat across thresholds 0.45–0.60
(0.65 loses 5) — the wide plateau confirms **keep 0.45**.

| Session (old → fixed) | recovery | sim min |
|---|---|---|
| vCA1 961-420 (the 0/17 session) | 0.00 → **0.82** | 0.757 |
| vCA1 921-880-A | 0.07 → **1.00** | 0.605 |
| vCA1 921-880-B | 0.24 → **0.96** | 0.701 |
| vCA1 96-172um AA | 0.03 → **0.93** (permissive: 1.00) | 0.733 |
| vCA1 962 CTA 12012023-A | 0.39 → **1.00** | 0.666 |
| vCA1 962 Valence 12082023-B (282 cur) | 0.63 → **1.00** | 0.698 |
| vCA1 100-300um 4CS (control) | 0.57 → **0.98** | 0.626 |
| BLA bla7-812um | 0.09 → **1.00** | 0.942 |
| BLA bla7-778um | 0.13 → **1.00** (permissive: 1.00) | 0.881 |
| BLA bla3-667um (legacy 9) | 0.64 → **1.00** | 0.760 |
| BLA bla15-183um (control) | 0.56 → **1.00** | 0.808 |
| BLA bla21-313um (agent GT session) | — → **1.00** (50/50) | 0.618 |

Session dirs: **ALL VERIFIED UNTOUCHED** (before/after file snapshot per run).

## 2. Validation against humans and predictions

- **bla21 end-to-end**: the fresh fixed pipeline's 50 positives vs human labels =
  **46 verified-kept / 1 verified-deleted / 3 unknown** — matches the offline sandbox
  study (a2) and closes the loop: driver + MATLAB + fixed matcher reproduce the
  ground-truth result.
- The a3 damage model predicted catastrophic sessions were mirror-geometry victims, not
  data problems — confirmed: every one normalized to ≥0.82 with no parameter changes.
- The 10 still-unmatched neurons are concentrated where a3 predicted residual dimness:
  10 below-threshold partners are now explicitly recorded as
  `ambiguous_candidate_indices` (weight-0 at training).

## 3. H1 (permissive params): real but small — do NOT change global params

min_corr 0.30 / min_pnr 4.0 recovered the last 2/30 on 96-172um (0.93 → 1.00) but added
nothing on the other two variants (already 1.00) while multiplying candidates 2–5x and
runtime up to 2.3x (146 min). Recommendation: run the corpus at default bootstrap params;
optionally revisit the handful of sessions that stay <0.90 afterwards.

## 4. Operational findings

- **Runtimes**: vCA1 4–27 min, BLA 21–76 min per session. Corpus estimate (202 sessions):
  ≈ 107 h serial ≈ 4–5 days of MATLAB.
- **>2GB save stub**: a permissive run (1,344 candidates) exceeded MATLAB's v7 save limit
  for `spatial_footprints.mat` (warned "not saved", wrote a stub). Fixed by an A.txt
  fallback in `_load_candidates` (orientation-correct); the failed entry was recovered
  offline in 0.7 min — the 146-min CNMFe run was not repeated. Normal-params sessions
  stay well under the limit.
- **Duplicates**: 7–194 per session (median ~40) now recorded as
  `duplicate_candidate_indices` — under old rules these same-cell re-detections would be
  full-weight negatives.
- **Step 4 landed mid-pilot** (`1f8d685`, `712e9ce`): BLA FEATURE_VERSION=2 live,
  T=0.06, watcher restarted. Bootstrap re-runs now auto-emit the 35-col contract
  (verified in the recovery run) — Phase C output will be contract-consistent by
  construction. Shadow npz written before the flip are 13-col; irrelevant (shadow only).

## 5. Proposed Phase C (needs approval)

1. **Corpus re-run, all 202 sessions** (`--sessions-file` + `--keep-candidates`, default
   params), writing INTO the real session dirs (labels.mat + candidate_features.npz
   35-col + JSON v2 + bootstrap_candidates.npz). Watcher must be OFF for the duration
   (its labels.mat mtime trigger would otherwise auto-retrain mid-rollout on a
   half-relabeled corpus); ~4–5 days serial, batched to taste.
2. **Trainer updates before the final retrain** (train_classifier.py is unfrozen now):
   ambiguous mask prefers `ambiguous_candidate_indices` (legacy fallback), duplicates
   masked to weight 0. Retro cn_correlation fix deliberately deferred to a separate
   gated change so it doesn't confound the bootstrap gates.
3. **Retrain + gates, both areas**, vs the post-Step-4 pinned baseline (BLA 35-col
   @0.06): G1 AUC / G2 false-AR@matched-junk / G3 bootstrap-contribution delta /
   G4 weight re-sweep (4.0x floor + 0.4x expected to shrink) / G5 per-era false-AR;
   threshold re-derivation per area. Re-pin the Step 4 pool manifest afterwards.
4. Docs + memory + REVIEW_QUEUE updates; ACORN stats refresh if numbers move.
