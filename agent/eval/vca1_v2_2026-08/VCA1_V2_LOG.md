# vCA1 v2 (35-column contract) — project log

**DEFERRED: the swap is NOT executed in this project.** Scope is prep + gates +
a rehearsed swap kit; the freeze/swap/retrain/flip/watcher-restart waits on the
bootstrap red-team report and a clean reviewer-return watcher cycle. Deploy-day
procedure is in the plan's §3f.

Branch `vca1-v2-2026-08` off `feature-expansion-step4` (da0a22a).
All numbers below computed fresh; nothing carried from the brief.

## Step 0 — ingest npz-overwrite guard (commit 6bef472)

`ingest_returns.copy_session` copied every non-dot file from a returned bundle
unless the size matched. `candidate_features.npz` differs in size across feature
contracts, so a reviewer mirroring central folders could drop a stale 13-column
npz over a 35-column session — silently, because the same-size skip cannot fire
precisely when the contract differs. **BLA had been exposed since 2026-08-20.**

Fix: generalised the `labels_provenance.txt` skip into
`CENTRAL_ONLY = {PROVENANCE_NAME, "candidate_features.npz"}`. Reviewer bundles
never carry either file (`push_review_bundle` stages `review_neuron.mat` +
Cn/pnr/Ybg_weights/pdf/summary), so nothing legitimate is refused.

`agent/test_ingest_guard.py`, temp-dir fixture, **9/9 pass**: stale 13-col npz
and stale provenance both left byte-identical (and the stale npz *did* differ in
size, so the old code would have copied it); `labels.mat` and `review_neuron.mat`
still copied; `--force` does not override the guard; dot-prefixed parked trees
still skipped.

## Step 0b — parked `3odor/AVG5x-TSeries-030426-pnb88-187um-35z-000`

Aborted final save consumed by the trainer as ground truth: `labels.mat`
(2026-06-16 12:10) holds 29 labels / 1 keep over the 29-candidate review set
(46 candidates, 17 auto-rejected), but `C_df.txt` is 0 bytes at that same mtime,
there is no final `ROIs.jpg`, and `neuron.mat` / `C_raw.txt` /
`spatial_footprints.mat` are still the 2026-04-08 candidate-level files.
`CNMFe_final_save.m` writes `labels.mat` before `extract_DF_F` → `C_df.txt` →
`C_raw.txt` → `neuron.mat` → `ROIs.jpg`, so the save aborted right after the
labels. It was also still "Out for review" to Alisia since 2026-06-25 with no
`labels_provenance.txt`, so her return would have overwritten it unrefused.

Contents verified against that description before touching anything, then moved
to `D:\Julian_CNMFe\vCA1\.excluded\3odor\` (rename within D:, 70 files before ==
70 after) with `EXCLUDED_REASON.txt`. **Operator action: tell Alisia to skip it.**

## Pool re-baseline (2026-08-26 ~12:40)

The survey behind the plan measured 22 labeled agent sessions. The live pool
disagreed, and the cause was **an operator `ingest_returns` run at 12:16 today**,
mid-planning, which landed two of Aneesh's pnb97 returns:

| session | N | reviewed | positives | labels.mat | ingested |
|---|---|---|---|---|---|
| 6odorDualDiffRew/…061826-pnb97-679um-24z-000 | 135 | 135 | 17 | 08-25 15:07 | 08-26 12:16 (Aneesh) |
| 6odorDualDiffRew/…061926-pnb97-610um-24z-000 | 107 | 107 | 21 | 08-26 11:30 | 08-26 12:16 (Aneesh) |

(`labels.mat` mtimes are the reviewer's own — ingest `copy2` preserves them — so
the ingest timestamp comes from `labels_provenance.txt`.)

Reconciled exactly: 22 surveyed + 2 ingested − 1 parked = **23 agent**;
pending 31 − 2 = **29**; agent rows 1,026 + 135 + 107 − 46 = **1,222**;
CV-usable (≥5 positives) 14 + 2 = **16**, and the thin pnb97 prep goes 3 → **5**
test sessions. Bootstrap untouched (111 sessions / 50,370 rows / 6,662 masked).
Parallel-file totals are unchanged at 163 files / 55,007 rows, because the two
sessions moved from the pending policy to the labeled policy (242 rows) and the
parked session removed 46.

Measured via the trainer's own scanners (`find_prospective_sessions`,
`_is_bootstrap_session`, `_get_bootstrap_ambiguous_mask`), not by counting dirs.

Two consequences recorded for the deploy step:
1. The deployed joblib (2026-08-24 03:38) is now **stale** — the newest
   `labels.mat` is 2026-08-26 11:30 — so restarting `watcher_vCA1.py` would fire
   an auto-retrain at the 13-column contract. That is fine and expected, but it
   means the "no spurious retrain" check belongs *after* that retrain, not before.
2. **A manual `ingest_returns` moves the pool exactly as a watcher would.** The
   D5 "watchers stay down" decision does not by itself freeze the corpus. The
   server inbox is currently fully drained (64/64 returns ingested, none queued),
   so the pool is stable as of now.

Both new sessions are finalized (`ROIs.jpg` present, `C_raw.txt` down to 17 and
21 rows), so like the other 21 they need MATLAB extraction from
`review_neuron.mat` — extraction list is **23**.
