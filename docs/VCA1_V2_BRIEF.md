# Brief: bringing vCA1 onto the 35-column (v2) feature contract

> **STATUS 2026-08-26 — prep and gates COMPLETE; deploy deferred.**
> Executed on branch `retire-bla-rollback` (the `vca1-v2-2026-08` branch is an ancestor;
> the substantive commits are here); full record in
> `agent/eval/vca1_v2_2026-08/VCA1_V2_LOG.md`, decision in `gate_decision.json`.
>
> **Decided: ship arm b0** — real v2b on bootstrap rows (`v2_present=1`) with
> `ring_contrast` forced to 0. Reviewed AUC 0.8786 → 0.9013 (+0.0227, 8/8 seeds),
> T stays at the deployed **0.05**, and junk auto-caught rises **+7.3pp at
> matched false-AR**. The §"What is different" option (a) zero-fill arm was
> measured and *rejected*: it gains +0.018 AUC but is **-1.2pp at matched
> false-AR, i.e. operationally worse than the 13-column model in production**.
> Real `ring_contrast` (arm b1) adds +0.0005 on 5/8 seeds — nothing — so the
> Cn-regeneration question this brief raised is **answered: not worth doing**.
>
> Production follow-on is DONE: `config.BOOTSTRAP_V2B` is area-scoped and
> `bootstrap_preagent` writes matching rows (BLA/DG_AL untouched).
>
> **Red-team report delivered 2026-08-26** (`agent/eval/bootstrap_redteam_2026-08/redteam_report.md`):
> arm b0's ranking gain is real and survives 2-animal leave-one-animal-out (0.8679 vs
> 0.8625 for b13); the rule's T = 0.05 sits at the 1% worst-seed ceiling (0.60% mean /
> 0.96% max) and flips to 0.04 under a different xgboost thread count; for a held-out
> animal 0.05 runs at ~5% false-AR (as does the deployed 13-col model). Re-derive T on
> deploy day with the thread count pinned and quote 0.04 vs 0.05 side by side.
>
> **Two blockers remain before the swap:** one
> clean reviewer-return watcher cycle since the 2026-08-24 redeploy (user's call), and the
> deploy commit (`FEATURE_VERSION = 2`, `BOOTSTRAP_V2B = "b0"`,
> `_VALIDATED_THRESHOLD = 0.05`). Deploy day starts with a mandatory re-run of
> extract → pin → hiconf → backfill → gate, because that second blocker
> guarantees the pool has moved.
>
> Two corrections to this brief, both measured:
> - the agent pool is **23 sessions / 16 CV folds** (an operator ingest landed two
>   pnb97 returns on 08-26 and one aborted-save session was parked), not 22/14;
> - the §"Definition of done" per-prep breakdown "pnb vs 2022–23 animals" is **not
>   measurable** — every 2022–23 animal is bootstrap-only and therefore train-only,
>   so no agent test fold exists for it. The real split is pnb88 (11 sessions) vs
>   pnb97 (5), both of which improve, plus a 2-animal LOAO.

Written 2026-08-25 at the end of the bootstrap-matching fix. For a fresh session (or
the feature-expansion chat, which holds the Step 4 context). Read
`docs/FEATURE_EXPANSION_STEP4_BRIEF.md` and `agent/eval/step4_2026-08/STEP4_LOG.md`
first — this brief lists only what differs for vCA1.

## Why

BLA runs FEATURE_VERSION=2 (35 cols: 13 base + 13 within-session ranks + 8 v2b trace
features + `v2_present`) since 2026-08-20 with measured gains (+0.02 reviewed-pool AUC,
false-AR at matched junk 0.86 → 0.11%, biggest wins on early-era sessions). vCA1 is
the only production area still on the 13-column contract. `features.py`'s v2 machinery
is area-generic and committed.

## Sequencing (do NOT start before these)

1. The BLA post-fix gates are done and the corpus is stable (this session).
2. The global GRIN-lens model evaluation has reported: if a pooled 35-col model wins,
   vCA1 v2 must land as part of one coordinated two-area swap, not standalone.
3. vCA1 was redeployed 2026-08-24 (w=5.0, T=0.05); let at least one reviewer-return
   cycle settle before stacking a contract swap on top.

## Invariants to carry over from Step 4

- First 13 columns bit-identical to v1 (parity check like `step4_2026-08/parity_check.py`).
- Backfill to parallel `candidate_features_v2.npz` files first; pin a pool manifest
  (`repin_manifest.py` pattern); swap atomically with the watcher stopped; verify.
- Companion 13-col first-pass model in the same joblib (curator two-pass nb scoring).
- Threshold re-derived from an 8-seed OOF sweep (`threshold_sweep_v2.py` rule: largest
  T with mean false-AR ≤ 0.85% and worst-seed ≤ 1.0%), never carried over.
- `config_vCA1.FEATURE_VERSION = 2` is the switch; `bootstrap_preagent` and the
  trainer already branch on it.

## What is different / new for vCA1

- **Agent-session backfill needs vCA1's own trace extraction** (Step 0 pattern:
  `review_neuron.mat` → `.feature_expansion` .mat per session, MATLAB, seconds each).
  Confirm every vCA1 agent session has `review_neuron.mat`; the pnb sessions curated
  at threshold 0 have unusual auto-reject structure — verify `review_indices` parity.
- **Bootstrap rows now have persisted candidate traces** (`bootstrap_candidates.npz`,
  C_raw for all 111 sessions). Step 4 zero-filled v2b for bootstrap because re-run
  traces were not label-faithful; post-fix the labels come from the same run as the
  traces. Decide, with a gate, between (a) keep zero-fill + `v2_present=0` (Step 4
  behavior, `assemble_v2_bootstrap`) and (b) real v2b for bootstrap rows with
  `v2_present=1`. Risk in (b): train/inference distribution shift (curator scores
  agent-run traces, not re-run traces) — measure with the agent-only test folds; the
  `v2_present` flag exists precisely to let the model separate the regimes.
- **Small, hard agent pool**: 22 sessions, 14 usable for CV, dominated by the
  pnb/tdTomato prep. Expect ±0.02 seed noise; use 8 seeds and paired deltas only.
- **Weight recipe**: `AGENT_WEIGHT_OVERRIDE = 5.0` is set for vCA1; re-sweep after the
  contract change (the sweep tool: `agent/eval/bootstrap_matching_2026-08/c3_weight_sweep.py --area vCA1`).
- **Watchers**: `watcher_vCA1.py` must be stopped for the swap; it does not set
  `curator.THRESHOLD_OVERRIDE` (only DG does) — the pnb threshold-0 re-curation
  procedure stays manual.

## Definition of done

Parity 13-col bit-exact on all vCA1 sessions; 8-seed gates (reviewed-pool AUC delta,
false-AR at matched junk, per-prep breakdown pnb vs 2022–23 animals) pass; joblib
verified (35 features, companion model present, threshold from the sweep); watcher
restarted with a clean first cycle; docs + memory updated.
