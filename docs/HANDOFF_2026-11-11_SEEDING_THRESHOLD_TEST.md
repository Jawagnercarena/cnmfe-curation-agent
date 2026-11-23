# HANDOFF 2026-11-11: test the per-session seeding-threshold reasoning (3odor intake)

Written 2026-11-11 for a fresh session whose job is to try to BREAK the reasoning
below before anyone changes how ACORN picks min_corr / min_pnr. The session that
wrote this also produced the claims, so treat every claim as unproven. Where a test
says "independently", write your own code from `greedyROI_endoscope.m`; do not reuse
the scripts in `agent/eval/3odor_intake_2026-11/`, because they may share the
author's mistakes. One mistake was already found while writing this (section 3).

## 0. Purpose and ground rules

The user (2026-11-11): the 3odor sessions should get Cn and PNR thresholds computed
from the sessions themselves, not hard-coded per-animal numbers inherited from
6odorDualDiffRew, especially on the dim-lens sessions. A per-session rule was
proposed the same day. It is NOT deployed. Decide which parts of the argument hold.

- Do not edit `agent/params.py`, `agent/config/param_overrides.json`, any watcher,
  or any model. Do not stage, curate or push a 3odor session. The outcome is a
  report for the user, who decides.
- The server is read-only (CLAUDE.md section 1). Everything here is read-only on
  data. Write only under `agent/eval/3odor_intake_2026-11/redteam/` or a scratchpad.
- MATLAB through `_run_matlab()`: no `%` comments, ASCII only. One headless run uses
  ~75 GB; at most 2 concurrent MATLAB runs, and a busy watcher counts as one. Check
  what is running before starting anything heavy. Reading many tifs from D: competes
  with the watchers.
- Report N, exclusions, and whether each number is measured, simulated or assumed.

## 1. State on 2026-11-11 (each line checked that day)

- 71 AVG5x tifs in `D:\2p data\3odor`, six animals (pnb99/102/104/105/106/109), not
  staged in ACORN. The same animals have 40 processed sessions in
  `DATA_PARENT\vCA1\6odorDualDiffRew`.
- Labels: only the 12 pilot sessions (082626 / 082726 / 082826) carry Taylor's
  labels. The 28 batch sessions are with Taylor and were not in the inbox on
  2026-11-11. Check again, read-only, through `local_config.EXCHANGE_ROOT`.
- How thresholds are set today:
  - `params.py:114-213` estimates min_corr and min_pnr as the 85th percentile of the
    pre-pass Cn (> 0.1) and PNR (> 1), clamped to [0.6, 0.92] and [5, 15].
  - `agent/config/param_overrides.json` then overrides min_corr: 0.70 for
    pnb99/102/104/106/109, 0.60 for pnb105, 0.60 for session 090526-pnb99.
- Pre-pass (`run_cnmfe.py:48-66`): Cn/PNR on 1000 frames of the half-rate (tsub 2),
  256 x 256 movie, spread over the session with linspace.
- Full run (`CNMFe_Biane_headless.m`): ssub 2, tsub 2, one patch (`patch_par [1,1]`,
  line 82). Seeding is `greedyROI_endoscope.m:84-181` on the full movie. A later
  `pickNeurons` pass (line 146) seeds again on the residual.
- The session's `Cn.mat` is the full-movie Cn returned by the seeding step (line 92),
  resized to 512 (line 238) and saved (line 296). `pnr.mat` stays 256.
- Record of the analysis: `agent/eval/3odor_intake_2026-11/README.md`, with scripts
  and tables. The `.npz` images and noise nulls are not kept; the README has the
  commands (null runs are seeded by T).

## 2. The proposal under test

    min_corr = floor(T) + f * (pk50 - floor(T)),   f = 0.444;   min_pnr unchanged

- `floor(T)`: the lowest min_corr at which a pure-noise movie of the session's
  length gives zero first-pass seeds. Simulated: 0.55 full length, 0.575 day-0.
- `pk50`: the Cn of the session's 50th strongest Cn peak (11 x 11 local maxima,
  border 10, at 256 x 256).
- `f`: chosen so the 31 bright 6odor sessions that ran at 0.70 stay near 0.70.

Per-session values are in `agent/eval/3odor_intake_2026-11/floor/rule_sessions.tsv`.

## 3. Already corrected

The author first claimed that at the pre-pass length the noise floor is 0.65 and
noise Cn reaches 0.87. That simulation used 500 frames; the pre-pass uses 1000
half-rate frames. At T 1000 the simulated values are: Cn p50 0.475, p99.9 0.681,
max 0.743; noise seeds at min_corr 0.55 are 170 (min_pnr 4) and 3 (min_pnr 5); at
0.60, 1 and 0; at 0.65, 0. Regenerate with
`seedfloor.py null --T 1000 --rhox 0.18 --rhoy 0.13 --seed 1000`.

## 4. Claims, why each may be wrong, and the test

Pass criteria were fixed before testing. If you change one, say why.

### C1. The Python port reproduces ACORN's Cn/PNR, and both tasks were measured alike

- Evidence: Cn p90 within 0.002 of `Cn.mat` on three 6odor sessions.
- Weak points:
  - Only p90 was compared. The rule uses peak values, which smoothing affects more.
  - 6odor pk50 came from `Cn.mat` (resized to 512, then averaged back to 256).
    3odor pk50 came from the port directly. `f` was calibrated on one side and
    applied to the other.
- Test: run the port on the full movie of at least 6 6odor sessions (one per animal)
  and compare pk25/pk50/pk100 and the full PNR image against Cn.mat/pnr.mat.
- Pass: |pk50 difference| <= 0.01 on every session. Otherwise recompute `f` with
  port-derived pk50 on both sides and report how the 3odor values move.

### C2. The noise model is adequate

- Model: Gaussian, white in time, neighbour correlation 0.18 (x) and 0.13 (y),
  nothing beyond one pixel.
- Evidence: second-difference residual on a 1000-frame block of 6 sessions
  (`floor/noise_pilot.tsv`). The brighter sessions show more lag-2 correlation,
  read as signal leaking into the residual.
- Weak points:
  - Temporal whiteness was never checked. `get_noise_fft` reads sigma from the upper
    half of the spectrum, so coloured noise shifts both PNR and the 3-sigma cut.
  - The tifs are 8-bit with per-session scaling; quantization and the floor at 0
    were ignored. Shot noise scales with signal; the model does not.
  - Six sessions only.
- Tests:
  - Spectrum flatness and lag-1 temporal correlation on low-PNR pixels of dim
    sessions.
  - The spatial measurement on all 111 sessions, restricted to low-PNR pixels.
  - Re-simulate with 8-bit quantization at measured noise levels (`noise_u8` in
    `vfull/video_compare.tsv`) and with Poisson noise.
- Pass: no variant moves the floor by more than 0.025.

### C3. The noise floor does not depend on brightness

- Reasoning: PNR is max/sigma and Cn is computed after a 3-sigma cut, so noise of any
  amplitude gives the same values.
- Tests: the quantized simulations from C2 at several amplitudes. Empirically,
  compare the Cn of low-PNR pixels between dim and bright sessions of one animal.
- Pass: floors agree within 0.025, and the low-PNR Cn bulk does not differ
  systematically between dim and bright sessions.

### C4. The floor values themselves

- Claim: zero noise seeds at min_corr 0.55 (T 6400 and 8445) and 0.575 (T 2400),
  even at min_pnr 3.
- Weak points:
  - One simulated field per length; grid step 0.025.
  - First pass only. greedyROI re-thresholds locally after each extraction
    (lines 340-360), and `pickNeurons` seeds again on the residual.
- Tests:
  - Independently: at least 10 noise fields per length, step 0.01. Report expected
    noise seeds per field against min_corr.
  - In MATLAB: run the real `greedyROI_endoscope` on a pure-noise movie at the floor
    and count components.
- Pass: at most 1 noise component per field at the stated floor.

### C5. The first-pass replay stands in for ACORN's seeding

- Evidence: Spearman 0.68 between replayed seeds and ACORN's candidate counts
  (6odor, N 40); median candidates/seeds 0.75.
- Weak points: 0.68 is moderate. The replay assumes `bd` 10 applies at 256
  resolution, while `params.estimate_bd` reads the 512 Cn.mat; check the units.
- Test: the instrumented copy in `agent/eval/missed_somata_2026-11/shadow/`
  (SEEDTRACE) records real first-pass seeds. Run it on 3-5 sessions at two
  thresholds and compare with an independent replay.
- Pass: replay within 10% of the real first-pass count on every session.

### C6. pk50 measures how bright a field is, not how many cells it has

This is the weakest link. A sparse but bright field also has a low 50th peak, and
the rule would then lower its threshold for the wrong reason.

- Tests:
  - For session pairs of the same field (`fov/pairs.tsv`, score > 0.354), does the
    pk50 difference follow dimness (`noise_rel`)?
  - Across sessions, how much of pk50 is explained by the number of strong peaks
    instead of their strength?
  - Compare alternatives that use only the brightest cells: the top-10 peaks, or the
    same cells tracked across sessions of one field.
- Pass: within same-field pairs, pk50 tracks `noise_rel` (Spearman >= 0.7), and
  rule values do not move by more than 0.02 from cell count alone.

### C7. A straight-line scaling tracks what dimming does to real cells

This is the strongest test that needs no new review.

- Method: degrade labelled bright sessions. Take 3-4 of the 12 pilots. Add noise to
  the tif movie until `noise_rel` matches the dim 3odor sessions (0.2-0.35); try a
  version with blur too, since a dirty lens also scatters. Recompute Cn/PNR. Use
  Taylor's kept cells (footprints in the review set) as ground truth.
- Measure at each degradation level:
  - where the kept cells' Cn peaks land;
  - the rule's threshold computed on the degraded movie;
  - the share of kept cells that can still seed (>= 13 of 25 pixels in the 5 x 5
    window pass) at 0.70, at the rule and at the floor.
- Pass: the rule keeps >= 95% of kept cells seedable at every level while 0.70 loses
  them. If it fails, fit the relation from these data and report it.
- Cautions:
  - Added noise is not a dirty lens.
  - The session movie `.mat` may be pruned; use the `.tif` (CLAUDE.md section 4).
  - Check that `labels.mat` covers all candidates on these threshold-0 pilots before
    joining (CLAUDE.md section 6).

### C8. f = 0.444 is the right constant

- It inherits 0.70 from the 12-pilot labels (1 of 410 kept cells below Cn 0.75;
  `agent/eval/cohort5_2026-09/analysis_2026-10-28/kept_by_cn_band.txt`).
- The "check" that the rule reproduces the hand-set 0.60 for 090526-pnb99 (0.59) and
  090526-pnb105 (0.62) is weak: those hand choices are not ground truth.
- Test, once the 28-session batch is back and ingested (ingest is the operator's
  step): kept-rate by Cn band measured from each session's rule threshold.
- Pass: under 1% of kept cells have a Cn peak below their session's rule threshold.

### C9. The animal-wide 0.60 over-seeds pnb105's bright sessions

- Evidence: candidate counts only (304-372 on its four brightest sessions).
- Test with the same returned labels: the kept-rate of candidates between 0.60 and
  the rule's value on those sessions. Near zero supports the claim.

### C10. PNR is not the gate; min_pnr can stay as estimated

- Evidence: above the floor, noise gives no seeds at min_pnr 3-4. Dim sessions' top
  peaks have PNR near the noise level (median 5.7-7.4 against noise p99.9 6.8).
- Known flaw: the 3odor `mp_now` column is a stand-in. It used 500 half-rate
  consecutive frames, not the pre-pass's 1000 spread frames, so it probably
  understates what ACORN would pick. "Dim sessions sit at the 5.0 clamp" rests on it.
- Tests:
  - Compute the exact pre-pass for all 71 3odor sessions, apply `params.py`'s
    estimator exactly, and redo seed counts and top-50 seedability at the rule's
    min_corr.
  - On the labelled pilots, check kept-rate against PNR margin: does min_pnr reject
    non-cells?
- Pass: with the exact min_pnr, dim sessions keep >= 90% of their top-50 peaks
  seedable at the rule's min_corr.

### C11. Lower priority: the comparison itself

Spot-check the sheet-note categories (regex in `summarize.py:sheet_notes`) and the
field-match cut-offs (0.354 and 0.202, from within-6odor labelled pairs).

## 5. Suggested order

1. C1 and C10 (cheap; they can change every number downstream).
2. C2, C3, C4 simulations (CPU only).
3. C6 (tables already exist).
4. C7 (reads full tifs; hours).
5. C5 and the MATLAB half of C4 (heavy; check the watchers first).
6. C8, C9 when labels return.

## 6. Deliverable

`agent/eval/3odor_intake_2026-11/redteam/REPORT.md`:
- one verdict per claim: holds, holds with a correction, or fails, with the numbers;
- the scripts that produced them;
- a revised rule if C1, C6 or C7 fail;
- what still cannot be settled without human labels. The cleanest experiment there
  is 2-3 dim sessions run at both 0.70 and the rule, reviewed at threshold 0; that
  needs the user's go-ahead.

## 7. Numbers to re-derive, not trust

| Number | Value | Source |
|---|---|---|
| Noise neighbour correlation | 0.18 x, 0.13 y | floor/noise_pilot.tsv |
| Floor, full length / day-0 | 0.55 / 0.575 | floor/floor_report.txt section 1 |
| pk50 median, 6odor / dim-noted 3odor | 0.886 / 0.652 | floor/floor_report.txt section 3 |
| Top-50 seedable at 0.70 / 0.60, dim-noted | 0.38 / 0.98 | floor/floor_report.txt section 3 |
| f | 0.444 | floor/rule_table.txt |
| Rule, 31 calibration sessions | 0.67-0.72 | floor/rule_sessions.tsv |
| Rule, 3odor | median 0.68, 0.57-0.74, 16 of 71 < 0.65 | floor/rule_table.txt |
| Replay vs candidates | Spearman 0.68, N 40 | floor/floor_report.txt section 2 |

Paths in the table are relative to `agent/eval/3odor_intake_2026-11/`.
