# Red team of the per-session seeding-threshold reasoning (3odor intake)

2026-11-11. Follows `docs/HANDOFF_2026-11-11_SEEDING_THRESHOLD_TEST.md`. Read-only on
data. Nothing in ACORN was changed: `agent/params.py`, `param_overrides.json`, the
watchers, the models, every session folder and every tif are untouched (checked by
modification time at the end), and no 3odor session was staged. Everything written is
in this folder.

The proposal under test: `min_corr = floor(T) + 0.444 * (pk50 - floor(T))`, with
`min_pnr` left as estimated.

## 1. Bottom line

The direction of the proposal is right. The rule itself does not pass its own tests.

- **0.70 loses real cells in dim sessions.** On five labelled pilots dimmed
  artificially across the noise range of the dim 3odor sessions, 23% of Taylor's kept
  cells can no longer seed at 0.70 at noise_rel 0.25, and 46% at 0.35 (section 4, C7).
  In three real dim pnb99 sessions of a pilot field, 63% of the pilot's kept cells can
  seed at 0.70, against 92% in bright sessions of the same fields.
- **The rule recovers part of them, not enough.** It keeps 82-94% at the three dimmest
  levels (criterion: 95%) and 77% in the real dim sessions. Kept cells fall about twice
  as fast as the rule lets the threshold fall.
- **Only a threshold near the noise floor recovers them**: 95-99% artificially, 88% in
  the real dim sessions (95% with the PNR gate off).
- **pk50 measures brightness and cell count together.** Computing it on half of the
  same field lowers it by 0.084, which moves the rule by -0.037.
- **The noise floor holds for the dim sessions** (0.55 simulated; 0.55-0.60 on their
  real noise). It is not a constant for bright sessions.
- **The port and the two-task comparison are sound**, and so is the first-pass replay.
- **Two things cannot be settled** until Taylor's 28-session batch returns (not in the
  inbox on 2026-11-11): whether 0.444 is right, and whether pnb105 at 0.60 over-seeds.

What this leaves you to decide is in section 6.

## 2. Verdicts

| Claim | Verdict | Deciding number |
|---|---|---|
| C1 port = ACORN, tasks measured alike | Holds | pk50 differs by at most 0.0065 on 40 of 40 sessions |
| C2 noise model adequate | Holds for dim sessions, with a correction; fails for bright ones | Real dim-session noise: floor 0.55-0.60. Bright-session statistics move the floor by +0.02 to +0.04 |
| C3 floor independent of brightness | Holds, with a caveat | Simulated amplitude changes the floor by at most 0.01; quiet-pixel Cn is 0.004-0.020 higher in bright sessions, in all six animals |
| C4 floor values | Holds (Gaussian model) | Real greedyROI on noise: 0 to 1 components at the stated floors |
| C5 replay stands in for seeding | Holds for the first pass; a weak proxy for candidate counts | Replay = real seeds in 18 of 18 cases; components grow 48-107% where seeds grow 7-35% |
| C6 pk50 = brightness, not count | Fails on count; holds on brightness | Half a field: pk50 -0.084, rule -0.037 (criterion 0.02) |
| C7 linear scaling tracks dimming | Fails | Rule keeps under 95% in 29 of 50 dimmed conditions |
| C8 f = 0.444 | Not settled (needs the batch) | C7 says too large in the dim range |
| C9 pnb105 over-seeded at 0.60 | Not settled (needs the batch) | One labelled session: 0 kept of 19 candidates between 0.60 and its rule value |
| C10 PNR is not the gate | Holds, with corrections | Exact min_pnr is 0.4 higher than the stand-in; no session at the 5.0 clamp |
| C11 sheet categories, field cut-offs | Holds | Own registration agrees on 235 of 236 same-field pairs |

Every number in the hand-off's "re-derive" table reproduced with independent code
(noise correlation, floors, pk50 medians 0.887 / 0.652, top-50 seedable 0.38 / 0.98,
f, both rule ranges, Spearman 0.68).

## 3. What was run

- **Own port** (`rt_core.py`), written from the MATLAB sources, nothing imported from
  the earlier scripts. Full-movie seeding images, the pre-pass images, `params.py`'s
  estimator and the first-pass seed screen.
- **All 111 sessions** (71 3odor, 40 6odor), each tif read once: seeding images,
  pre-pass images, noise diagnostics. No session failed or was excluded.
- **Noise fields**: 12 Gaussian fields at each of five lengths, plus 3-4 fields each of
  twelve variants (lag-2 correlation, two stronger correlations, 8-bit quantisation at
  three amplitudes, Poisson at two rates, three coloured-in-time, white). Simulated.
- **MATLAB**: the production `greedyROI_endoscope.m` on five pure-noise movies; the
  SEEDTRACE shadow copy, initialisation stage only, at min_corr 0.60 / 0.70 / 0.80 on
  four sessions. All runs returned without error.
- **Real-data noise check**: the half-difference movie of 14 sessions (section 4, C2).
- **Artificial dimming**: 5 labelled pilots (one per animal except pnb106, which has
  only 10-11 kept cells) x 5 noise levels x 2 variants (noise; blur then noise).
- **Labels**: the 12 threshold-0 pilots, 2327 candidates, 410 kept. The join between
  candidate footprints and labels was verified on all 12 (the reviewer's final kept
  footprints map onto exactly the label-1 candidates).

Terms used below:
- A cell **can seed** at (min_corr, min_pnr) when some pixel of its footprint core
  passes both gates and survives the 5 x 5 median filter there.
- A cell's **ceiling** is the highest min_corr at which that still holds. It is the
  quantity that decides whether a cell is lost, and it sits below the cell's peak Cn.
- **noise_rel** is the frame-to-frame noise divided by the 1st-99th percentile range of
  the mean image, as in the session table. Bright sessions: 0.05-0.15. Dim: 0.19-0.35.
- **Floor** means the value the rule uses: 0.575 for day-0 length, 0.55 otherwise.

## 4. Claim by claim

### C1. The port reproduces ACORN; both tasks were measured alike -- holds

Measured, 40 of 40 6odor sessions.
- Pixel agreement with the session's own files: PNR within 0.0098 of `pnr.mat` (worst
  session); Cn within 0.0037 of `Cn.mat` after the same resize.
- pk50 from the port minus pk50 from `Cn.mat` (as the earlier analysis measured 6odor):
  median +0.0013, range -0.0013 to +0.0065. None above 0.01. pk25 and pk100 agree
  within 0.009 and 0.018.
- The earlier port agrees with this one on 3odor: largest pk50 difference 0.0071, N 71.
- f recomputed with port-derived pk50 on both sides: 0.441 instead of 0.444. One 3odor
  rule value moves, by 0.01.

### C2. The noise model -- holds for dim sessions with a correction; fails for bright ones

Model: Gaussian, white in time, neighbour correlation 0.18 / 0.13.

Measured on quiet pixels (lowest quarter by PNR) of all 111 sessions:

| | Model | Dim sessions (noise_rel >= 0.19, N 11) | Bright sessions |
|---|---|---|---|
| Spatial correlation, 1 px (x / y) | 0.18 / 0.13 | 0.170 / 0.123 | up to 0.30 / 0.26 |
| Spatial correlation, 2 px | 0.008 | 0.014 | up to 0.17 |
| Temporal lag-1 of the residual | -0.667 | -0.665 | -0.649 at worst |
| Spread of the trace over the fft sigma | 1.335 | 1.360 | up to 1.47 |
| Power below a quarter of Nyquist, relative | 1 | 1.21 | up to 2.7 |
| Cn of quiet pixels | 0.478 | 0.494 | up to 0.56 |

24 of 111 sessions fall outside the model; all 24 have noise_rel <= 0.15 (mostly
pnb102 and pnb104). The likely cause is weak real signal in their "quiet" pixels; that
is an interpretation, not something measured.

Simulated variants, lowest min_corr with no seed in any field, day-0 / full length:

| Variant | Floor | Moves it by more than 0.025? |
|---|---|---|
| Baseline | 0.58 / 0.55 | |
| 8-bit quantisation, noise 1, 7 or 20 grey levels | 0.57-0.58 / 0.54-0.55 | No |
| Poisson, 17 or 3 counts per pixel | 0.57-0.58 / 0.54 | No |
| Lag-2 correlation 0.015 | 0.58 / 0.54 | No |
| Temporal spectrum of the dimmest session | - / 0.54 | No |
| Spatial correlation 0.24 / 0.18 | 0.59 / 0.56 | No |
| Spatial correlation 0.32 / 0.26 | 0.61 / 0.58 | Yes (+0.03) |
| Temporal spectrum of a bright session | 0.60 / 0.57 | No (+0.02) |
| Both, as measured in that bright session | 0.62 / 0.59 | Yes (+0.04) |

Real noise. The half-difference movie `(frame 2t - frame 2t+1) / 2` keeps the real
noise and cancels everything slower than one frame. It also keeps transient onsets, so
its seeds are an upper bound on noise seeds. Restricted to the half of the field with
no cell-like structure:
- The five dimmest sessions tested: Cn median 0.476-0.480 (model 0.478). The lowest
  min_corr with no seed is 0.55, 0.55, 0.575, 0.575 and 0.60.
- Bright sessions: seeds persist up to 0.70, almost all on active cells. That is
  leftover signal, so the bright sessions give no clean reading.

**Correction:** for dim sessions take the floor as 0.55-0.60, not 0.55. The rule's
values for the dimmest sessions (0.57-0.60) therefore sit at the real floor with no
margin.

### C3. The floor does not depend on brightness -- holds, with a caveat

- Simulated: noise amplitude from 1 to 20 grey levels, quantised, changes the floor by
  at most 0.01.
- Measured: within every one of the six animals, quiet-pixel Cn is higher in the
  brightest sessions than in the dimmest (by 0.004 to 0.020; Spearman with noise_rel
  -0.24 to -0.52). That is inside the 0.025 tolerance but it is systematic, so the
  second half of the pass criterion is not met as written. The direction is harmless
  for the proposal: the model fits best where it is used.

### C4. The floor values -- holds in the Gaussian model

- 12 fields per length, 0.01 steps, first pass, min_pnr 3. No seed in any field from:
  0.58 (T 2392), 0.55 (5062), 0.55 (6370), 0.54 (8480), 0.53 (10515). The claimed
  0.575 and 0.55 are within 0.01 of these.
- The real `greedyROI_endoscope` (production copy, all greedy passes), min_pnr 3:

| T | Field | min_corr 0.53 | 0.55 | 0.575 | 0.60 |
|---|---|---|---|---|---|
| 2392 | 1 | | 11 | 1 | 0 |
| 2392 | 2 | | 6 | 0 | 0 |
| 6370 | 1 | 2 | 0 | 0 | |
| 6370 | 2 | 1 | 0 | 0 | |
| 8480 | 1 | 0 | 0 | | |

- The `pickNeurons` stage on the same noise gave 0 components in all six runs.
- Later greedy passes add nothing on noise: MATLAB's component count never exceeded
  the first-pass replay (11 vs 18, 6 vs 9, 2 vs 3).
- With real dim-session noise, add up to 0.05 (C2).

### C5. The first-pass replay -- holds for the first pass; weak as a candidate proxy

- Real seeds from the SEEDTRACE shadow copy: 6 production runs plus 12 initialisation
  runs (4 sessions x 3 thresholds). The replay returns exactly the real pass-1 seeds in
  18 of 18 cases, the same pixels, both on MATLAB's own images and on this port's.
- `bd`: the run passes 10 to greedyROI at 256 resolution, and `estimate_bd` reads the
  256 pre-pass image. The units agree.
- The caveat is what the first pass stands for. Components initialised over all passes
  are 0.43 to 1.33 times the pass-1 seeds, and they respond much more to the threshold:

| Session | min_corr 0.80 | 0.70 | 0.60 |
|---|---|---|---|
| 090826-pnb102: seeds / components | 282 / 155 | 379 / 346 | 407 / 512 |
| 090826-pnb105 | 155 / 84 | 244 / 186 | 276 / 368 |
| 090826-pnb109 | 223 / 95 | 344 / 241 | 390 / 473 |
| 090826-pnb99 | 199 / 90 | 290 / 205 | 392 / 424 |

  From 0.70 to 0.60, seeds rise 7-35% and components 48-107%. Any first-pass seed count
  quoted for a lower threshold understates the candidates it will produce.
- The reported Spearman of 0.68 against candidate counts reproduces (N 40).

### C6. pk50 measures brightness, not cell count -- fails on count

Brightness half, holds:
- 235 same-field pairs over 107 sessions. Spearman between the pk50 difference and the
  noise_rel difference: -0.67 on all pairs, -0.73 and -0.74 on the 3odor-3odor and
  6odor-3odor pairs, -0.93 on the 37 pairs whose noise_rel differs by 0.05 or more.
  Within fields, field means removed: -0.77 (90 sessions, 10 fields).

Count half, fails:
- Same session, same brightness, half the field: pk50 is lower by a median 0.084
  (444 half-fields from 111 sessions; 10th-90th percentile 0.035 to 0.163). The rule
  moves by -0.037, and by more than 0.02 in 81% of half-fields. A quarter of the field:
  pk50 -0.245, rule -0.109.
- The alternatives do not escape it. pk10 moves less with count (-0.025) but must be
  multiplied by a steeper slope to predict the needed threshold; in threshold units a
  halving costs 0.04-0.07 whichever of pk10, pk25 or pk50 is used.
- A real instance: pnb109 on 9/29 and 9/30. The sheet says the lens is badly damaged
  and few cells are visible, yet the image is not noisy (noise_rel 0.091 and 0.065).
  pk10 is 0.72 and 0.85, pk50 0.635 and 0.669, and the rule gives 0.59 and 0.60.
  Whether the extra seeds there are cells is unknown.
- Among the 12 pilots pk50 is unrelated to the number of kept cells (Spearman -0.07).
  Only 18% of candidates are kept, so the 50 strongest peaks are mostly not cells.

### C7. A straight-line scaling tracks dimming -- fails

Five labelled pilots, 216 kept cells that could seed in the original movie. Noise was
added to the tif (and, in a second variant, a 2-px blur first) until noise_rel reached
0.15 to 0.35, then everything was recomputed, including the min_pnr ACORN would pick.
Share of those cells that can still seed, cells pooled over both variants:

| noise_rel | at 0.70 | at 0.65 | at 0.60 | at 0.575 | at 0.55 | Rule, median over sessions (noise / blur) |
|---|---|---|---|---|---|---|
| original (0.08-0.12) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.15 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| 0.20 | 0.85 | 0.95 | 0.99 | 0.99 | 0.99 | 0.98 / 0.97 |
| 0.25 | 0.77 | 0.85 | 0.96 | 0.98 | 0.99 | 0.91 / 0.94 |
| 0.30 | 0.66 | 0.78 | 0.88 | 0.94 | 0.97 | 0.91 / 0.84 |
| 0.35 | 0.54 | 0.71 | 0.81 | 0.87 | 0.95 | 0.84 / 0.82 |

- The rule keeps under 95% in 29 of the 50 dimmed conditions.
- The min_corr that keeps 95% falls about one for one with pk50 inside each session
  (slopes 1.00 to 1.24); the rule lets it fall 0.444 for one. Fitted over all
  conditions: needed = 0.032 + 0.835 * pk50, rms 0.042. It reaches the floor at pk50
  about 0.62.
- The PNR gate plays no part here: the shares are the same with min_pnr set to 3.
- Cost in this test: at noise_rel 0.25-0.35 the dimmed movies give 1.7 to 2.3 times as
  many first-pass seeds at 0.55 as at 0.70. A median 96% of the seeds at 0.55 (at
  least 88%) fall inside candidates Taylor already reviewed.

Real-data cross-check. The pilots' kept cells were carried by image registration into
every other session of the same field (58 pilot-target pairs, 43 target sessions):

| Target sessions | Cell-instances | at 0.70 | at the rule | at 0.60 | at the floor | floor, min_pnr 3 |
|---|---|---|---|---|---|---|
| Bright (40 sessions) | 1953 | 0.92 | 0.93 | 0.97 | 0.97 | 0.99 |
| Dim (3 sessions, all pnb99) | 185 | 0.63 | 0.77 | 0.83 | 0.88 | 0.95 |

The bright row is the baseline for cells that are simply silent that day or slightly
misregistered. The dim row is one animal and one field, so it is a small sample; it
agrees with the artificial result in direction and roughly in size.

Limits of this test:
- Added noise is not a dirty lens. At equal noise_rel the real dim sessions have a
  lower pk50 (0.58-0.65) than the dimmed pilots (0.64-0.72), so real dimming removes
  more than added noise does.
- The ground truth is cells kept in the bright movie. A reviewer might not keep all
  of them in a dim movie.
- It says nothing about how many non-cells a lower threshold admits in a real dim
  session.

### C8. f = 0.444 -- not settled

Needs the 28-session batch. What exists today:
- On the 12 pilots no kept cell has a peak Cn or a ceiling below its session's rule
  value (0 of 410). This is weak evidence: 10 of the 12 are calibration sessions and
  all are bright.
- C7 indicates 0.444 is too large where it matters. In the dimmed conditions at
  noise_rel 0.25-0.35, f 0.444 keeps a median 88% (worst 79%), f 0.30 keeps 92% (82%)
  and f 0.20 keeps 94% (85%).

### C9. The animal-wide 0.60 over-seeds pnb105's bright sessions -- not settled

Needs the batch. One labelled pnb105 session ran at 0.60 (082726; rule value 0.64):
19 of its candidates have a ceiling between 0.60 and 0.64 and Taylor kept none. Across
all 12 pilots, kept-rate by ceiling is 0 of 21 at 0.60-0.65 and 1 of 32 at 0.65-0.70.
This supports the claim on bright sessions; it is one session, not the four in question.

### C10. PNR is not the gate -- holds, with corrections

- The pre-pass port returns exactly the min_corr and min_pnr ACORN logged, in 40 of 40
  sessions.
- Exact min_pnr for the 71 3odor sessions against the stand-in: median +0.40, higher
  in 63. **No session sits at the 5.0 clamp** (the stand-in put 9 there). Dim-noted
  sessions get 5.1 to 6.6, median 5.4.
- At the rule's min_corr the exact value changes little: first-pass seeds change by a
  median 0.7%, and the dim-noted sessions with at least 90% of their top-50 peaks
  seedable go from 7 of 10 to 6 or 7 of 10 depending on the definition. The sessions
  that fail (093026-pnb99, 101626-pnb99, 092926-pnb105, 092926-pnb109) fail on Cn:
  switching PNR off changes their share by 0.00 to 0.06. The 90% criterion as written
  is therefore not met in 3-4 of 10 dim sessions, for a reason unrelated to min_pnr.
- One exception: 100826-pnb106 would get min_pnr 11.4 (stand-in 6.4), which cuts its
  first-pass seeds at 0.70 from 261 to 166.
- In bright sessions min_pnr does separate cells from non-cells. Raising it by 2 above
  the run's value would have removed 13.4% of rejected candidates and 0.7% of kept
  cells (3 of 410); by 3, 24.0% and 1.5%.
- In real dim sessions PNR starts to bind once min_corr is at the floor: 88% of
  tracked cells can seed with the estimated min_pnr, 95% with it off.

### C11. The comparison itself -- holds

- Sheet categories: read the Notes of all 72 rows for the six animals in the local
  export. The 10 dim/hazy sessions are the right ones; "less dim" and "past was really
  bad" are correctly left out.
- Field matching: an independent registration (own band-pass and cross-correlation)
  agrees with the 0.354 cut on 235 of 236 pairs and with the 0.202 cut on 566 of 566.

## 5. Corrections of fact

These are wrong or incomplete in the README, the hand-off or the memory notes. None
changes a rule value; the first changes an explanation that was given to you.

1. **The pre-pass does not compute the image the seeding uses.** It applies a 5-sigma
   cut (`correlation_image_endoscope.m:22`) in 3 x 3 patches; seeding applies 3 sigma
   (`greedyROI_endoscope.m:61`) on the whole field.
   - Pure noise through the real pre-pass is exactly 0 in 91% of pixels, and
     `params.py` drops those pixels. Its estimate on pure noise is 0.50, clamped to 0.60.
   - So neither "noise reaches 0.87" nor the corrected "noise reaches 0.74, floor
     0.60-0.65" describes the pre-pass. Both simulated a 3-sigma image.
   - On real sessions the pre-pass leaves a median 58-60% of pixels at 0 (74% in
     dim-noted sessions), and its 85th percentile is higher than the seeding image's
     by a median 0.08. On 082626-pnb102: 0.848 pre-pass, 0.769 for the same frames at
     3 sigma, 0.764 for the full movie. The cut explains the gap; movie length does not.
2. **The pre-pass min_pnr runs low, not high.** Its 85th percentile is below the
   full-movie value by a median 2.6 (6odor) and 1.7 (3odor).
3. **"Dim sessions sit at the 5.0 clamp" is false** (C10).
4. **ACORN's own estimator already lowers min_corr for the dimmest sessions.** Without
   the overrides it would give the dim-noted 3odor sessions 0.63 to 0.81 (0.63 for
   093026-pnb99 and 101626-pnb99) and the 3odor set a median 0.79.
5. **"3 sigma" is about 2.25 true sigma.** `get_noise_fft` with `logmexp` reads 0.749
   of the true sigma on white Gaussian noise, so PNR is inflated by 1.335. This applies
   equally to simulations and data and changes no result.
6. **First-pass seed counts understate candidates at lower thresholds** (C5).

## 6. Options for a revised rule, and what remains for you to decide

C6 and C7 fail, so the rule should not be deployed as proposed. Candidates scored on
the same data. "Dimmed" is the 30 conditions at noise_rel 0.25-0.35; "real dim" is the
185 tracked cell-instances (bright baseline 0.91-0.99).

| Rule | Dimmed: kept, median (worst) | Real dim: kept | min_corr it gives the 10 dim-noted sessions |
|---|---|---|---|
| 0.70 | 0.66 (0.47) | 0.63 | 0.70 |
| Proposal, f 0.444 | 0.88 (0.79) | 0.77 | 0.57-0.69, median 0.59 |
| Same form, f 0.30 | 0.92 (0.82) | 0.81 | 0.56-0.65 |
| Same form, f 0.20 | 0.94 (0.85) | 0.84 | 0.56-0.61 |
| pk50 - 0.12, limited to [floor, 0.70] | 0.96 (0.89) | 0.79 | 0.55-0.70, median 0.55 |
| Floor + 0.025 | 0.93 (0.76) | 0.84 | 0.575 (0.60 day-0) |
| Floor | 0.97 (0.91) | 0.88 | 0.55 |

Reading of the table:
- No function of pk50 tested recovers the real dim-session cells as well as seeding
  near the floor. The real dim sessions lose cells faster than their pk50 suggests.
- Lower f has a cost the table does not show: it lowers every bright session too
  (f 0.30 gives the bright pilots 0.61-0.66). In bright sessions the 0.60-0.70 band is
  where Taylor kept 1 of 53 candidates, so that is added review work for nothing.
- A rule that leaves bright sessions at 0.70 and takes dim sessions to just above the
  floor avoids that, and limits the damage from pk50's count sensitivity to sessions
  near the switch.

My recommendation, for a test and not for deployment: keep 0.70 for bright sessions and
seed dim sessions at floor + 0.025 (0.575, or 0.60 at day-0 length), with "dim" defined
on pk50. The switch point is the open design choice; the three real dim sessions that
lost cells have pk50 0.71-0.75. I have not tested a switch point against labels,
because there are none for dim sessions.

Review load is bounded: the 10 dim-noted sessions give a median 53 first-pass seeds at
0.70, 128 at 0.60, 171 at 0.575 and 228 at 0.55, against 236 for an ordinary 3odor
session at 0.70. Expect the candidate counts to grow more than these (C5).

Decisions that are yours:
1. Whether to drop f 0.444 in favour of a two-level or steeper rule.
2. Whether to run the labelled experiment below before choosing.
3. Whether to raise min_pnr in bright sessions (C10: +2 removes 13% of rejected
   candidates for 0.7% of kept cells). This is a separate lever from min_corr.
4. Whether 100826-pnb106 should really run at min_pnr 11.4.

Any adopted change is a new regime for the classifier (threshold-0 first pass, then
retrain), as the hand-off already notes. A per-session rule also needs the full-movie
seeding image before the run; the pre-pass image cannot supply pk50 (section 5, item 1).

## 7. What cannot be settled without labels

- Whether the extra candidates a lower threshold produces in a real dim session are
  cells. Everything above measures cells that were found in bright sessions.
- C8 and C9.

The cleanest experiment is one run, not two: take 2-3 dim sessions, run them once at
floor + 0.025 and have them reviewed at threshold 0. Every candidate then has a
ceiling, so kept-rate by ceiling band tells what 0.60, 0.65 and 0.70 would each have
kept and what they would have spared the reviewer. Good candidates: 092926-pnb99 (the
dim session of a labelled pilot field, so tracked cells give a second check),
092926-pnb102, 100126-pnb99. That needs your go-ahead and stages 3odor sessions, which
this task was told not to do.

When Taylor's batch returns: C8 is the kept-rate of candidates below each session's
rule value; C9 is the kept-rate between 0.60 and the rule value on pnb105's bright
sessions. `c8_c9_pilots.py` already computes both for labelled sessions and only needs
its session list widened.

## 8. Not done, and limits

- Dimming used additive Gaussian noise and one blur width. A signal-dependent variant
  is coded (`--variants shot`) but was not run.
- The real-data cross-check has three dim target sessions, all pnb99.
- The half-difference check is clean only for dim sessions.
- Floors were simulated at a fixed neighbour correlation. Sessions vary (C2).
- The local sheet export is older than your online fixes (README), so C11 was checked
  against the export.
- While the tifs were being read (17:38-19:05) the D: disk was saturated. Both
  watchers stayed idle and healthy throughout ("No new .tif files" every 60 s).

## 9. Files

Scripts (run from this folder with `PYTHON_EXE`, `-I`):

| Script | Output | Claim |
|---|---|---|
| `rt_core.py`, `rt_labels.py`, `rt_fov.py` | (modules) | port, labels, registration |
| `validate_port.py <session>` | console | port vs one session |
| `pass_all.py` | `cache/<session>.npz`, `pass_all*.log` | all |
| `c1_port_vs_mat.py` | `c1_port_vs_mat.txt`, `c1_sessions.tsv` | C1 |
| `c2_noise_model.py` | `c2_noise_model.txt`, `c2_sessions.tsv` | C2, C3 |
| `noise_null.py`, `run_null_variants.sh`, `null_report.py` | `cache/null/`, `null_report.txt` | C2, C3, C4 |
| `halfdiff_null.py`, `halfdiff_report.py` | `halfdiff_report.txt` | C2, C4 |
| `run_matlab_rt.py c4` with `c4_noise_greedy.m` | `cache/matlab/c4_*.mat`, `matlab_rt.log` | C4 |
| `run_matlab_rt.py c5` with `c5_seedtrace_init.m`, `c5_replay.py` | `cache/matlab/c5_*.mat`, `c5_replay.txt` | C5 |
| `c6_pk50.py` | `c6_pk50.txt`, `c6_sessions.tsv`, `c6_tracked.tsv` | C6, C7 cross-check, C11 |
| `c7_degrade.py`, `c7_report.py`, `c7_rules.py` | `cache/degrade/`, `c7_report.txt`, `c7_conditions.tsv`, `c7_rules.txt` | C7, C8 |
| `curves.py` | `curves.txt` | sections 4 and 6 tables |
| `c8_c9_pilots.py` | `c8_c9_pilots.txt` | C8, C9 |
| `c10_minpnr.py` | `c10_minpnr.txt`, `c10_sessions.tsv` | C10 |

`cache/` (422 MB of images) is regenerable and excluded by the `.gitignore` here. The
whole `3odor_intake_2026-11` folder is untracked; nothing was committed.
