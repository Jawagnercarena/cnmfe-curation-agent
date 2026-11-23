# 3odor intake, 2026-11: 3odor vs 6odorDualDiffRew, and per-session seeding thresholds

Read-only analysis, 2026-11-11. Subject: the AVG5x tifs in `D:\2p data\3odor` (shock
task; 71 sessions after the changes below; NOT yet staged in ACORN) compared with the
same six animals' 40 processed `vCA1\6odorDualDiffRew` sessions (pnb99, pnb102,
pnb104, pnb105, pnb106, pnb109). The scripts write only inside this folder.

## Changes to the data made on 2026-11-11 (user-requested)

- `AVG5x-TSeries-100926-pnb109-427um-25z-000.tif` deleted locally (laser failure
  after trial ~13). The server raw remains. Its row is still in
  `vfull/video_compare.tsv`; `summarize.py` excludes it.
- 100826-pnb106 renamed 158um -> 258um (XML Z Focus -257.7). This covers the local
  tif and, with one-time user permission, the server TSeries folder. The XML inside
  keeps the acquisition name; `xml_compare.py` falls back to it.
- 092426-pnb99 made by the user and added.
- The two z mismatches printed by `summarize.py` (100926-pnb102 452 vs 470,
  100226-pnb102 618 vs 617) come from the local sheet export
  (`D:\Analysis\3odor\...`, saved 11:26). That export predates the user's sheet fix.

## Scripts and outputs

Run from this folder with `PYTHON_EXE` from `agent/local_config.py`, flag `-I`.

| Script | Output | What |
|---|---|---|
| xml_compare.py | xml_compare.tsv | Bruker XML settings and trial structure; reads the server read-only. `fs_avg` = within-trial AVG rate (the earlier inverted formula was fixed 2026-11-11). |
| video_compare.py | vfull/video_compare.tsv (+ vfull/img/*.npz, not kept) | Image statistics from each tif. For 3odor it ports greedyROI_endoscope.m to compute full-movie Cn/PNR (checked against Cn.mat on three 6odor sessions: Cn p90 within 0.002). For 6odor it reads the pipeline's Cn.mat/pnr.mat. |
| timecourse.py | tc/timecourse.tsv | Within-session stability over 300 spread frames. |
| fov_match.py | fov/pairs.tsv, fov/best.tsv | Same-field test: DoG mean-image cross-correlation. |
| summarize.py | summary/summary.txt, summary/merged.tsv | **merged.tsv is the per-session table**; summary.txt holds the group, animal and sheet-note tables. |
| seedfloor.py | floor/noise_pilot.tsv (+ floor/null_T*_m.npz, not kept) | Noise spatial structure, and the pure-noise seeding floor. |
| floor_report.py | floor/floor_report.txt, floor/floor_sessions.tsv | Noise floor vs every session. |
| rule_table.py | floor/rule_table.txt, floor/rule_sessions.tsv | Per-session min_corr computed from the session itself. |
| cn_length_test.py | (console) | Does the Cn port depend on movie length (vs Cn.mat)? |

```
$PY -I xml_compare.py --out xml_compare.tsv
$PY -I video_compare.py --out vfull                  # ~1-2 min per session; writes vfull/img/*.npz
$PY -I timecourse.py --out tc
$PY -I fov_match.py --img vfull/img --xml xml_compare.tsv --out fov
$PY -I summarize.py --xml xml_compare.tsv --video vfull/video_compare.tsv --fov fov/best.tsv --out summary > summary/summary.txt
$PY -I seedfloor.py noise --only 082826-pnb99 082726-pnb105 100226-pnb99 093026-pnb99 092426-pnb104 093026-pnb102 --out floor/noise_pilot.tsv
for T in 8445 6400 2400 1000 500: $PY -I seedfloor.py null --T $T --rhox 0.18 --rhoy 0.13 --seed $T --out floor/null_T${T}_m.npz
$PY -I floor_report.py > floor/floor_report.txt
$PY -I rule_table.py > floor/rule_table.txt
```

The `.npz` files are binary and can be regenerated (the null runs are seeded by T), so
they stay out of git. fov_match, floor_report and rule_table need them. Constants at
the top of each script: the raw tif folder, server XML roots and sheet CSVs. The 6odor
root comes from `local_config.DATA_PARENT`.

## Findings: the 3odor vs 6odor comparison

See summary/summary.txt.
- **Same in both tasks:** acquisition pipeline, frame rate (6.04-6.07 Hz within
  trials), motion and stability.
- **Different:** 3odor sessions are shorter (day-0 sessions are about 4.8k frames)
  and Pockels is higher on every session.
- **Image quality:** the dim/hazy sheet notes mark the worst sessions. pnb99 is dim in
  every 3odor session, cleaning included.
- **Fields:** 43 of 71 3odor sessions match a 6odor field above the highest
  different-field score within 6odor (0.354). 7 score below 0.202.

## Findings: the seeding threshold

- **The noise floor depends on movie length, not brightness.**
  - Cn and PNR are both normalised to each pixel's own noise, so the floor doesn't
    move with dimness.
  - The measured noise is spatially correlated: lag-1 0.18 across and 0.13 down,
    about 0.015 at lag 2.
  - A pure-noise movie with that correlation, run through the port, gives zero
    first-pass seeds at min_corr 0.55 for full-length movies and 0.575 for day-0
    length, even at min_pnr 3.
  - ACORN's pre-pass (`run_cnmfe.py:62`) computes Cn/PNR on 1000 frames of the
    half-rate movie, spread over the session. At that length (simulated T 1000) noise
    reaches Cn 0.74 (p99.9 0.68) and the floor is 0.60-0.65. This is one reason the
    85th-percentile pre-pass estimate ran high. (Correction made the same day: the
    T 500 row in floor_report.txt is not the pre-pass; an earlier note quoting
    "0.65 / 0.87" for the pre-pass was wrong.)
  - At these min_corr values PNR does no noise rejection, so min_pnr is not the gate.
- **The session's own field sets where cells sit.** `pk50` is the Cn of the 50th
  strongest Cn peak.
  - 6odor median: 0.886. 3odor dim-noted median: 0.652.
  - In the dim-noted sessions only 38% (median) of the 50 strongest peaks survive
    the 5x5 median filter at 0.70. At 0.60, 98% do.
- **Per-session rule (not deployed):** `min_corr = floor(T) + f * (pk50 - floor(T))`,
  with f = 0.444.
  - f is set so the 31 bright 6odor sessions that ran at 0.70 stay at 0.70 (they
    land at 0.67-0.72).
  - Unseen check: it gives 090526-pnb99 0.59 (hand-set 0.60) and 090526-pnb105 0.62
    (hand-set 0.60).
  - The 7 pnb105 sessions that ran at the animal-wide 0.60 get 0.62-0.71, session by
    session.
  - For 3odor: median 0.68, range 0.57-0.74, below 0.65 in 16 of 71 sessions.
- **Not yet known.** No dim session has labels, so whether the extra seeds are cells
  is unmeasured. The first test is Taylor's return of the 28-session batch
  (090526-pnb99 and 090526-pnb105 ran at 0.60). First-pass seed counts are a proxy:
  Spearman 0.68 against ACORN's candidate counts, 6odor N 40.
- **Known flaw in the 3odor min_pnr column:** `mp_now` in floor_sessions.tsv and
  rule_sessions.tsv stands in for ACORN's estimate. It is the 85th percentile of PNR
  on 1000 consecutive full-rate (= 500 half-rate) mid-session frames, whereas ACORN
  uses 1000 spread half-rate frames. PNR grows with length, so the stand-in probably
  understates ACORN's min_pnr. The claim that dim sessions sit at the 5.0 clamp rests
  on it. Untested; see docs/HANDOFF_2026-11-11_SEEDING_THRESHOLD_TEST.md.
- **Near noise:** 093026-pnb99 (pk50 0.584, below the noise maximum 0.589) and
  101626-pnb99 (0.595) have almost nothing above noise beyond their top ~25 peaks.
