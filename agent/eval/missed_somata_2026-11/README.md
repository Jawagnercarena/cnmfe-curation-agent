# missed_somata_2026-11 -- why visible GCaMP somata get no ACORN candidate

Question (user, 2026-11-04): cell bodies visible in the session mean (red guide
panel 3) are often not candidates. Are they never seeded, or seeded and then
merged/deleted? Re-run on the 9/8/26 FOVA session of each of the six red-guide
animals (pnb99/102/104/105/106/109). Diagnostic only: no production file, session
folder, server path or model was written.

## Files

| file | what |
|---|---|
| `shadow/greedyROI_endoscope.m` | production copy + `% SEEDTRACE` lines that record every seed attempt (one line changed only to set `st_end`) |
| `seedtrace_run.m`, `st_snap.m` | headless pipeline copy that snapshots the component set after every merge/delete step; stops before any session output |
| `run_seedtrace.py`, `run_seedtrace.log` | driver (one MATLAB at a time, shadow must win `which -all`), log of the 2026-11-04/05 batch |
| `seedtrace/<session>/seedtrace.mat` | attempts, pass-1 Cn*PNR, 38 snapshots, final A |
| `somata/<date-animal>/blobs.json, reds.json` | the 2026-11-04 GCaMP (DoG) and red soma lists; `somata/scripts/` made them |
| `analyze_seedtrace.py` -> `seedtrace_analysis.txt`, `seedtrace_somata.csv` | reproduction check, snapshot linking (exact merge bookkeeping), per-soma fate |
| `qc_seedtrace.py` -> `seedtrace_qc_gatepass.png` | one panel per gate-passing missed soma |
| `red_exempt_yield.py` -> `red_exempt_yield.txt` | option 1 (red auto-reject exemption) yield on the 12 threshold-0 pilots |

## Result (2026-11-05, all numbers in seedtrace_analysis.txt)

- Reproduction exact on 6/6 sessions (final A identical to production
  spatial_footprints.mat, max |diff| 0); 0 of 600 somata change category.
- 49 GCaMP somata with no candidate pass both seeding gates. Merged away 2,
  deleted 1. Never covered by any component 18: the gate passes on only a sparse
  patch (median 10 of 37 disk pixels) that the 5x5 median filter on Cn*PNR
  removes (10), or Cn*PNR fell below threshold after a neighbour's initialisation
  was subtracted (7, inferred: only pass-1 Cn*PNR is saved). Covered only
  transiently 28: a footprint seeded a median 5 um away (a neighbour or a
  process) overlapped the soma, then contracted in trimSpatial/compactSpatial
  (20), updateSpatial (6) or the full-res pass (2).
- pnb106 (zoom 4) supplies 24 of the 49; there the DoG detector picks
  sub-cellular texture, and 7 of its 24 have an unclaimed candidate centred
  <= 3 um away (same cell, ring/offset footprint). Without pnb106: 25 = 16 never
  covered, 9 transient, 0 merged.
- 29 gate-failing somata: 19 never covered, 10 transient. 38 red somata with no
  candidate: 19 never covered (12 have no pixel passing the gate), 17 transient
  (12 of those footprints end on another visible GCaMP soma), 2 merged.
- Side observations: the K cap was never reached; MergeNeighbors keeps the
  survivor's own footprint (124/124 survivors unchanged; it writes the merged
  footprint only to a local copy, MergeNeighbors.m:93).
