# Motion correction of the 2-photon movies: what changed in November 2026

## Summary

Since 21 November 2026 the 2-photon pipeline registers each recording with a wider
search range and a re-centred reference image. It also keeps a record of every
frame's shift and checks every movie for poorly registered frames before the raw
data are deleted. Movies made before that date have not been changed, with four
exceptions. Two sessions that reviewers had flagged for motion, 090826-pnb104 and
090826-pnb102, were re-made with the new settings and re-extracted. Two 3odor
sessions with the same fault, 093026-pnb102 and 100926-pnb102, were re-made
before extraction.

## What stayed the same

Raw Prairie TSeries (512 x 512 px) -> suite2p 0.14.6 registration -> 5-frame
average and 8-bit conversion in Fiji -> AVG5x tif -> CNMF-E.

The non-rigid settings are unchanged: 32 x 64 px blocks, maximum non-rigid shift
7 px, spatial smoothing 1.15, no temporal smoothing, 300 frames for the reference,
bidirectional phase correction off. The averaging and the 8-bit conversion are
also unchanged.

## What changed

1. **Wider rigid search.** suite2p's `maxregshift` went from 0.1 to 0.2. The
   largest rigid shift suite2p looks for is now +/-102 px instead of +/-51 px.
2. **Re-centred reference image.** suite2p builds its reference from 300 frames
   spread over the session.
   - The pipeline now measures where those 300 frames sit relative to that
     reference, leaving out the worst-correlated 10%.
   - If their median position is more than 5 px away, the reference is rebuilt
     at that position. The frames within 5 px of it are each shifted onto it and
     averaged (within 10 px if fewer than 15 frames are that close).
   - This repeats until the median is within 2 px, at most 4 times. suite2p then
     uses the rebuilt image as a fixed reference.
3. **Shift record kept.** The raw TSeries folder, with suite2p's per-frame
   shifts, is still deleted once the averaged movie is verified. Before that, the
   pipeline now saves the following to `mc_record\<movie name>_mc.npz` in the task
   folder:
   - the per-frame x/y shifts and registration correlation
   - the reference and mean images
   - the settings used
4. **Quality check before delete.** Each movie gets one row in `mc_qc.csv` in the
   task folder. The raw TSeries folder is kept, not deleted, if either:
   - more than 5% of the averaged frames are "bad"
   - more than 1% of the raw frames sit at the search limit

   A kept folder means the session can be re-examined and re-processed.

   A frame is "bad" when its correlation with the movie's own template is below
   half of the movie's quiet level. In detail: each averaged frame is band-pass
   filtered (2-15 px) and correlated with a template made from the
   better-correlated frames of the same movie. The quiet level is the 90th
   percentile of that correlation. Frames below 0.75 x the quiet level are
   reported as "degraded".

The saved suite2p GUI defaults were not changed. Running suite2p by hand from the
GUI still uses `maxregshift` 0.1 and suite2p's own reference. The new behaviour
comes from the pipeline scripts.

## Why

Reviewers flagged 090826-pnb104 and 090826-pnb102 for "too much motion". Re-running
the old settings on raw trials from these sessions showed:

- **The reference was off-centre.** suite2p's reference image sat 27-45 px from
  where the field usually was (4 sessions tested). Ordinary frames therefore
  needed shifts close to or beyond the +/-51 px limit.
- **Many frames were never registered.** 30.3% (pnb104) and 6.4% (pnb102) of raw
  frames were stuck at the +/-51 px limit.
- **Large shifts left a visible edge strip.** suite2p applies shifts by circular
  rolling, so every large shift wraps a strip of the image to the opposite edge.
  The strip is visible in the old pnb104 movie.
- **Some frames cannot be fixed by any x-y shift.** About 3% (pnb104) and 7%
  (pnb102) of raw frames matched the template at no x-y shift at all. They come
  as multi-second episodes after reward in some trials, consistent with
  focal-plane (z) motion.

## What the change did

The table gives bad / degraded averaged frames for each whole session.

| Session | Old settings | New settings |
|---|---|---|
| 090826-pnb104 | 16.6% / 29.3% | 0.6% / 8.0% |
| 090826-pnb102 | 10.2% / 16.7% | 4.3% / 10.3% |
| 082726-pnb104 (control, a good session) | 0.9% / 2.2% | 0.0% / 0.6% |

- **The wider search does most of the work.** Re-centring the reference changes
  the bad-frame rate little. It reduces how many frames need large shifts, and so
  the wrapped strip at the frame edge.
- **The old pipeline is reproducible.** Re-running the old settings on the
  control session reproduced the delivered movie byte for byte. The differences
  above come from the settings change alone.
- **Clean sessions are not harmed.** On five clean sessions (vCA1, BLA and DG;
  12-trial samples) the new settings gave the same result as the old in four.
  The fifth, a BLA session, came out slightly better (0.3% -> 0.0% bad).
- **pnb102 keeps 4.3% bad frames.** They fall mostly 10-12 s into the trial in a
  subset of trials. These are the focal-plane episodes described above.

## Effect on extraction (the two re-made sessions)

Both sessions were re-run through CNMF-E with the same parameters as before. The
exception is `min_pnr`, which is estimated from each movie (6.6 -> 6.4 and
7.0 -> 7.1).

- **090826-pnb104:** 276 -> 228 candidate neurons. Pairs flagged as possible
  duplicates (overlapping footprints with correlated traces) fell from 58 to 3.
- **090826-pnb102:** 333 -> 346 candidates; possible duplicate pairs 29 -> 40.

Manual review of the new candidates is still to come; until then we do not know
whether they are better.

## What this means for existing data

- **Old movies are untouched.** Movies made before 21 November 2026 were not
  changed, except the four re-made sessions named in this note. Their old
  versions are kept.
- **Two earlier pnb104 sessions were left as they are.** 090726-pnb104 (4.6% bad)
  and 091026-pnb104 (3.8% bad) had already been curated.
- **The 3odor movies use the old settings.** Across the 71 movies the median
  bad-frame share is 0.02%; 4 of them are above 5%. Three of those four were
  tested on raw trials from their bad stretch:
  - **093026-pnb102 and 100926-pnb102** have the same registration failure, and
    the new settings fix it. In the affected trials bad frames fall from 74% to
    0.1% and from 45% to 0.7%. Both were re-made from the raw data on
    22 November, before CNMF-E. Over the whole session, bad / degraded frames
    went from 5.2% / 6.4% to 0.01% / 0.4% (093026-pnb102) and from
    7.7% / 11.2% to 0.09% / 3.0% (100926-pnb102).
  - **100826-pnb106** is different: its imaged field drifts slowly through the
    session and differs completely by the last third, so no registration can fix
    it.
- **Old and new versions of a re-made session do not mix.**
  - The field sits at a different position in the frame: shifted by about
    (+14, +47) px in pnb104 and (+4, +36) px in pnb102.
  - The 8-bit intensity scale also differs.
  - Pixel coordinates, footprints and absolute intensities from the old version
    therefore do not carry over to the new one.
- **The QC limits are a starting point.** The 5% bad and 1% at-limit lines come
  from 17 scored movies. Reviewers rejected two of them (10.2% and 16.6% bad) and
  accepted the rest (up to 4.6%).
