# Reviewer Training -- guide for trainees

This guide is for **trainees** learning to curate CNMFe sessions with ACORN
before their decisions count. You re-review sessions that an experienced
reviewer has already labelled, see where you agree and disagree with them (and
with the classifier), and watch your own numbers across attempts. Nothing you
do in training reaches the real data: the folders you work in are sandboxes.

Everything on your side is **pure MATLAB**, exactly like a real review
(`REVIEW_SETUP.md`). Julian decides when you move from one stage to the next;
the tools only report numbers. (Operator procedures: `TRAINING_SETUP.md`.)

---

## 1. Prerequisites

Identical to a reviewer machine (`REVIEW_SETUP.md` sections 1-2):

- **MATLAB R2023b** with Image Processing, Signal Processing, Statistics,
  Optimization, Curve Fitting.
- The repo cloned and on your path: `addpath(genpath(<repo>))` (and
  `cvx_setup` once, needed for stage 4). Pull the branch the operator names;
  the training code lives in the repo's `training/` folder.
- Access to `\\kheirbek-nas.cin.ucsf.edu\kheirbek1\Julian\cnmfe_review`
  (often mapped as `X:\Julian\cnmfe_review`).
- Stages 1-3 load the raw video as 8-bit (about 2 GB): a machine with ~16 GB
  RAM and ~3 GB of local disk per session is enough. Stage 4 runs the real
  review tool (~64 GB RAM) -- use a reviewer-class machine for that.

---

## 2. Per-session workflow

1. **Pull your bundle** from *your own* training folder to a local disk
   (never work directly on the share):
   ```
   X:\Julian\cnmfe_review\training\<your name>\{area}\{task}\{session}\
        ->  D:\review_work\training\{area}\{task}\{session}\
   ```
   A bundle holds `review_neuron.mat` (the candidates), `training_key.mat`
   (the answer key), `TRAINING_SESSION.txt`, `run_training.m`, the raw
   `{session}.mat` video, and optionally `Cn.mat`, `pnr.mat`, `Ybg_weights.mat`.

2. **Run `run_training.m`** in MATLAB (open it in the session folder and run).
   It asks your name once, then shows a menu:
   ```
   1  Video drill: every candidate in the movie panel          [stages 1-2]
   2  Workflow drill: static triage, then video on your keeps   [stage 3]
   3  Dress rehearsal: the real review tool                     [stage 4]
   4  My progress
   5  Open the example gallery
   6  Static warm-up drill (footprint + trace only, no video)
   ```
   `TRAINING_SESSION.txt` carries the stage Julian suggested for this bundle;
   the menu preselects it but never stops you choosing another option.

3. **Return your results.** Copy the whole `training_results\` folder back:
   ```
   D:\review_work\training\{area}\{task}\{session}\training_results\
        ->  X:\Julian\cnmfe_review\training\<your name>\returns\{session}\training_results\
   ```
   **Never copy a training folder into `inbox\`.** The central ingest refuses
   anything marked `TRAINING_SESSION.txt`, but keep the habit: `inbox\` is for
   real reviews only.

---

## 3. Stages

| Stage | What you do | Settings |
|---|---|---|
| 0 | Study the example gallery (menu 5): clear real cells, dim-but-real cells, motion artefacts, duplicates, diffuse footprints, noise | |
| 1 | **Video drill** on the clear-cut candidates only, feedback after every decision | menu 1, `coach`, items `clear` |
| 2 | **Video drill** on the whole session, feedback after every decision | menu 1, `coach`, `all` |
| 3 | **Workflow drill**: static triage pass, then the video pass over your keeps (the real tool's order), feedback at the end | menu 2, `exam`, `all` |
| 4 | **Dress rehearsal**: the real `CNMFe_final_save` on this sandbox copy (video reload, background, updates, merges), scored afterwards | menu 3 |

Every stage uses the raw video. You get three sessions per area, easy,
medium and hard; repeat a stage on them until Julian moves you on.

---

## 4. The drills

**Why the video comes first.** The answer key is the reference reviewer's
decision at the END of their review, after the video pass. In the real tool
the static pass is a triage -- you keep anything plausible -- and the movie
is where you decide. So the training default (menu 1) shows every candidate
in the **movie panel**: the raw video at the frame where the trace peaks,
every outline in black and the current one in red, the zoomed footprint, and
the raw trace (blue) with the fitted calcium trace (red) and a yellow time
marker. A real cell brightens in place; a motion artefact is the whole
neighbourhood jumping (x-y) or the plane changing (z) at that moment.

The **workflow drill** (menu 2) runs the real two-pass order: first the
static view (footprint, zoom, trace, plus a correlation-image panel the real
tool does not have), where the only mistake that counts is deleting a real
cell -- it never reaches the video; then the movie panel over your keeps,
which is scored like the video drill.

Keys at the prompt (the real tool's keys, minus the editing ones):

| key | meaning |
|---|---|
| `k` or Enter | keep |
| `d` | delete |
| `m` | delete as a **motion artefact** (use it whenever motion is the reason) |
| `n` | jump to the next-biggest transient, landing 2 s **before** its onset so you can scrub or play through the rise; `t` = back to the biggest (movie panel) |
| `p` | play from 2 s before the onset through the peak to 3 s after it, twice, then rest on the peak (movie panel) |
| `b` | back one |
| `e` | end the pass (unvisited candidates count as keep, like the real tool) |
| `x` | flag "I disagree with the key" (keeps your decision; tells Julian to look) |
| a number | jump to that neuron number |

The slider under the trace scrubs through the whole movie. Each candidate
opens on the frame of its biggest transient so you can find the cell; press
`t` or `n` to step back to just before it starts, then `p` to watch it rise
and decay. The real review tool has the same `n` and `p` keys in its video
pass (there `t` is trim, and `n` cycles round to the biggest transient).

In **coach** mode a line appears after each decision:
`You: keep | Reference: delete (motion) | Model: 0.12 | DISAGREE | hint ...`.
In the static triage pass of the workflow drill a keep gets no verdict (the
video decides); a delete of a reference keep is called out as a real cell
lost. In **exam** mode you see nothing until the report.

**Coach mode scores your first decision.** If you press `b` after the
feedback and change your answer, the change is listed as "changed after the
feedback" and does not move the score. The score is what you knew before the
answer; the redo is the learning. In exam mode the decision you end with is
scored.

The **static warm-up** (menu 6) is the old-style image-only drill. Its score
is against post-video labels, so liberal keeps count against you there; use
it only when a bundle has no video.

---

## 5. Reading the report

After every attempt a report opens (`training_results\report_<session>_<time>.html`):

- **agreement** = fraction of scored candidates where you and the reference
  reviewer decided the same; **kappa** = agreement corrected for chance (1 =
  perfect, 0 = chance).
- **false keeps** = candidates you kept that the reference deleted (out of the
  reference deletes); **false deletes** = candidates you deleted that the
  reference kept (out of the reference keeps). False deletes of real cells are
  the costly mistake: a real cell lost is data lost.
- **contested** = candidates where the reference reviewer and the classifier
  disagree. They are shown, with both opinions, but **never counted**,
  whatever you decide. The report still says how often you sided with the
  reference and how often with the model on them -- worth watching over time.
- Every disagreement gets a picture and a **hint**: which feature of the
  candidate (SNR, footprint shape, overlap with a neighbour, contrast with
  its surround, number of plausible transients, match to Cn) sits at the
  extreme of this session. Hints are heuristics that explain the reference
  decision; they do not define it.
- Motion rows: how many of the reference's motion deletes you also deleted,
  and how many of those you tagged with `m`.
- In the workflow drill the first column is the static triage, where only
  "real cells lost before the video" matters; the final column is the result.

**My progress** (menu 4) plots agreement, kappa and the two error rates across
all your attempts and writes `progress.html` under your MATLAB user folder.

---

## 6. Troubleshooting

- **"acorn_training.m not found"** -- the repo is not on your path:
  `addpath(genpath(<repo>))`.
- **"review_neuron.mat has N candidates but training_key.mat expects M"** --
  the bundle is inconsistent; ask the operator to rebuild the key and re-push.
- **"the raw video {session}.mat is not in ..."** -- the bundle was pushed
  without the raw video; only the static warm-up works. Ask for a bundle with
  the video.
- **"progress.csv header extended ..."** -- normal after an update: your
  older progress file gained columns; nothing was lost.
- **Dress rehearsal is slow** -- the real tool reloads the video (~20 min)
  before the first prompt; that is normal. If `Ybg_weights.mat` is missing it
  also rebuilds the background (~10 min more).
