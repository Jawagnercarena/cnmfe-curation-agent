---
name: push-training-bundle
description: Stage reviewer-TRAINING bundles (answer key + candidates, optional video, gallery) on the lab-server training folder for a named trainee - build the key first, dry run, never inbox/outbox, never marks a session out for review.
disable-model-invocation: true
argument-hint: "(--from-plan | <area>\\<task>\\<session> [...]) --trainee <name> [--stage N] [--no-video] [--gallery] [--dry-run]"
---

# Push training bundles

Manual-only: it writes to the lab server (copy-only, into
`<exchange>/training/<trainee>/` and `training/_shared/gallery/`). Central
machine only. `$ARGUMENTS` are passed to `agent/push_training_bundle.py`.
Trainee-facing instructions live in `docs/TRAINING.md`; the full operator
procedure (one-time setup, per-trainee cycle, parameters) in
`docs/TRAINING_SETUP.md`.

## 0. What the script does and does not do

- Copies `review_neuron.mat`, `Cn.mat`, `pnr.mat`, `Ybg_weights.mat`, the raw
  `{session}.mat` (unless `--no-video`) and `training_key.mat` (from
  `D:\Julian_CNMFe\.training\keys\`) with `shutil.copy2`, and generates
  `run_training.m` + `TRAINING_SESSION.txt` into
  `<exchange>/training/<trainee>/<area>/<task>/<session>/`. Same-size files
  are skipped, so a re-run resumes. `--gallery` copies `.training\gallery\`
  to `training/_shared/gallery/`.
- Never writes `review_assigned.txt`, never writes into the production session
  folder, never touches `inbox/` or `outbox/`, never deletes (CLAUDE.md s1).
- Exits if `<exchange>/training/` is missing: the operator creates it (and
  `training\_shared\`) by hand, once.
- BLA and vCA1 only; a DG_AL session is refused.

## 1. Pre-flight

1. **Server folder.** Confirm `X:\Julian\cnmfe_review\training\` exists (read-only
   check). If not, the operator creates `training\` and `training\_shared\` by
   hand; do not create them.
2. **Key.** Build or refresh the key(s):

```bash
"<PYTHON_EXE>" agent/build_training_key.py <AREA>\<TASK>\<SESSION> [--force]
```

   Needs the four pool files (`review_neuron.mat`, `labels.mat`,
   `candidate_features.npz`, `labels_provenance.txt`). It launches one short
   headless MATLAB for the spatial overlap (check the machine is not already at
   its MATLAB limit; `--no-matlab` skips it, losing the spatial duplicate hint).
   Read the console line: `n_review`, `keep`, `contested`, `clear` -- a session
   with few clear-cut items is a poor stage-1 pick.
3. **Choose the sessions with the planner, not by hand-waving:**

```bash
"<PYTHON_EXE>" agent/plan_training_curriculum.py [--area <AREA>] [--pick 6]
```

   It describes every keyed session by its reference-kept cells (median
   transient length in seconds, events per minute, event SNR, crowding,
   contrast, motion share) and by candidate ambiguity, ranks them easy ->
   hard within the area (heuristic tertiles), flags "typical dynamics"
   (inside the area's inter-quartile range) and proposes `--pick` sessions
   spread over the tiers with distinct animals and tasks. Read
   `.training\curriculum_plan.md`; a session with atypical dynamics (for
   example the longest transients in the area) is a poor first bundle even if
   it ranks easy. The proposal is a starting point; the supervisor picks.
4. **Gallery** (first push only, or after a refresh):
   `"<PYTHON_EXE>" agent/build_training_gallery.py --all`, then edit the
   `TODO(Julian)` captions in `.training\gallery\index.html` before pushing.
5. **Stage and video.** Every stage uses the raw video (2-4 GB per bundle):
   the default drill is the VIDEO drill. `--no-video` leaves only the static
   warm-up usable -- use it only for a machine that cannot hold the video.
   Stage 4 (the real tool) needs a 64 GB trainee machine; stages 1-3 run on
   ~16 GB.
6. **Dry run always:**

```bash
"<PYTHON_EXE>" agent/push_training_bundle.py --from-plan --trainee <Name> [--gallery] --dry-run
"<PYTHON_EXE>" agent/push_training_bundle.py <AREA>\<TASK>\<SESSION> --trainee <Name> --stage <N> [--gallery] --dry-run
```

   `--from-plan` takes every `pick` row of `.training\curriculum_plan.csv`
   (three per area) with the stage from its tier (easy 1, medium 2, hard 3).

   Read the list: right trainee folder, right files, `WARNING: raw video ...
   not found` only when `--no-video` was intended.

## 2. Run

Same command without `--dry-run`. Success per session: `copy <file> (N MB)` /
`skip (already staged)` lines, `generated run_training.m +
TRAINING_SESSION.txt`, `staged N MB`; a `pushed.csv` row lands under
`.training\<trainee>\`. Anything listed under `item(s) not staged` (missing
key, missing `review_neuron.mat`, unsupported area) was not staged.

## 3. Afterwards

- Tell the operator what to tell the trainee: the bundle path under
  `training\<name>\`, the suggested stage, and that results go back to
  `training\<name>\returns\<session>\training_results\` -- never `inbox\`.
- Update memory: trainee, sessions, stages, whether the gallery was pushed,
  and any key parameter overrides used.
