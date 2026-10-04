# Reviewer Training -- setup and operation (central operator)

How to stand up and run the reviewer training program. The trainee side is in
`TRAINING.md`. Everything here runs on the central machine with the `valence`
Python (`agent/local_config.py:PYTHON_EXE`); the only server writes are the
copy-only pushes, and two folders you create by hand once.

## What it is, in one paragraph

A trainee re-reviews sessions that an experienced reviewer already labelled.
For each such session the central machine builds an **answer key**
(`training_key.mat`): the reference reviewer's keep / delete / motion decision
per candidate, the deployed classifier's score, a "contested" flag where the
two disagree, the candidate's 35 features with within-session percentile ranks,
and a plain-language hint. A **bundle** (candidates, key, raw video, launcher)
goes to the trainee's own folder on the exchange; they run `run_training.m`
in MATLAB, drill in the movie panel, get a report per attempt, and copy their
`training_results\` back. You bring the results home and read one summary.
Progression between stages is your call; nothing in the code sets a pass mark.

## Where things live

| What | Where |
|---|---|
| Central scripts | `agent/build_training_key.py`, `plan_training_curriculum.py`, `build_training_gallery.py`, `push_training_bundle.py`, `summarize_training.py`, shared `training_common.py` |
| Trainee MATLAB | `training/` (`acorn_training.m` menu, `_drill`, `_score`, `_report`, `_progress`, `_rehearsal`, helpers, `tests/acorn_training_selftest.m`) |
| Shared with the real tool | `ca_source_extraction/utilities/find_trace_transients.m` (the `n` / `p` keys in `viewNeuronsVideo` and in the drill) |
| Central artefacts | `D:\Julian_CNMFe\.training\` (dot folder: every pipeline scanner skips it): `keys\<area>\<task>\<session>\training_key.mat`, `curriculum.csv`, `curriculum_plan.csv/.md`, `gallery\`, `<trainee>\pushed.csv`, `<trainee>\returns\`, `progress_all.csv`, `summary.md` |
| Exchange | `X:\Julian\cnmfe_review\training\<trainee>\<area>\<task>\<session>\` (bundles), `training\<trainee>\returns\<session>\training_results\` (what comes back), `training\_shared\gallery\` |
| Skills | `/push-training-bundle`, `/summarize-training` |

Areas: BLA and vCA1 only (both on the 35-column feature contract).

## 1. One-time setup

1. **Trainee machines** need the repo at a commit that contains `training/`
   and the `viewNeuronsVideo` change, on the MATLAB path with
   `addpath(genpath(<repo>))` (`REVIEW_SETUP.md` section 2).
2. **Create two folders on the server by hand** (the scripts never create
   them; they exit if either is missing):
   ```
   X:\Julian\cnmfe_review\training\
   X:\Julian\cnmfe_review\training\_shared\
   ```
3. **Build the keys** for every eligible session (has `review_neuron.mat`,
   `labels.mat`, `candidate_features.npz`, `labels_provenance.txt`):
   ```
   <PYTHON_EXE> agent/build_training_key.py --all
   ```
   One headless MATLAB launch follows (spatial overlap per session); it is
   light, but count it against the machine's MATLAB limit. Rebuild with
   `--force` after a model retrain (scores are the deployed model's, in-sample)
   or when new reviewed sessions arrive. `--reviewer <name>` restricts the key
   to one reference reviewer.
4. **Plan the curriculum:**
   ```
   <PYTHON_EXE> agent/plan_training_curriculum.py --pick 3
   ```
   Writes `.training\curriculum_plan.md`: every keyed session described by its
   reference-kept cells (median transient length in seconds, events per
   minute, event SNR, crowding, contrast with surround, motion share), ranked
   easy -> hard within the area (heuristic tertiles), with a clarity rank and a
   "typical dynamics" flag, and a proposal of one clear, typical session per
   tier with distinct animals and tasks. Sessions with fewer than 10 reference
   keeps are not proposed (`--min-keeps`). Read it; the proposal is a starting
   point. A session with atypical dynamics (for example the longest transients
   in the area) is a poor first bundle even if it ranks easy.
5. **Build the gallery and edit its captions:**
   ```
   <PYTHON_EXE> agent/build_training_gallery.py --all
   ```
   Renders 12 examples per category (clear real, dim-but-real, motion,
   duplicate, diffuse, low SNR, few transients, Cn mismatch, contested) into
   `.training\gallery\`. Open `index.html` and replace every `TODO(Julian)`
   caption; they are first drafts. Then push it once:
   ```
   <PYTHON_EXE> agent/push_training_bundle.py --gallery --trainee <any name>
   ```

## 2. Per trainee

1. **Push the bundle set** (dry run first). The planned set is the sessions
   marked `pick` in `.training\curriculum_plan.csv`, three per area; the stage
   hint comes from the tier (easy 1, medium 2, hard 3). Each session carries
   its raw video, which every drill uses, so a full set is roughly 18 GB:
   ```
   <PYTHON_EXE> agent/push_training_bundle.py --from-plan --trainee <Name> --gallery --dry-run
   <PYTHON_EXE> agent/push_training_bundle.py --from-plan --trainee <Name> --gallery
   ```
   Single sessions work too (`<area>\<task>\<session> --stage N`); `--stage`
   with `--from-plan` overrides every tier-derived stage (for example to re-use
   the same sessions at stage 4). The stage is only a hint shown in the
   trainee's menu. The script writes `run_training.m` and
   `TRAINING_SESSION.txt` into each bundle, logs to `.training\<Name>\pushed.csv`,
   never writes `review_assigned.txt`, never touches the production session
   folder, `inbox\` or `outbox\`. Same-size files already on the server are
   skipped, so a second trainee's push of the same sessions still copies the
   full set (different folder) but an interrupted push resumes.
2. **Tell the trainee** (template):
   > Your training bundles are under `X:\Julian\cnmfe_review\training\<Name>\`.
   > Copy a session folder to a local disk, open `run_training.m` in MATLAB
   > and run it. Start with the gallery (menu 5) and the easy session, menu 1
   > in coach mode. When you are done with a session copy its
   > `training_results\` folder to `training\<Name>\returns\<session>\`.
   > Never put anything in `inbox\`. The guide is `docs/TRAINING.md`.
3. **Collect and read** when results come back:
   ```
   <PYTHON_EXE> agent/summarize_training.py [--trainee <Name>] --dry-run
   <PYTHON_EXE> agent/summarize_training.py [--trainee <Name>]
   ```
   Reads the server (read-only), copies result files under
   `.training\<Name>\returns\`, writes `.training\summary.md` and
   `progress_all.csv`. Per trainee and attempt: shown, scored, contested,
   agreement, kappa, false keeps / reference deletes, false deletes /
   reference keeps, m-tags on motion items, contested sided with reference /
   model, decisions changed after feedback, median seconds per decision.
4. **Decide progression.** Suggested reading order: false deletes of real
   cells first (the costly error), then false keeps, then motion tagging, then
   the trend across attempts; flagged items (`n_flagged`) are the trainee
   saying the key is wrong -- open that session's report under
   `returns\<session>\training_results\` before judging. Coach-mode rows score
   the first decision, before the feedback. Then push the next stage's bundle.

## 3. What the numbers mean (and do not)

- The reference labels are one reviewer's post-video decisions. A trainee
  "error" can be the reference's error; the `x` flag and the contested list
  are the only mitigations. `--reviewer` on the key builder and the
  `reviewer` column of the curriculum make reviewer differences visible.
- Model scores are **in-sample** (the deployed model was trained on these
  labels), recorded as `score_kind = insample` in every key. "Contested" is
  therefore rarer than it would be with out-of-fold scores.
- Contested items are never counted, whatever the trainee decides; the side
  numbers are descriptive.
- Auto-rejected candidates are absent from `review_neuron.mat` and cannot be
  drilled; only threshold-0 sessions show every candidate.
- Seven BLA sessions were reviewed before the `m` key existed; their motion
  numbers are empty. The curriculum flags them (`has_motion_field`).
- Exam mode is honour-based: the key travels in the bundle.

## 4. Parameters (named assumptions)

All are flags, stored in every key (`key.params`) and printed in every report
footer. Defaults were chosen on 2026-10-01 without measurement.

| Flag | Default | Role |
|---|---|---|
| `--contest-score` | 0.5 | model keep / delete midpoint used for "contested"; not the joblib `reject_threshold` |
| `--clear-margin` | 0.25 | clear-cut = model at least this far from the midpoint (stage-1 subset) |
| `--hint-low-pct` / `--hint-high-pct` | 0.2 / 0.8 | within-session rank cut-offs for hints |
| `--dup-overlap` | 0.5 | cosine footprint overlap for the duplicate hint |
| gallery `--per-category` / `--max-per-session` | 12 / 2 | gallery size |
| planner `--min-keeps` | 10 | sessions with fewer reference keeps are not proposed |
| drill `pre_sec` / `post_sec` | 2 s / 3 s | lead-in before a transient's onset, run-out after the peak (`n` / `p`); the real tool uses the same values |

Across the 93 keys built on 2026-10-01 the contested fraction at the 0.5
midpoint had a median of 0.16 in BLA and 0.10 in vCA1.

## 5. Safety contract

- The server is read-only for the code except the copy-only push
  (`shutil.copy2` + two generated text files) into `training\`.
- Nothing writes into a production session folder, `inbox\`, `outbox\`, or a
  `review_assigned.txt`.
- `agent/ingest_returns.py` refuses any folder carrying `TRAINING_SESSION.txt`
  (name matching would otherwise land practice labels on the real session).
- Everything central lives under the dot folder `.training\`, which the
  watcher, ingest, trainer and push scripts all skip.
- Progress-file columns are only ever appended; older files are read as a
  prefix and extended in place by the MATLAB appender.

## 6. Tests to run after a change

```
<PYTHON_EXE> agent/test_training_key.py
<PYTHON_EXE> agent/test_training_guard.py
<PYTHON_EXE> agent/test_push_training_bundle.py
<PYTHON_EXE> agent/test_ingest_guard.py
"<MATLAB_EXE>" -batch "addpath(genpath('<repo>/ca_source_extraction')); addpath(genpath('<repo>/training')); acorn_training_selftest('<scratch>', '<a pool session dir>', '<its training_key.mat>', '<a staged bundle dir with the video>');"
```
The MATLAB self-test covers the scorer, the progress file, the static, video
and workflow drills with scripted answers and redos, the report and the
transient finder; the `-batch` string must not contain `%`. The interactive
parts (slider, `n` / `t` / `p`, coach prompts, the dress rehearsal) need a
human run.

## 7. Known limits

- The dress rehearsal runs the real `CNMFe_final_save` in the base workspace,
  exactly as `run_final_review.m` does; it needs the 64 GB machine and ~20 min
  before the first prompt.
- The drill's movie panel mirrors `viewNeuronsVideo` including its contour
  scaling; check it visually on one session before the first stage-3 trainee.
- Gallery images are static; motion is best learned in the drills.
