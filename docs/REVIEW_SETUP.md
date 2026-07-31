# Remote Review Setup

This guide is for **reviewers** doing the final manual curation of CNMFe sessions
on their own machine. The heavy compute (headless CNMFe + agent auto-scoring)
runs on the central machine; you only do the interactive MATLAB review and send
the curated folder back. The classifier retrains **centrally** — you never train.

The review step is **pure MATLAB**. You do not need Python, conda, or the model.

---

## 1. Prerequisites

- **MATLAB R2023b** with toolboxes: Image Processing, Signal Processing,
  Statistics, Optimization, Curve Fitting.
- **Git** (to clone/update this repo).
- Access to the lab server exchange folder. Use the **drive-independent UNC
  path** `\\kheirbek-nas.cin.ucsf.edu\kheirbek1\Julian\cnmfe_review` (the
  `kheirbek1` share). It may be mapped to a different drive letter (e.g. `X:`) on
  your machine — the UNC path works regardless of the letter.
- ~64 GB RAM recommended; a fast **local** disk with room for ~3 GB per session.

> CVX is vendored in this repo with the free SeDuMi / SDPT3 solvers. The
> commercial Gurobi / MOSEK solvers are intentionally **not** included and are
> not needed.

---

## 2. One-time setup

1. **Clone the repo** (then `git pull` later to get updates):
   ```
   git clone <REPO_URL> CNMF_E_LEGACY_BIANE_CLAUDE
   ```
2. **Put it on your MATLAB path.** In MATLAB:
   ```matlab
   cd <path-to>\CNMF_E_LEGACY_BIANE_CLAUDE
   addpath(genpath(pwd));   % repo root + all libraries
   savepath;                % persist across MATLAB restarts (optional)
   cvx_setup                % one-time CVX registration (SeDuMi/SDPT3)
   ```
   `addpath(genpath(...))` is what lets the per-session launcher find
   `CNMFe_final_save` automatically. (`cnmfe_setup.m` alone does **not** add the
   repo root, so the launcher would not resolve.)

---

## 3. Per-session workflow

1. **Pull the bundle** from *your own* folder in the server outbox to a **local**
   working folder (do not run the review directly off the network share — the
   ~2.4 GB video needs local disk):
   ```
   \\kheirbek-nas.cin.ucsf.edu\kheirbek1\Julian\cnmfe_review\outbox\<your name>\{area}\{task}\{session}\
        ->  D:\review_work\{area}\{task}\{session}\
   ```
   Only the sessions under **your name** are assigned to you. Each session is
   placed in exactly one reviewer's folder, so two machines never pick up the
   same one - don't pull from another reviewer's folder. (If your folder is
   empty, nothing is assigned to you yet; ask the central operator to stage some.)
   The bundle contains: `{session}.mat` (raw video), `review_neuron.mat`,
   `Cn.mat`, `pnr.mat`, `Ybg_weights.mat`, `run_final_review.m`, and
   `review_report.pdf` / `review_summary.txt` (the agent's guidance).

2. **Run the review.** In MATLAB, open the session's `run_final_review.m` and
   run it (it self-locates to its own folder and launches `CNMFe_final_save`).
   Work through the steps: `viewNeurons` (delete bad cells) → optional update →
   `viewNeuronsVideo` (catch motion artifacts) → merges → save.

   > **Motion deletes:** at the per-neuron prompt you can now press **`m`**
   > instead of `d`. It deletes the neuron exactly like `d`, but also logs it as
   > a *motion artifact* (most useful in the `viewNeuronsVideo` pass, where you
   > see spikes that track brain motion). This trains the agent to pre-catch
   > motion — use `m` whenever the reason you're deleting is motion, and plain
   > `d` for everything else. No extra steps; it's saved automatically.

3. **Push the finished folder back** to **your own** inbox folder (whole folder
   is fine) - same name as your outbox folder:
   ```
   D:\review_work\{area}\{task}\{session}\
        ->  \\kheirbek-nas.cin.ucsf.edu\kheirbek1\Julian\cnmfe_review\inbox\<your name>\{area}\{task}\{session}\
   ```
   The key new file is `labels.mat` (your keep/delete decisions); the curated
   `neuron.mat`, traces, and `ROIs.jpg` come along for the archive.

   > **Keep the full `{area}\{task}\{session}` path** under your inbox name — the
   > same folder layout you pulled from your outbox. Don't drop the `{area}` level
   > (e.g. push to `inbox\<you>\{area}\{task}\...`, not `inbox\<you>\{task}\...`).
   > The simplest safe habit: copy the whole `{area}` folder as-is. (Ingest now
   > also matches returns by session name, so a slip is caught rather than
   > misfiled, but keeping the layout avoids the warning.)

**Knowing what's done (no extra bookkeeping).** Your two server folders are a
matched pair, both named after you:

- `outbox\<your name>\`  =  your **to-do** list (sessions assigned to you)
- `inbox\<your name>\`   =  your **done** list (sessions you've returned)

Returning a session in step 3 is the only "mark done" there is: anything still
only in your **outbox** is outstanding; once it's in your **inbox** it's
finished. There's no checklist to keep. (If you ever want to confirm a single
folder, a reviewed session contains `labels.mat` and `ROIs.jpg`; an un-reviewed
one only has `review_neuron.mat` and `ROIs_candidates.jpg`.)

That's it. The central operator runs `python agent/ingest_returns.py`, which
copies your curated folder into the canonical data tree and triggers an
automatic retrain.

---

## 4. Troubleshooting

- **Error about a missing `review_neuron.mat`, or `session_dir` pointing at `D:\...`** - your `run_final_review.m` is an old hardcoded launcher. Quick fix: in MATLAB, `cd` to the local session folder and run `session_dir = pwd; CNMFe_final_save`. Freshly staged bundles already self-locate, so re-pulling the session also fixes it.
- **"CNMFe_final_save.m not found ... Add the CNMFe repo to your MATLAB path"** —
  you skipped step 2. Run `addpath(genpath(<repo>))`.
- **CVX / solver errors during an update pass** — run `cvx_setup` once (step 2).
- **Slow startup / "recomputing background"** — make sure `Ybg_weights.mat` was
  included in the bundle and is in the session folder.

---

## Central operator notes (not for reviewers)

- Stage a bundle to a reviewer:  `python agent/push_review_bundle.py {area}\{task}\{session} --assignee <name>`
- Stage a whole batch, split across reviewers:  `python agent/push_review_bundle.py --all --assignee Alisia,Julian`
- `--assignee` is required: it routes the bundle into `outbox\<name>\` so each session lands in exactly one reviewer's folder.
- Bring results home + retrain:  `python agent/ingest_returns.py` (auto-discovers every `inbox\<name>\` folder; the reviewer-name prefix is dropped into the canonical tree)
- Set the server path once via `CNMFE_EXCHANGE_ROOT` (see `agent/.env.example`).
- **Server safety:** these scripts never delete anything from the server. `push`
  only copies to the outbox; `ingest` only reads the inbox and writes to the
  local `D:` tree. Any cleanup of the server is manual — it is never automated.
