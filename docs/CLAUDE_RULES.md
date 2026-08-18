# Rules for Claude on this project

A portable set of standing constraints for anyone running Claude (or any coding
agent) against this pipeline, this data, or the lab server -- on any machine.

**How to use it:** copy this file to `CLAUDE.md` at the root of your checkout, or
paste it into your Claude at the start of a session. Section 1 is the part that
matters even if you never touch this repo's code, because it is about the shared
lab server.

Rules marked **[explicit]** were stated directly by Julian and are not Claude's
inference. Treat them as non-negotiable. Rules marked **[derived]** come from bugs
that actually happened; they are strong defaults, not absolutes.

---

## 1. Never write to the lab server

**[explicit, 2026-06-25]** The lab server holds everyone's irreplaceable data and
has no undo.

- Server root: `\\kheirbek-nas.cin.ucsf.edu\kheirbek1` (often mapped to `X:`).
- **Claude must not create, write, move, rename, or delete anything under it** --
  not files, not folders, not "just an empty directory so the script works."
  Julian rejected exactly that (a `New-Item` folder creation) and asked that the
  rule be abundantly clear.
- A **human operator** performs every server write. Claude may read, and may
  *propose* a command for the operator to run, clearly labelled as such.

Concretely, none of these may target a server path:

| Tool | Forbidden against the server |
|------|------------------------------|
| PowerShell | `New-Item`, `Remove-Item`, `Move-Item`, `Rename-Item`, `Copy-Item -Destination`, `Set-Content`, `Add-Content`, `Out-File`, `>`, `>>` |
| Bash | `mkdir`, `rm`, `mv`, `cp`, `touch`, `>`, `>>`, `rsync`, `tee` |
| Python | `Path.mkdir`, `open(..., "w"/"a")`, `shutil.copy*`, `shutil.move`, `shutil.rmtree`, `Path.unlink`, `Path.rename`, `os.remove`, `os.rmdir` |

This applies to the *destination*. Reading from the server is fine.

**Exception, for humans only:** reviewers pushing a finished folder back to their
server inbox is a normal part of the workflow. That is a person doing it by hand.
It is not a licence for Claude or a script to write there.

## 2. Exchange tooling must never delete from the server

**[explicit, 2026-06-25]** The two scripts that talk to the server have a fixed
contract:

- `agent/push_review_bundle.py` -- **copy-only**, into `outbox/<reviewer>/...`.
- `agent/ingest_returns.py` -- **read-only on the server**; it writes only into
  the local data root.

If you edit either one, or add any new exchange tooling, grep it for delete verbs
before you call the work done. (Re-verified clean on 2026-08-08: the only
`rmtree`/`unlink`/`move` calls in `agent/` are in `bootstrap_preagent.py`,
`validate_threshold.py`, `curator.py`, `train_classifier.py`, and `watcher.py`,
all against local scratch paths.)

Corollary: a missing server folder is not an error to fix by creating it.
`ingest_returns.py` exits gracefully when the inbox is absent, and should stay
that way.

## 3. Test path-touching logic against temp directories

**[explicit]** Anything that copies, moves, or resolves destinations gets
exercised against a scratch directory or a dry-run first -- never against the
server and never against the real data root on its first outing.

---

## 4. Repo and code constraints

These matter if you are editing this codebase. They are all rules that were
learned from silent, hard-to-diagnose breakage.

**4a. No `%` comments in MATLAB passed to `_run_matlab()`. [explicit]**
That function collapses the script to a single line, so `%` comments out
everything after it. Use semicolons only. Symptom when violated: MATLAB appears
to do nothing and reports no error.

**4b. No non-ASCII characters anywhere in source files. [explicit]**
Python and MATLAB alike, including string literals that get written to disk.
Use `--` not an em dash, `->` not an arrow, straight quotes, no box-drawing
characters. Windows `cp1252` raises `UnicodeDecodeError`/`UnicodeEncodeError`
when Python reads a file MATLAB wrote, or logs a Unicode arrow. This has broken
the watcher twice, and in one case made a *successful* training run log
"Training failed."
When reading Windows-generated files (e.g. `.m` files the watcher wrote), use
`encoding='latin-1'`, not `utf-8`.

**4c. Do not modify `C:\code\CNMF_E_legacy_Biane\`. [explicit]**
That is the legacy checkout. The active repo is
`C:\code\CNMF_E_LEGACY_BIANE_CLAUDE\`. The legacy copy is also still on the saved
MATLAB path on the central machine, so it can shadow functions -- if MATLAB calls
a version of a function you did not edit, run `which -all <fnname>`.

**4d. Do not use `conda run` or `conda activate` from a tool call. [explicit]**
They are not reachable from the agent's shell. Call the interpreter by full path
(see section 6).

**4e. The project belongs to the Kheirbek Lab at UCSF. [explicit, 2026-07-30]**
Do not describe it as a Biane-lab project or a Biane-lab fork in any paper,
README, figure, or commit message. `LEGACY_BIANE` in the directory name is
historical only. The system is called **ACORN** (Automated CNMFe Of
Recording-Networks).

**4f. Review whitespace-heavy MATLAB diffs with `git diff -w`. [derived]**
Wrapping a loop in `try/catch` re-indents the whole block; without `-w` the real
change is invisible.

---

## 5. Workflow rules

**5a. Training is central and single-canonical. [explicit]**
One model per brain area, trained on the central machine. Reviewers review; they
never retrain, and a reviewer machine should not be running
`train_classifier.py`. If you are on a reviewer machine, your outputs are
labels, not models.

**5b. Scripts never delete assignment markers. [explicit]**
`review_assigned.txt` marks a session as out for review. Re-opening a session by
deleting that marker is a deliberate human action.

**5c. Ingest must never invent a brain-area folder. [derived, 2026-07-12]**
A reviewer once returned a session at `inbox/<name>/<task>/<session>`, dropping
the `{area}` level. The old "last three path parts" heuristic turned the
reviewer's *name* into a phantom brain area, which silently stranded five
sessions' labels away from their features and excluded them from training. The
fix resolves the destination by **session name** against the local tree and
**skips with a loud warning** if it cannot. Preserve that behaviour: skipping is
correct, guessing is not.
Reviewers: keep the full `{area}/{task}/{session}` layout under your inbox name.

**5d. Ingest matches sessions by name, so a reviewer rename forks the session.
[derived, 2026-08-07]** A reviewer corrected a session folder name (adding a
missing depth field); ingest could no longer match it, created a second folder,
and the labels were stranded from their features -- silently dropping the session
from training. Before concluding "this is a new session," check whether it is an
existing one under a new name.

**5e. A new imaging prep is a new regime -- do not assume the model transfers.
[derived, 2026-08-05]** Four sessions from a retro-tdTomato prep were 0.999
separable from the vCA1 training distribution and only 0.655 from BLA: the model
was extrapolating, and wanted to auto-reject 44% of candidates. They were
repackaged at `--threshold 0` so a human saw everything. For any new area,
prep, or indicator: start with a conservative or zero auto-reject threshold and
only tighten once there are enough reviewed sessions to measure a false
auto-reject rate.

**5f. Report faithfully. [explicit, standing]**
If a run failed, say so with the output. If a step was skipped, say that. Do not
report a pipeline stage as complete on the basis that the command exited zero --
on this project a timed-out MATLAB job has more than once kept running and
succeeded while the log said "failed."

**5g. Keep memory current without being asked. [explicit]**
At the end of a substantive task or a context switch, update the memory files.
Julian has had to ask for this more than once; it should be automatic.

---

## 6. Values you must re-derive on your own machine

Do **not** copy these paths verbatim. They are the central machine's defaults,
resolved by `agent/local_config.py`; every one is overridable by an environment
variable or an entry in `agent/.env` (gitignored -- see `.env.example`).

| Env var | Central machine default | What it is |
|---------|------------------------|------------|
| `CNMFE_REPO_ROOT` | self-locating (parent of `agent/`) | repo checkout |
| `CNMFE_DATA_PARENT` | `D:\Julian_CNMFe` | parent of one folder per brain area |
| `CNMFE_MATLAB_EXE` | `C:\Program Files\MATLAB\R2023b\bin\matlab.exe` | R2023b is the lab's working version; R2025b is installed but errored |
| `CNMFE_PYTHON_EXE` | `C:\ProgramData\anaconda3\envs\valence\python.exe` | the `valence` env, Python 3.10 |
| `CNMFE_EXCHANGE_ROOT` | `\\kheirbek-nas.cin.ucsf.edu\kheirbek1\Julian\cnmfe_review` | server exchange root -- **read-only to Claude** |

Never hardcode an absolute path in a script. Import it from `local_config.py`;
MATLAB reads `getenv('CNMFE_DATA_PARENT')`.

Run Python by full path:

```bash
"/c/ProgramData/anaconda3/envs/valence/python.exe" agent/some_script.py
```

---

## 7. Context worth having before you analyse anything

Not rules -- orientation, so your Claude does not re-derive or contradict them.

- **The data are motion-corrected, averaged 2-photon** recordings, not 1-photon,
  despite CNMF-E's usual association.
- **13 features**, in a fixed positional order: 6 spatial (area, circularity,
  eccentricity, compactness, max_weight, weight_spread), 5 temporal (peak_snr,
  transient_freq, events_per_min, baseline_stability, skewness), 1 motion
  (motion_correlation), 1 correlation-image (cn_correlation). The order is a
  contract between the extractor and the model; changing it requires an atomic
  swap of both.
- **Per-area models and configs**: `agent/config.py` (BLA) and
  `agent/config_vCA1.py` (vCA1) define `DATA_ROOT` and `MODEL_DIR`; models live
  in `agent/model/{AREA}/classifier.joblib`.
- **`candidate_features.npz` holds all N candidates; `labels.mat` covers only the
  non-auto-rejected subset.** Any code joining them must reconstruct the full
  label vector (auto-rejected implies label 0). Length mismatches here are a
  recurring source of silent bugs.
- **Historical candidate traces are gone** -- overwritten at finalize. Any
  trace-based feature can only be computed forward, on new sessions.
- **Bootstrap sessions are load-bearing for calibration.** An XGBoost model
  trained on agent sessions alone is uncalibrated (>= 4.2% false auto-reject at
  any threshold). The small AUC cost of including bootstrap data is the price of
  a trustworthy probability scale -- do not "clean up" the training set by
  dropping them.
- **Docs**: [SETUP.md](SETUP.md) for the central role,
  [REVIEW_SETUP.md](REVIEW_SETUP.md) for the reviewer role (MATLAB only, no
  Python).
