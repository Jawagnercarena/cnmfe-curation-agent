# Setup Guide

How to set up this repo on a new machine and run the CNMFe curation system.
(For people who only do the **manual MATLAB review**, see [REVIEW_SETUP.md](REVIEW_SETUP.md) instead — that role needs MATLAB only, no Python.)

This supersedes the older `SETUP_INSTRUCTIONS.txt`.

---

## What this is

A pipeline that turns motion-corrected, averaged 2‑photon miniscope videos into curated neurons:

1. **Headless CNMFe** (MATLAB) extracts candidate neurons from each `.tif`.
2. **Agent auto-curation** (Python) scores candidates with a trained classifier and pre-rejects the obvious junk, producing a small review package.
3. **Human review** (MATLAB) — a person makes the final keep/delete/merge decisions.
4. **Auto-retrain** (Python) — the classifier retrains on the new human labels.

There are two machine **roles**:
- **Central machine** — runs steps 1, 2, 4 (heavy compute + the single canonical model), stages/retrieves work via the lab server, and still does some of the review itself.
- **Reviewer machines** — run step 3 (MATLAB) and take the bulk of the reviewing. They never train the model.

---

## Prerequisites

- **MATLAB R2023b** + toolboxes: Image Processing, Signal Processing, Statistics, Optimization, Curve Fitting.
- **Python 3.10** (conda recommended) — central machine only.
- **Git**.
- A data tree laid out as `DATA_PARENT/{AREA}/{TASK}/{SESSION}/` (default `D:\Julian_CNMFe`).
- Access to the lab server exchange folder (UNC path, see step 6).

---

## 1. Clone

```
git clone https://github.com/Jawagnercarena/cnmfe-curation-agent.git
```

## 2. Python environment (central machine)

```
conda env create -f environment.yml
conda activate cnmfe
```
(or `pip install -r requirements.txt` into an existing Python 3.10 env.)

## 3. MATLAB path

In MATLAB:
```matlab
cd <path-to>\cnmfe-curation-agent
addpath(genpath(pwd));   % repo root + bundled libraries
savepath;                % persist across restarts (optional)
cvx_setup                % one-time CVX registration
```
CVX ships here with the free **SeDuMi / SDPT3** solvers; the commercial Gurobi/MOSEK solvers are intentionally not included.

## 4. Configure machine paths

Copy the template and edit it (the file is gitignored — never committed):
```
copy agent\.env.example agent\.env
```
Set whichever differ from this machine's defaults:
```
CNMFE_DATA_PARENT=D:\Julian_CNMFe
CNMFE_MATLAB_EXE=C:\Program Files\MATLAB\R2023b\bin\matlab.exe
CNMFE_PYTHON_EXE=C:\ProgramData\anaconda3\envs\valence\python.exe
CNMFE_EXCHANGE_ROOT=\\kheirbek-nas.cin.ucsf.edu\kheirbek1\Julian\cnmfe_review
```
All paths resolve through [agent/local_config.py](agent/local_config.py); anything left unset falls back to its default there.

## 5. Run the pipeline (central machine)

```
python agent\watcher.py          # BLA
python agent\watcher_vCA1.py     # vCA1
```
Each watcher polls for new `.tif` files, runs headless CNMFe, auto-scores, and writes a review package (`review_neuron.mat`, `run_final_review.m`, `review_report.pdf`, …) into the session folder. When a reviewer's `labels.mat` lands in a session folder, the watcher **auto-retrains** the classifier (`agent/model/{AREA}/classifier.joblib`).

---

## 6. Moving data to / from the server  *(temporary manual process)*

> This is the current interim workflow. It will be streamlined later (e.g. a published package / automated sync). For now, data hops through the lab server's `outbox` (to reviewers) and `inbox` (back from reviewers).
>
> **Server safety:** the scripts here **never delete anything on the server** — `push` only copies out, `ingest` only reads the inbox and writes to your local tree. Any server cleanup is manual.

**Send a session out to a reviewer** (run on the central machine). `--assignee`
names the reviewer and routes the bundle into **their** outbox folder, so no two
machines ever pick up the same session:
```
python agent\push_review_bundle.py {AREA}\{TASK}\{SESSION} --assignee Alisia --dry-run   # preview the ~2.6 GB bundle
python agent\push_review_bundle.py {AREA}\{TASK}\{SESSION} --assignee Alisia             # actually stage it
```
To hand out many at once, batch every awaiting session and split it across
reviewers round-robin (comma-separated names):
```
python agent\push_review_bundle.py --all --assignee Alisia,Julian               # all areas
python agent\push_review_bundle.py --all --area BLA --task 2tones --assignee Alisia
```
This copies the bundle to `…\cnmfe_review\outbox\{REVIEWER}\{AREA}\{TASK}\{SESSION}\`. The bundle is:
`{SESSION}.mat` (raw video), `review_neuron.mat`, `run_final_review.m`, `Cn.mat`, `pnr.mat`, `Ybg_weights.mat`, `review_report.pdf`, `review_summary.txt`.

*Manual alternative:* copy those files into the same `outbox\{REVIEWER}\{AREA}\{TASK}\{SESSION}\` path by hand.

**Bring a reviewer's curated session back** (run on the central machine):
```
python agent\ingest_returns.py            # ingest everything waiting in inbox
python agent\ingest_returns.py --dry-run  # preview first
```
This auto-discovers each reviewer's `…\cnmfe_review\inbox\{REVIEWER}\{AREA}\{TASK}\{SESSION}\` and mirrors it into your local `DATA_PARENT\{AREA}\{TASK}\{SESSION}\` (the reviewer-name prefix is dropped; the unchanged video is skipped), and the watcher then auto-retrains on the new `labels.mat`. Reviewers track their own progress for free this way: `outbox\{REVIEWER}\` is their to-do pile, `inbox\{REVIEWER}\` is their done pile.

*Manual alternative:* copy the returned folder (`neuron.mat`, `labels.mat`, `A.txt`, traces, `spatial_footprints.mat`, `ROIs.jpg`, `{SESSION}_neurons/`, …) into your local data tree by hand.

---

## 7. Reviewer machines

Hand reviewers the repo and [REVIEW_SETUP.md](REVIEW_SETUP.md). They need MATLAB only — no Python, no model. They pull a bundle from **their own** `outbox\<name>\` folder, run `run_final_review.m`, and drop the finished folder in their own `inbox\<name>\` folder. (Their `outbox\<name>\` is their to-do list, `inbox\<name>\` is their done list — see [REVIEW_SETUP.md](REVIEW_SETUP.md).)

---

## Notes

- The trained model lives at `agent/model/{AREA}/classifier.joblib` (gitignored — regenerated by training; snapshot to a GitHub Release at milestones).
- Adding a new brain area: duplicate `config_vCA1.py` → `config_{AREA}.py`, plus the matching `watcher_{AREA}.py` / `train_classifier_{AREA}.py` wrappers.
