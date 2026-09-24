---
name: recurate-sessions
description: Re-score unreviewed sessions with the current model, usually at --threshold 0 for a new prep, animal or indicator so a human sees every candidate. Rewrites the review package; refuses labeled sessions.
disable-model-invocation: true
argument-hint: "<BLA|vCA1|DG_AL> [<session_dir> ...] [--threshold X]"
---

# Re-curate sessions (threshold override)

Manual-only: it rewrites `candidate_features.npz`, `review_neuron.mat`,
`review_report.pdf` and `review_summary.txt` in place and there is **no dry run**.
Central machine only. `$ARGUMENTS` = area, then optional explicit session dirs and
`--threshold X`.

## 0. Resolve the area and the scope

| Area | Script |
|---|---|
| BLA | `agent/recurate_sessions.py` |
| vCA1 | `agent/recurate_sessions_vCA1.py` |
| DG_AL | `agent/recurate_sessions_DG_AL.py` |

A bare `recurate_sessions.py` is BLA: it would scan BLA's data root and score with
BLA's model. Python by full path (`local_config.PYTHON_EXE`).

**Scope matters more than the flag.** With no session paths the script re-curates
*every* pending session of the area (`find_pending_sessions`: has
`candidate_features.npz` + `review_neuron.mat` + `ROIs_candidates.jpg`, no
`labels.mat`, no `bootstrap_match_stats.json`, dot-dirs skipped). A threshold
override applied that way hits sessions it was never meant for. For a new prep or
animal, pass the explicit session directories (this is how the retro-tdTomato
sessions were handled, 2026-08-05 and 2026-08-18).

## 1. When threshold 0 is the right call

CLAUDE.md rule 4: a new imaging prep, animal, indicator or objective is a new
regime; the calibrated cutoff is extrapolation there. The 2026-08-26 red team
measured about 4.5% false auto-reject on a held-out animal at the deployed
thresholds, so the first pass on anything new is `--threshold 0` (nothing
auto-rejected). Tighten only after a false auto-reject rate has been measured on
reviewed sessions of that regime.

This is different from `curator.THRESHOLD_OVERRIDE` (`agent/curator.py:52`),
which pins the *watcher's* live curation and is set only by `watcher_DG_AL.py`.
`recurate_sessions.py` does not read it. Consequence for vCA1 and BLA: every new
session of the same prep that the watcher curates later is back at the joblib
threshold and must be re-curated by hand *before* it is pushed for review.

## 2. Pre-flight, per session

- **Not labeled.** The script refuses with `[GUARD] labels.mat exists` and that is
  correct: regenerating the review package on a labeled session breaks its
  pre-decision provenance. Do not work around it.
- **Not out for review.** If `review_assigned.txt` exists, the bundle on the server
  is about to become stale. Either the reviewer has not started (then re-push with
  `/push-review-bundle ... --force`, operator deletes the stale outbox copy) or they
  have (then do not re-curate: their labels would come back against a different
  candidate set). Ask the operator.
- **MATLAB slot.** `_write_review_mat` shells out to MATLAB per session; the machine
  caps at two concurrent MATLAB runs and a running watcher holds one.
- Record the current `auto_rejected` count from the npz for the before/after:

```bash
"<PYTHON_EXE>" -c "import numpy as np,sys; d=np.load(sys.argv[1]); a=d['auto_rejected']; print(sys.argv[1], 'auto_rejected', int(a.sum()) if a.size else 0, 'of', int(d['n_candidates'].ravel()[0]))" "<session>/candidate_features.npz"
```

## 3. Run

```bash
"<PYTHON_EXE>" agent/recurate_sessions_<AREA>.py "<session_dir>" ["<session_dir>" ...] --threshold 0
```

Per session expect `Threshold override: 0.000 (model's calibrated value was ...)`,
`Newly rejected (removed from review): N`, `Rescued from auto-reject (added to
review): N`, `Review set: N neurons (was M with old model)`, then
`All done. Open MATLAB and run run_final_review.m for each session.`

## 4. Verify from the files, not the summary

Re-run the npz one-liner: at threshold 0 `auto_rejected` must be 0 and the review
set must equal `n_candidates`. Do not take the threshold from
`review_summary.txt`; it has misreported it before (2026-08-18). Training
consequence to state in the report: with an empty `auto_rejected` array every
candidate of that session carries a human decision (`train_classifier`'s
"sizes match" path), which is the point.

## 5. Afterwards

- Push (or re-push with `--force`) via `/push-review-bundle`.
- Update memory: sessions, area, threshold, before/after counts, and whether the
  watcher will re-curate later same-prep sessions at the joblib threshold (vCA1,
  BLA: yes).
