---
name: push-review-bundle
description: Stage curated sessions on the lab-server outbox for a named reviewer - dry run, whole-FOV dealing, right threshold and gSig first, --force only with the operator's server delete.
disable-model-invocation: true
argument-hint: "--assignee <name> (<area>\\<task>\\<session> | --all [--area A] [--task T] [--animal a1,a2]) [--force] [--dry-run]"
---

# Push review bundles

Manual-only: it writes to the lab server (copy-only, into `outbox/<assignee>/`)
and marks sessions out for review. Central machine only. `$ARGUMENTS` are passed
to `agent/push_review_bundle.py`.

## 0. What the script does and does not do

- Copies `{SESSION}.mat`, `review_neuron.mat`, `Cn.mat`, `pnr.mat`,
  `Ybg_weights.mat`, `review_report.pdf`, `review_summary.txt` with `shutil.copy2`
  and generates a self-locating `run_final_review.m` into
  `<exchange>/outbox/<assignee>/<area>/<task>/<session>/`. A file already there
  is skipped only if byte-identical (size alone is not trusted), so a re-run
  resumes and an in-place re-push replaces anything that changed; the check
  reads an identical raw video back from the server in full.
- For pnb* sessions, first runs the retro hook `python -m retro_align
  guide-for-push <session> [--dry-run]` (cwd = the retro project), which reuses
  a current guide or rebuilds it there (needs X: mapped). Then adds
  `red_guide.png` from `CNMFE_RETRO_RESULTS` only if the hook said `red guide ok`
  and its `candidates_shown.npz` matches the live review set (same N, same
  auto-rejected indices). Prints the hook's line, then `red guide: included` or
  `STALE, NOT included`. A failed, timed-out or missing hook means no guide,
  never a blocked push. `CNMFE_RETRO_GUIDE_BUILD=0` skips the hook (lookup
  only). Two matching guides is an error.
- Writes `review_assigned.txt` into the **local** session folder (assigned, to, by).
- Never deletes anything anywhere (CLAUDE.md section 1). Re-assignment needs the
  operator to delete the stale `outbox/<old>/` folder on the server first.
- `--assignee` is required. `--all` picks sessions with `ROIs_candidates.jpg` and
  `review_neuron.mat` but no `ROIs.jpg`, skipping dot-dirs and, without `--force`,
  anything already marked out.
- `--animal` matches lowercase letters + digits only (`bla12`, `pnb97`). DG animals
  (`DG6D`) do not match; use `agent/push_DG_AL_batch.py --plan` / `--dry-run`, which
  also refuses sessions processed at the wrong gSig.

## 1. Pre-flight

1. **Threshold.** A session from a new prep, animal or indicator must have been
   re-curated at threshold 0 first (`/recurate-sessions`); otherwise the reviewer
   never sees what the model extrapolated away. vCA1 and BLA watchers curate at
   the joblib threshold regardless of prep.
2. **Whole FOVs, one reviewer.** Never `--assignee A,B` (the docstring's
   round-robin): it splits a field of view across people and the same cells get
   judged twice, inconsistently, forever (2026-08-18). Deal whole FOV groups; if
   `agent/plan_fov_assignment.py` is present it plans that and prints the operator
   delete list. Then run one push per assignee with `--animal` / `--task` filters.
3. **Dry run always:**

```bash
"<PYTHON_EXE>" agent/push_review_bundle.py --all --area <AREA> --animal <a1,a2> --assignee <Name> --dry-run
```

   Read the list: is every session one you meant to send, and none already out?
   A `WARNING: raw video ... not found` means the reviewer cannot do the video
   pass; fix before staging. For pnb* sessions, `red guide: would be built at
   push (...)` is normal (the real push builds it; one timed build took 48 s);
   `red guide NOT MADE: <reason>` means the session will go without a guide, so
   fix the reason first if the guide matters (e.g. map X:). Never write into
   the retro project or call its `red-guide --force` from here.
4. **Single-session mode** re-stages even when a marker exists (prints `NOTE:
   already out for review ...`) without `--force`. Check the marker yourself.

## 2. Run

Same command without `--dry-run`. Success per session: `copy <file> (N MB)` /
`skip (already staged)` lines, `generated run_final_review.m; marked
out-for-review`, `-> staged N MB to <dest>`; for `--all`: `Done: staged X/Y
session(s), N MB total.` Anything in `failures` (missing `review_neuron.mat`) is
listed at the end and was not staged.

## 3. Re-assigning (`--force`)

Only when the operator has already deleted `outbox/<old reviewer>/...` for those
sessions on the server and told that reviewer. `--force` re-stages marked sessions
and rewrites the marker to the new name. Never delete the marker or the outbox
copy yourself.

## 4. Afterwards

- Tell the operator what to tell the reviewer: bundle path and session list.
- Update memory: reviewer, area, sessions, any `--force` re-deal, and the
  remaining awaiting count.
