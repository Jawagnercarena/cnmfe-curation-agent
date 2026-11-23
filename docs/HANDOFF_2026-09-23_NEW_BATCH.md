# Handoff: qualify the new sessions, validate the v2 threshold, run the next batch

Written 2026-09-23 21:00 for a fresh session. Every number below was read from
the joblibs, logs or result files at that time; re-check anything older than a
day (CLAUDE.md section 6: never quote a threshold or AUC from docs). Start in
plan mode; nothing here authorises a write to the lab server.

## 1. State on 2026-09-23 21:00 (verified)

| area | deployed model (`agent/model/<AREA>/classifier.joblib`) | today's agent-pool report (watcher log, one seed) |
|---|---|---|
| BLA | xgboost, 35 columns (v2), T = 0.04, agent weight 4.0, 179 sessions, retrained 10:52 | reviewed AUC 0.912; at 0.04: false auto-reject 0.57% (15 of 2643 kept cells), junk 41.9% |
| vCA1 | xgboost, 35 columns (v2), T = 0.04, agent weight 5.0 (override), 161 sessions, retrained 10:28 | reviewed AUC 0.925; at 0.04: **false auto-reject 1.43% (14 of 976)**, junk 48.5% |
| DG_AL | xgboost, 13 columns, joblib T = 0.04 but the watcher overrides to 0.0, 11 sessions (09-04) | 7 sessions curated and awaiting labels |

- All three watchers were alive (heartbeat 20:59 in `agent/logs/watcher*.log`).
- **No unprocessed `.tif` exists under `D:\Julian_CNMFe` for any area** (no loose
  tifs, no session dir without `neuron.mat`). The 50 sessions to process are not
  staged yet. The vCA1 queue's "Complete -- agent pipeline (50 sessions)" list is
  the set of already-labelled agent sessions, not a backlog. Step 2 starts with
  finding out where the 50 are and what they are (area, animals, prep, task, AVG
  factor). Assumption until told otherwise: vCA1, retro-tdTomato prep (pnb97/98),
  the prep the v2 threshold question is about.
- Pool growth since the last 8-seed evaluations: vCA1 158 -> 161 labelled
  (3 pnb98 sessions, Taylor, since 09-19); BLA 176 -> 181 (5 bla36 sessions since
  09-06). Two vCA1 sessions are out with Aneesh (071526-pnb98, 080626-pnb98), one
  BLA with a reviewer (07202026-bla36).
- Branch `claude-config`. Uncommitted and live: `agent/train_classifier.py` (the
  agent-pool report, 09-19), `agent/eval_motion.py` (docstring), `agent/ingest_returns.py`
  (earlier, not by this line of work), `agent/config/animal_params.json`,
  `agent/plan_fov_assignment.py`, `docs/CNN_BLEND_STAGE3_BRIEF.md`, everything under
  `agent/eval/algo_bakeoff_2026-09/`. Nothing in this handoff needs a commit first.

## 2. The threshold question, answered as far as the data go

The vCA1 v2 model at T = 0.04 has been outside the 1%-worst-seed posture on every
evaluation since the 09-06 bake-off. On the 09-19 pool (39 CV sessions, 934 kept
cells, 8 seeds, threads pinned; `agent/eval/algo_bakeoff_2026-09/results_0919/`):

| T | session-grouped false auto-reject mean / worst | junk | animal-held-out mean / worst | junk |
|---|---|---|---|---|
| 0.03 | 0.74% / 0.96% | 40.9% | 0.67% / 0.86% | 40.4% |
| 0.04 (deployed) | 1.06% / 1.28% | 45.2% | 0.91% / 1.28% | 44.7% |
| 0.05 | 1.50% / 1.71% | 48.6% | 1.20% / 1.61% | 48.1% |

The rule (largest T in [0.03, 0.10] with mean false auto-reject <= 0.85% and
worst seed <= 1.0%, junk >= 30%) admits **0.03 under both groupings** and rejects
0.04. The misses at 0.04 sit on pnb97 (2.3% of its kept cells), not pnb98 (0.8%)
or pnb88 (0%). Today's one-seed 1.43% on 161 sessions points the same way. BLA at
0.04 is inside the posture (0.57% today; 8-seed 0.64% / 0.96% worst on 08-26).

**Recommendation for the batch:** vCA1 at **T = 0.03** for sessions of animals
already in training (pnb88/97/98); **T = 0 (manual re-curate) for any new
animal, prep, indicator or objective** (CLAUDE.md section 4; the 08-26 red team
measured ~4.5% false auto-reject on a held-out animal at the deployed
thresholds). Cost of 0.03 vs 0.04: about 4 points less junk removed before a
human sees the session (41% vs 45%), i.e. roughly 3 more candidates per 100.
Step 4 re-derives this on the 161-session pool before it is applied.

## 3. The classifier question

- Tabular: nothing beats the deployed XGBoost (12 contenders, null; nested tuning
  hurts). Keep it. Never bare `--model auto`.
- CNN blend (`fixed12+nbtrace+ens3`, 50/50 with XGBoost) meets its pre-registered
  gate on BLA (+0.010/+0.014 AUC, +5 to +8 pp junk at matched false auto-reject)
  and on vCA1 as "not worse" (+0.011/+0.012 AUC, but only +1.9 / -0.4 pp junk on
  the 09-19 pool; for a held-out vCA1 animal its admissible T catches 38.9% of
  junk vs the tabular model's 40.4%). It is **not deployed**; deploying it is
  Stage 3 (`docs/CNN_BLEND_STAGE3_BRIEF.md`: 2-3 working days plus rehearsal, a
  torch subprocess bridge from the `ACORN` env, thresholds re-derived on the blend).
- **Do not hold the batch for it.** On vCA1 the blend is a ranking gain with no
  auto-reject gain, so for reviewers of a vCA1 batch it changes little. On BLA it
  would remove 5-8 more junk per 100 at the same safety; if the 50 sessions are
  BLA, that is the case for doing Stage 3 first. Either way the decision is the
  user's; ask, do not assume.

## 4. Checklist for the new session

1. **Locate and characterise the batch** (user input). Area, task folder, animals,
   prep, objective, AVG prefix in the file names (the features assume the AVG5x
   3.75 Hz rate, `agent/features.py:171`; one existing pnb98 session is AVG2x and
   is in training as is), gSig per animal (`agent/config/animal_params.json`;
   `agent/eval/sweep_gsig.py` if unknown). Anything new -> threshold 0 first pass.
   The operator copies tifs from the NAS into `D:\Julian_CNMFe\<AREA>\<task>\`;
   the watcher picks up loose tifs (`agent/watcher.py:62 find_new_tifs`). Capacity:
   two concurrent MATLAB runs on this machine (a watcher is one), ~75 GB each; a
   headless run was 139 min before the 1.62x background fix, so budget roughly
   1.5-2 h per session (estimate) -> about two days of compute for 50.
2. **Qualify the sessions already added** (16 vCA1 on 09-19, 3 more since, 5 BLA
   bla36). Per session: reviewer in `labels_provenance.txt` (all present); kept /
   reviewed / auto-rejected counts (`candidate_features.npz` holds all N,
   `labels.mat` only the reviewed subset -- reconstruct, do not zip); gSig used
   (`agent_run.log` "Final parameters" vs `animal_params.json`); motion QC flags
   in `review_summary.txt`; out-of-fold false auto-rejects per session from
   `agent/diagnose_model_vCA1.py` (honours the weight override). Flag any session
   whose kept cells score far below the pool (label or run problem, not model).
3. **Fresh pool snapshot and 8-seed control on the 161-session vCA1 pool**
   (and BLA on 181):
   ```
   cd agent/eval/algo_bakeoff_2026-09
   set BAKEOFF_RESULTS=results_0923      (PowerShell: $env:BAKEOFF_RESULTS="results_0923")
   C:\Users\julia\.conda\envs\ACORN\python.exe bakeoff_tabular.py --area vCA1 --grouping session,animal --contenders xgb_deployed --threads 8
   C:\Users\julia\.conda\envs\ACORN\python.exe bakeoff_tabular.py --area vCA1 --report
   ```
   `report_vCA1_t8.md` gives rule_T and false auto-reject at each T for both
   groupings with threads pinned (an unpinned run moves scores by up to 0.4 and
   flips the rule; pin it). `results_0919/` is the reference to compare against.
   The valence env also works for the tabular scripts.
4. **Apply the threshold** only if step 3 confirms it. vCA1's deployed T is a
   constant the wrapper injects on every retrain, watcher retrains included:
   `agent/train_classifier_vCA1.py:59-60` (`_BOOTSTRAP_THRESHOLD`,
   `_VALIDATED_THRESHOLD`). Change both, then `/retrain-area vCA1 --threshold 0.03`
   (the skill covers the watcher race and the joblib snapshot), then
   `agent/eval/vca1_v2_2026-08/verify_vca1.py` (it compares the joblib T with
   `gate_decision.json`, which still records 0.04 -- update that file's decided T
   in the same change or the check will fail for the right reason). Re-curate the
   sessions still unreviewed after the change if their auto-reject set matters.
5. **Threshold 0 for anything new** is manual on vCA1: `watcher_vCA1.py` sets no
   override, so a new animal is curated at the deployed T on arrival. Run
   `/recurate-sessions vCA1 <explicit session dirs> --threshold 0` *before*
   `/push-review-bundle`. Explicit paths only; the bare form re-scores every
   pending session.
6. **Push and ingest.** `/push-review-bundle --assignee <name> --dry-run` first;
   whole FOVs to one reviewer, never round-robin (FOV is free text in the retro
   imaging sheet Notes); safe vCA1 filter and reviewer map in
   `memory/project_pnb_area_misfiling.md`. Returns come back with
   `/ingest-returns --dry-run` then for real; the watcher retrains on the next
   poll; ingest preserves label mtimes, so after a manual batch retrain by hand.
7. **After every ingest, re-read the agent-pool line in the watcher log**
   (`AUC reviewed candidates` and `Agent-pool operating point`). The standing
   dynamic: each batch adds dim real cells, so false auto-reject drifts up at a
   fixed T while AUC stays flat. The all-session "CV AUC=0.94" line above it is
   model selection only (93% bootstrap rows on vCA1) and must not be quoted as
   performance.
8. **CNN Stage 3** is a separate track; if the user wants it, open
   `docs/CNN_BLEND_STAGE3_BRIEF.md` and plan it in its own session.

## 5. Where the evidence lives

- Bake-off, CNN rounds, refresh: `agent/eval/algo_bakeoff_2026-09/RESULTS.md`
  (README holds the pre-registrations and amendments; `results/` = 09-06 pool,
  `results_0919/` = 09-19 vCA1 pool; env `BAKEOFF_RESULTS` and
  `BAKEOFF_CORPUS_TAG_vCA1` select them).
- Threshold rule and harness: `agent/eval/bootstrap_redteam_2026-08/rt_lib.py`
  (`rule_T`, `matched_operating_point`), `agent/eval/README.md`.
- vCA1 v2 deploy record and verifier: `agent/eval/vca1_v2_2026-08/` (`VCA1_V2_LOG.md`,
  `gate_decision.json`, `verify_vca1.py`), `docs/VCA1_V2_BRIEF.md`.
- New-prep regime, threshold-0 mechanics, reviewer map: memory
  `project_pnb_area_misfiling.md`; FOV dealing: `project_fov_reviewer_assignment.md`.
- Trainer report change (uncommitted): `agent/train_classifier.py`
  (`_agent_pool_eval`, `bootstrap_report`, stored in the joblib as
  `agent_pool_eval`, `cv_breakdown`, `bootstrap_report`).

## 6. Things that will bite

- The 50 sessions are not on `D:` -- confirm with the user before touching anything.
- vCA1 threshold 0 is manual; BLA/vCA1 watchers curate new animals at the deployed T.
- `_VALIDATED_THRESHOLD` in the vCA1 wrapper silently redeploys the old T on the
  next reviewer return if it is not changed with the joblib.
- Thread count changes XGBoost scores and rule picks; every gate number above is t8.
- The 09-19 vCA1 numbers rest on three animals; leave-one-animal-out is dominated
  by pnb88 (old prep). A fourth animal is the first real test of any T.
- Reviewer renames fork sessions on ingest; check for an existing session under a
  new name before calling one new.
- No non-ASCII in code you write; `%` comments break `_run_matlab()`.
