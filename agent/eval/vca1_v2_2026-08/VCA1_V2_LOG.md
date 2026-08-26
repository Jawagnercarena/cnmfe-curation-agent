# vCA1 v2 (35-column contract) — project log

**DEFERRED: the swap is NOT executed in this project.** Scope is prep + gates +
a rehearsed swap kit; the freeze/swap/retrain/flip/watcher-restart waits on the
bootstrap red-team report and a clean reviewer-return watcher cycle. Deploy-day
procedure is in the plan's §3f.

Branch `vca1-v2-2026-08` off `feature-expansion-step4` (da0a22a).
All numbers below computed fresh; nothing carried from the brief.

## Step 0 — ingest npz-overwrite guard (commit 6bef472)

`ingest_returns.copy_session` copied every non-dot file from a returned bundle
unless the size matched. `candidate_features.npz` differs in size across feature
contracts, so a reviewer mirroring central folders could drop a stale 13-column
npz over a 35-column session — silently, because the same-size skip cannot fire
precisely when the contract differs. **BLA had been exposed since 2026-08-20.**

Fix: generalised the `labels_provenance.txt` skip into
`CENTRAL_ONLY = {PROVENANCE_NAME, "candidate_features.npz"}`. Reviewer bundles
never carry either file (`push_review_bundle` stages `review_neuron.mat` +
Cn/pnr/Ybg_weights/pdf/summary), so nothing legitimate is refused.

`agent/test_ingest_guard.py`, temp-dir fixture, **9/9 pass**: stale 13-col npz
and stale provenance both left byte-identical (and the stale npz *did* differ in
size, so the old code would have copied it); `labels.mat` and `review_neuron.mat`
still copied; `--force` does not override the guard; dot-prefixed parked trees
still skipped.

## Step 0b — parked `3odor/AVG5x-TSeries-030426-pnb88-187um-35z-000`

Aborted final save consumed by the trainer as ground truth: `labels.mat`
(2026-06-16 12:10) holds 29 labels / 1 keep over the 29-candidate review set
(46 candidates, 17 auto-rejected), but `C_df.txt` is 0 bytes at that same mtime,
there is no final `ROIs.jpg`, and `neuron.mat` / `C_raw.txt` /
`spatial_footprints.mat` are still the 2026-04-08 candidate-level files.
`CNMFe_final_save.m` writes `labels.mat` before `extract_DF_F` → `C_df.txt` →
`C_raw.txt` → `neuron.mat` → `ROIs.jpg`, so the save aborted right after the
labels. It was also still "Out for review" to Alisia since 2026-06-25 with no
`labels_provenance.txt`, so her return would have overwritten it unrefused.

Contents verified against that description before touching anything, then moved
to `D:\Julian_CNMFe\vCA1\.excluded\3odor\` (rename within D:, 70 files before ==
70 after) with `EXCLUDED_REASON.txt`. **Operator action: tell Alisia to skip it.**

## Pool re-baseline (2026-08-26 ~12:40)

The survey behind the plan measured 22 labeled agent sessions. The live pool
disagreed, and the cause was **an operator `ingest_returns` run at 12:16 today**,
mid-planning, which landed two of Aneesh's pnb97 returns:

| session | N | reviewed | positives | labels.mat | ingested |
|---|---|---|---|---|---|
| 6odorDualDiffRew/…061826-pnb97-679um-24z-000 | 135 | 135 | 17 | 08-25 15:07 | 08-26 12:16 (Aneesh) |
| 6odorDualDiffRew/…061926-pnb97-610um-24z-000 | 107 | 107 | 21 | 08-26 11:30 | 08-26 12:16 (Aneesh) |

(`labels.mat` mtimes are the reviewer's own — ingest `copy2` preserves them — so
the ingest timestamp comes from `labels_provenance.txt`.)

Reconciled exactly: 22 surveyed + 2 ingested − 1 parked = **23 agent**;
pending 31 − 2 = **29**; agent rows 1,026 + 135 + 107 − 46 = **1,222**;
CV-usable (≥5 positives) 14 + 2 = **16**, and the thin pnb97 prep goes 3 → **5**
test sessions. Bootstrap untouched (111 sessions / 50,370 rows / 6,662 masked).
Parallel-file totals are unchanged at 163 files / 55,007 rows, because the two
sessions moved from the pending policy to the labeled policy (242 rows) and the
parked session removed 46.

Measured via the trainer's own scanners (`find_prospective_sessions`,
`_is_bootstrap_session`, `_get_bootstrap_ambiguous_mask`), not by counting dirs.

Two consequences recorded for the deploy step:
1. The deployed joblib (2026-08-24 03:38) is now **stale** — the newest
   `labels.mat` is 2026-08-26 11:30 — so restarting `watcher_vCA1.py` would fire
   an auto-retrain at the 13-column contract. That is fine and expected, but it
   means the "no spurious retrain" check belongs *after* that retrain, not before.
2. **A manual `ingest_returns` moves the pool exactly as a watcher would.** The
   D5 "watchers stay down" decision does not by itself freeze the corpus. The
   server inbox is currently fully drained (64/64 returns ingested, none queued),
   so the pool is stable as of now.

Both new sessions are finalized (`ROIs.jpg` present, `C_raw.txt` down to 17 and
21 rows), so like the other 21 they need MATLAB extraction from
`review_neuron.mat` — extraction list is **23**.

## Step 3a — pin, provenance gate, MATLAB extraction (commit C1)

`vca1_common.py` is the shared module every script here imports first: it injects
`config_vCA1` as `config` before any pipeline import, holds the area constants,
and `configure()` re-points the Step 4 modules (which are never modified).
Two traps it exists to close, both verified by its self-test:

- `swap_v2.MANIFEST` is import-bound as `BK / "backup_manifest.json"`
  (swap_v2.py:44). Overriding only `BK` would leave it addressing
  `D:\Julian_CNMFe\BLA\...\_v1_backup\backup_manifest.json` — the sole index of
  BLA's still-armed rollback, which `do_backup()` writes (:106). `configure()`
  sets it explicitly, and `assert_no_bla()` walks every module global and refuses
  if any `Path` still mentions BLA (self-test: guard fires on an injected BLA
  path, clears when removed).
- `backfill_v2` does `from parity_check import EXT, PIN, DATA_ROOT`, binding the
  *values*; re-pointing `parity_check` alone would not move them, so `configure()`
  sets the names on both modules.

`bootstrap_footprints()` reshapes **order='C'** (bootstrap_preagent.py:274/470),
the opposite of `parity_check.footprints_from_sparse`'s order='F' for
review_neuron extractions. Self-test on 4CS/05172022-100: stored `area` and
`circularity` reproduce at reldiff 0, `max_weight` at 3.2e-08 (the npz stores
footprints as float32). `load_cn_any()` adds an eval-only h5py fallback;
`features.load_cn` is untouched.

### `repin_vca1.py` (first run creates the pin)
- pool **134 labeled sessions (23 agent / 111 bootstrap)**; 030426 asserted absent.
- provenance gate: **23 need extraction, 0 failures** (every labeled session has
  `review_neuron.mat` older than `labels.mat`, and review set == labels length).
- pending: **29 sessions / 3,415 rows, 0 missing a candidate file**.
- bootstrap Cn inventory: **71 same-resolution** (59 v5 + 12 v7.3), 26 wrong-res
  (256x256 vs 512x512 candidates), 14 missing. So arm b1 gets real
  `ring_contrast` on 71 and zero on 40; arm b0 gets zero on all 111.

### Extraction + verification
`extract_vca1.m` (port of `extract_step2.m`, base/outdir/list changed, resumable,
never writes into session dirs) via `matlab -batch`:
**23 ok, 0 skipped, 0 failed** into `D:\Julian_CNMFe\vCA1\.feature_expansion\`.
N ranges 14–138, T 4,788–17,084, all dims 512x512.

`check_extract_vca1.m` (port of the red team's `c1_matlab_full79.m`) reopens each
`review_neuron.mat` and diffs it against the extraction — the one check Python
cannot do, since `review_neuron.mat` holds an opaque MCOS Sources2D object:
**23 pass, 0 fail; max|C_raw diff| = 0 and max|A diff| = 0 on every session.**

`repin_vca1.py --check-extract` (Python N-parity): **23/23 pass** — `C_raw` rows
== review set == labels length, `A` cols == `C_raw` rows, `A` rows == d1*d2, all
finite.

## Step 3b — the vCA1 pin (commit C2)

### Baseline (`baseline_oof_vca1.py` -> `PIN/baseline_oof.npz`, `baseline_repin_vca1.json`)
8 seeds x StratifiedGroupKFold(5), bootstrap always in train at the deployed
weighting, agent rows at the **fixed 5.0** override (not the sqrt/4.0 recipe
`threshold_sweep_v2.run_oof:104` hard-codes, which resolves to ~7x here).
Pool identity asserted against the deployed joblib: **masked rows 6,662 == the
joblib's `n_excluded_ambiguous`**.

OOF pool 1,002 rows / 208 real / 794 junk (737 reviewed junk), 16 CV sessions.

- **13-col AUC full 0.8852 +/- 0.0037**, reviewed **0.8786 +/- 0.0037**.

| T | false-AR % (mean +/- sd, max seed) | junk full % | junk reviewed % |
|---|---|---|---|
| 0.03 | 0.18 +/- 0.33 (max 0.96) | 37.2 | 34.7 |
| 0.04 | 0.36 +/- 0.47 (max 1.44) | 41.1 | 38.5 |
| **0.05 (deployed)** | **0.84 +/- 0.75 (max 2.40)** | 44.9 | 42.3 |
| 0.06 | 1.38 +/- 1.00 (max 3.37) | 47.4 | 44.8 |
| 0.07 | 1.98 +/- 0.91 (max 3.85) | 49.8 | 47.3 |
| 0.12 | 4.33 +/- 0.96 (max 6.25) | 57.5 | 55.3 |

**Finding worth flagging:** applying the Step 4 threshold rule (mean FAR <= 0.85%
AND worst-seed <= 1.0%) to *today's deployed 13-column model* selects **T = 0.03**,
not the deployed 0.05 — 0.05 passes on the mean (0.84%) but fails badly on the
worst seed (2.40%). With 208 reals one false-AR cell is 0.48%, so per-seed
variance is inherently coarse. This says the rule is satisfiable on vCA1 (the
plan's "STOP is likely" was pessimistic), but that the deployed 0.05 is already
outside the posture the rule encodes. Recorded for the threshold decision; not
acted on here.

### Hi-confidence neighbour scores (`hiconf_vca1.py` -> `PIN/hiconf_scores.npz`)
`nb_corr_max` needs neighbour scores before a 35-col model exists. Step 2's first
attempt used the deployed model's in-sample scores — the leak the red team caught.
vCA1 has no OOF pin at all, so one source per row class, **none in-sample on that
row's own labels**: 163 sessions, 54,787 rows, 13,397 hi-conf (24.4%).

| source | sessions | hi-conf | note |
|---|---|---|---|
| `oof8` | 16 | 314/1,002 (31.3%) | 8-seed grouped OOF from the pin |
| `loso` | 7 | 75/220 (34.1%) | leave-session-out (train-only agent sessions) |
| `lsobs` | 111 | 11,774/50,370 (23.4%) | leave-session-out over bootstrap, 111 fits |
| `deployed` | 29 | 1,234/3,415 (36.1%) | pending; unlabelled, so nothing to leak |

Resume is keyed on a hash of the pool manifest, so a reviewer return discards
stale partials rather than mixing regimes. Second-order exposure stated plainly
in the script: leave-session-out models see other sessions' labels — the same
bounded path Step 4 accepted for its 4 non-OOF sessions, not the in-sample leak.

### Parity (`ref_v2b_vca1.py` + `parity_vca1.py --pending-all`): **ALL PASS**
`ref_v2b_vca1.py` vendors the Step 2 reference functions verbatim (copied, not
imported — `compute_v2_features.py` loads a BLA joblib at module scope, and a
shared object would make the check vacuous).

- **Phase 1: 23/23 sessions, worst relative diff 0.00e+00.** The shipping
  `features.compute_v2b_features` and the independent reference agree bit-for-bit,
  the same result BLA got.
- **Phase 2:** order-F rel diff 5.7e-06 / 7.1e-06 (A.txt text precision) vs
  order-C 1.00 on 2 pending sessions — clean rejection of the wrong geometry.
- **Phase 3 (new, vCA1-specific).** Redesigned after noticing the planned control
  was vacuous: all 111 vCA1 bootstrap frames are 512x512, so order-F of a C-order
  vector is exactly the transpose, and every spatial feature is
  transpose-invariant. The control therefore runs on `cn_correlation`, which is
  not transpose-symmetric: **order-C error 0.0000 vs order-F 0.3157 / 0.4002**.
  Faithfulness: spatial and temporal columns recompute from the persisted npz to
  max rel **4e-08** — i.e. **the persisted bootstrap traces and footprints ARE the
  ones the labels were matched against**, which is the premise arm (b) rests on.
- **Bonus finding for arm b1:** recomputing `cn_correlation` from the *session*
  `Cn.mat` reproduces the stored column to ~5e-09 (float32 footprint round-off) on
  8/8 sampled same-res sessions. So for those 71 sessions the session Cn.mat is
  the very image the bootstrap run used — b1's `ring_contrast` is not an
  approximation there.
- Pending pre-check: all 29 load with the production loaders, so the backfill
  cannot fail mid-write.

## Step 3c — backfill of the parallel files (commit C3)

`backfill_vca1.py`. Nothing deployed reads any of these; the swap (deferred)
renames the winning arm into place.

**163 `candidate_features_v2.npz` + 222 arm files, 55,007 rows** — exactly the
totals the re-baselined plan predicted (the 12:16 ingest moved 242 rows from the
pending policy to the labeled policy, and the parked session removed 46, so the
totals were unchanged by both events).

| class | n | policy | hiconf source |
|---|---|---|---|
| labeled agent | 23 | reviewed rows real v2b (order='F' extraction), flag=1; auto-rejected zeros + flag=0 | `oof8` / `loso` |
| bootstrap arm (a) | 111 | `assemble_v2_bootstrap`: real ranks, zero v2b, flag=0 — what `bootstrap_preagent.py:402-404` writes today | n/a |
| bootstrap arm (b0) | 111 | real v2b from `bootstrap_candidates.npz` (order='C'), `Cn=None` so ring 0 everywhere, flag=1 | `lsobs` |
| bootstrap arm (b1) | 111 | as b0 but with the session Cn where same-resolution — **real ring on 71, zero on 40** | `lsobs` |
| pending | 29 | real v2b all rows via the production loaders, flag=1, `auto_rejected` VERBATIM | `deployed` |

Arm files live in `EXT/_arms/{key}__b0.npz` / `__b1.npz`, never in session dirs:
they are intermediates, and only the winning arm is ever materialized into a
session. That also keeps the deployed corpus free of orphan files.

Hard checks, all passed on every file: width 13 in / 35 out, row count unchanged,
**first 13 columns bit-identical to v1**, ranks deterministic on recompute, flag
and zero patterns, the three bootstrap arms identical in columns 0-25, b0 vs b1
differing **only** in the ring column, ring provably 0 wherever no usable Cn
exists, and `auto_rejected` re-read after writing on every pending session.
All 23 labeled sessions were additionally re-verified against the vendored
reference *from the written values* (rtol 1e-6).

`backfill_report_vca1.json` carries a sha256 per file; the swap kit and
`verify_vca1.py` compare against it.

## Step 3d — gate harness (commit C3)

`gate_vca1.py --arm {a,b0,b1} --agent-weight W` runs b13 and rankv2b_35 through
one OOF harness so deltas are paired per seed, and asserts b13 reproduces the pin
(the arm only touches bootstrap columns 26-34, which b13 never sees). Reports the
reference eval, false-AR at matched junk, the threshold table with the Step 4
rule, per-prep pnb88 vs pnb97, and a 2-animal LOAO (informational — 2 animals is
not a gate). LOAO calls `clf.set_params(random_state=seed)` because the factory
hard-codes 42 and the seeds would otherwise be identical fits.

`decide_vca1.py` holds the pre-registered rules and their constants, fixed before
any gate ran. **Self-test passes: the threshold rule reproduces Step 4's published
choice of 0.06 from `step5_results.json`.** Decision order is arm -> adopt-v2 on
the winning arm -> weight -> threshold, and an arm-(b) win sets
`production_followon_required` (bootstrap_preagent zero-fills unconditionally,
and the leave-session-out hiconf has no production analogue).

## Step 3d results — STOP AND REPORT

All three arms ran at weight 5.0, 8 seeds. In every arm **b13 reproduced the pin
exactly (max|score diff| 0.00e+00)**, so the arms are strictly comparable.

| arm | bootstrap rows flagged | reviewed AUC b13 -> v35 | paired delta | rule T | junk @ rule T |
|---|---|---|---|---|---|
| a (zero-fill) | 0 / 50,370 | 0.8786 -> 0.8965 | **+0.0179** (min +0.0120, 8/8) | 0.03 | 45.3% |
| b0 (real v2b, ring 0) | 50,370 | 0.8786 -> 0.9013 | **+0.0227** (min +0.0177, 8/8) | 0.05 | 51.6% |
| b1 (real v2b + ring on 71) | 50,370 | 0.8786 -> 0.9018 | **+0.0232** (min +0.0191, 8/8) | 0.05 | 52.0% |

Per-prep and LOAO improve in every arm and on both animals (arm b0: pnb88
+0.0199, pnb97 +0.0258; LOAO pnb88 +0.0264, pnb97 +0.0408).

### The pre-registered rule selects arm (a) — and that is the wrong answer

`decide_vca1.py` applied literally: **arm a, weight 5.0, T=0.03, adopt=True,
deploy_ok=True**. But three things make that outcome untrustworthy, and none of
them is a reason to retune the rule after the fact:

1. **b0 misses the bar by 0.0003.** b0 vs a is +0.0047 with se 0.0009 (a ~5-sigma
   effect) and **8/8 seeds positive**; the bar is `max(0.005, 2*se) = 0.005`, the
   absolute floor. It fails only that floor.
2. **The rule contradicts itself.** b1 vs a is +0.0052, which *clears* the same
   bar (7/8 seeds). So "b1 beats a" and "b0 does not beat a" are both true, while
   b1 vs b0 is +0.0005 (5/8, nothing). The winner therefore depends on the
   comparison ORDER, not on the evidence — an artifact of the sequential
   b0-vs-a-then-b1-vs-b0 structure I wrote into the plan.
3. **The decisive metric was broken, and it is mine.** `gate_vca1.matched_junk_far`
   computed on the seed-MEAN OOF vector, which smooths away individual seed dips:
   b13's false-AR at T=0.05 evaluated to 0.00% on the mean vector against an
   honest per-seed mean of 0.84%. Both sides printed 0.00%, so the ship criterion's
   "false-AR not worse" check was **vacuous**. Fixed to compute per seed;
   `operating_point_vca1.py` re-derives the numbers from the stored OOF fixtures.

### The honest operating point (per seed, vs the deployed 13-col model at T=0.05)

| arm | false-AR at matched junk | junk at matched false-AR | safety gain | **yield gain** |
|---|---|---|---|---|
| b13 (today) | 0.84% | 44.9% | — | — |
| **a** | 0.60% | **43.7%** | +0.24pp | **-1.2pp** |
| **b0** | 0.36% | **52.1%** | +0.48pp | **+7.3pp** |
| **b1** | 0.30% | **53.4%** | +0.54pp | **+8.5pp** |

**Arm (a) — the rule's winner — catches 1.2 points LESS junk than the model
already in production, at matched false-AR.** Its +0.018 AUC does not reach the
operating region that matters. Arms b0/b1 move the operating point substantially
(+7.3 / +8.5 points of junk auto-caught at today's false-AR).

So the evidence says: **ship arm b0, or ship nothing.** Shipping arm (a) would
spend a freeze, a swap and a rollback window to make production slightly worse
where it counts.

b1 adds nothing over b0 (+0.0005, 5/8 seeds), so **the Cn-regeneration follow-on
for the 40 sessions is NOT warranted** — that question is now answered.

Arm b0 requires the production follow-on before any deploy:
`bootstrap_preagent.py:402-404` zero-fills unconditionally, so a future vCA1
bootstrap run would write arm-(a) rows into an arm-(b) corpus, and the
leave-session-out hiconf used in the gate has no production analogue.

**Deploy was already deferred (D1); this decision is now referred to the user.**

## Step 3e — swap kit built and rehearsed (commit C4)

`swap_vca1.py` implements the WRITE paths (`materialize`/`backup`/`swap`/
`rollback`) locally against `vca1_common`'s constants instead of calling
`swap_v2`'s functions with patched module globals. `swap_v2.MANIFEST` is
import-bound to the BLA backup dir and `do_backup()` writes it, and BLA's
rollback manifest is live — `configure()` re-points it correctly, but for the
functions that actually write, a local implementation removes the failure mode
rather than guarding it. Only `sha256` is reused.

- `record_preswap_vca1.py`: deployed 13-col scores for **163 sessions / 55,007
  rows** -> `preswap_scores.npz`.
- `swap_vca1.py backup`: **163 v1 npz + the joblib** into `_v1_backup/`,
  sha-verified, plus a local `classifier_v1_2026-08-24.joblib`.
- `swap_vca1.py rehearse`: **PASS** — scores recomputed from the BACKUP BYTES
  with the BACKED-UP joblib reproduce the preswap fixture **exactly** on the 3
  widest sessions (N = 1387 / 1003 / 930). The rollback path is proven.
- `swap_vca1.py preflight`: correctly **NOT READY**, and for the right reason —
  the only failing check is the absent bootstrap red-team report, which is the
  deferred gate. Everything else passes: session set matches the manifest, no
  live npz drifted, joblib unchanged, every session has a v2 sibling.

### Two self-inflicted bugs found and fixed in the preflight's watcher check
Worth recording because both would have mattered on deploy day:
1. `wmic` no longer ships on Windows 11 (raises `FileNotFoundError`), so the
   original check silently fell through to a fallback that tested for **any**
   `python.exe` — and the preflight *is* a python.exe, so it reported the watcher
   running forever and would have blocked every legitimate swap.
2. Fixing that, I dropped the process-name filter, so the query matched **all**
   processes — including the shell invoking it, whose command line can contain
   the literal string `watcher_vCA1` (a grep, this script's own source). Same
   false positive by a different route.

Now filtered to python processes, excluding the current PID, and verified in both
directions: `False` with nothing running, `True` against a decoy process whose
command line carries the name, `False` again once it exits. `running` is `None`
when it cannot be determined, and the caller must treat that as unknown, never as
stopped.

### Rollback guard adopted from the BLA retirement (2026-08-26)
While this project ran, the operator retired BLA's Step 4 rollback (commit
d4b3185): its 08-20 backup stopped being a restore point once the
bootstrap-matching fix rewrote 91 sessions' `labels.mat`, so putting the 13-col
features back would have paired them with labels they no longer match.
`swap_v2.do_rollback` now refuses whenever any `labels.mat` postdates the backup.

The same hazard applies here, so `swap_vca1.do_rollback` carries the same guard.
Checked today: backup manifest 2026-08-26 16:41, **0 sessions with newer labels
-> the vCA1 rollback is currently VALID**. It will refuse itself the moment a
reviewer return lands, which is the correct behaviour: at that point the backup
is historical and the pool must be re-backed-up.
