# Red-team report: the bootstrap pixel-order fix and the BLA / vCA1 / global-model decisions

_2026-08-26, fresh-session adversarial review per `docs/BOOTSTRAP_REDTEAM_BRIEF.md`.
Every number below was computed this session by an independent evaluator
(`rt_lib.py`: own discovery, labels, weights, CV, metrics, loaders, matchers; it imports
only the feature MATH from `agent/features.py` and never the harnesses it checks) unless
marked "claimed". Corpus pinned first (`pin_manifest.json`, hash `cf761cafdcc0…`); every
attack refuses to run on drift and stamps the hash into its `results/aNN.json`. Each attack
was then re-checked by a refuter (`refute_all.py`: clean-process reproduction of the JSON,
pin check, and a flipped-control or alternative computation) — `results/aNN_refute.json`._

<!-- SYNTH:HEADER -->
Pin `cf761cafdcc0` -- BLA 170 labeled (79 agent / 91 bootstrap), vCA1 134 (23 / 111), DG_AL 9; joblib md5 BLA be42080f, vCA1 00768667.
<!-- /SYNTH:HEADER -->

## Verdict summary

**The bug is real, the fix is complete, the labels are right, and the ranking decisions
survive every attack. Two things do not survive: the thresholds' worst-seed guarantee under
animal-level grouping (#8), and the seed-robustness of the "5.0 vs 7.01 doubles false-AR"
claim behind the vCA1 weight (#9). One animal (bla21) got slightly worse (#11). The §D
question — why were the gains so small — has an evidence-backed answer: the old positives
were mostly cell-like blobs (D13), and with clean labels the bootstrap corpus supplies
score *calibration*, not ranking information, on the agent test distribution (D15).**

### Verdict table

<!-- SYNTH:VERDICT_TABLE -->
| # | attack | verdict | refuter | one-line evidence |
|---|---|---|---|---|
| 1 | Orientation re-derived without bmlib | **PASS** | agrees | mixed metric reproduces stored March sims on 4/4 sandboxes (bla21 37/50, max|diff| 5.0e-05; matches 6.8 px from the mirror vs 130 px from the truth); consistent metric 50/50 at 0.98 px; live labels consistent 202/202 |
| 2 | An 11th orientation site | **PASS** | agrees | no live production defect beyond the 2 known; `features.load_spatial` A.txt fallback has 2 latent defects (transpose + square-only) firing on 0 sessions |
| 3 | Agent labels unaffected | **PASS** | agrees | 102/102 sessions reproduce labels.mat exactly from footprints (both rules); transposed control agreement 0.77 and 6 vs 26 positives |
| 4 | Retro cn_correlation transpose | **PASS** | agrees | 6 BLA sessions (rows 705, reals 217) carry a transposed cn_correlation (Spearman vs corrected ~0); fixing it: deployed 35-col +0.0007 pool / +0.0096 on those sessions; b13 +0.0024, b13 rule T 0.05->0.05; sizing MATERIAL |
| 5 | Matched pairs are the right cells | **PASS** | agrees | dist p50 0.69/0.75 px, p95 4.4/4.4; IoU20 p05 0.54/0.62; flagged 1+0 of 7885; residual transposes 0; human sheets pending |
| 6 | Masked duplicates | **INCONCLUSIVE** | agrees | same-cell 27%/30%, distinct 7.8%/9.9%, other 63%/59%; trace r with the matched candidate p50 0.39/0.40 (>=0.7 only 1%/2%) |
| 7 | Unrecovered neurons + section-4 table | **PASS** | agrees | table reproduces exactly; 39+4 unrecovered: vCA1 {'detection_miss': 5, 'merge_split': 25, 'partner_taken': 9}, BLA {'detection_miss': 0, 'merge_split': 0, 'partner_taken': 4}; all partners in the ambiguous set |
| 8 | Animal / FOV leakage | **FAIL** | agrees | BLA v2b AUC 0.9283 session -> 0.9158 animal; FAR@0.04 0.64% -> 0.92% (max 1.09); vCA1 13-col FAR@0.05 0.84% -> 4.45% LOAO (max 5.29); arm b0 5.41%; rankings (v2 > b13) hold under both |
| 9 | 3-seed decisions at 8 seeds | **FAIL** | agrees | 3-seed repro w=5 AUC 0.8843 / w=7.01 0.8861; at 8 seeds FAR@0.05(7.01)-FAR(5.0) = +0.30 pp (4/8 seeds, se 0.24), AUC -0.0007; dups: label1 -0.0067 (BLA) / -0.0020 (vCA1) |
| 10 | Threshold rule self-consistency | **PASS** | agrees | rule gives 0.06 pre-fix (table + vectors) and 0.04 post-fix; at 0.045 FAR 0.80% max 1.29%; animal-grouped BLA -> 0.03; vCA1 13-col -> 0.035, arm b0 -> 0.05, LOAO -> None |
| 11 | BLA G5 per-animal / early-era | **FAIL** | agrees | paired per-animal deltas (pre->post fix) positive for 5/6 animals; negative cell [['animal', 'bla21', -0.003760991023997129]]; early era +0.0018 (8/8) vs late +0.0052 |
| 12 | Cond A/B vs LOO sign flip | **PASS** | agrees | both protocols reproduced (eval 0.873->0.877; LOO 0.890->0.888); fixed 5.0 not worse than sqrt on FAR in 8/8 protocol cells |
| 13 | D13 old positives were cells | **PASS** | agrees | simulated old positives 2579+1639: 76%/64% score >= T out-of-sample (unmasked negatives 49%/41%; true positives 99%/99%); 26%/19% were the right cell |
| 14 | D14 BLA bootstrap v2b degenerate | **PASS** | agrees | real v2b on BLA bootstrap rows: reviewed +0.0011 (8/8; bar 0.0050); junk at matched FAR 43.2% -> 45.3%; fixture reproduced: True |
| 15 | D15 bootstrap redundant (learning curve) | **PASS** | agrees | AUC flat 0->100% bootstrap (BLA 0.9281->0.9284; vCA1 0.8846->0.8848) but FAR at fixed T 1.34->0.66% / 8.35->0.96% |
| 16 | D16 bootstrap positives easy | **PASS** | agrees | hard band [0.02,0.10): bootstrap pos 0.028 vs agent pos 0.018 (BLA); 0.018 vs 0.034 (vCA1) |
| 17 | D17 reviewer ceiling | **INCONCLUSIVE** | agrees | no different-reviewer overlap on the server (0 candidates); proxies: reals < 0.02 = 0.0021 (BLA), 0.0000 (vCA1) |
| 18 | Global model, animal-grouped | **PASS** | agrees | session grouping reproduces the log; DG_AL +0.3 pooled: +0.0204 session, +0.0794 LOAO (8/8), +0.0097 FOV (8/8) |
| 19 | Pooled DG calibration | **PASS** | agrees | junk@0.05 own 33.4% vs pooled 6.8%; junk at matched FAR 33.4% -> 39.4%; leak-free isotonic at FAR<=1%: own 15.9% vs pooled 13.2%; not-usable-at-fixed-T |
<!-- /SYNTH:VERDICT_TABLE -->

Coverage: #1 (all 202 sessions + 4 sandboxes), #3 (all 102 agent sessions), #4 (all 102),
#5/#6/#7 (all 7,885 pairs, 11,113 duplicates, 43 unrecovered neurons), #8–#16 (the full
pinned pools, 8 seeds), #2 (every reshape/flatten/h5py site in `agent/*.py`). Sampled:
the human contact sheets for #5/#6 (30 sessions × 5 pairs; 30 duplicates; all 43
unrecovered), which are **pending the user's eyes** in `contact_sheets/`. #17 is
INCONCLUSIVE because the data to answer it (a two-reviewer overlap) does not exist.

## Details

### A. Is the bug real, is the fix complete?

**#1 Orientation re-derived without bmlib — PASS.** (i) On one bootstrap session per area
plus a v7.3 stack, `spatial_footprints.mat` flattened column-major equals `A.txt` to text
precision (rel. diff 6.1e-06 / 7.3e-06 / 5.1e-06) and the C-order flatten does not (0.95–1.00).
(ii) On the four real `_bootstrap_validate` sandboxes (the fifth, `2tones/100125-bla8-735um`,
is an empty directory) the MIXED-order cosine (candidates C-order, curated F-order) →
Hungarian reproduces the stored March pair similarities to 4-decimal rounding on 4/4
(max|diff| 4.98e-05 / 4.72e-05 / 4.82e-05 / 4.98e-05; matched 37/50, 19/45, 25/63, 34/66 = the
stored counts; bla21 top-5 0.9439 / 0.9241 / 0.8807 / 0.8646 / 0.8025). Under that metric the
matched candidates sit 6.8 / 7.4 / 4.7 / 5.7 px from the *mirror* of the curated centroid and
130 / 216 / 69 / 136 px from the true one. The consistent metric recovers 50/50, 45/45,
63/63, 66/66 at median similarity 0.971 / 0.985 / 0.985 / 0.982 and median true-position
distance 0.98 / 0.94 / 0.54 / 0.70 px. (iii) All 202 live sessions: the recomputed fixed-metric
cosine equals the stored `sim_matrix` (atol 1e-5), the Hungarian pairs equal the JSON pairs,
`labels.mat` positives equal `candidate_indices[:n_matched]`, and N agrees across npz / JSON /
candidate npz / labels — 111/111 vCA1, 91/91 BLA. (iv) **INCONCLUSIVE-by-loss, stated not
scored:** the pre-fix corpus signature ("median best pair 0.852, never 0.99") cannot be
reproduced because the pre-fix JSONs and candidate caches were overwritten by the corpus
re-run and never versioned; only per-session counts survive (`a3_damage.json`,
`b2_gate_b.json`). Its post-fix mirror image is measured: median best pair 0.9981 (vCA1) /
0.9992 (BLA), with 106/111 and 91/91 sessions reaching ≥ 0.99.
Refuter: agrees (clean re-run identical; the CONSISTENT metric does *not* reproduce the stored
sims; A.txt as the candidate source gives the same mixed counts; the transposed metric does
not match the stored `sim_matrix` on 5 random sessions).

**#2 An 11th site — PASS.** Twelve sites audited beyond AUDIT.md (`results/a02.json`, each
with the quoted source line): the `.feature_expansion` extraction A is proven F-order (S1,
via #4's bit-exact reproduction of the stored `cn_correlation` on 96/102 sessions);
`features.load_spatial`'s A.txt fallback has **two** latent defects (transpose *and* a
square-frame assumption, `side = sqrt(n_pixels)`) but fires on 0 of 369 sessions scanned
(S2); `bootstrap_preagent`'s >2 GB A.txt fallback reindexes F→C and equals the primary path
on a synthetic asymmetric footprint (S3); `validate_threshold` reindexes `A_final` at the
call site (S4); `decision_margins.py:83` / `sweep_gsig.py:99` flatten C-order and use the
result column-wise or against itself — the "matching neuron.A" comment is misleading but
nothing compares it to `neuron.A` (S5); the curator PDF / one-class fallback go through the
primary loader (S6); `Coor.mat` has no Python consumer (S7); all three h5py readers transpose
(S8); the two MATLAB writers build `[n, row, col]` stacks (S9); `motion_qc` /
`extract_cand_traces.m` are MATLAB-side only (S10); the v2b neighbour distance works in image
coordinates on both sides (S11); `run_cnmfe.py:61` is a MATLAB statement inside a string
(S12). No live production-path defect beyond the two already known. Refuter: agrees after
one round (its independent census found S12; added to the audit).

**#3 Agent labels unaffected — PASS.** Labels recomputed from footprints alone (review side =
the `.feature_expansion` extraction A, F-order; final side = `spatial_footprints.mat`, the same
`neuron.A` the labels were matched against, `CNMFe_final_save.m:628-655`) reproduce
`labels.mat` **exactly on 102/102 sessions** under both rules (greedy per-final at 0.60 and
Hungarian at 0.60). The transposed control (review side reshaped C-order) drops agreement to
0.77 and finds 6.0 positives per session instead of 25.8. Refuter: agrees (10 random
sessions stay exact at 0.60 with `A.txt` as the final side; the alternative thresholds 0.50 /
0.70 degrade only mildly). The MATLAB load-only link (`matlab/m03_loadonly_check.m`) was
written but not executed this session — the extraction was already proven bit-identical
to `review_neuron.mat` by the prior projects (79/79, 23/23), and the final-side stack was
proven equal to `A.txt` in #1.

**#4 Retro `cn_correlation` transpose — PASS (bug confirmed; follow-up sized MATERIAL).**
`prep_retro_ids.py` classified every agent session by recomputing column 12 from the
extraction A under both reshapes: 96 sessions match the order-F recomputation and **6 BLA
sessions match the order-C recomputation bit-for-bit** (|stored − C| = 0.0):
`2tones/093025-bla12-660um`, `2tones/093025-bla8-765um`, `4odorDO/02092026-bla12-681um`,
`4odorDO/02092026-bla16-314um`, `4odorDO/02092026-bla8-745um`, `Valence/121225-bla12-652um`
— 705 rows / 217 reals, all six in the CV pool; 0 in vCA1. On those sessions the stored
column is uncorrelated with the truth (Spearman −0.06…+0.11; 88–100% of rows move by >0.1;
positives' median `cn_correlation` ≈ 0.00 instead of 0.71–0.74). Correcting column 12 and
recomputing the 13 rank columns (8 seeds, unrestricted threads): deployed 35-col +0.0007 pool
AUC (7/8 seeds) and **+0.0096 on the six sessions** (their AUC 0.9103 → 0.9199); b13 +0.0024
(8/8) and +0.0076. Junk at matched false-AR 43.2% → 41.9% (35-col) / 35.0% → 35.7% (b13).
The corrected 35-col's rule pick moves 0.04 → 0.03 because its worst seed at 0.04 becomes
1.09% — one more instance of a rule pick sitting on the 1% ceiling (see #8/#10). Sized
MATERIAL for the six sessions, cosmetic for the pool. Refuter: agrees (session `Cn.mat` ==
extraction Cn; retro set == prep's; clean re-run identical).

### B. Are the new labels correct?

**#5 Matched pairs are the right cells — PASS (human check pending).** All 5,148 + 2,737
pairs: centroid distance p50 0.69 / 0.75 px, p95 4.42 / 4.38 px; IoU at 20% p05 0.54 / 0.62;
mirror-position distance p05 15.7 / 14.3 px. Flagged: 1 pair (IoU20 < 0.2) of 7,885; 0 pairs
farther than gSiz; **0 residual transposes** (mirror closer than true AND true > gSig AND
mirror < half the true distance). 19 pairs have a mirror distance marginally below a
sub-4-px true distance — cells on the image diagonal, where the mirror *is* the true
position; they are counted, not flagged. Contact sheets: `contact_sheets/a05_*.png` (15
sessions × 5 pairs per area, lowest-similarity sessions and pairs over-sampled). Refuter:
agrees (tighter alternatives: pairs beyond gSig or IoU50 < 0.2 are also rare).

**#6 Masked duplicates — INCONCLUSIVE on composition; the decision stands (#9).** The 6,623 + 4,490
masked rows are *not* mostly same-cell re-detections in the temporal sense: trace correlation
with the matched candidate p50 0.39 / 0.40, ≥ 0.7 on only 1% / 2%; IoU50 with it p50 0.30 /
0.31; offset p50 5.3 / 6.7 px. The class split depends on the thresholds: strict (IoU50 ≥ 0.5
or r ≥ 0.7 = same cell; offset > gSig and r < 0.3 = distinct) gives same-cell 27% / 30%,
distinct 7.8% / 9.9%, other 63% / 59%; a looser reading (IoU50 ≥ 0.3 or r ≥ 0.5; offset >
0.5·gSig) gives same-cell 62% / 63% and distinct 36% / 36% — so the geometry cannot settle
how many are second neurons. What is robust: they are spatially overlapping (cosine > 0.45)
but mostly temporally distinct components; "same-cell re-detection" is a minority reading
under any threshold; every duplicate row is label 0 / weight 0 in the trainer's view; and the
masking *decision* rests on model evidence (#9: label-0 vs masked +0.0000 / +0.0003, label-1
−0.0067 / −0.0020 and −11.6 pp junk at matched false-AR on BLA), not on this composition.
Contact sheets: `contact_sheets/a06_*_duplicates.png`. Refuter: agrees.

**#7 Unrecovered neurons + the section-4 table — PASS.** The corpus table now has a backing
artifact (`results/corpus_table.json`): vCA1 5,148 / 5,187 (99.25%), 39 ambiguous + 6,623
duplicates; BLA 2,737 / 2,741 (99.85%), 4 + 4,490; 0 sessions below 0.40 recovery in either
area; `recovery_by_threshold` corpus totals included. The 43 unrecovered neurons (39 + 4) are
classified: vCA1 25 merge/split (a candidate within gSiz but similarity ≤ 0.45), 9 partner
taken, 5 detection misses; BLA 4 partner taken. **All 43 have their Hungarian partner in the
ambiguous set**, so none is a full-weight negative. Unrecovered vCA1 neurons are dimmer than
recovered ones (curated max weight p50 23.0 vs 27.5). Sheets: `contact_sheets/a07_*.png`.
Refuter: agrees (independent recount; the JSON tail == the ambiguous set on every session).

### C. Are the gates honest?

**#8 Animal / FOV leakage — FAIL on the threshold ceilings, PASS on the rankings.**
Leak channels measured (`results/animal_map.json`): BLA bootstrap shares animals with the
CV pool (bla8 ×3, bla16 ×2, bla12 ×1 sessions); vCA1's 15 bootstrap animals are disjoint from
pnb88 / pnb97, so only the agent-agent channel exists there; DG_AL has 2 animals / 4 FOVs.
Re-run with animal grouping (BLA: 5 folds over 6 animals; vCA1: 2-animal LOAO) and the test
animals' bootstrap sessions dropped from training, 8 seeds, unrestricted threads:

#### Deploy-verdict inputs (8 seeds, unrestricted threads)

<!-- SYNTH:DEPLOY_TABLE -->
| model | grouping | AUC full | AUC reviewed | FAR@T mean (max) | junk@T | rule T |
|---|---|---|---|---|---|---|
| BLA 35-col @0.04 | session | 0.9283 | 0.9149 | 0.64% (0.96) | 43.2% | 0.04 |
| BLA 35-col @0.04 | animal | 0.9158 | 0.9003 | 0.92% (1.09) | 42.4% | 0.03 |
| vCA1 13-col @0.05 | session | 0.8852 | 0.8786 | 0.84% (2.40) | 44.9% | 0.03 |
| vCA1 13-col @0.05 | animal | 0.8625 | 0.8544 | 4.45% (5.29) | 47.5% | None |
| vCA1 arm b0 @0.05 | session | 0.9081 | 0.9013 | 0.66% (0.96) | 51.6% | 0.05 |
| vCA1 arm b0 @0.05 | animal | 0.8679 | 0.8589 | 5.41% (6.73) | 51.1% | None |
<!-- /SYNTH:DEPLOY_TABLE -->

The ranking decisions hold under animal grouping (BLA v2b > b13: 0.9158 vs 0.9040; vCA1 arm
b0 > b13: 0.8679 vs 0.8625). The absolute AUC deflation is 0.0125 (BLA), 0.0227 (vCA1 13-col),
0.0402 (vCA1 b0). **The thresholds do not carry:** BLA FAR@0.04 rises to 0.92% mean / 1.09%
worst seed (the rule would pick 0.03), and vCA1's deployed 0.05 runs at **4.45% (max 5.29%)**
when the other animal is the only agent data — the same for arm b0 (5.41%, max 6.73%). Read
this as "the deployed thresholds are calibrated to animals already in the training set; a
brand-new animal at 0.05 starts near 4–5% false-AR" — which is exactly why new preps are
curated at threshold 0 first. The bootstrap contribution is ≈ 0 for BLA under both groupings
(−0.0001 / −0.0006) and large for vCA1 only under LOAO (+0.11), i.e. bootstrap matters when
agent data for the animal is absent. Refuter: agrees (explicit 6-fold LOAO reproduces the BLA
ceiling breach).

**#9 The 3-seed decisions at 8 seeds — FAIL on the pre-registered test.** The logged 3-seed
sweep reproduces on the as-of-08-24 pool (22 agent incl. the parked session, 14 CV folds; w=5
AUC 0.8843 vs logged 0.8857, w=7.01 0.8861 vs 0.8843) — but its false-AR numbers do not: they
came out 2.0% / 2.2% here vs the logged 1.2% / 2.5%, and 1.4% / 1.8% in a 16-thread run, i.e.
**the 3-seed FAR figures move by ±0.5 pp with xgboost's thread count alone** (a couple of
cells out of ~170 reals). At 8 seeds on the pinned pool: FAR@0.05(7.01) − FAR@0.05(5.0) =
+0.30 pp with 4/8 seeds in that direction (se 0.24) under the sweep script's protocol; AUC
−0.0007. The harness protocol (scaler, weight-0 rows kept) shows the same direction with a
larger gap (0.84% vs 1.32%), and #12 shows the direction is protocol-invariant. The fair
comparison — junk caught at MATCHED false-AR vs w=5.0 — is 39.8% (w=1), 43.7% (w=2), 43.4%
(w=3.5), 44.2% (w=7.01) against 44.9–46.8% for w=5.0: **5.0 is the best or tied-best
operating point**, so the *choice* stands; the *claim* ("7.01 doubles false-AR") does not.
Duplicate handling at 8 seeds, both areas: label-0 vs masked +0.0000 (BLA) / +0.0003 (vCA1),
within noise; label-1 −0.0067 (BLA, 0/8) / −0.0020 (vCA1, 1/8); junk at matched FAR: label-1
costs 11.6 pp on BLA (43.2% → 31.6%) and is neutral on vCA1 (44.9% → 45.6%). Refuter: agrees.

**#10 The threshold rule — PASS.** An independent implementation of the Step-5 rule gives
0.06 on the pinned pre-fix table *and* on its raw OOF vectors, and 0.04 on the post-fix
vectors; a second implementation in the refuter agrees. On the 0.005 grid: **0.045 fails the
worst-seed ceiling (FAR 0.80% mean, 1.29% max)** — the undocumented sentence in the write-up
is now a computed fact; b13 alone would pick 0.055. Under animal grouping BLA's rule pick is
0.03. vCA1: the deployed 13-col model's rule pick is 0.035 (fine grid; 0.03 on the coarse
one), not the deployed 0.05 (0.84% mean but 2.40% worst seed); arm b0's pick is 0.05 (0.60% /
0.96%); under LOAO no T in [0.03, 0.10] satisfies the rule for either. The reviewed-stratum
gate (junk_reviewed ≥ 30%) does not change any pick.

**#11 BLA G5 per-animal / early-era — FAIL on one cell.** The pre-fix `step5_oof.npz` and the
post-fix `c3_bla_gate8_oof.npz` share rows, groups and seeds, so the corpus fix can be judged
per animal and era without refits (rankv2b_35): bla12 +0.0015 (7/8), bla16 +0.0076 (8/8),
bla36 +0.0060 (8/8), bla37 +0.0034 (8/8), bla8 +0.0028 (8/8), **bla21 −0.0038 (min −0.0092,
1/8; 4 sessions / 106 reals)**; era a (recorded 2025) +0.0018 (8/8) vs 2026 +0.0052 (8/8);
era b (labels before 2026-04-15) +0.0068 (8/8) vs +0.0034. The early era improved, but less
than the late era — the "2022–25 labels are where the corpus should show its value" intuition
is not borne out on the agent test folds. Leave-one-animal-out refits on the live files (8
xgb seeds): rankv2b_35 beats b13 for every held-out animal (bla8 +0.021, bla37 +0.010, bla36
+0.009, bla21 +0.009, bla16 +0.008, bla12 +0.003) and for the held-out early era (+0.007);
dropping the held-out animal's own bootstrap changes AUC by ≤ 0.002. bla21 is the hardest
animal (LOAO 0.853) and the retro-era one; its negative paired delta is small but beyond seed
noise. Refuter: agrees.

**#12 Cond A/B vs LOO sign flip — PASS (attributed).** Both claimed numbers reproduce under
their own protocols on the 08-24 pool (`--eval`: 0.873 → 0.877 vs claimed 0.869 → 0.878; LOO:
0.890 → 0.888 vs claimed 0.888 → 0.877). The factorial (protocol × aggregation × weight ×
scaler × 8 seeds) shows the sign is set by **aggregation**: the mean of per-unit AUCs is
negative for the bootstrap condition under LOO (−0.004 sqrt, −0.007 fixed-5.0, −0.022
uniform) and ≈ 0 under 5-fold, while the POOLED AUC is positive (+0.002…+0.004) in every
sqrt/fixed cell — small per-session test sets (5–20 positives) make the per-unit mean a
noisy, pessimistic estimator. The deploy-relevant comparison holds in all 8 cells: fixed 5.0
runs at 0.74–0.96% FAR@0.05 vs sqrt's 1.44–2.13%, with AUC within 0.002. Refuter: agrees.

### D. Why were the gains so small?

**#13 D13 — SUPPORTED.** Re-applying the transposed metric to today's candidates yields
2,579 (vCA1) + 1,639 (BLA) "would-be old positives". Of these 26% / 19% *are* the true cell
(near-diagonal or symmetric footprints), 15% / 17% are masked duplicates, 60% / 63% are plain
negatives — yet **76% / 64% score ≥ the deployed T out-of-sample** (unmasked negatives: 49% /
41%; true positives: 99% / 99%), with medians 0.30 / 0.13. The literal check on the four
sandboxes agrees (old positives score ≥ T 73% / 68% / 52% / 71%; human-transferred labels
11 kept / 4 deleted / 22 unknown on bla21). So the old positive class was mostly cell-shaped
material, and the fix's main effect was on the ~7,900 true cells that sat at label 0 — which
predicts the observed pattern (AUC +0.004, junk at matched FAR +9.4 pp). A correction to the
damage model: a3 predicted ~94% wrong cells; the simulated rate is 74–81% wrong (the a2
sandboxes: 73%, 90%, 68%, 71%). Direction unchanged, magnitude overstated.

**#14 D14 — NOT supported.** Real v2b for the 91 BLA bootstrap sessions from
`bootstrap_candidates.npz` (hiconf from leave-session-out 13-col scores, or from the deployed
companion model; agreement 98.6%), 8-seed gate paired against the fixture (arm a reproduces
it to 0.00e+00): reviewed AUC +0.0011 (b0_lso, 8/8, min +0.0004) / +0.0012 (b0prod) / +0.0014
(b1 with real ring on 72 same-res sessions) — all far below the +0.005 bar; junk at matched
false-AR 43.2% → 45.3–46.3%; the bla21 smoke cell N25 rises from 0.046 to 0.093 (safer). A
small, real, low-priority lever — not vCA1's +0.023.

**#15 D15 — SUPPORTED, with a twist.** Learning curve over bootstrap sessions (0 / 25 / 50 /
100%, fixed weight, 8 seeds): reviewed AUC is flat — BLA 0.9140 (0%) → 0.9149 (25%) → 0.9148
(50%) → 0.9150 (100%); vCA1 0.8788 → 0.8800 → 0.8797 → 0.8781 — and junk at matched false-AR is
flat (43.7–45.2% / 43.8–48.3%). But false-AR at the *fixed* deployed T moves enormously: BLA
1.34% → 0.66%, **vCA1 8.35% → 0.96%**. With clean labels the bootstrap corpus contributes
score-scale calibration, not ranking information, on the agent test distribution — AUC
cannot see it, and the operating point depends on it completely.

**#16 D16 — NOT supported.** Out-of-sample bootstrap positives are not easier than agent
positives: hard band [0.02, 0.10) 2.8% vs 1.8% (BLA, 35-col; medians 0.890 vs 0.889); 1.8% vs
3.4% (vCA1; medians 0.914 vs 0.834) — the latter just short of the "half" bar, and driven by
the pnb prep's harder agent cells rather than by bootstrap being easy. Score correlates with
pair similarity (Spearman 0.53–0.65).

**#17 D17 — INCONCLUSIVE.** No two-reviewer overlap exists: all 64 returns on the server
inbox carry the same reviewer as the local provenance (the refused duplicate the guard once
caught is no longer there), and the DG AVG4x / non-averaged pairs are two CNMFe runs, not two
reviews of one candidate set. Proxies from the reproduced pins: reals scoring < 0.02
(confident-wrong) are 0.21% of BLA reals (5 cells in 4 sessions) and 0.00% of vCA1's; reals
below T 0.54% / 0.00%. The a2 self-consistency bound (207 kept / 2 deleted / 15 unknown) is
the same reviewer against a re-run, not inter-rater. A reviewer ceiling cannot be measured
until a session is deliberately reviewed twice.

### E. The global model

**#18 Global model, animal-grouped, 8 seeds — PASS (reproduced; two findings).** The
re-implementation of c5's Cond A / Cond B (shared 13 columns; other areas' rows at 1.0 and 0.3)
reproduces the logged session-grouped deltas: vCA1 −0.0013 / −0.0010 (claimed −0.0023 /
−0.0038), BLA −0.0008 / +0.0000 (claimed −0.0011 / +0.0003); DG_AL +0.0179 / +0.0204 against
the claimed +0.0110 / +0.0145 — same sign, larger, and within the seed noise of a 9-session
pool measured at 3 seeds. Under animal grouping: BLA stays null (−0.0013 / −0.0002); **the
DG_AL gain grows to +0.0781 / +0.0794 under 2-animal LOAO (min +0.0729, 8/8) and holds at
+0.0073 / +0.0097 under FOV grouping (4 FOVs, 8/8)** — it is not a 2-animal artifact, it is
what a data-starved area looks like when its only other animal is held out. And a finding
the brief did not ask for: **pooling helps a held-out vCA1 animal too (+0.0130 at w=1.0,
+0.0067 at w=0.3, 8/8)** while being null when both pnb animals are in training — the same
"new animal" regime that #8 exposed. Refuter: agrees (4 DG FOV groups recounted).

**#19 Pooled DG usability — PASS (descriptive): not usable at a fixed threshold.** The pooled
model's score scale collapses: at T = 0.05 it catches 4.0–10.1% of DG junk vs 33.4–46.3% for
the own-only model (false-AR 0.00–0.07% vs 2.5–4.2%). At matched false-AR the pooled model is
comparable or better (LOAO: 33.4% → 39.4%; session: 46.3% → 49.0% at w=0.3), so the gain is a
ranking gain. A leak-free calibration fitted inside the training folds does not recover a
usable operating point on this little data: junk caught at false-AR ≤ 1% under LOAO is 15.9%
own-only vs 13.2% isotonic / 9.7% Platt (raw pooled 18.6%), and the calibrated AUCs (0.83–0.84)
give back part of the gain. DG_AL runs `THRESHOLD_OVERRIDE = 0` today, so ranking is the
operative metric there; the pooled prior would improve the review order, not the auto-reject.
Refuter: agrees (an in-sample isotonic map scores higher than the leak-free one, as it must).


## Section-D scoreboard

<!-- SYNTH:SCOREBOARD -->
| hyp | supported? | number |
|---|---|---|
| D13 old positives were cells | {'vCA1': True, 'BLA': True} | simulated old positives 2579+1639: 76%/64% score >= T out-of-sample (unmasked negatives 49%/41%; true positives 99%/99%); 26%/19% were the right cell |
| D14 BLA bootstrap rows feature-degenerate | False | real v2b on BLA bootstrap rows: reviewed +0.0011 (8/8; bar 0.0050); junk at matched FAR 43.2% -> 45.3%; fixture reproduced: True |
| D15 bootstrap redundant with agent data (AUC) | {'BLA': True, 'vCA1': True} | AUC flat 0->100% bootstrap (BLA 0.9281->0.9284; vCA1 0.8846->0.8848) but FAR at fixed T 1.34->0.66% / 8.35->0.96% |
| D16 bootstrap positives easy | {'BLA': False, 'vCA1': False} | hard band [0.02,0.10): bootstrap pos 0.028 vs agent pos 0.018 (BLA); 0.018 vs 0.034 (vCA1) |
| D17 reviewer ceiling | INCONCLUSIVE | no different-reviewer overlap on the server (0 candidates); proxies: reals < 0.02 = 0.0021 (BLA), 0.0000 (vCA1) |
<!-- /SYNTH:SCOREBOARD -->

## Verdict on each deploy

**BLA, 35-col rankv2b_35 @ T = 0.04 (2026-08-26): STANDS.** The label fix is genuine (#1–#7),
the gate reproduces exactly (#8 session rows == the pins), the corpus improved for 5 of 6
animals and both eras (#11), and the rule's 0.04 is self-consistent (#10; 0.045 fails). Two
caveats to record: bla21 (4 sessions) is −0.004 after the fix, and under animal-level
grouping FAR@0.04 is 0.92% mean / 1.09% worst seed (rule → 0.03) — the 1% worst-case posture
holds for animals in the training set, not necessarily for a new one.

**vCA1, 13-col, agent weight 5.0 @ T = 0.05 (2026-08-24): STANDS as a ranking model; the
threshold is the weak point.** 5.0 is the best operating point among the weights tested
(junk at matched false-AR), but the recorded rationale ("7.01× doubles false-AR, 1.2 → 2.5%")
is a 3-seed, thread-count-sensitive number; at 8 seeds the gap is +0.3–0.5 pp and not
seed-robust (#9), though its direction is protocol-invariant (#12). The deployed 0.05 fails
the Step-5 rule on the deployed model itself (2.40% worst seed; rule → 0.035) — already
acknowledged as a deliberate 1.8%-posture choice — and runs at 4.45% (max 5.29%) for a
held-out animal (#8). Nothing in the fix changes this; it says the threshold protects
scarce cells for known animals and should not be assumed for new ones.

**vCA1 arm b0 (pending 35-col deploy, gate T = 0.05): the ranking gain is real and survives
LOAO (+0.023 reviewed session-grouped; 0.8679 vs 0.8625 under LOAO); the rule's 0.05 is at
the edge** (0.60% mean, 0.96% worst seed against a 1.0% ceiling) and **flips to 0.04 under a
different xgboost thread count** (0.84% / 1.44% at 16 threads). The deploy decision should
quote both and re-derive T on the day with the thread count pinned.

**Global model: null for BLA / vCA1 confirmed under animal grouping; the DG_AL gain is real
and larger than claimed under LOAO, but unusable at a fixed threshold without calibration.**
Under animal grouping BLA is null (−0.0013 / −0.0002), the DG_AL gain is +0.079 under LOAO and
+0.010 under FOV grouping (8/8), and vCA1 gains +0.007–0.013 for a held-out animal; but the
pooled score scale collapses (junk@0.05 33–46% → 4–10%) and leak-free calibration on 9 DG
sessions does not recover it (13.2% vs 15.9% junk at false-AR ≤ 1%). Read: a real ranking
prior for DG_AL (and for new vCA1 animals), not a threshold-ready model.

## Ranked recommendations (expected AUC / false-AR impact)

1. **Fix the retro `cn_correlation` transpose and refresh the 6 BLA sessions** (#4):
   +0.008–0.009 AUC on those sessions, +0.0008 (35-col) / +0.0026 (b13) on the pool; sized
   MATERIAL; code fix protects future areas (`train_classifier.py:235`, `features.py:32-38`).
2. **Report the animal-grouped false-AR beside every threshold decision** (#8/#10) and keep
   the existing practice of curating a new animal/prep at threshold 0 before any auto-reject:
   the deployed thresholds are 4–5× outside the 1% posture for a held-out vCA1 animal.
3. **Pin xgboost's thread count in every gate/threshold script** (`n_jobs`, `OMP_NUM_THREADS`)
   and record it: individual OOF scores move by up to 0.37 and rule picks flip (vCA1 b0
   0.05 ↔ 0.04; 3-seed FAR figures ±0.5 pp) with the thread count alone.
4. **vCA1 arm b0 deploy: proceed on ranking; re-derive T on deploy day with the thread count
   pinned and quote 0.04 vs 0.05 side by side** (+0.023 reviewed AUC; 0.05 at the ceiling).
5. **Real v2b on BLA bootstrap rows** (`BOOTSTRAP_V2B="b0"` for BLA, #14): +0.001 AUC, +2–3 pp
   junk at matched false-AR, safer smoke cell — cheap, low impact; after 1–4.
6. **Docs**: replace "same-cell re-detections" (#6), the "~94% wrong" damage figure (#13:
   74–81%), back the T=0.045 sentence with `results/a10.json`, cite `results/corpus_table.json`
   for the section-4 table, and note the bla21 cell (#11).
7. **DG_AL pooled prior with a leak-free calibration layer** (#18/#19) — outside this
   report's execution scope; numbers below.
8. `diagnose_model.py` / `sweep_weights.py` honouring `AGENT_WEIGHT_OVERRIDE` (tooling).

## New holes found while attacking

- **xgboost thread-count nondeterminism** is large enough to move operating-point decisions
  (above). None of the earlier gates recorded the thread count they ran with.
- The 3-seed vCA1 weight sweep's FAR figures (1.2% / 2.5%) are within numerical noise of
  each other; the decision should have rested on junk at matched false-AR (it does hold there).
- `_bootstrap_validate` count: 4 real sandboxes + 1 empty shell, not 4.
- The "duplicate" mask's rationale text does not describe most of the rows it masks.
- Attack #3's MATLAB load-only step was written (`matlab/m03_loadonly_check.m`) but not
  run; the link it closes was already proven by the prior projects' extraction checks.

## Open items / honesty

- Human spot-checks of #5 / #6 / #7 contact sheets are pending (the automated geometry is
  complete; the sheets are for the user).
- #17 cannot be closed without a deliberate double review of one session.
- The refuters reproduce every fast attack from a clean process (max relative difference
  0 on all numeric leaves); the long attacks (#8, #9, #11, #12, #15, #18, #19) are covered by
  their own pin / fixture consistency checks and probes, not by a second full run.
- Read-only invariants: closing `rt_pin.py --check` and a recursive mtime scan of the three
  area roots (see the end of `refute.log` / `closing_check.log`); no session dir, npz,
  labels, JSON, joblib, watcher, server object or git state was touched.

<!-- SYNTH:CLOSING -->
Closing check: pin unchanged = True; 49104 files scanned under the three area roots + `.bootstrap_diag`, 0 newer than the pin; tracked files modified: none; unexpected untracked: none -> **PASS**.
<!-- /SYNTH:CLOSING -->
