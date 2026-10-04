# Pixel-order (orientation) audit — every A-matrix consumer

Date: 2026-08-20. Verified by `a1_orientation_audit.py` (all checks PASS; re-run any time,
read-only). Conventions: MATLAB `neuron.A` columns are **F-order** (pixel = row + col*d1);
`spatial_footprints.mat` stacks are correct `[n, row, col]` images (writers:
`CNMFe_Biane_headless.m:318`, `CNMFe_final_save.m:728`); numpy C-order flattening of an
image stack is **not** MATLAB's linearization — mixing the two compares against
transposed images.

| # | Site | Sides compared | Verdict |
|---|------|----------------|---------|
| 1 | `bootstrap_preagent.py:245` `_load_candidates` → `_match_and_save` L282–299 | candidates **C-order** vs `A_final` **F-order** | **BROKEN** — the transpose bug. All 202 bootstrap sessions' labels scrambled. |
| 2 | `bootstrap_preagent.py:318` feature path (`A_review.T.reshape(N,d1,d2)`) | round-trips the C-order flatten back to images | OK — bootstrap features correctly oriented |
| 3 | `train_classifier.py:213–215` retro matching | `A_review` (MATLAB) vs `A_final` (MATLAB), both **F-order** | OK — retro/agent labels are clean |
| 4 | `train_classifier.py:235` retro feature path | F-order columns reshaped **C-order** → transposed images | **BUG (secondary)** — `cn_correlation` corrupted for retro-labeled sessions; the other 12 features are transpose-invariant. Fix post-Step-4 (file owned by Step 4). |
| 5 | `validate_threshold.py` `_spatial_matrix` L152–156 | replicates site 1 | **BROKEN** (proven: stored March scores == mismatched replica to 3rd decimal). March-2026 study conclusions void. |
| 6 | `features.py:20–39` `load_spatial` primary path | (N,H,W) stack used as images | OK |
| 7 | `features.py:32–38` `load_spatial` A.txt fallback | F-order columns reshaped C-order | **BUG (latent)** — transposed footprints, but scan found **0 sessions** where the fallback fires. Fix when touching features.py post-Step-4. |
| 8 | `curator.py:287–291` PDF rendering | via `load_spatial` primary | OK (display only) |
| 9 | `refresh_features.py`, `retarget_labels.py`, `check_match_stats.py`, `diagnose_model.py`, `sweep_weights.py` | npz/JSON level only | OK (no pixel data) — but they consume the scrambled JSON/labels produced by site 1 |
| 10 | v7.3 spatial_footprints (h5py path) | h5py reverses MATLAB dims → must transpose(2,1,0) | Handled in `bmlib.load_stack`; verified vs A.txt (cosine 1.0000) |

## The fix (Phase B)

Convert `A_final` to the candidates' pixel order once at load (or flatten candidates
F-order); one line either way, plus the same fix in `validate_threshold.py`. The
synthetic red/green test in `a1_orientation_audit.py §1` becomes the regression test:
production formula gives sim=0.0000 on an asymmetric footprint that should score 1.0;
the consistent formula gives exactly 1.0 (square and rectangular cases).
