# agent/eval — evaluation & sweep harness scripts

These scripts produced the pinned BLA classifier numbers (multi-seed OOF AUC,
threshold sweeps, reviewer-quality checks). They were rescued from a session
scratchpad on 2026-08-18 so the reference methodology survives; byte-identical
to the versions that produced the 2026-08 numbers.

## The harness contract (do not break)

1. **Never fork weighting or model code from `train_classifier.py`.** Get
   records and weights via `diagnose_model.load_all_records()`, model factories
   via `diagnose_model.make_clf` (which delegates to
   `train_classifier._make_clf`), and constants (`MIN_AGENT_WEIGHT`,
   `BAD_SESSION_*`) from `train_classifier`. A forked copy of the
   ambiguous-mask helper once read a JSON key no producer writes and silently
   ran all-False for months — that is why this rule exists.
2. **XGBoost semantics only for BLA** — never `--model auto` (it can flip the
   deployed model to lightgbm on a CV tie).
3. **Numbers are not comparable across harness fixes.** Every number produced
   before 2026-08-18 used the broken (all-False) ambiguous mask. The corrected
   baseline lives in the feature-expansion gate report; re-derive, don't quote.
4. In-file threshold labels in comments/docstrings (e.g. "deployed 0.14") are
   historical — the deployed threshold lives in the joblib metadata.

## Scripts

| Script | What it measures |
|---|---|
| `threshold_robustness.py` | Multi-seed (8) OOF threshold sweep — false-AR / garbage-caught stability across CV seeds |
| `reviewer_quality.py` | Seed-averaged OOF scores of real-labeled cells; recent-vs-established false-AR split |
| `session_quality.py` | Per-session 5-fold CV (LR, own labels) — session separability ceiling |
| `counterfactual_reclassify.py` | What the deployed-config model would auto-reject among labeled reals |
| `counterfactual_drop6.py` | Pool sensitivity: metrics with the 6 newest sessions dropped |
| `scope_misclassified.py` | Score bands of misclassified reals (confident-vs-boundary errors) |
| `count_corpus.py` | Corpus census: sessions/candidates/reals by agent-vs-bootstrap |

All scripts insert `agent/` on `sys.path` themselves and scan `DATA_ROOT` live —
if the pool can change under you (watcher ingests), pin a manifest first.
