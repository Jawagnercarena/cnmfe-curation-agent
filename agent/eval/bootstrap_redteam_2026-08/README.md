# Bootstrap-fix red team (2026-08)

Adversarial review of the bootstrap pixel-order fix and the three model decisions,
per `docs/BOOTSTRAP_REDTEAM_BRIEF.md`.  Everything here is **read-only toward the
pipeline**: no session dir, npz, labels.mat, JSON, joblib, watcher, server or git
state is touched.  Outputs live only in this directory (`results/` committed;
`fixtures/`, `contact_sheets/`, `*.log` gitignored and regenerable).

Run order (valence python, from this directory):

```
python rt_pin.py                 # pin the corpus -> pin_manifest.json
python rt_selftest.py            # evaluator reproduces both OOF pins; synthetic red/green
python prep_*.py                 # shared fixtures (animal map, OOS scores, geometry, arms, retro ids)
python aNN_*.py                  # one script per attack -> results/aNN.json
python refute/rNN.py             # adversarial re-check -> results/aNN_refute.json
python synth_report.py           # tables from results/*.json
python check_report.py           # completeness critic
python rt_pin.py --check         # corpus unchanged since the pin
```

`rt_lib.py` is the independent evaluator (own discovery, labels, weights, CV,
metrics, loaders, matchers).  It imports only the feature MATH from
`agent/features.py`; it never imports the harnesses it is checking.  Every
attack script calls `rt_lib.assert_pinned()` first and refuses to run on drift;
every results JSON carries the pin hash and the script's sha256.
