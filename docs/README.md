# Documentation

Docs for **ACORN — Automated CNMFe Of Recording-Networks**, the automation-assisted
curation layer this repository adds on top of CNMF-E.

[![ACORN — the full pipeline at a glance](acorn/acorn_schematic.png)](acorn/acorn_schematic.pdf)

_Click the figure for the print-resolution PDF, or open the [interactive version](acorn/acorn-schematic.html)._

## The system at a glance

- **[acorn/acorn-schematic.html](acorn/acorn-schematic.html)** — interactive schematic of the whole pipeline (open in a browser).
- **[acorn/acorn_schematic.pdf](acorn/acorn_schematic.pdf)** — the same figure as a print-resolution vector, for the manuscript / slides.
- **[acorn/acorn_schematic.png](acorn/acorn_schematic.png)** — a raster preview of the figure.
- **[acorn/make_acorn_figure.py](acorn/make_acorn_figure.py)** — regenerates the PDF/PNG (run in the `valence` env).

ACORN wraps CNMF-E with: automatic per-session parameter estimation, headless
extraction, a per-area XGBoost curator that pre-rejects confident junk and flags
motion / split-cell candidates, a short human review of the survivors, and
**online retraining** — every review sharpens the model, so the next session needs
less work. Reviewing distributes across a network of machines, one canonical model
per brain area.

## Setup & operation

- **[SETUP.md](SETUP.md)** — set up the repo on a new machine and run the pipeline (central machine: heavy compute + the single canonical model).
- **[REVIEW_SETUP.md](REVIEW_SETUP.md)** — the reviewer role: MATLAB-only, no Python. Pull a bundle, run the review, push it back.
- **[../CLAUDE.md](../CLAUDE.md)** — standing constraints for anyone running Claude (or any coding agent) against this pipeline or the lab server. Loaded automatically from the repo root; `CLAUDE_RULES.md` is now a pointer to it.
- **[SETUP_INSTRUCTIONS.txt](SETUP_INSTRUCTIONS.txt)** — legacy setup notes, superseded by `SETUP.md` (kept for reference).

## Records

Completed work, kept for its evidence and reasoning. Each file opens with a status
banner saying what superseded it; do not take numbers from these files as current.

- **[VCA1_V2_BRIEF.md](VCA1_V2_BRIEF.md)** — vCA1 35-column contract: gates, arm decision, deploy record (2026-08-26).
- **[CNN_BLEND_STAGE3_BRIEF.md](CNN_BLEND_STAGE3_BRIEF.md)** — CNN/XGBoost blend, stage 3 decision brief (open).
- **[archive/](archive/)** — feature-expansion set (handoff, gate, step 2, step 4 brief, red-team brief), motion handoffs, the March 2026 curator upgrade note, the bootstrap-matching bug write-up and its red-team brief, and the 2026-08 ACORN stats refresh. Superseded; banners at the top of each.

> Links in these files that point at code (e.g. `../agent/...`) are written relative to
> this `docs/` folder.
