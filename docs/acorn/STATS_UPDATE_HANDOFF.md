# ACORN figure — stats refresh handoff (2026-08-06)

Refreshed BLA model numbers after a batch of reviewer returns landed. The BLA
training pool grew **49 → 58 agent sessions** (140 BLA sessions total, incl. 82
bootstrap). Re-swept with 8 CV seeds. Numbers below are the ones to update in the
schematic. **Threshold is unchanged at 0.12** — no wording about the operating
point needs to move.

## Provenance (one sentence for the manuscript / caption)

> BLA out-of-fold AUC **0.893 ± 0.001** (8-seed StratifiedGroupKFold, whole
> session held out and re-scored by a model trained on the rest; 58 agent + 82
> bootstrap sessions, real deployed weights). At the deployed operating point
> (0.12): **21.1%** of garbage candidates auto-removed, **0.85%** of real
> neurons auto-rejected (false-AR).

## Stat tiles — old → new

| tile | current | new | notes |
|---|---|---|---|
| Real-vs-junk skill (big) | `0.88` | `0.89` | BLA OOF 0.893 ± 0.001 (was 0.884). **See AUC caveat.** |
| …caption | `~88% of the time` | `~89% of the time` | same number, plain-language form |
| Junk auto-removed (big) | `~18%` | `~21%` | garbage caught at 0.12 = 21.1% (was ~18%) |
| …caption | "Roughly **a fifth**…" | keep, or "just over a fifth" | 21% is still ~a fifth; phrasing still true |
| Real ever auto-rejected (big) | `<1%` | `<1%` — **no change** | now 0.85% (was ~0.9%); claim still holds comfortably |

## Exact locations

**docs/acorn/acorn-schematic.html** (interactive HTML)
- line 451 — `<div class="big">0.88</div>` → `0.89`
- line 452 — `~88% of the time` → `~89% of the time`
- line 455 — `<div class="big">~18%</div>` → `~21%`
- line 456 — "Roughly a fifth…" — optional, still accurate
- line 459 — `<div class="big">&lt;1%</div>` — **leave as-is**
- line 472 — `88,000<span class="plus">+</span>` → `90,000<span class="plus">+</span>`
- line 474 — `across 261 sessions` → `across 276 sessions`
- line 483 — `about <b>8,400 cells reviewed directly</b>` → `about <b>10,800 cells reviewed directly</b>`
- line 488 — `~7,700 neurons` — **leave as-is** (bootstrap pool unchanged)

**docs/acorn/make_acorn_figure.py** (PDF/PNG vector figure)
- line 223 — `88,000+ human-labeled cells  ·  261 sessions` → `90,000+ human-labeled cells  ·  276 sessions`
- line 235 — `("0.88", "…ranks a real neuron above a junk one ~88% of the time…")` → `0.89` / `~89%`
- line 236 — `("~18%", "Of the junk, auto-removed…")` → `~21%`
- line 237 — `("<1%", …)` — **leave as-is**
- (re-run the script to regenerate `acorn_schematic.pdf` / `.png`)

## Two caveats — don't apply blind

1. **AUC is a cross-area tile (BLA + vCA1); 0.893 is BLA only.** vCA1 was NOT
   re-measured this session (last validated ~0.881, threshold locked). So:
   - If the tile represents *both* areas, `0.88` is still a true floor for both.
   - `0.89` is accurate **for BLA**. Safest options: keep `0.88`, show a range
     `0.88–0.89`, or label the bump as BLA-specific. Pick based on whether the
     figure is BLA-centric. The `~18%` and `<1%` tiles are BLA-model numbers.

2. **Corpus counts — RECOUNTED 2026-08-06** (both areas, via
   `scratchpad/count_training_cells.py`). Values above are final; apply them.
   Full before→after:

   | corpus metric | old (07-31) | new (08-06) |
   |---|---|---|
   | human-labeled candidate cells | 88,316 (`88,000+`) | **90,942 (`90,000+`)** |
   | sessions | 261 | **276** |
   | direct-review (agent) decisions | ~8,434 (`~8,400`) | **10,802 (`~10,800`)** |
   | prior hand-curated (bootstrap) | ~7,722 (`~7,700`) | **7,722 — unchanged** |
   | confirmed-real positives | 5,598 | **6,101** |

   All growth is BLA (131→146 sessions, 38,213→40,839 cells); **vCA1 is
   byte-identical** (130 sessions / 50,103 cells), consistent with it being
   locked and only BLA re-run. Per area, new: BLA 146 sessions / 40,839 cells
   (64 agent + 82 bootstrap); vCA1 130 / 50,103 (19 agent + 111 bootstrap).

   Note: the corpus **agent-session** count (BLA 64) is higher than the
   classifier **CV pool** (58 agent) on purpose — the corpus counts every
   human-labeled session, while the CV pool applies training filters. The `276`
   figure number is the corpus total and does not conflict with the `58`/`140`
   used for the AUC provenance.

## Not changing
- Threshold / operating point: still 0.12.
- Feature count: still 13.
- Params removed: still 5 → 0.
- Deployed model file: may not yet include the 13 new sessions (needs
  `train_classifier.py --prospective-only`); does not affect these figure stats,
  which are cross-validated on the current data.
