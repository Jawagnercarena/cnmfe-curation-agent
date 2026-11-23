# Round 2 confirmation: `fixed12+nbtrace` vs the deployed control (2026-09-19 16:42)

Combiner: fixed 50/50 probability blend (pre-declared). Columns as in stage2_*.md.

## vCA1 session-grouped (8 seeds; 3669 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 3669 / 934 | 0.9206 / 0.9206 | 0.9264 / 0.9264 | +0.0059 (+0.0014; 8/8) | 0.07 (0.96 worst) | 0.92 vs 1.06 | 47.4 vs 44.2 (+3.2) |
| (a') round-1 CNN alone, same rows [no round-1 run in this results dir: same as the variant] | 3669 / 934 | 0.9206 / 0.9206 | 0.9264 / 0.9264 | +0.0059 (+0.0014; 8/8) | 0.07 (0.96 worst) | 0.92 vs 1.06 | 47.4 vs 44.2 (+3.2) |
| (b) blend 50/50, all rows | 3726 / 934 | 0.9221 / 0.9206 | 0.9322 / 0.9308 | +0.0102 (+0.0082; 8/8) | 0.06 (0.96 worst) | 0.96 vs 1.06 | 46.6 vs 45.2 (+1.5) |
| (b') round-1 blend 50/50, same seeds [no round-1 run in this results dir: same as the variant] | 3726 / 934 | 0.9221 / 0.9206 | 0.9322 / 0.9308 | +0.0102 (+0.0082; 8/8) | 0.06 (0.96 worst) | 0.96 vs 1.06 | 46.6 vs 45.2 (+1.5) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8714 | 0.8520 | 0.8781 | 0.8781 |
| pnb97 | 1200 | 206 | 0.9199 | 0.9281 | 0.9309 | 0.9309 |
| pnb98 | 2141 | 618 | 0.9221 | 0.9310 | 0.9328 | 0.9328 |

## vCA1 animal-grouped (8 seeds; 3669 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 3669 / 934 | 0.9125 / 0.9125 | 0.9168 / 0.9168 | +0.0043 (+0.0013; 8/8) | 0.05 (0.96 worst) | 1.14 vs 0.91 | 39.7 vs 44.0 (-4.3) |
| (a') round-1 CNN alone, same rows [no round-1 run in this results dir: same as the variant] | 3669 / 934 | 0.9125 / 0.9125 | 0.9168 / 0.9168 | +0.0043 (+0.0013; 8/8) | 0.05 (0.96 worst) | 1.14 vs 0.91 | 39.7 vs 44.0 (-4.3) |
| (b) blend 50/50, all rows | 3726 / 934 | 0.9142 / 0.9125 | 0.9259 / 0.9244 | +0.0119 (+0.0104; 8/8) | 0.06 (0.96 worst) | 1.00 vs 0.91 | 43.9 vs 44.7 (-0.9) |
| (b') round-1 blend 50/50, same seeds [no round-1 run in this results dir: same as the variant] | 3726 / 934 | 0.9142 / 0.9125 | 0.9259 / 0.9244 | +0.0119 (+0.0104; 8/8) | 0.06 (0.96 worst) | 1.00 vs 0.91 | 43.9 vs 44.7 (-0.9) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8647 | 0.8349 | 0.8663 | 0.8663 |
| pnb97 | 1200 | 206 | 0.9188 | 0.9253 | 0.9294 | 0.9294 |
| pnb98 | 2141 | 618 | 0.9116 | 0.9249 | 0.9257 | 0.9257 |

## Verdict

| cell | gate |
|---|---|
| BLA session | FAIL / not run |
| BLA animal | FAIL / not run |
| vCA1 session | PASS |
| vCA1 animal | PASS |

**Pre-registered gate (BLA both groupings pass, vCA1 not worse under both): NOT MET**
