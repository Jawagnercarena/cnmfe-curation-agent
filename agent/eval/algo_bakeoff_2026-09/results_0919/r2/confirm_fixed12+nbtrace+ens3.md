# Round 2 confirmation: `fixed12+nbtrace+ens3` vs the deployed control (2026-09-19 20:26)

Combiner: fixed 50/50 probability blend (pre-declared). Columns as in stage2_*.md.

## vCA1 session-grouped (8 seeds; 3669 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 3669 / 934 | 0.9206 / 0.9206 | 0.9275 / 0.9275 | +0.0070 (+0.0042; 8/8) | 0.07 (0.96 worst) | 0.87 vs 1.06 | 47.5 vs 44.2 (+3.3) |
| (a') round-1 CNN alone, same rows [no round-1 run in this results dir: same as the variant] | 3669 / 934 | 0.9206 / 0.9206 | 0.9275 / 0.9275 | +0.0070 (+0.0042; 8/8) | 0.07 (0.96 worst) | 0.87 vs 1.06 | 47.5 vs 44.2 (+3.3) |
| (b) blend 50/50, all rows | 3726 / 934 | 0.9221 / 0.9206 | 0.9325 / 0.9311 | +0.0106 (+0.0091; 8/8) | 0.06 (0.96 worst) | 1.00 vs 1.06 | 47.1 vs 45.2 (+1.9) |
| (b') round-1 blend 50/50, same seeds [no round-1 run in this results dir: same as the variant] | 3726 / 934 | 0.9221 / 0.9206 | 0.9325 / 0.9311 | +0.0106 (+0.0091; 8/8) | 0.06 (0.96 worst) | 1.00 vs 1.06 | 47.1 vs 45.2 (+1.9) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8714 | 0.8596 | 0.8806 | 0.8806 |
| pnb97 | 1200 | 206 | 0.9199 | 0.9294 | 0.9315 | 0.9315 |
| pnb98 | 2141 | 618 | 0.9221 | 0.9314 | 0.9329 | 0.9329 |

## vCA1 animal-grouped (8 seeds; 3669 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 3669 / 934 | 0.9125 / 0.9125 | 0.9182 / 0.9182 | +0.0057 (+0.0021; 8/8) | 0.05 (0.75 worst) | 1.04 vs 0.91 | 42.4 vs 44.0 (-1.6) |
| (a') round-1 CNN alone, same rows [no round-1 run in this results dir: same as the variant] | 3669 / 934 | 0.9125 / 0.9125 | 0.9182 / 0.9182 | +0.0057 (+0.0021; 8/8) | 0.05 (0.75 worst) | 1.04 vs 0.91 | 42.4 vs 44.0 (-1.6) |
| (b) blend 50/50, all rows | 3726 / 934 | 0.9142 / 0.9125 | 0.9263 / 0.9249 | +0.0124 (+0.0106; 8/8) | 0.05 (0.86 worst) | 0.95 vs 0.91 | 44.3 vs 44.7 (-0.4) |
| (b') round-1 blend 50/50, same seeds [no round-1 run in this results dir: same as the variant] | 3726 / 934 | 0.9142 / 0.9125 | 0.9263 / 0.9249 | +0.0124 (+0.0106; 8/8) | 0.05 (0.86 worst) | 0.95 vs 0.91 | 44.3 vs 44.7 (-0.4) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8647 | 0.8306 | 0.8656 | 0.8656 |
| pnb97 | 1200 | 206 | 0.9188 | 0.9268 | 0.9298 | 0.9298 |
| pnb98 | 2141 | 618 | 0.9116 | 0.9270 | 0.9263 | 0.9263 |

## Verdict

| cell | gate |
|---|---|
| BLA session | FAIL / not run |
| BLA animal | FAIL / not run |
| vCA1 session | PASS |
| vCA1 animal | PASS |

**Pre-registered gate (BLA both groupings pass, vCA1 not worse under both): NOT MET**
