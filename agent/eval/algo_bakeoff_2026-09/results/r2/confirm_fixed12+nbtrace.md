# Round 2 confirmation: `fixed12+nbtrace` vs the deployed control (2026-09-08 04:43)

Combiner: fixed 50/50 probability blend (pre-declared). Columns as in stage2_*.md.

## BLA session-grouped (8 seeds; 11877 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 11877 / 2618 | 0.9099 / 0.9099 | 0.9094 / 0.9094 | -0.0006 (-0.0033; 2/8) | 0.07 (0.92 worst) | 0.80 vs 0.76 | 31.2 vs 32.3 (-1.0) |
| (a') round-1 CNN alone, same rows | 11877 / 2618 | 0.9099 / 0.9099 | 0.8944 / 0.8944 | -0.0155 (-0.0296; 0/8) | 0.05 (0.92 worst) | 1.03 vs 0.76 | 28.6 vs 32.3 (-3.6) |
| (b) blend 50/50, all rows | 13766 / 2618 | 0.9245 / 0.9099 | 0.9330 / 0.9198 | +0.0098 (+0.0083; 8/8) | 0.09 (0.95 worst) | 0.51 vs 0.76 | 46.8 vs 41.4 (+5.4) |
| (b') round-1 blend 50/50, same seeds | 13766 / 2618 | 0.9245 / 0.9099 | 0.9286 / 0.9144 | +0.0045 (+0.0021; 8/8) | 0.09 (0.95 worst) | 0.54 vs 0.76 | 45.8 vs 41.4 (+4.4) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| bla12 | 2079 | 541 | 0.9206 | 0.9181 | 0.9237 | 0.9296 |
| bla16 | 1056 | 169 | 0.9110 | 0.8831 | 0.9037 | 0.9081 |
| bla21 | 312 | 106 | 0.8515 | 0.9178 | 0.8977 | 0.9058 |
| bla36 | 4724 | 690 | 0.9023 | 0.9013 | 0.9064 | 0.9111 |
| bla37 | 3507 | 783 | 0.9219 | 0.9138 | 0.9207 | 0.9257 |
| bla8 | 2088 | 329 | 0.9087 | 0.9162 | 0.9140 | 0.9228 |

## BLA animal-grouped (8 seeds; 11877 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 11877 / 2618 | 0.8989 / 0.8989 | 0.9034 / 0.9034 | +0.0046 (+0.0018; 8/8) | 0.04 (0.88 worst) | 0.81 vs 1.01 | 35.2 vs 31.5 (+3.7) |
| (a') round-1 CNN alone, same rows | 11877 / 2618 | 0.8989 / 0.8989 | 0.8881 / 0.8881 | -0.0108 (-0.0254; 1/8) | 0.04 (0.92 worst) | 1.13 vs 1.01 | 30.2 vs 31.5 (-1.3) |
| (b) blend 50/50, all rows | 13766 / 2618 | 0.9151 / 0.8989 | 0.9272 / 0.9128 | +0.0140 (+0.0123; 8/8) | 0.08 (0.88 worst) | 0.54 vs 1.01 | 48.6 vs 40.6 (+8.0) |
| (b') round-1 blend 50/50, same seeds | 13766 / 2618 | 0.9151 / 0.8989 | 0.9221 / 0.9067 | +0.0078 (+0.0010; 8/8) | 0.09 (0.95 worst) | 0.65 vs 1.01 | 46.8 vs 40.6 (+6.2) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| bla12 | 2079 | 541 | 0.9097 | 0.9153 | 0.9192 | 0.9237 |
| bla16 | 1056 | 169 | 0.9109 | 0.8849 | 0.9028 | 0.9068 |
| bla21 | 312 | 106 | 0.8570 | 0.9190 | 0.9039 | 0.9100 |
| bla36 | 4724 | 690 | 0.8958 | 0.8984 | 0.9014 | 0.9063 |
| bla37 | 3507 | 783 | 0.9151 | 0.9087 | 0.9176 | 0.9207 |
| bla8 | 2088 | 329 | 0.9053 | 0.9162 | 0.9150 | 0.9216 |

## vCA1 session-grouped (8 seeds; 2011 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 2011 / 483 | 0.9227 / 0.9227 | 0.9292 / 0.9292 | +0.0065 (+0.0010; 8/8) | 0.04 (0.83 worst) | 1.11 vs 1.16 | 46.0 vs 44.4 (+1.7) |
| (a') round-1 CNN alone, same rows | 2011 / 483 | 0.9227 / 0.9227 | 0.9205 / 0.9205 | -0.0022 (-0.0194; 4/8) | None (nan worst) | 1.14 vs 1.16 | 44.7 vs 44.4 (+0.4) |
| (b) blend 50/50, all rows | 2068 / 483 | 0.9252 / 0.9227 | 0.9368 / 0.9346 | +0.0119 (+0.0086; 8/8) | 0.04 (0.83 worst) | 0.98 vs 1.16 | 49.7 vs 46.0 (+3.7) |
| (b') round-1 blend 50/50, same seeds | 2068 / 483 | 0.9252 / 0.9227 | 0.9343 / 0.9319 | +0.0093 (+0.0049; 8/8) | 0.07 (0.83 worst) | 0.91 vs 1.16 | 51.2 vs 46.0 (+5.2) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8736 | 0.8589 | 0.8839 | 0.8818 |
| pnb97 | 1200 | 206 | 0.9192 | 0.9293 | 0.9287 | 0.9319 |
| pnb98 | 483 | 167 | 0.9476 | 0.9513 | 0.9539 | 0.9567 |

## vCA1 animal-grouped (8 seeds; 2011 rows with raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 2011 / 483 | 0.9071 / 0.9071 | 0.9151 / 0.9151 | +0.0080 (+0.0033; 8/8) | 0.03 (0.83 worst) | 1.01 vs 0.91 | 43.1 vs 43.1 (-0.0) |
| (a') round-1 CNN alone, same rows | 2011 / 483 | 0.9071 / 0.9071 | 0.8930 / 0.8930 | -0.0142 (-0.0323; 2/8) | None (nan worst) | 2.28 vs 0.91 | 30.2 vs 43.1 (-12.8) |
| (b) blend 50/50, all rows | 2068 / 483 | 0.9101 / 0.9071 | 0.9264 / 0.9239 | +0.0167 (+0.0141; 8/8) | 0.04 (0.83 worst) | 0.78 vs 0.91 | 47.4 vs 44.5 (+3.0) |
| (b') round-1 blend 50/50, same seeds | 2068 / 483 | 0.9101 / 0.9071 | 0.9180 / 0.9152 | +0.0081 (-0.0025; 7/8) | 0.04 (0.83 worst) | 1.27 vs 0.91 | 42.7 vs 44.5 (-1.8) |

kill (CNN alone < control - 0.02): no; gate: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control | CNN | round-1 blend | blend |
|---|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8590 | 0.8191 | 0.8556 | 0.8609 |
| pnb97 | 1200 | 206 | 0.9150 | 0.9272 | 0.9251 | 0.9287 |
| pnb98 | 483 | 167 | 0.9459 | 0.9485 | 0.9535 | 0.9554 |

## Verdict

| cell | gate |
|---|---|
| BLA session | PASS |
| BLA animal | PASS |
| vCA1 session | PASS |
| vCA1 animal | PASS |

**Pre-registered gate (BLA both groupings pass, vCA1 not worse under both): MET**
