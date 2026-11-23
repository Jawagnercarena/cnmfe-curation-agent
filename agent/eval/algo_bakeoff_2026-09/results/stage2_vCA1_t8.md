# Stage 2: raw-data CNN vs deployed XGBoost -- vCA1 (2026-09-06 22:22)

Columns: rows / reals used; control AUC full / reviewed; new AUC full / reviewed; paired reviewed-AUC delta (min; seeds positive); rule_T (worst-seed FAR at it); FAR of new at the control's matched junk vs the control's FAR at T; junk of new at the control's matched FAR vs the control's junk at T (delta pp).

## session-grouped (8 seeds; 2011 of 2068 rows have raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 2011 / 483 | 0.9227 / 0.9227 | 0.9205 / 0.9205 | -0.0022 (-0.0194; 4/8) | None (nan worst) | 1.14 vs 1.16 | 44.7 vs 44.4 (+0.4) |
| (b) blend mean(control, CNN), all rows | 2068 / 483 | 0.9252 / 0.9227 | 0.9343 / 0.9319 | +0.0093 (+0.0049; 8/8) | 0.07 (0.83 worst) | 0.91 vs 1.16 | 51.2 vs 46.0 (+5.2) |
| (c) rank blend (T is a quantile here) | 2068 / 483 | 0.9252 / 0.9227 | 0.9355 / 0.9332 | +0.0105 (+0.0053; 8/8) | 0.1 (0.00 worst) | 0.88 vs 1.16 | 49.9 vs 46.0 (+3.9) |

kill criterion (CNN alone < control - 0.02 on reviewed AUC): no; gate for the blend: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control AUC | CNN AUC | blend AUC |
|---|---|---|---|---|---|
| pnb88 | 328 | 110 | 0.8736 | 0.8567 | 0.8839 |
| pnb97 | 1200 | 206 | 0.9192 | 0.9218 | 0.9287 |
| pnb98 | 483 | 167 | 0.9476 | 0.9402 | 0.9539 |

## animal-grouped (8 seeds; 2011 of 2068 rows have raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 2011 / 483 | 0.9071 / 0.9071 | 0.8930 / 0.8930 | -0.0142 (-0.0323; 2/8) | None (nan worst) | 2.28 vs 0.91 | 30.2 vs 43.1 (-12.8) |
| (b) blend mean(control, CNN), all rows | 2068 / 483 | 0.9101 / 0.9071 | 0.9180 / 0.9152 | +0.0081 (-0.0025; 7/8) | 0.04 (0.83 worst) | 1.27 vs 0.91 | 42.7 vs 44.5 (-1.8) |
| (c) rank blend (T is a quantile here) | 2068 / 483 | 0.9101 / 0.9071 | 0.9192 / 0.9165 | +0.0093 (-0.0033; 7/8) | 0.1 (0.00 worst) | 1.19 vs 0.91 | 44.7 vs 44.5 (+0.3) |

kill criterion (CNN alone < control - 0.02 on reviewed AUC): no; gate for the blend: AUC fail, junk fail, rule pass -> **FAIL**

| animal | rows | reals | control AUC | CNN AUC | blend AUC |
|---|---|---|---|---|---|
| pnb88 | 328 | 110 | 0.8590 | 0.8090 | 0.8556 |
| pnb97 | 1200 | 206 | 0.9150 | 0.9166 | 0.9251 |
| pnb98 | 483 | 167 | 0.9459 | 0.9421 | 0.9535 |
