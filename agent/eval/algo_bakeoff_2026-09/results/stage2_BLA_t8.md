# Stage 2: raw-data CNN vs deployed XGBoost -- BLA (2026-09-06 22:40)

Columns: rows / reals used; control AUC full / reviewed; new AUC full / reviewed; paired reviewed-AUC delta (min; seeds positive); rule_T (worst-seed FAR at it); FAR of new at the control's matched junk vs the control's FAR at T; junk of new at the control's matched FAR vs the control's junk at T (delta pp).

## session-grouped (8 seeds; 11877 of 13766 rows have raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 11877 / 2618 | 0.9099 / 0.9099 | 0.8944 / 0.8944 | -0.0155 (-0.0296; 0/8) | 0.05 (0.92 worst) | 1.03 vs 0.76 | 28.6 vs 32.3 (-3.6) |
| (b) blend mean(control, CNN), all rows | 13766 / 2618 | 0.9245 / 0.9099 | 0.9286 / 0.9144 | +0.0045 (+0.0021; 8/8) | 0.09 (0.95 worst) | 0.54 vs 0.76 | 45.8 vs 41.4 (+4.4) |
| (c) rank blend (T is a quantile here) | 13766 / 2618 | 0.9245 / 0.9099 | 0.9285 / 0.9143 | +0.0044 (+0.0007; 8/8) | 0.1 (0.00 worst) | 0.48 vs 0.76 | 47.3 vs 41.4 (+5.9) |

kill criterion (CNN alone < control - 0.02 on reviewed AUC): no; gate for the blend: AUC fail, junk pass, rule pass -> **FAIL**

| animal | rows | reals | control AUC | CNN AUC | blend AUC |
|---|---|---|---|---|---|
| bla12 | 1857 | 541 | 0.9206 | 0.9019 | 0.9237 |
| bla16 | 840 | 169 | 0.9110 | 0.8712 | 0.9037 |
| bla21 | 307 | 106 | 0.8515 | 0.9000 | 0.8977 |
| bla36 | 4114 | 690 | 0.9023 | 0.8871 | 0.9064 |
| bla37 | 3038 | 783 | 0.9219 | 0.8993 | 0.9207 |
| bla8 | 1721 | 329 | 0.9087 | 0.8947 | 0.9140 |

## animal-grouped (8 seeds; 11877 of 13766 rows have raw data)

| comparison | rows / reals | control AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| (a) CNN alone, raw-data rows | 11877 / 2618 | 0.8989 / 0.8989 | 0.8881 / 0.8881 | -0.0108 (-0.0254; 1/8) | 0.04 (0.92 worst) | 1.13 vs 1.01 | 30.2 vs 31.5 (-1.3) |
| (b) blend mean(control, CNN), all rows | 13766 / 2618 | 0.9151 / 0.8989 | 0.9221 / 0.9067 | +0.0078 (+0.0010; 8/8) | 0.09 (0.95 worst) | 0.65 vs 1.01 | 46.8 vs 40.6 (+6.2) |
| (c) rank blend (T is a quantile here) | 13766 / 2618 | 0.9151 / 0.8989 | 0.9222 / 0.9069 | +0.0080 (+0.0003; 8/8) | 0.1 (0.00 worst) | 0.66 vs 1.01 | 47.3 vs 40.6 (+6.7) |

kill criterion (CNN alone < control - 0.02 on reviewed AUC): no; gate for the blend: AUC pass, junk pass, rule pass -> **PASS**

| animal | rows | reals | control AUC | CNN AUC | blend AUC |
|---|---|---|---|---|---|
| bla12 | 1857 | 541 | 0.9097 | 0.9070 | 0.9192 |
| bla16 | 840 | 169 | 0.9109 | 0.8745 | 0.9028 |
| bla21 | 307 | 106 | 0.8570 | 0.9127 | 0.9039 |
| bla36 | 4114 | 690 | 0.8958 | 0.8868 | 0.9014 |
| bla37 | 3038 | 783 | 0.9151 | 0.9021 | 0.9176 |
| bla8 | 1721 | 329 | 0.9053 | 0.9035 | 0.9150 |
