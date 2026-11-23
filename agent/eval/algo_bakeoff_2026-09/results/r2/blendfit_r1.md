## BLA session-grouped, CNN = BLA_session_cnn_t8.npz, 8 seeds (2026-09-07 14:54)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| fixed 50/50 blend vs control | 13766 / 2618 | 0.9245 / 0.9099 | 0.9286 / 0.9144 | +0.0045 (+0.0021; 8/8) | 0.09 (0.95 worst) | 0.54 vs 0.76 | 45.8 vs 41.4 (+4.4) |
| fitted logistic combiner vs control | 13766 / 2618 | 0.9245 / 0.9099 | 0.9257 / 0.9134 | +0.0034 (-0.0055; 7/8) | None (nan worst) | 0.73 vs 0.76 | 42.3 vs 41.4 (+0.9) |
| fitted combiner vs fixed blend | 13766 / 2618 | 0.9286 / 0.9144 | 0.9257 / 0.9134 | -0.0010 (-0.0076; 2/8) | None (nan worst) | 0.40 vs 0.27 | 27.5 vs 31.9 (-4.3) |

combiner coefficients (mean over seeds x folds): control-logit 0.629, CNN-logit 0.420, intercept -1.557; CNN share of logit weight 0.40 [0.18, 0.50]

## BLA animal-grouped, CNN = BLA_animal_cnn_t8.npz, 8 seeds (2026-09-07 14:54)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| fixed 50/50 blend vs control | 13766 / 2618 | 0.9151 / 0.8989 | 0.9221 / 0.9067 | +0.0078 (+0.0010; 8/8) | 0.09 (0.95 worst) | 0.65 vs 1.01 | 46.8 vs 40.6 (+6.2) |
| fitted logistic combiner vs control | 13766 / 2618 | 0.9151 / 0.8989 | 0.9160 / 0.9022 | +0.0033 (-0.0059; 6/8) | None (nan worst) | 1.03 vs 1.01 | 41.8 vs 40.6 (+1.2) |
| fitted combiner vs fixed blend | 13766 / 2618 | 0.9221 / 0.9067 | 0.9160 / 0.9022 | -0.0045 (-0.0075; 0/8) | None (nan worst) | 0.54 vs 0.32 | 25.6 vs 31.4 (-5.8) |

combiner coefficients (mean over seeds x folds): control-logit 0.553, CNN-logit 0.456, intercept -1.531; CNN share of logit weight 0.45 [0.21, 0.60]

## vCA1 session-grouped, CNN = vCA1_session_cnn_t8.npz, 8 seeds (2026-09-07 14:54)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| fixed 50/50 blend vs control | 2068 / 483 | 0.9252 / 0.9227 | 0.9343 / 0.9319 | +0.0093 (+0.0049; 8/8) | 0.07 (0.83 worst) | 0.91 vs 1.16 | 51.2 vs 46.0 (+5.2) |
| fitted logistic combiner vs control | 2068 / 483 | 0.9252 / 0.9227 | 0.9323 / 0.9303 | +0.0077 (+0.0024; 8/8) | None (nan worst) | 1.06 vs 1.16 | 48.8 vs 46.0 (+2.9) |
| fitted combiner vs fixed blend | 2068 / 483 | 0.9343 / 0.9319 | 0.9323 / 0.9303 | -0.0016 (-0.0047; 2/8) | None (nan worst) | 0.52 vs 0.34 | 27.6 vs 29.7 (-2.1) |

combiner coefficients (mean over seeds x folds): control-logit 0.475, CNN-logit 0.550, intercept -1.464; CNN share of logit weight 0.54 [0.34, 0.66]

## vCA1 animal-grouped, CNN = vCA1_animal_cnn_t8.npz, 8 seeds (2026-09-07 14:54)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| fixed 50/50 blend vs control | 2068 / 483 | 0.9101 / 0.9071 | 0.9180 / 0.9152 | +0.0081 (-0.0025; 7/8) | 0.04 (0.83 worst) | 1.27 vs 0.91 | 42.7 vs 44.5 (-1.8) |
| fitted logistic combiner vs control | 2068 / 483 | 0.9101 / 0.9071 | 0.8928 / 0.8894 | -0.0177 (-0.0324; 1/8) | None (nan worst) | 1.76 vs 0.91 | 36.6 vs 44.5 (-7.9) |
| fitted combiner vs fixed blend | 2068 / 483 | 0.9180 / 0.9152 | 0.8928 / 0.8894 | -0.0258 (-0.0329; 0/8) | None (nan worst) | 0.93 vs 0.39 | 29.5 vs 33.0 (-3.4) |

combiner coefficients (mean over seeds x folds): control-logit 0.542, CNN-logit 0.394, intercept -1.009; CNN share of logit weight 0.42 [0.07, 0.66]

