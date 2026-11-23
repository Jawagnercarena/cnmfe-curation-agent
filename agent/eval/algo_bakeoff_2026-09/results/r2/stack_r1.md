## BLA session-grouped, CNN = BLA_session_cnn_t8.npz, 8 seeds (2026-09-07 14:50)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| stack (35 + CNN + rank) vs pinned control | 13766 / 2618 | 0.9245 / 0.9099 | 0.9272 / 0.9134 | +0.0034 (-0.0042; 7/8) | 0.03 (0.65 worst) | 0.62 vs 0.76 | 44.3 vs 41.4 (+2.9) |
| stack vs ctrl35 (same code path) | 13766 / 2618 | 0.9245 / 0.9099 | 0.9272 / 0.9134 | +0.0034 (-0.0042; 7/8) | 0.03 (0.65 worst) | 0.62 vs 0.76 | 44.3 vs 41.4 (+2.9) |
| ctrl35 vs pinned control (code-path check) | 13766 / 2618 | 0.9245 / 0.9099 | 0.9245 / 0.9099 | +0.0000 (+0.0000; 0/8) | 0.04 (0.92 worst) | 0.76 vs 0.76 | 41.6 vs 41.4 (+0.2) |
| blend 50/50 vs pinned control (same seeds) | 13766 / 2618 | 0.9245 / 0.9099 | 0.9286 / 0.9144 | +0.0045 (+0.0021; 8/8) | 0.09 (0.95 worst) | 0.54 vs 0.76 | 45.8 vs 41.4 (+4.4) |

gate (stack vs pinned control): AUC fail, junk fail, rule pass -> **FAIL**

| animal | rows | reals | control | blend | stack |
|---|---|---|---|---|---|
| bla12 | 2079 | 541 | 0.9206 | 0.9237 | 0.9228 |
| bla16 | 1056 | 169 | 0.9110 | 0.9037 | 0.9096 |
| bla21 | 312 | 106 | 0.8515 | 0.8977 | 0.8749 |
| bla36 | 4724 | 690 | 0.9023 | 0.9064 | 0.9055 |
| bla37 | 3507 | 783 | 0.9219 | 0.9207 | 0.9211 |
| bla8 | 2088 | 329 | 0.9087 | 0.9140 | 0.9131 |

## vCA1 session-grouped, CNN = vCA1_session_cnn_t8.npz, 8 seeds (2026-09-07 14:50)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| stack (35 + CNN + rank) vs pinned control | 2068 / 483 | 0.9252 / 0.9227 | 0.9289 / 0.9267 | +0.0040 (+0.0014; 8/8) | None (nan worst) | 1.50 vs 1.16 | 42.9 vs 46.0 (-3.1) |
| stack vs ctrl35 (same code path) | 2068 / 483 | 0.9252 / 0.9227 | 0.9289 / 0.9267 | +0.0040 (+0.0014; 8/8) | None (nan worst) | 1.50 vs 1.16 | 42.9 vs 46.0 (-3.1) |
| ctrl35 vs pinned control (code-path check) | 2068 / 483 | 0.9252 / 0.9227 | 0.9252 / 0.9227 | +0.0000 (+0.0000; 0/8) | None (nan worst) | 1.16 vs 1.16 | 50.8 vs 46.0 (+4.8) |
| blend 50/50 vs pinned control (same seeds) | 2068 / 483 | 0.9252 / 0.9227 | 0.9343 / 0.9319 | +0.0093 (+0.0049; 8/8) | 0.07 (0.83 worst) | 0.91 vs 1.16 | 51.2 vs 46.0 (+5.2) |

gate (stack vs pinned control): AUC fail, junk fail, rule pass -> **FAIL**

| animal | rows | reals | control | blend | stack |
|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8736 | 0.8839 | 0.8882 |
| pnb97 | 1200 | 206 | 0.9192 | 0.9287 | 0.9209 |
| pnb98 | 483 | 167 | 0.9476 | 0.9539 | 0.9476 |

## vCA1 animal-grouped, CNN = vCA1_animal_cnn_t8.npz, 8 seeds (2026-09-07 14:51)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| stack (35 + CNN + rank) vs pinned control | 2068 / 483 | 0.9101 / 0.9071 | 0.9091 / 0.9063 | -0.0009 (-0.0129; 3/8) | None (nan worst) | 1.22 vs 0.91 | 43.1 vs 44.5 (-1.3) |
| stack vs ctrl35 (same code path) | 2068 / 483 | 0.9101 / 0.9071 | 0.9091 / 0.9063 | -0.0009 (-0.0129; 3/8) | None (nan worst) | 1.22 vs 0.91 | 43.1 vs 44.5 (-1.3) |
| ctrl35 vs pinned control (code-path check) | 2068 / 483 | 0.9101 / 0.9071 | 0.9101 / 0.9071 | +0.0000 (+0.0000; 0/8) | None (nan worst) | 0.91 vs 0.91 | 47.3 vs 44.5 (+2.8) |
| blend 50/50 vs pinned control (same seeds) | 2068 / 483 | 0.9101 / 0.9071 | 0.9180 / 0.9152 | +0.0081 (-0.0025; 7/8) | 0.04 (0.83 worst) | 1.27 vs 0.91 | 42.7 vs 44.5 (-1.8) |

gate (stack vs pinned control): AUC fail, junk fail, rule fail -> **FAIL**

| animal | rows | reals | control | blend | stack |
|---|---|---|---|---|---|
| pnb88 | 385 | 110 | 0.8590 | 0.8556 | 0.8576 |
| pnb97 | 1200 | 206 | 0.9150 | 0.9251 | 0.9203 |
| pnb98 | 483 | 167 | 0.9459 | 0.9535 | 0.9444 |

## BLA animal-grouped, CNN = BLA_animal_cnn_t8.npz, 8 seeds (2026-09-07 14:51)

| comparison | rows / reals | ref AUC | new AUC | d reviewed AUC | rule_T | FAR @matched junk | junk @matched FAR |
|---|---|---|---|---|---|---|---|
| stack (35 + CNN + rank) vs pinned control | 13766 / 2618 | 0.9151 / 0.8989 | 0.9174 / 0.9021 | +0.0032 (-0.0027; 7/8) | None (nan worst) | 0.90 vs 1.01 | 42.8 vs 40.6 (+2.2) |
| stack vs ctrl35 (same code path) | 13766 / 2618 | 0.9151 / 0.8989 | 0.9174 / 0.9021 | +0.0032 (-0.0027; 7/8) | None (nan worst) | 0.90 vs 1.01 | 42.8 vs 40.6 (+2.2) |
| ctrl35 vs pinned control (code-path check) | 13766 / 2618 | 0.9151 / 0.8989 | 0.9151 / 0.8989 | +0.0000 (+0.0000; 0/8) | 0.03 (0.76 worst) | 1.01 vs 1.01 | 41.0 vs 40.6 (+0.4) |
| blend 50/50 vs pinned control (same seeds) | 13766 / 2618 | 0.9151 / 0.8989 | 0.9221 / 0.9067 | +0.0078 (+0.0010; 8/8) | 0.09 (0.95 worst) | 0.65 vs 1.01 | 46.8 vs 40.6 (+6.2) |

gate (stack vs pinned control): AUC fail, junk fail, rule fail -> **FAIL**

| animal | rows | reals | control | blend | stack |
|---|---|---|---|---|---|
| bla12 | 2079 | 541 | 0.9097 | 0.9192 | 0.9151 |
| bla16 | 1056 | 169 | 0.9109 | 0.9028 | 0.9113 |
| bla21 | 312 | 106 | 0.8570 | 0.9039 | 0.8825 |
| bla36 | 4724 | 690 | 0.8958 | 0.9014 | 0.8975 |
| bla37 | 3507 | 783 | 0.9151 | 0.9176 | 0.9165 |
| bla8 | 2088 | 329 | 0.9053 | 0.9150 | 0.9143 |

