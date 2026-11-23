# vCA1 bake-off report (t8, 2026-09-06 16:54)

## session-grouped: 24 CV sessions, 2068 rows, 483 real, 50370 bootstrap rows, 5 folds over 24 groups, T = 0.04

| contender | seeds | AUC full | AUC reviewed | dAUC rev vs control (min; n+/n) | rule T | FAR@T mean / worst | junk@T | FAR at matched junk | junk at matched FAR (dpp) | min |
|---|---|---|---|---|---|---|---|---|---|---|
| xgb_deployed | 8 | 0.9252 +- 0.0026 | 0.9227 | +0.0000 (+0.0000; 0/8) | None | 1.16 / 1.45 | 46.0 | 1.16 | 50.8 (+4.8) | 0 |
| xgb_tuned | 3 | 0.9198 +- 0.0021 | 0.9171 | -0.0051 (-0.0092; 0/3) | None | 1.04 / 1.45 | 40.9 | 1.31 | 39.2 (-6.6) | 7 |
| xgb_winner | 8 | 0.9246 +- 0.0032 | 0.9221 | -0.0006 (-0.0020; 1/8) | None | 1.27 / 1.66 | 42.9 | 1.53 | 42.6 (-3.4) | 0 |
| lgbm_deployed | 8 | 0.9241 +- 0.0019 | 0.9216 | -0.0011 (-0.0030; 2/8) | None | 1.01 / 1.24 | 43.6 | 1.14 | 49.6 (+3.6) | 0 |
| lgbm_tuned | 3 | 0.9186 +- 0.0041 | 0.9159 | -0.0064 (-0.0087; 0/3) | None | 2.35 / 2.90 | 49.7 | 1.93 | 36.4 (-9.4) | 3 |
| lgbm_winner | 8 | 0.9235 +- 0.0034 | 0.9210 | -0.0017 (-0.0035; 0/8) | None | 3.31 / 3.73 | 58.2 | 1.66 | 38.5 (-7.5) | 0 |
| cat_default | 8 | 0.9261 +- 0.0034 | 0.9236 | +0.0009 (-0.0020; 6/8) | None | 3.31 / 4.14 | 60.3 | 1.79 | 37.8 (-8.1) | 6 |
| cat_tuned | 3 | 0.9233 +- 0.0010 | 0.9206 | -0.0017 (-0.0051; 1/3) | None | 1.17 / 1.45 | 38.5 | 1.38 | 39.1 (-6.7) | 7 |
| cat_winner | 8 | 0.9233 +- 0.0032 | 0.9205 | -0.0021 (-0.0067; 1/8) | None | 0.93 / 1.66 | 40.5 | 1.42 | 45.2 (-0.8) | 1 |
| histgb | 8 | 0.9232 +- 0.0019 | 0.9206 | -0.0021 (-0.0036; 0/8) | None | 0.85 / 1.04 | 42.1 | 1.19 | 47.8 (+1.8) | 0 |
| extratrees | 8 | 0.9125 +- 0.0014 | 0.9095 | -0.0132 (-0.0162; 0/8) | 0.08 | 0.10 / 0.21 | 12.9 | 1.92 | 39.8 (-6.2) | 1 |
| mlp | 8 | 0.9198 +- 0.0064 | 0.9168 | -0.0058 (-0.0136; 1/8) | None | 1.99 / 2.69 | 43.4 | 2.28 | 32.6 (-13.4) | 4 |
| ens_mean | 8 | 0.9279 +- 0.0025 | 0.9255 | +0.0028 (+0.0017; 8/8) | None | 1.19 / 1.45 | 47.5 | 1.16 | 49.4 (+3.4) | 0 |

## animal-grouped: 24 CV sessions, 2068 rows, 483 real, 50370 bootstrap rows, 3 folds over 3 groups, T = 0.04

| contender | seeds | AUC full | AUC reviewed | dAUC rev vs control (min; n+/n) | rule T | FAR@T mean / worst | junk@T | FAR at matched junk | junk at matched FAR (dpp) | min |
|---|---|---|---|---|---|---|---|---|---|---|
| xgb_deployed | 8 | 0.9101 +- 0.0016 | 0.9071 | +0.0000 (+0.0000; 0/8) | None | 0.91 / 1.24 | 44.5 | 0.91 | 47.3 (+2.8) | 0 |
| xgb_tuned | 3 | 0.9005 +- 0.0183 | 0.8971 | -0.0098 (-0.0288; 1/3) | None | 0.97 / 1.24 | 30.1 | 2.90 | 30.1 (-14.8) | 4 |
| xgb_winner | 8 | 0.9099 +- 0.0018 | 0.9070 | -0.0002 (-0.0025; 4/8) | 0.03 | 0.88 / 1.04 | 40.5 | 1.11 | 43.1 (-1.4) | 0 |
| lgbm_deployed | 8 | 0.9098 +- 0.0010 | 0.9069 | -0.0003 (-0.0028; 3/8) | 0.03 | 1.09 / 1.45 | 42.0 | 1.19 | 42.2 (-2.2) | 0 |
| lgbm_tuned | 3 | 0.8933 +- 0.0214 | 0.8898 | -0.0170 (-0.0417; 1/3) | None | 2.00 / 3.11 | 32.3 | 3.80 | 24.1 (-20.9) | 1 |
| lgbm_winner | 8 | 0.9037 +- 0.0000 | 0.9006 | -0.0065 (-0.0095; 0/8) | None | 2.48 / 2.48 | 53.1 | 1.24 | 43.8 (-0.6) | 0 |
| cat_default | 8 | 0.9055 +- 0.0027 | 0.9023 | -0.0048 (-0.0082; 1/8) | None | 3.83 / 4.55 | 58.9 | 1.48 | 39.2 (-5.3) | 4 |
| cat_tuned | 3 | 0.8852 +- 0.0251 | 0.8811 | -0.0258 (-0.0517; 0/3) | None | 1.10 / 1.66 | 28.5 | 5.18 | 28.0 (-16.9) | 4 |
| cat_winner | 8 | 0.9098 +- 0.0015 | 0.9067 | -0.0004 (-0.0038; 3/8) | None | 0.98 / 1.24 | 41.7 | 1.19 | 39.9 (-4.6) | 1 |
| histgb | 8 | 0.9098 +- 0.0000 | 0.9069 | -0.0003 (-0.0032; 4/8) | 0.06 | 0.83 / 0.83 | 40.0 | 0.83 | 48.1 (+3.6) | 0 |
| extratrees | 8 | 0.8961 +- 0.0009 | 0.8926 | -0.0145 (-0.0179; 0/8) | 0.1 | 0.00 / 0.00 | 9.3 | 2.12 | 34.2 (-10.3) | 1 |
| mlp | 8 | 0.8958 +- 0.0062 | 0.8923 | -0.0148 (-0.0250; 0/8) | None | 2.51 / 5.18 | 46.1 | 2.20 | 31.9 (-12.6) | 3 |
| ens_mean | 8 | 0.9114 +- 0.0014 | 0.9084 | +0.0013 (-0.0004; 7/8) | 0.03 | 0.93 / 1.24 | 46.0 | 0.85 | 47.7 (+3.3) | 0 |

## Decision rule (pre-registered in README.md)

- **cat_default**: null -- session: dAUC +0.0009 (6/8), junk@matchedFAR -8.1 pp; animal: dAUC -0.0048 (1/8), junk@matchedFAR -5.3 pp
- **cat_tuned**: null -- session: dAUC -0.0017 (1/3), junk@matchedFAR -6.7 pp; animal: dAUC -0.0258 (0/3), junk@matchedFAR -16.9 pp
- **cat_winner**: null -- session: dAUC -0.0021 (1/8), junk@matchedFAR -0.8 pp; animal: dAUC -0.0004 (3/8), junk@matchedFAR -4.6 pp
- **ens_mean**: null -- session: dAUC +0.0028 (8/8), junk@matchedFAR +3.4 pp; animal: dAUC +0.0013 (7/8), junk@matchedFAR +3.3 pp
- **extratrees**: null -- session: dAUC -0.0132 (0/8), junk@matchedFAR -6.2 pp; animal: dAUC -0.0145 (0/8), junk@matchedFAR -10.3 pp
- **histgb**: null -- session: dAUC -0.0021 (0/8), junk@matchedFAR +1.8 pp; animal: dAUC -0.0003 (4/8), junk@matchedFAR +3.6 pp
- **lgbm_deployed**: null -- session: dAUC -0.0011 (2/8), junk@matchedFAR +3.6 pp; animal: dAUC -0.0003 (3/8), junk@matchedFAR -2.2 pp
- **lgbm_tuned**: null -- session: dAUC -0.0064 (0/3), junk@matchedFAR -9.4 pp; animal: dAUC -0.0170 (1/3), junk@matchedFAR -20.9 pp
- **lgbm_winner**: null -- session: dAUC -0.0017 (0/8), junk@matchedFAR -7.5 pp; animal: dAUC -0.0065 (0/8), junk@matchedFAR -0.6 pp
- **mlp**: null -- session: dAUC -0.0058 (1/8), junk@matchedFAR -13.4 pp; animal: dAUC -0.0148 (0/8), junk@matchedFAR -12.6 pp
- **xgb_tuned**: null -- session: dAUC -0.0051 (0/3), junk@matchedFAR -6.6 pp; animal: dAUC -0.0098 (1/3), junk@matchedFAR -14.8 pp
- **xgb_winner**: null -- session: dAUC -0.0006 (1/8), junk@matchedFAR -3.4 pp; animal: dAUC -0.0002 (4/8), junk@matchedFAR -1.4 pp
