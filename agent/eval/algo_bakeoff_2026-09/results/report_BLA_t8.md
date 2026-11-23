# BLA bake-off report (t8, 2026-09-06 17:03)

## session-grouped: 81 CV sessions, 13766 rows, 2618 real, 33448 bootstrap rows, 5 folds over 81 groups, T = 0.04

| contender | seeds | AUC full | AUC reviewed | dAUC rev vs control (min; n+/n) | rule T | FAR@T mean / worst | junk@T | FAR at matched junk | junk at matched FAR (dpp) | min |
|---|---|---|---|---|---|---|---|---|---|---|
| xgb_deployed | 8 | 0.9245 +- 0.0013 | 0.9099 | +0.0000 (+0.0000; 0/8) | 0.04 | 0.76 / 0.92 | 41.4 | 0.76 | 41.6 (+0.2) | 0 |
| xgb_tuned | 3 | 0.9201 +- 0.0027 | 0.9050 | -0.0035 (-0.0047; 0/3) | None | 1.39 / 2.06 | 43.7 | 1.06 | 38.7 (-2.9) | 9 |
| xgb_winner | 8 | 0.9253 +- 0.0013 | 0.9110 | +0.0010 (+0.0002; 8/8) | 0.03 | 0.81 / 1.03 | 42.4 | 0.78 | 41.8 (+0.4) | 1 |
| lgbm_deployed | 8 | 0.9240 +- 0.0014 | 0.9092 | -0.0007 (-0.0018; 2/8) | 0.04 | 0.73 / 0.99 | 41.0 | 0.76 | 42.3 (+0.9) | 0 |
| lgbm_tuned | 3 | 0.9139 +- 0.0065 | 0.8985 | -0.0100 (-0.0154; 0/3) | None | 1.59 / 2.02 | 44.7 | 1.41 | 36.0 (-5.5) | 4 |
| lgbm_winner | 8 | 0.9204 +- 0.0013 | 0.9053 | -0.0046 (-0.0058; 0/8) | 0.05 | 0.43 / 0.53 | 34.0 | 0.81 | 41.4 (+0.0) | 0 |
| cat_default | 8 | 0.9229 +- 0.0016 | 0.9088 | -0.0011 (-0.0021; 0/8) | None | 2.70 / 2.98 | 56.5 | 1.01 | 38.2 (-3.1) | 6 |
| cat_tuned | 3 | 0.9238 +- 0.0016 | 0.9091 | +0.0006 (-0.0001; 2/3) | 0.05 | 0.48 / 0.57 | 40.0 | 0.62 | 43.8 (+2.3) | 10 |
| cat_winner | 8 | 0.9247 +- 0.0011 | 0.9101 | +0.0001 (-0.0009; 5/8) | 0.05 | 0.42 / 0.57 | 38.0 | 0.65 | 42.8 (+1.4) | 4 |
| histgb | 8 | 0.9238 +- 0.0015 | 0.9090 | -0.0009 (-0.0019; 1/8) | 0.04 | 0.73 / 0.80 | 39.9 | 0.84 | 41.0 (-0.4) | 0 |
| extratrees | 8 | 0.9160 +- 0.0007 | 0.8998 | -0.0102 (-0.0112; 0/8) | 0.1 | 0.01 / 0.04 | 22.9 | 0.72 | 42.2 (+0.8) | 1 |
| mlp | 8 | 0.9204 +- 0.0023 | 0.9056 | -0.0043 (-0.0077; 1/8) | None | 0.97 / 1.60 | 43.3 | 0.83 | 40.7 (-0.7) | 3 |
| ens_mean | 8 | 0.9261 +- 0.0014 | 0.9118 | +0.0019 (+0.0014; 8/8) | 0.03 | 0.97 / 1.07 | 44.4 | 0.75 | 41.5 (+0.1) | 0 |

## animal-grouped: 81 CV sessions, 13766 rows, 2618 real, 33448 bootstrap rows, 5 folds over 6 groups, T = 0.04

| contender | seeds | AUC full | AUC reviewed | dAUC rev vs control (min; n+/n) | rule T | FAR@T mean / worst | junk@T | FAR at matched junk | junk at matched FAR (dpp) | min |
|---|---|---|---|---|---|---|---|---|---|---|
| xgb_deployed | 8 | 0.9151 +- 0.0010 | 0.8989 | +0.0000 (+0.0000; 0/8) | 0.03 | 1.01 / 1.15 | 40.6 | 1.01 | 41.0 (+0.4) | 0 |
| xgb_tuned | 3 | 0.9134 +- 0.0019 | 0.8973 | -0.0015 (-0.0054; 2/3) | 0.03 | 0.81 / 1.03 | 34.2 | 1.17 | 37.2 (-2.9) | 8 |
| xgb_winner | 8 | 0.9174 +- 0.0010 | 0.9017 | +0.0028 (+0.0025; 8/8) | 0.03 | 0.88 / 0.99 | 40.2 | 0.93 | 41.6 (+1.0) | 1 |
| lgbm_deployed | 8 | 0.9156 +- 0.0010 | 0.8994 | +0.0005 (-0.0001; 7/8) | 0.03 | 0.89 / 1.03 | 40.1 | 0.91 | 42.2 (+1.6) | 0 |
| lgbm_tuned | 3 | 0.9018 +- 0.0140 | 0.8845 | -0.0143 (-0.0260; 1/3) | None | 1.30 / 1.49 | 36.0 | 1.68 | 31.4 (-8.7) | 4 |
| lgbm_winner | 8 | 0.9110 +- 0.0016 | 0.8944 | -0.0045 (-0.0063; 0/8) | 0.05 | 0.37 / 0.46 | 30.2 | 1.01 | 40.7 (+0.1) | 0 |
| cat_default | 8 | 0.9116 +- 0.0012 | 0.8958 | -0.0030 (-0.0038; 0/8) | None | 3.32 / 3.51 | 57.4 | 1.24 | 38.1 (-2.5) | 6 |
| cat_tuned | 3 | 0.9161 +- 0.0010 | 0.9001 | +0.0013 (+0.0004; 3/3) | 0.04 | 0.48 / 0.92 | 35.3 | 0.78 | 43.4 (+3.3) | 7 |
| cat_winner | 8 | 0.9170 +- 0.0009 | 0.9010 | +0.0022 (+0.0016; 8/8) | 0.06 | 0.39 / 0.46 | 34.9 | 0.74 | 45.0 (+4.4) | 1 |
| histgb | 8 | 0.9154 +- 0.0010 | 0.8991 | +0.0003 (-0.0006; 5/8) | 0.03 | 0.92 / 0.99 | 39.6 | 1.00 | 41.8 (+1.2) | 0 |
| extratrees | 8 | 0.9059 +- 0.0013 | 0.8880 | -0.0109 (-0.0124; 0/8) | 0.1 | 0.00 / 0.00 | 21.4 | 0.81 | 43.9 (+3.3) | 1 |
| mlp | 8 | 0.9142 +- 0.0037 | 0.8987 | -0.0002 (-0.0059; 3/8) | None | 0.83 / 1.57 | 39.0 | 0.90 | 42.5 (+1.9) | 3 |
| ens_mean | 8 | 0.9165 +- 0.0010 | 0.9006 | +0.0018 (+0.0014; 8/8) | 0.03 | 1.12 / 1.30 | 43.7 | 0.89 | 42.3 (+1.7) | 0 |

## Decision rule (pre-registered in README.md)

- **cat_default**: null -- session: dAUC -0.0011 (0/8), junk@matchedFAR -3.1 pp; animal: dAUC -0.0030 (0/8), junk@matchedFAR -2.5 pp
- **cat_tuned**: null -- session: dAUC +0.0006 (2/3), junk@matchedFAR +2.3 pp; animal: dAUC +0.0013 (3/3), junk@matchedFAR +3.3 pp
- **cat_winner**: null -- session: dAUC +0.0001 (5/8), junk@matchedFAR +1.4 pp; animal: dAUC +0.0022 (8/8), junk@matchedFAR +4.4 pp
- **ens_mean**: null -- session: dAUC +0.0019 (8/8), junk@matchedFAR +0.1 pp; animal: dAUC +0.0018 (8/8), junk@matchedFAR +1.7 pp
- **extratrees**: null -- session: dAUC -0.0102 (0/8), junk@matchedFAR +0.8 pp; animal: dAUC -0.0109 (0/8), junk@matchedFAR +3.3 pp
- **histgb**: null -- session: dAUC -0.0009 (1/8), junk@matchedFAR -0.4 pp; animal: dAUC +0.0003 (5/8), junk@matchedFAR +1.2 pp
- **lgbm_deployed**: null -- session: dAUC -0.0007 (2/8), junk@matchedFAR +0.9 pp; animal: dAUC +0.0005 (7/8), junk@matchedFAR +1.6 pp
- **lgbm_tuned**: null -- session: dAUC -0.0100 (0/3), junk@matchedFAR -5.5 pp; animal: dAUC -0.0143 (1/3), junk@matchedFAR -8.7 pp
- **lgbm_winner**: null -- session: dAUC -0.0046 (0/8), junk@matchedFAR +0.0 pp; animal: dAUC -0.0045 (0/8), junk@matchedFAR +0.1 pp
- **mlp**: null -- session: dAUC -0.0043 (1/8), junk@matchedFAR -0.7 pp; animal: dAUC -0.0002 (3/8), junk@matchedFAR +1.9 pp
- **xgb_tuned**: null -- session: dAUC -0.0035 (0/3), junk@matchedFAR -2.9 pp; animal: dAUC -0.0015 (2/3), junk@matchedFAR -2.9 pp
- **xgb_winner**: null -- session: dAUC +0.0010 (8/8), junk@matchedFAR +0.4 pp; animal: dAUC +0.0028 (8/8), junk@matchedFAR +1.0 pp
