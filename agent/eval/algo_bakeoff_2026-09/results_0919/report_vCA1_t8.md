# vCA1 bake-off report (t8, 2026-09-19 13:27)

## session-grouped: 39 CV sessions, 3726 rows, 934 real, 50370 bootstrap rows, 5 folds over 39 groups, T = 0.04

| contender | seeds | AUC full | AUC reviewed | dAUC rev vs control (min; n+/n) | rule T | FAR@T mean / worst | junk@T | FAR at matched junk | junk at matched FAR (dpp) | min |
|---|---|---|---|---|---|---|---|---|---|---|
| xgb_deployed | 8 | 0.9221 +- 0.0023 | 0.9206 | +0.0000 (+0.0000; 0/8) | 0.03 | 1.06 / 1.28 | 45.2 | 1.06 | 45.4 (+0.3) | 0 |
| lgbm_deployed | 8 | 0.9210 +- 0.0024 | 0.9195 | -0.0011 (-0.0029; 1/8) | 0.03 | 0.83 / 1.07 | 43.3 | 1.06 | 45.9 (+0.7) | 0 |
| histgb | 8 | 0.9211 +- 0.0016 | 0.9195 | -0.0010 (-0.0033; 2/8) | 0.04 | 0.76 / 0.96 | 42.5 | 1.11 | 45.2 (-0.0) | 0 |
| mlp | 8 | 0.9182 +- 0.0025 | 0.9165 | -0.0040 (-0.0069; 1/8) | None | 1.42 / 1.93 | 40.0 | 1.79 | 36.2 (-9.0) | 2 |
| ens_mean | 8 | 0.9222 +- 0.0022 | 0.9207 | +0.0001 (-0.0008; 5/8) | 0.03 | 0.90 / 1.07 | 43.9 | 1.03 | 46.2 (+1.0) | 0 |

## Decision rule (pre-registered in README.md)

- **ens_mean**: null -- session: dAUC +0.0001 (5/8), junk@matchedFAR +1.0 pp
- **histgb**: null -- session: dAUC -0.0010 (2/8), junk@matchedFAR -0.0 pp
- **lgbm_deployed**: null -- session: dAUC -0.0011 (1/8), junk@matchedFAR +0.7 pp
- **mlp**: null -- session: dAUC -0.0040 (1/8), junk@matchedFAR -9.0 pp
