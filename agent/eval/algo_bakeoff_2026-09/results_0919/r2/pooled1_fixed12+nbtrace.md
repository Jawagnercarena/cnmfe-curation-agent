# Pooled BLA+vCA1 CNN vs per-area CNN — `fixed12+nbtrace`, other area x1 (2026-09-19 16:41)

CNN alone: reviewed AUC on rows both scored, pooled minus per-area, paired per seed. Blend: 50/50 with the deployed control, delta vs control (per-area -> pooled) and junk at matched false-AR / FAR at matched junk.

| cell | seeds | CNN alone per-area -> pooled | d alone (min; +) | blend d AUC per-area -> pooled | junk pp per-area -> pooled | FAR @matched junk per-area -> pooled | per animal (per-area -> pooled, CNN alone) |
|---|---|---|---|---|---|---|---|
| BLA session | (not run) | | | | | | |
| BLA animal | (not run) | | | | | | |
| vCA1 session | 3 | 0.9267 -> 0.9266 | -0.0001 (-0.0021; 2/3) | +0.0100 -> +0.0106 (3/3) | +2.7 -> +5.0 | 0.89 -> 0.93 | pnb88 0.848->0.835; pnb97 0.928->0.931; pnb98 0.932->0.933 |
| vCA1 animal | 3 | 0.9165 -> 0.9223 | +0.0058 (+0.0020; 3/3) | +0.0127 -> +0.0142 (3/3) | -1.2 -> -2.3 | 0.96 -> 1.03 | pnb88 0.837->0.814; pnb97 0.926->0.930; pnb98 0.925->0.933 |
