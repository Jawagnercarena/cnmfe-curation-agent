# Pooled BLA+vCA1 CNN vs per-area CNN — `fixed12+nbtrace`, other area x1 (2026-09-08 20:27)

CNN alone: reviewed AUC on rows both scored, pooled minus per-area, paired per seed. Blend: 50/50 with the deployed control, delta vs control (per-area -> pooled) and junk at matched false-AR / FAR at matched junk.

| cell | seeds | CNN alone per-area -> pooled | d alone (min; +) | blend d AUC per-area -> pooled | junk pp per-area -> pooled | FAR @matched junk per-area -> pooled | per animal (per-area -> pooled, CNN alone) |
|---|---|---|---|---|---|---|---|
| BLA session | 3 | 0.9083 -> 0.9114 | +0.0032 (+0.0017; 3/3) | +0.0101 -> +0.0109 (3/3) | +5.3 -> +6.0 | 0.55 -> 0.51 | bla12 0.917->0.918; bla16 0.881->0.897; bla21 0.914->0.914; bla36 0.900->0.903; bla37 0.913->0.918; bla8 0.915->0.917 |
| BLA animal | 3 | 0.9027 -> 0.9074 | +0.0047 (+0.0034; 3/3) | +0.0140 -> +0.0156 (3/3) | +7.4 -> +9.5 | 0.50 -> 0.48 | bla12 0.915->0.917; bla16 0.884->0.891; bla21 0.920->0.918; bla36 0.897->0.902; bla37 0.908->0.918; bla8 0.917->0.915 |
| vCA1 session | 3 | 0.9295 -> 0.9246 | -0.0048 (-0.0081; 0/3) | +0.0128 -> +0.0117 (3/3) | +5.2 -> +0.8 | 0.90 -> 1.10 | pnb88 0.861->0.833; pnb97 0.930->0.932; pnb98 0.952->0.946 |
| vCA1 animal | 3 | 0.9150 -> 0.9214 | +0.0064 (+0.0025; 3/3) | +0.0162 -> +0.0200 (3/3) | +1.1 -> -0.8 | 0.90 -> 1.04 | pnb88 0.820->0.815; pnb97 0.926->0.931; pnb98 0.948->0.946 |
