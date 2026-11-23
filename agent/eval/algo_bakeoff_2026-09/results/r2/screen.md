# Round 2 screening (2026-09-07 22:53)

CNN alone = reviewed AUC on the rows both CNNs scored, paired per seed; blend = 50/50 with the control, delta vs the control (round-1 blend -> variant blend) and junk at the control's matched false-AR; stack = 35 + CNN + rank XGBoost vs the pinned control when computed.

| variant | cell | seeds | CNN alone r1 -> v | d alone (min; +) | blend d AUC r1 -> v | blend junk pp r1 -> v | stack d AUC (junk pp) | best ep | cap hits | inner val rows | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| valfix | BLA session | 3 | 0.8979 -> 0.8957 | -0.0022 (-0.0103; 2/3) | +0.0054 -> +0.0057 (3/3) | +4.3 -> +4.0 | - | 16 | 0/15 | 4535-6053 | 76 |
| valfix | vCA1 animal | 3 | 0.9027 -> 0.9142 | +0.0115 (+0.0035; 3/3) | +0.0117 -> +0.0172 (3/3) | -2.5 -> +0.3 | - | 11 | 0/9 | 4438-7215 | 50 |
| fixed12 | BLA session | 3 | 0.8979 -> 0.9014 | +0.0035 (+0.0023; 3/3) | +0.0054 -> +0.0067 (3/3) | +4.3 -> +5.9 | - | 11 | 15/15 | 0-0 | 46 |
| fixed12 | vCA1 animal | 3 | 0.9027 -> 0.9131 | +0.0104 (-0.0030; 2/3) | +0.0117 -> +0.0143 (3/3) | -2.5 -> +2.2 | - | 11 | 9/9 | 0-0 | 32 |
| nbtrace | BLA session | 3 | 0.8979 -> 0.9054 | +0.0075 (+0.0049; 3/3) | +0.0054 -> +0.0101 (3/3) | +4.3 -> +5.6 | - | 11 | 4/15 | 1202-1880 | 75 |
| nbtrace | vCA1 animal | 3 | 0.9027 -> 0.8916 | -0.0111 (-0.0258; 0/3) | +0.0117 -> +0.0120 (3/3) | -2.5 -> +1.2 | - | 7 | 0/9 | 27-264 | 38 |
| patch48 | BLA session | 3 | 0.8979 -> 0.9005 | +0.0026 (+0.0020; 3/3) | +0.0054 -> +0.0065 (3/3) | +4.3 -> +3.9 | - | 14 | 4/15 | 1202-1880 | 87 |
| patch48 | vCA1 animal | 3 | 0.9027 -> 0.8845 | -0.0182 (-0.0302; 0/3) | +0.0117 -> +0.0061 (3/3) | -2.5 -> -6.4 | - | 9 | 1/9 | 27-264 | 46 |
| fixed12+nbtrace | BLA session | 3 | 0.8979 -> 0.9083 | +0.0104 (+0.0086; 3/3) | +0.0054 -> +0.0101 (3/3) | +4.3 -> +5.3 | - | 11 | 15/15 | 0-0 | 45 |
| fixed12+nbtrace | vCA1 animal | 3 | 0.9027 -> 0.9150 | +0.0123 (+0.0007; 3/3) | +0.0117 -> +0.0162 (3/3) | -2.5 -> +1.1 | - | 11 | 9/9 | 0-0 | 31 |
| ens3 | BLA session | 3 | 0.8979 -> 0.9016 | +0.0036 (+0.0013; 3/3) | +0.0054 -> +0.0064 (3/3) | +4.3 -> +6.1 | - | 11 | 0/15 | 1202-1880 | 162 |
| ens3 | vCA1 animal | 3 | 0.9027 -> 0.9156 | +0.0129 (+0.0042; 3/3) | +0.0117 -> +0.0169 (3/3) | -2.5 -> +3.1 | - | 8 | 0/9 | 27-264 | 116 |

Selection rule: CNN-alone delta >= 0 on both cells and >= +0.003 on at least one.

- `valfix`: not carried
- `fixed12`: CARRIED FORWARD
- `nbtrace`: not carried
- `patch48`: not carried
- `fixed12+nbtrace`: CARRIED FORWARD
- `ens3`: CARRIED FORWARD
