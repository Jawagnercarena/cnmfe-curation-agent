#!/bin/sh
# Noise-floor variants (CPU only).  Run from redteam/ after null_gauss has finished.
PY="/c/ProgramData/anaconda3/envs/valence/python.exe"
$PY -I noise_null.py --variant gauss2 --T 2392 6370 --seeds 1 2 3 4 --workers 4 --fft-workers 4
$PY -I noise_null.py --variant white --T 2392 6370 --seeds 1 2 3 --workers 4 --fft-workers 4
$PY -I noise_null.py --variant gauss --rhox 0.24 --rhoy 0.18 --T 2392 6370 --seeds 1 2 3 4 --workers 4 --fft-workers 4
$PY -I noise_null.py --variant gauss --rhox 0.32 --rhoy 0.26 --T 2392 6370 --seeds 1 2 3 4 --workers 4 --fft-workers 4
for S in 1 7 20; do
  $PY -I noise_null.py --variant u8 --level 80 --sigma $S --T 2392 6370 --seeds 1 2 3 --workers 3 --fft-workers 4
done
for L in 3 17; do
  $PY -I noise_null.py --variant poisson --level 80 --lam $L --T 2392 6370 --seeds 1 2 3 --workers 3 --fft-workers 4
done
