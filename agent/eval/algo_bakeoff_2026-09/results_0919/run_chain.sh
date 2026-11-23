#!/bin/bash
# vCA1 refresh chain (2026-09-19): member 0 -> members 1,2 -> pooled (3 seeds). Sequential on one GPU.
cd "C:/code/CNMF_E_LEGACY_BIANE_CLAUDE/agent/eval/algo_bakeoff_2026-09"
export BAKEOFF_RESULTS=results_0919 BAKEOFF_CORPUS_TAG_vCA1=v3
PY="C:/Users/julia/.conda/envs/ACORN/python.exe"
"$PY" train_cnn2.py --variant fixed12+nbtrace --cells vCA1:session,vCA1:animal --seeds 8 > results_0919/r2/confirm_single.log 2> results_0919/r2/confirm_single.err
"$PY" train_cnn_pooled.py --variant fixed12+nbtrace --cells vCA1:session,vCA1:animal --seeds 3 > results_0919/r2/pooled1.log 2> results_0919/r2/pooled1.err
"$PY" train_cnn2.py --variant fixed12+nbtrace+ens3 --members 1,2 --cells vCA1:session,vCA1:animal --seeds 8 > results_0919/r2/confirm_members12.log 2> results_0919/r2/confirm_members12.err
echo CHAIN_DONE > results_0919/r2/chain_done.txt
