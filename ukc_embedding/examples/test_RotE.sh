#!/bin/bash
cd .. 
# source set_env.sh
python test.py \
            --model_dir hyp/logs/12_01/UKC_CUT_1_hyp_t/RotE_12_25_20/model_checkpoint/model_20251201_122520_best.pt 

cd examples/
