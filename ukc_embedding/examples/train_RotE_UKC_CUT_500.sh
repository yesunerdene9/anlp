#!/bin/bash
cd .. 
# source set_env.sh. UKC_CUT_1_hyp_simlex.  UKC_CUT_1_hyp_t
python run.py \
            --dataset UKC_CUT_1_hyp_t_simlex \
            --model RotE \
            --rank 500 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 200 \
            --patience 5 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size 300 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --dropout 0 \
            # --train_c 

cd examples/
