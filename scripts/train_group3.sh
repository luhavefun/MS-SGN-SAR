#!/usr/bin/env bash

cd ..

GPU_ID=0
GROUP_ID=3

train_class='Visakhapatnam HK Barcelone Chittaagong ShangHai-HH Singapore SH SaoPauloHV PanamaCanal BayPlenty-Sulphur IS SaoPauloHH'
# val_class='AswanDam StraitGibraltar QD Houston'
batch_size=3
save_interval=10000
lr=0.01
max_steps=300001
# resume_step=100000
# start_count=100000


CUDA_VISIBLE_DEVICES=$GPU_ID python train_frn.py \
    --group ${GROUP_ID} \
    --train_class ${train_class} \
    --batch_size ${batch_size} \
    --max_steps ${max_steps} \
    --save_interval ${save_interval}
    --lr ${lr}
    # --resume_step ${resume_step} \
    # --start_count ${start_count} \
    # --resume 