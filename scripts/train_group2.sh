#!/usr/bin/env bash

cd ..

GPU_ID=0
GROUP_ID=0

train_class='Visakhapatnam HK Barcelone Chittaagong ShangHai-HH Singapore SH SaoPauloHV AswanDam StraitGibraltar QD Houston'
# val_class='PanamaCanal BayPlenty-Sulphur IS SaoPauloHH'
batch_size=3


CUDA_VISIBLE_DEVICES=$GPU_ID python train_frn.py \
    --group ${GROUP_ID} \
    --train_class ${train_class} \
    --batch_size ${batch_size} \
    # --val_class ${val_class}