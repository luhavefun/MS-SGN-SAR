#!/usr/bin/env bash

cd ..

GPU_ID=1
GROUP_ID=3

test_class='inshore'
batch_size=1
k_shot=1
restore_step=300000


CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --group ${GROUP_ID} \
    --test_class ${test_class} \
    --batch_size ${batch_size} \
    --k_shot ${k_shot} \
    --restore_step ${restore_step} \
    # --vis

# CUDA_VISIBLE_DEVICES=$GPU_ID python inference_kshot.py \
#     --group ${GROUP_ID} \
#     --test_class ${test_class} \
#     --batch_size ${batch_size} \
#     --k_shot ${k_shot} \
#     --restore_step ${restore_step}