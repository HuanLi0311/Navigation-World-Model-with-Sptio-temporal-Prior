#!/bin/bash

set -e

workdir='..'
model_names=('StreamVGGT') 

ckpt_name='checkpoints'
model_weights="/home/ma-user/modelarts/user-job-dir/ys/project/InfiniteVGGT/checkpoints.pth"


# datasets=('sintel')
datasets=('tum_90')

for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose_evaluation/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29602 ./eval/pose_evaluation/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        # --model_update_type "$model_name"
done
done