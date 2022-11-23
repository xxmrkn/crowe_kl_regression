#!/bin/bash

PROJECT=''
NUM_GPU=1
NUM_CORE=4
SIMG=''

mkdir -p "${PROJECT}/slurm"

NODE='cl-viking'
sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
  --wrap="singularity exec --nv -B /win/salmon/user ${SIMG} \
    python src/train.py \
    --sign '1121_for_paper_results' \
    --datalist 10 \
    --model 'VisionTransformer_Base16' \
    --epoch 300 \
    --fold 4 \
    --image_size 224 \
    --batch_size 32 \
    --valid_batch_size 8 \
    --lr 5e-5 \
    --min_lr 5e-6 \
    --t_max 2700 \
    --t_0 25 \
    --wd 1e-4 \
    --num_classes 1 \
    --num_workers 4 \
    --num_sampling 1 \
    --optimizer 'Adam' \
    --criterion 'MAE Loss' \
    --seed 42 \
    --df_path '' \
    --datalist_path  '' \
    --image_path '' \
    --result_path ''"
# NORMAL DRR image or BONE DRR iamge
#--image_path '/win/salmon/user/masuda/project/vit_kl_crowe/20220511_DRR_with_Crowe_Kl/DRR_AP' \
#--image_path '/win/salmon/user/masuda/project/makedrr/150_Extracted_2D_DRR_944_masuda/DRR_AP' \

# t_max = epoch x batchsize
# 200 epoch t_max 1800
# 300 epoch t_max 2700
