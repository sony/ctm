#!/bin/bash

# MODEL_FLAGS="--data_name=imagenet64 --microbatch=18 --global_batch_size=1440 --lr=0.0004 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
# CKPT_FLAGS="--out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440 --dm_sample_path_seed_42=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/DM/samples_ver2 --ref_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/home/ubuntu/dataset/ImageNet/train"

# SET guidance_scale > 0 for using doob's h-transform!

MODEL_FLAGS="--data_name=cifar10 --microbatch=64 --batch_size=256 --global_batch_size=256 
--lr=0.0004 --class_cond=False --eval_interval=1000 --save_interval=1000 --num_classes=10 
--eval_batch=256 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100
--diffusion_weight_schedule=karras --pred_mode=ve --inner_parametrization=cm
--skip_final_ctm_step=False --use_fp16=False
--guidance_scale=0.5 --loss_norm=pseudo-huber
--ema_rate=0.9999
--start_ema=0.999
"
# --loss_norm=pseudo-huber --condition_mode=concat
# --teacher_model_path='' --self_learn=True
# --diffusion_weight_schedule=bridge_karras

# DATASET_NAME="cifar10"
# EXP_DATE=0518
# EXP_NAME="hybrid_ddbm"

DATASET_NAME="cifar10"
EXP_DATE=0527
EXP_NAME="hybrid_teacher"

CKPT_FLAGS="--out_dir /root/code/results/${EXP_DATE}/${DATASET_NAME}/${EXP_NAME}
--dm_sample_path_seed_42=/root/code/results/${EXP_DATE}/${DATASET_NAME}/${EXP_NAME}/heun_18_seed_42_ver2
--ref_path=/root/data/FID_stats/cifar10-32x32.npz 
--data_dir=/root/data/cifar10/ 
--eval_fid=True 
--save_png=True
--eval_sampler=hybrid
--eval_num_samples=10000
"
# --large_nfe_eval=True 

# CKPT_FLAGS="--out_dir /root/code/results/0503/cifar10 --data_dir=/root/data/cifar10/"

# export OMPI_COMM_WORLD_SIZE=1
# export OMPI_COMM_WORLD_RANK=0
# export OMPI_COMM_WORLD_LOCAL_RANK=0

export OMPI_COMM_WORLD_SIZE=8
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0

export NUM_PROC=4
export CUDA_IDX=4

# next exp: use hybrid sampler, but with churn step ratio > 0 
# (the problem cld have been caused cz of the ratio being eq to 0 )

export NCCL_P2P_DISABLE=1

mpiexec -n ${NUM_PROC} python /root/code/code/cm_train.py \
--device_id=${CUDA_IDX} \
$MODEL_FLAGS $CKPT_FLAGS

# CUDA_VISIBLE_DEVICES=4 mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS
# mpiexec -n 5 --allow-run-as-root python -m mpi4py.bench helloworld
