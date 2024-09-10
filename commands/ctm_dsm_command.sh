#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/root/code/
export LD_LIBRARY_PATH=/root/miniforge3/envs/ctm/lib

# MODEL_FLAGS="--data_name=imagenet64 --microbatch=18 --global_batch_size=1440 --lr=0.0004 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
# CKPT_FLAGS="--out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440 --dm_sample_path_seed_42=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/DM/samples_ver2 --ref_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/home/ubuntu/dataset/ImageNet/train"

# # EMA is CTM Like + LPIPS
# MODEL_FLAGS="--data_name=cifar10 --microbatch=64 --batch_size=64 --global_batch_size=64
# --class_cond=False --eval_interval=1000 --save_interval=5000 --num_classes=10 
# --eval_batch=2048 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100
# --diffusion_weight_schedule=karras --pred_mode=ve --inner_parametrization=edm
# --skip_final_ctm_step=False --use_fp16=True --use_bf16=False
# --guidance_scale=0. --loss_norm=lpips
# --teacher_model_path='' --self_learn=True --self_learn_iterative=True --condition_mode=concat --weight_schedule=ict
# --qpart_loss=False --do_xT_precond=False --use_x0_as_denoised_in_solver=True
# --ema_rate=0.999 --start_ema=0.999 --num_heun_step=1 --churn_step_ratio=0
# --is_I2I=False --gamma=1. --lr=0.0002 
# --compute_ema_fids=True --sample_s_strategy=sigma_s_is_zero --schedule_sampler=ict
# --diffusion_training=True --target_ema_mode=fixed --scale_mode=ict_exp --start_scales=10 --end_scales=1280
# "

# Comb:
MODEL_FLAGS="--data_name=cifar10 --microbatch=128 --batch_size=128 --global_batch_size=128
--class_cond=False --eval_interval=1000 --save_interval=5000 --num_classes=10 
--eval_batch=4096 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100
--diffusion_weight_schedule=karras --pred_mode=ve --inner_parametrization=edm
--skip_final_ctm_step=False --use_fp16=False --use_bf16=True
--guidance_scale=0. --loss_norm=pseudo-huber
--teacher_model_path='' --self_learn=True --self_learn_iterative=True --condition_mode=concat --weight_schedule=ict
--qpart_loss=False --do_xT_precond=True --use_x0_as_denoised_in_solver=True
--ema_rate=0.99993 --start_ema=0. --num_heun_step=1 --churn_step_ratio=0
--is_I2I=False --gamma=1. --lr=0.0002 --total_training_steps=400000
--compute_ema_fids=True --sample_s_strategy=sigma_s_is_zero --schedule_sampler=ict
--target_ema_mode=fixed --scale_mode=ict_exp --start_scales=10 --end_scales=1280
--traditional_ctm=False --apply_adaptive_weight=False
--diffusion_training=True
"
# 0.99993

# # CM:
# MODEL_FLAGS="--data_name=cifar10 --microbatch=128 --batch_size=128 --global_batch_size=128
# --class_cond=False --eval_interval=1000 --save_interval=5000 --num_classes=10 
# --eval_batch=4096 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100
# --diffusion_weight_schedule=karras --pred_mode=ve --inner_parametrization=edm
# --skip_final_ctm_step=False --use_fp16=False --use_bf16=True
# --guidance_scale=0. --loss_norm=pseudo-huber
# --teacher_model_path='' --self_learn=True --self_learn_iterative=True --condition_mode=concat --weight_schedule=ict
# --qpart_loss=False --do_xT_precond=True --use_x0_as_denoised_in_solver=True
# --ema_rate=0.99993 --start_ema=0. --num_heun_step=1 --churn_step_ratio=0
# --is_I2I=False --gamma=1. --lr=0.0002 --total_training_steps=400000
# --compute_ema_fids=True --sample_s_strategy=sigma_s_is_zero --schedule_sampler=ict
# --target_ema_mode=fixed --scale_mode=ict_exp --start_scales=10 --end_scales=1280
# --traditional_ctm=False --apply_adaptive_weight=False
# --diffusion_training=False
# "

# 23rd aug TODO: --apply_adaptive_weight=True but try making it false.
# # TRADITIONAL CTM:
# MODEL_FLAGS="--data_name=cifar10 --microbatch=64 --batch_size=64 --global_batch_size=64
# --lr=0.0004 --class_cond=False --eval_interval=1000 --save_interval=5000 --num_classes=10 
# --eval_batch=2048 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100
# --diffusion_weight_schedule=karras --pred_mode=ve --inner_parametrization=edm
# --skip_final_ctm_step=False --use_fp16=False --use_bf16=False
# --guidance_scale=0. --loss_norm=lpips
# --teacher_model_path='' --self_learn=True --self_learn_iterative=False --condition_mode=concat --weight_schedule=uniform
# --qpart_loss=False --do_xT_precond=False --use_x0_as_denoised_in_solver=True
# --ema_rate=0.999 --start_ema=0.999 --num_heun_step=17 --churn_step_ratio=0
# --is_I2I=False --gamma=1. --lr=0.0004 --diffusion_schedule_sampler=halflognormal
# --compute_ema_fids=True --sample_s_strategy=uniform --schedule_sampler=uniform
# --diffusion_training=True --target_ema_mode=fixed --scale_mode=fixed --start_scales=18 --end_scales=18 --traditional_ctm=True
# "

# 22nd Aug NOTE For CTM: 
# --teacher_model_path='' --self_learn=True --self_learn_iterative=False --condition_mode=concat --weight_schedule=uniform 
# --gamma=0. --loss_norm=lpips --schedule_sampler=uniform --diffusion_schedule_sampler=halflognormal
# --start_ema=0.999 ema_rate=0.999 --scale_mode=fixed --lr=0.0004

# ORG: --eval_batch=512 --compute_ema_fids=False also, remove ema_rate=0.0 command.
# 7/4: Next exp: weight_sched = uniform, and self_learn_iterative = True (<- this shld make it similar to gctm)
# --use_bf16=False
# --loss_norm=pseudo-huber
# --teacher_model_path= --self_learn=True --self_learn_iterative=True --condition_mode=concat
# --diffusion_weight_schedule=bridge_karras --weight_schedule=ict
# --ema_rate="0.0" --start_ema=0.0
# --do_xT_precond=True --gamma=1.
# --sample_s_strategy=smallest --schedule_sampler=ict --use_milstein_method=True
# --sampling_steps=4

# 7/31: Set self_learn to False, and self_learn_iterative to False as well.
# DATASET_NAME="cifar10"
# EXP_DATE=0518
# EXP_NAME="hybrid_ddbm"

# DATASET_NAME="cifar10_qPartloss"
# DATASET_NAME="cifar10"
DATASET_NAME="cifar10"
EXP_DATE=0910
SAMPLER=contri_ddpm_pp
SOLVER=no_solver
# CM: contri_ddpm_pp_cm
# COMB: contri_ddpm_pp

# EXP_NAME="bridge/ema0_iterative_contri_sampler"
# EXP_NAME="bridge3/ema0_csr0_1iter_contri_sampler2_contri_solver_g1_ictScaleMode_f16"
# EXP_NAME="bridge3/lpips_emaCTMlike_csr0_1iter_contri_sampler2_heun_solver_g1_ictScaleMode_f16"
# EXP_NAME="bridge3/total100k_lpips_ema0_csr0_1iter_contri_sampler2_heun_solver_g1_ictScaleMode_f16"
# EXP_NAME="bridge/ema0_csr0_1iter_contri_sampler2_heun_solver_g1_bf16"
# EXP_NAME="no_diff_bridge/cm/ema0_csr0_1iter_${SAMPLER}_${SOLVER}_g1_bf16"
EXP_NAME="no_diff_bridge/comb/ema0_csr0_1iter_${SAMPLER}_${SOLVER}_g1_bf16"
# EXP_NAME="no_diff_bridge_or_bridgeSampling/comb/ema0_csr0_1iter_${SAMPLER}_${SOLVER}_g1_bf16"

# EXP_NAME="test/cm"

CKPT_FLAGS="--out_dir /root/code/results/${EXP_DATE}/${DATASET_NAME}/${EXP_NAME}
--dm_sample_path_seed_42=/root/code/results/${EXP_DATE}/${DATASET_NAME}/${EXP_NAME}/heun_18_seed_42_ver2
--ref_path=/root/data/FID_stats/cifar10-32x32.npz 
--data_dir=/root/data/cifar10/ 
--eval_fid=True
--save_png=True
--eval_sampler=${SAMPLER}
--eval_num_samples=5000
"
# --eval_sampler=contri_sampler2
# --eval_sampler=bridge_gamma_multistep # CTM
# --eval_num_samples=50000
# --large_nfe_eval=True 

# CKPT_FLAGS="--out_dir /root/code/results/0503/cifar10 --data_dir=/root/data/cifar10/"

# export OMPI_COMM_WORLD_SIZE=1
# export OMPI_COMM_WORLD_RANK=0
# export OMPI_COMM_WORLD_LOCAL_RANK=0

# export OMPI_COMM_WORLD_SIZE=8
# export OMPI_COMM_WORLD_RANK=0
# export OMPI_COMM_WORLD_LOCAL_RANK=0

NUM_PROC=1
CUDA_IDX=0

# next exp: use hybrid sampler, but with churn step ratio > 0 
# (the problem cld have been caused cz of the ratio being eq to 0 )
# export OMPI_MCA_opel_cuda_support=true

export NCCL_P2P_DISABLE=1


# OMP_PROC_BIND=CLOSE OMP_SCHEDULE=STATIC GOMP_CPU_AFFINITY="10-28"

# for gpus 4~7:
# OMP_NUM_THREAD=28 GOMP_CPU_AFFINITY="84-111" OMP_PROC_BIND=CLOSE OMP_SCHEDULE=STATIC \

mpiexec -n ${NUM_PROC} --allow-run-as-root python /root/code/code/cm_train.py --device_id=${CUDA_IDX} ${MODEL_FLAGS} ${CKPT_FLAGS}

# CUDA_VISIBLE_DEVICES=4 mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS
# mpiexec --allow-run-as-root -n 5 python -m mpi4py.bench helloworld
# mpiexec --allow-run-as-root --mca btl vader,self -n 5 python -m mpi4py.bench helloworld