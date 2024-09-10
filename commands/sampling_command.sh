#!/bin/bash

# export LD_LIBRARY_PATH=~/miniforge3/envs/ctm/lib

# GAIA
# ssh gaia
# salloc --partition=project142-a100-v2 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=32 --time=60 --account=project142

# source /etc/profile.d/modules.sh
# module load singularity/3.5.3
# module load openmpi/3.1.6
# singularity run --nv /sample/container/NGC/tensorflow2/nvcr.io-nvidia-tensorflow.22.11-tf2-py3.sif
# cd /home/fp084243/EighthArticle/consistency_models-main_ver3
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

# MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"

MODEL_FLAGS="--data_name=cifar10 --microbatch=256 --batch_size=256 --global_batch_size=256
--class_cond=False --eval_interval=1 --save_interval=1000 --num_classes=10 
--eval_batch=512 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100
--diffusion_weight_schedule=bridge_karras --pred_mode=ve --inner_parametrization=edm
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

--model_path=/root/code/results/0908/cifar10/no_diff_bridge_or_bridgesampling_DMUniform/comb/ema0_csr0_1iter_contri_ddpm_pp_heun_solver_g1_bf16/model125000.pt

--training_mode=ctm 
--eval_num_samples=5000
--batch_size=2048 
--device_id=0 
--stochastic_seed=False 
--save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True 
--sampler=contri_ddpm_pp
--save_png=True
--ref_path=/root/data/FID_stats/cifar10-32x32.npz 
"

# --model_path=/root/code/results/0910/cifar10/no_diff_bridge_or_bridgeSampling/comb/ema0_csr0_1iter_contri_ddpm_pp_no_solver_g1_bf16/ema_0.99993_015000.pt
# --model_path=/root/code/results/0909/cifar10/no_diff_bridge/comb/ema0_csr0_1iter_contri_ddpm_pp_cm_no_solver_g1_bf16/ema_0.99993_015000.pt


# CUDA_VISIBLE_DEVICES=0 nohup mpiexec -n 1 python image_sample.py $MODEL_FLAGS --class_cond=True --num_classes=1000 --out_dir /home/fp084243/EighthArticleExperimentalResults/ImageNet64/random_samples_to_Jesse --model_path=/home/fp084243/EighthArticleExperimentalResults/ImageNet64/GAN/fine_tune_ver3/M20_w1/ema_0.999_049000.pt --training_mode=ctm --class_cond=True --eval_num_samples=6400 --batch_size=800 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True --sampler=exact --sampling_steps=1 > log.txt 2>&1 &
mpiexec --allow-run-as-root -n 1 python /root/code/code/image_sample.py $MODEL_FLAGS --out_dir /root/code/results/sampling/0909/cifar10/1h --sampling_steps=1
mpiexec --allow-run-as-root -n 1 python /root/code/code/image_sample.py $MODEL_FLAGS --out_dir /root/code/results/sampling/0909/cifar10/4h --sampling_steps=4


# * This is an example of sampling command.
# Here, you could use few commands.
# - sampler: exact, gamma, gamma_multistep, onestep, ...
#   - To obtain CM samples, you put --training_mode=cm --sampler=onestep --sampling_steps=1
#   - To obtain EDM samples, you put --training_mode=edm --sampler=heun --sampling_steps=40
#   - To obtain CTM samples (NFE n), you put --training_mode=ctm --sampler=exact --sampling_steps=n
#   - To obtain Bridge CTM samples (NFE n), you put --training_mode=ctm --sampler=hybrid --sampling_steps=n
# - If stochastic_seed=False, then you can generate samples with same seeds.