#!/bin/bash

GAIA
ssh gaia
salloc --partition=project142-a100-v2 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=32 --time=60 --account=project142

source /etc/profile.d/modules.sh
module load singularity/3.5.3
module load openmpi/3.1.6
singularity run --nv /sample/container/NGC/tensorflow2/nvcr.io-nvidia-tensorflow.22.11-tf2-py3.sif
cd /home/fp084243/EighthArticle/consistency_models-main_ver3
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"

CUDA_VISIBLE_DEVICES=0 nohup mpiexec -n 1 python image_sample.py $MODEL_FLAGS --class_cond=True --num_classes=1000 --out_dir /home/fp084243/EighthArticleExperimentalResults/ImageNet64/random_samples_to_Jesse --model_path=/home/fp084243/EighthArticleExperimentalResults/ImageNet64/GAN/fine_tune_ver3/M20_w1/ema_0.999_049000.pt --training_mode=ctm --class_cond=True --eval_num_samples=6400 --batch_size=800 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True --sampler=exact --sampling_steps=1 > log.txt 2>&1 &


* This is an example of sampling command.
Here, you could use few commands.
- sampler: exact, gamma, gamma_multistep, onestep, ...
  - To obtain CM samples, you put --training_mode=cm --sampler=onestep --sampling_steps=1
  - To obtain EDM samples, you put --training_mode=edm --sampler=heun --sampling_steps=40
  - To obtain CTM samples (NFE n), you put --training_mode=ctm --sampler=exact --sampling_steps=n
- If stochastic_seed=False, then you can generate samples with same seeds.