#!/bin/bash
#SBATCH --time=1440
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=sampling-job-%j.log

source /etc/profile.d/modules.sh
module load singularity/3.5.3
module load openmpi/3.1.6
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

singularity exec /sample/pytorch/container/nvcr.io-nvidia-pytorch.21.10-py3.sif python -m pip install --user einops
MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
mpiexec -n 1 singularity exec --nv /sample/pytorch/container/nvcr.io-nvidia-pytorch.21.10-py3.sif python3 image_sample.py $MODEL_FLAGS --class_cond=True --num_classes=1000 --out_dir /home/fp084243/EighthArticleExperimentalResults/ImageNet64/GAN/fine_tune_ver3/M20_w1 --model_path=/home/fp084243/EighthArticleExperimentalResults/ImageNet64/GAN/fine_tune_ver3/M20_w1/ema_0.999_028000.pt --training_mode=ctm --class_cond=True --eval_num_samples=8000 --batch_size=1000 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --use_MPI=False --sampler=gamma --sampling_steps=2