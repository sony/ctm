#!/bin/bash
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
##SBATCH --cpus-per-task=10 # if you use V100
#SBATCH --output=nnabla-cifar10-single-task-job-%j.log

source /etc/profile.d/modules.sh
module load singularity/3.5.3
module load openmpi/3.1.6

export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8


mpiexec -n 8 singularity exec --nv "/sample/container/NGC/tensorflow2/nvcr.io-nvidia-tensorflow.22.11-tf2-py3.sif" \
        python "/home/fp084243/EighthArticle/consistency_models-main_ver3/cm_train.py \
        --data_name=imagenet64 --class_cond=True --num_classes=1000 --eval_interval=11 --eval_num_samples=2000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=2 \
        --teacher_model_path=/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pkl --data_dir=/group/project142/dataset/imagenet/imagenet_dir/train \
        --out_dir /home/fp084243/EighthArticleExperimentalResults/ImageNet64/test --microbatch=16 --global_batch_size=16 --model_path=/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm-imagenet-64x64-cond-adm.pkl" \
        --context cudnn