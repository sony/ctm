#!/bin/bash
#$-l rt_AF=1
#$ -l h_rt=31:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8
module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load intel-mpi/2021.8

cat $SGE_JOB_HOSTLIST > ./${JOB_ID}_hostfile
HOST=${HOSTNAME:0:5}
NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=8 # 4 for V, 8 for A
MPIOPTS="-n ${NUM_GPUS_PER_NODE}" # -x MASTER_ADDR=${HOSTNAME}" # --map-by ppr:${NUM_GPUS_PER_NODE}:node

export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export MASTER_PORT="29145"

MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --start_ema=0.999 --gan_different_augment=True --eval_interval=500 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=10 --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --use_MPI=True --num_workers=16"
CKPT_FLAGS="--resume_checkpoint=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/ImageNet64/fine_tune_ver3/M20_w1/model021000.pt --ref_path=/home/acf15618av/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=/home/acf15618av/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/groups/gce50978/dataset/imagenet_dir/train"

mpiexec -n 8 python3.11 cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/ImageNet64/fine_tune_ver3/M20_w0.75 --num_heun_step=20 --discriminator_weight=0.75 --gan_specific_time=True --microbatch=11 --global_batch_size=2112 --lr=0.000008