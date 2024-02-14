#!/bin/bash
MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
CKPT_FLAGS="--dm_sample_path_seed_42=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/DM/samples_ver2 --ref_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/home/ubuntu/dataset/ImageNet/train"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440 --microbatch=18 --global_batch_size=1440 --lr=0.0004