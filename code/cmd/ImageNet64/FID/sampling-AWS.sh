#!/bin/bash
MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
CKPT_FLAGS="--dm_sample_path_seed_42=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/DM/samples_ver2 --ref_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/home/ubuntu/dataset/ImageNet/train"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8


CUDA_VISIBLE_DEVICES=0 nohup python image_sample.py $MODEL_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440 --model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440/ema_0.999_010000.pt --class_cond=True --batch_size=800 --eval_num_samples=6400 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --use_MPI=True --device_id=0 --sampler=exact --sampling_steps=1 > log.txt 2>&1 &
nohup python image_sample.py $MODEL_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440 --model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440/ema_0.999_010000.pt --class_cond=True --batch_size=800 --eval_num_samples=25600 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --use_MPI=True --sampler=exact --sampling_steps=2 --device_id=0 > log.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python evaluations/evaluator.py /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/ctm_bs_1440/ctm_exact_sampler_1_steps_010000_itrs_0.999_ema_
