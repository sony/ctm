#!/bin/bash
MODEL_FLAGS="--data_name=imagenet64 --class_cond=True --start_ema=0.999 --gan_different_augment=True --eval_interval=1000 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=10 --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --num_workers=32"
CKPT_FLAGS="--ref_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --resume_checkpoint=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/GAN_bs_264/model015000.pt --teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/home/ubuntu/dataset/ImageNet/train"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/GAN_bs_1056 --num_heun_step=10 --gan_specific_time=True --microbatch=11 --global_batch_size=1056 --lr=0.000008