#!/bin/bash
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8
MODEL_FLAGS="--num_heun_step=20 --discriminator_weight=1.0 --gan_specific_time=True --microbatch=11 --global_batch_size=4224 --lr=0.000008 --data_name=imagenet64 --class_cond=True --start_ema=0.999 --gan_different_augment=True --eval_interval=500 --save_interval=1000 --num_classes=1000 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=10 --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --use_MPI=True --num_workers=32"
CKPT_FLAGS="--out_dir /home/ubuntu/EighthArticleExperimentalResults/ImageNet64/fine_tune_ver3/M20_w1_v2 --ref_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz --resume_checkpoint=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/fine_tune_ver3/M20_w1/model025000.pt --teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/ImageNet64/author_ckpt/edm_imagenet64_ema.pt --data_dir=/home/ubuntu/dataset/ImageNet/train"

mpiexec -n 8 python cm_train.py $MODEL_FLAGS $CKPT_FLAGS
