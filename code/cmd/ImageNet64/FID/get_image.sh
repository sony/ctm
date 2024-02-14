#!/bin/bash

qrsh -g gce50978 -l rt_F=1 -l h_rt=12:00:00
source /etc/profile.d/modules.sh

module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load intel-mpi/2021.8

python3.11 get_image.py --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/ImageNet_val/ --data_dir=/groups/gce50978/dataset/imagenet_dir/val --use_MPI=True --global_batch_size=1000