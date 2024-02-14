#!/bin/bash

#SBATCH --time=1:0:0
#SBATCH --output=tensorflow2-mnist-job-%j.log
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

#SBATCH --partition=project142-a100-v2
#SBATCH --cpus-per-task=32

set -uxeC

# log environment variables
JOB_DATE=$(date -u "+%Y%m%d %H:%M:%S")
HOSTNAME=$HOSTNAME
env | sort | grep SLURM || true
echo ""

# set variables
SAMPLE_ROOT_DIR_PATH="${1:-/sample/tensorflow2}"
WORKING_DIR_PATH="${2:-$HOME/sample-output/$SLURM_JOB_ID}"
LOCAL_WORKING_DIR="${3:-/local/job/$SLURM_JOB_ID}"

CODE_DIR_PATH="$SAMPLE_ROOT_DIR_PATH/training/horovod-0.25.0"
CODE_FILE_PATH="examples/tensorflow2/tensorflow2_keras_mnist.py"
CONTAINER_PATH="$SAMPLE_ROOT_DIR_PATH/container/nvcr.io-nvidia-tensorflow.21.10-tf2-py3.sif"

LOCAL_CONTAINER_DIR="$LOCAL_WORKING_DIR/container"
LOCAL_CONTAINER_FILE="$LOCAL_CONTAINER_DIR/container.sif"
LOCAL_CODE_DIR="$LOCAL_WORKING_DIR/codes"
LOCAL_CODE_FILE="$LOCAL_CODE_DIR/$CODE_FILE_PATH"

# load modules
set +x
set -uC
source /etc/profile.d/modules.sh
module load singularity/3.5.3
module load openmpi/3.1.6

# create resources dir and copy resources into local job dir
SECONDS=0
echo "start copy resources to each compute node"
set -x
mkdir -p "$WORKING_DIR_PATH"
mpirun -n "$SLURM_JOB_NUM_NODES" -npernode 1 mkdir -p "$LOCAL_CONTAINER_DIR"
mpirun -n "$SLURM_JOB_NUM_NODES" -npernode 1 mkdir -p "$LOCAL_CODE_DIR"
mpirun -n "$SLURM_JOB_NUM_NODES" -npernode 1 cp "$CONTAINER_PATH" "$LOCAL_CONTAINER_FILE"
mpirun -n "$SLURM_JOB_NUM_NODES" -npernode 1 cp -rT "$CODE_DIR_PATH" "$LOCAL_CODE_DIR"
set +x
echo "copy finished in $SECONDS sec"

# execute training
SECONDS=0
echo "start learning"
set -x
time mpirun -wdir "$LOCAL_WORKING_DIR" \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x NCCL_SOCKET_IFNAME=ib -x NCCL_DEBUG=INFO \
    singularity exec \
    --nv "$LOCAL_CONTAINER_FILE" \
    python \
    "$LOCAL_CODE_FILE"
set +x
echo "learning finished in $SECONDS sec"

# save checkpoints (one node only)
mpirun -n 1 -npernode 1 cp "$LOCAL_WORKING_DIR"/*.h5 "$WORKING_DIR_PATH"
