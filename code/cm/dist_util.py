"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(device_id):
    """
    Setup a distributed process group.
    """
    from mpi4py import MPI
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{(MPI.COMM_WORLD.Get_rank() + device_id) % GPUS_PER_NODE}"
    #os.environ["CUDA_VISIBLE_DEVICES"] = f"{(MPI.COMM_WORLD.Get_rank()) % GPUS_PER_NODE}"
    #os.environ["CUDA_VISIBLE_DEVICES"] = f"0,1,3,4"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    print("hostname, port: ", hostname, port)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def setup_dist_without_MPI(device_id):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
    os.environ["NCCL_SOCKET_NTHREADS"] = "8"

    world_size = os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    rank = os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    local_rank = os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    print("!!: ", world_size, rank, local_rank)
    world_size = int(world_size)
    rank = int(rank)
    local_rank = int(local_rank)

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    hostname = [hostname]
    #os.environ["MASTER_ADDR"] = hostname[0]
    num = 9
    port = str(65535 - num)
    os.environ["MASTER_PORT"] = port  # str(port[0])
    print("hostname, port, rank: ", hostname, port, rank)
    if backend == 'gloo':
        dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)
    else:
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    '''while flag:
        port = str(65535 - num)
        os.environ["MASTER_PORT"] = port#str(port[0])
        print("hostname, port: ", rank, hostname, port)
        try:
            dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
            flag = False
        except:
            num += 1
            flag = True'''
    assert dist.get_rank() == rank
    _device = th.device("cuda", local_rank) if th.cuda.is_available() else th.device("cpu")
    th.cuda.set_device(_device)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    from mpi4py import MPI
    chunk_size = 2**30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
