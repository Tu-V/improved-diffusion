"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

def _get_local_rank():
    """
    Get the node-local rank (which GPU to use on this node).
    Priority:
      1. OMPI_COMM_WORLD_LOCAL_RANK  (OpenMPI, used on Gadi)
      2. MPI_LOCALRANKID              (MPICH)
      3. LOCAL_RANK                   (PyTorch launcher)
      4. global_rank % device_count   (fallback)
    """
    for env_var in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "LOCAL_RANK"):
        val = os.environ.get(env_var)
        if val is not None:
            return int(val)
    gpus = th.cuda.device_count() if th.cuda.is_available() else 1
    return MPI.COMM_WORLD.Get_rank() % gpus

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

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
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    Uses local rank so each rank on a node maps to a unique GPU.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{_get_local_rank()}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
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
