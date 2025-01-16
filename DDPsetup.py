import torch.distributed as dist
import os

def setup(rank, world_size):
    """Sets up the process group for DistributedDataParallel (DDP)."""
    # Set environment variables required by PyTorch DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()