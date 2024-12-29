# model/parallelism/ddp.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(local_rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)

def apply_ddp(model, local_rank):
    model = DDP(model, device_ids=[local_rank])
    print("Distributed Data Parallelism 적용됨")
    return model