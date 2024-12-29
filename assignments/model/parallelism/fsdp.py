# model/parallelism/fsdp.py

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def apply_fsdp(model):
    model = FSDP(model)
    print("Fully Sharded Data Parallelism 적용됨")
    return model