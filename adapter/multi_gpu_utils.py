import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

def distribute_model(model, num_gpus):
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    return model.cuda()

def setup_distributed(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    torch.distributed.destroy_process_group()
