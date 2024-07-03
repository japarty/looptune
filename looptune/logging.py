import torch

def get_cuda_memory(device_no=0):
    returned = {}

    gpu_stats = torch.cuda.get_device_properties(device_no)

    returned['name'] = gpu_stats.name
    returned['device_no'] = device_no
    returned['total_memory'] = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    returned['reserved'] = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

    return returned