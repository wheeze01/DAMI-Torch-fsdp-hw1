# model/parallelism/dp.py

import torch
from torch.nn import DataParallel

def apply_dp(model):
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print("Data Parallelism 적용됨")
    else:
        print("Data Parallelism을 적용할 수 있는 GPU가 충분하지 않습니다.")
    return model