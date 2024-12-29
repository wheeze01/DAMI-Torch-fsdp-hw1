# src/main_train.py

import argparse
import yaml
import torch
from model.t5_model import T5Model
from model.parallelism.dp import apply_dp
from model.parallelism.ddp import setup_ddp, apply_ddp
from model.parallelism.fsdp import apply_fsdp
from data_loader.data_loader import WMTDataLoader
from training.train import train
from transformers import AdamW, get_linear_schedule_with_warmup

def main(config_path):
    # 구성 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']
    model_name = config['model_name']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    max_source_length = config['max_source_length']
    max_target_length = config['max_target_length']
    parallelism = config['parallelism']

    # 모델 초기화
    model = T5Model(model_name=model_name)

    device = model.device

    # 병렬 처리 적용
    if parallelism == 'dp':
        model.model = apply_dp(model.model)
    elif parallelism == 'ddp':
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        world_size = torch.cuda.device_count()
        setup_ddp(local_rank, world_size)
        model.model = apply_ddp(model.model, local_rank)
    elif parallelism == 'fsdp':
        model.model = apply_fsdp(model.model)
    else:
        print("병렬 처리 미적용")

    # 데이터 로더 초기화
    data_loader = WMTDataLoader(data_dir=data_dir, batch_size=batch_size)
    train_loader = data_loader.get_train_dataloader()

    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 훈련 실행
    train(model=model.model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs, device=device, parallelism=parallelism)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 Model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args.config)