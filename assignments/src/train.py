# src/main.py

import argparse
import torch
from utils import (
    load_config,
    load_model,
    preprocess_data,
    get_dataloader,
    train_model,
    save_checkpoint
)
import os

def main(config_path):
    # 구성 파일 로드
    config = load_config(config_path)

    data_dir = config['data_dir']
    model_name = config['model_name']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    max_source_length = config['max_source_length']
    max_target_length = config['max_target_length']
    checkpoint_dir = config['checkpoint_dir']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 및 토크나이저 로드
    model, tokenizer = load_model(model_name, device)

    # 데이터 다운로드 및 전처리
    print("데이터 다운로드 및 전처리 중...")
    tokenized_datasets = preprocess_data(
        tokenizer,
        dataset_name=config.get('dataset_name', 'wmt16'),
        subset=config.get('subset', 'de-en'),
        max_source_length=max_source_length,
        max_target_length=max_target_length
    )

    # 데이터로더 생성
    train_loader = get_dataloader(tokenized_datasets, split='train', batch_size=batch_size, shuffle=True)
    valid_loader = get_dataloader(tokenized_datasets, split='validation', batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(tokenized_datasets, split='test', batch_size=batch_size, shuffle=False)

    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 훈련 실행
    print("훈련 시작...")
    train_model(model, train_loader, optimizer, scheduler, num_epochs, device, checkpoint_dir)

    # 훈련 완료 후 체크포인트 저장
    final_checkpoint = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(model, tokenizer, final_checkpoint)
    print(f"최종 모델이 {final_checkpoint}에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T5 Model for Translation Task")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
    