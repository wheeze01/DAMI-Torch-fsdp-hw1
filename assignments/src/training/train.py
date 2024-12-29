# src/training/train.py

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

def train(model, dataloader, optimizer, scheduler, num_epochs, device, parallelism=None):
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        # 에포크 끝날 때마다 체크포인트 저장
        model.module.save_checkpoint(f"model/checkpoints/epoch_{epoch+1}.pth") if parallelism == 'dp' else model.save_checkpoint(f"model/checkpoints/epoch_{epoch+1}.pth")

    print("훈련 완료!")