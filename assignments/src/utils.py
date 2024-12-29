# src/utils.py

import os
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(model, tokenizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Checkpoint saved at {path}")

def load_model(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

def preprocess_data(tokenizer, dataset_name='wmt16', subset='de-en', max_source_length=128, max_target_length=128):
    dataset = load_dataset(dataset_name, subset)
    
    def tokenize_function(examples):
        inputs = examples['en']
        targets = examples['de']
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

def get_dataloader(tokenized_datasets, split='train', batch_size=16, shuffle=True):
    dataloader = DataLoader(tokenized_datasets[split], batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train_model(model, dataloader, optimizer, scheduler, num_epochs, device, checkpoint_dir):
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}")
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

            loop.set_postfix(loss=loss.item())

        # 체크포인트 저장
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        save_checkpoint(model, None, checkpoint_path)  # Tokenizer는 별도 저장하지 않음

    print("훈련 완료!")