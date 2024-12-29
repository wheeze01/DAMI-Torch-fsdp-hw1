# src/data_loader/data_loader.py

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

class WMTDataLoader:
    def __init__(self, data_dir, batch_size, shuffle=True):
        processed_dir = f"{data_dir}/processed/WMT16_de_en"
        self.dataset = load_from_disk(processed_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_train_dataloader(self):
        train_dataset = self.dataset['train']
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_valid_dataloader(self):
        valid_dataset = self.dataset['validation']
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        test_dataset = self.dataset['test']
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)