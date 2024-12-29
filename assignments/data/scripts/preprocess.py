# data/scripts/preprocess.py

import os
from datasets import load_dataset
from transformers import T5Tokenizer
import argparse

def preprocess(data_dir, tokenizer_name, max_source_length, max_target_length):
    # WMT 데이터셋 로드 (예: WMT16 영어-독일어)
    dataset = load_dataset('wmt16', 'de-en')

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        inputs = [ex for ex in examples['en']]
        targets = [ex for ex in examples['de']]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding='max_length')

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 저장 디렉토리 생성
    processed_dir = os.path.join(data_dir, 'processed', 'WMT16_de_en')
    os.makedirs(processed_dir, exist_ok=True)

    # 데이터셋 저장
    tokenized_datasets.save_to_disk(processed_dir)
    print(f"Processed data saved to {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WMT Dataset")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save processed data')
    parser.add_argument('--tokenizer_name', type=str, default='t5-small', help='Tokenizer name')
    parser.add_argument('--max_source_length', type=int, default=128, help='Max source sequence length')
    parser.add_argument('--max_target_length', type=int, default=128, help='Max target sequence length')

    args = parser.parse_args()
    preprocess(args.data_dir, args.tokenizer_name, args.max_source_length, args.max_target_length)