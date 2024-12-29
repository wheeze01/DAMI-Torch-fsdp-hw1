# Evaluation/evaluate.py

import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

def evaluate(model, dataloader, tokenizer, device, metric_fn):
    model.eval()
    total_loss = 0
    references = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            # 예측 생성
            generated_ids = model.generate(input_ids=batch['input_ids'].to(device),
                                          attention_mask=batch['attention_mask'].to(device),
                                          max_length=128)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            target = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(target)

    avg_loss = total_loss / len(dataloader)
    metric_score = metric_fn(references, predictions)

    return avg_loss, metric_score