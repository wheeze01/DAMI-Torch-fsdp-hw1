# Evaluation/scripts/evaluation_script.py

import argparse
import yaml
import torch
from model.t5_model import T5Model
from data_loader.data_loader import WMTDataLoader
from Evaluation.evaluate import evaluate
from Evaluation.metrics import get_bleu

def main(config_path):
    # 구성 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']
    model_checkpoint = config['model_checkpoint']
    batch_size = config['batch_size']
    metric = config['metric']

    # 모델 로드
    model = T5Model()
    model.load_checkpoint(model_checkpoint)
    device = model.device

    # 데이터 로더 초기화
    data_loader = WMTDataLoader(data_dir=data_dir, batch_size=batch_size, shuffle=False)
    test_loader = data_loader.get_test_dataloader()

    # 평가 메트릭 설정
    if metric == 'bleu':
        metric_fn = get_bleu()
    else:
        raise ValueError("지원하지 않는 메트릭입니다.")

    # 평가 실행
    avg_loss, metric_score = evaluate(model=model.model, dataloader=test_loader, tokenizer=model.tokenizer, device=device, metric_fn=metric_fn)

    print(f"평균 손실: {avg_loss}")
    print(f"BLEU 점수: {metric_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate T5 Model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args.config)