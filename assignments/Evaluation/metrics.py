# Evaluation/metrics.py

from datasets import load_metric

def get_bleu():
    bleu = load_metric('bleu')
    def compute_bleu(references, predictions):
        # BLEU는 참조가 여러 개일 수 있으므로 리스트의 리스트로 변환
        references = [[ref.split()] for ref in references]
        predictions = [pred.split() for pred in predictions]
        return bleu.compute(predictions=predictions, references=references)['bleu']
    return compute_bleu