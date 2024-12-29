# model/t5_model.py

from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Model:
    def __init__(self, model_name='t5-small', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

    def save_checkpoint(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_checkpoint(self, path):
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model.to(self.device)