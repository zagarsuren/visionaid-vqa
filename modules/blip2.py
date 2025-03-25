import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2Model:
    def __init__(self, model_path="./models/local_blip2"):
        self.processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tried to use MPS but it returned MPS backend out of memory error
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=40)
        answer = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return answer
