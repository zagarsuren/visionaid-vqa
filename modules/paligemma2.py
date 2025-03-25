# modules/paligemma.py

import torch
import time
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

class PaliGemmaModel:
    def __init__(self, model_path="./models/local_paligemma"):
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tried to use MPS but it returned MPS backend out of memory error
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, image, question):
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=40)

        end_time = time.time()
        inference_time = end_time - start_time

        answer = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        return answer, inference_time, gpu_memory
