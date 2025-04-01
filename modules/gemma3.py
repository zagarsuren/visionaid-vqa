import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

class Gemma3Model:
    def __init__(self, model_path="./models/local_gemma3"):
        # Load processor and model from local directory
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def generate_answer(self, image, question):
        # Format the input as chat messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]
            }
        ]

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32)

        input_len = inputs["input_ids"].shape[-1]

        # Free GPU memory and track inference time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        end_time = time.time()
        inference_time = end_time - start_time

        generated = outputs[0][input_len:]
        answer = self.processor.decode(generated, skip_special_tokens=True)

        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB

        return answer, inference_time, gpu_memory