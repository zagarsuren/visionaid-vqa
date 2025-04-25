#!/usr/bin/env python
import os
import json
import argparse
from collections import Counter
from PIL import Image
import pytesseract
import torch
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# ─── SETUP ──────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Update as needed
nltk.download("punkt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_processor(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config.vision_config, "model_type", None) != "davit":
        config.vision_config.model_type = "davit"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    ).to(device)
    return model, processor

class VizWizDataset:
    def __init__(self, image_dir, annotation_file):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error opening image {image_path}: {e}")
        ocr_text = pytesseract.image_to_string(image).strip()
        ocr_text = " ".join(ocr_text.split()) if ocr_text else "no visible text"
        prompt = f"{sample['question'].strip()} Context: {ocr_text}"
        answers = [a["answer"].strip().lower() for a in sample.get("answers", []) if a["answer"].strip()]
        ref_answer = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"
        return {"image": image, "prompt": prompt, "label": ref_answer}

def evaluate(args):
    model, processor = load_model_and_processor(args.model_path)
    dataset = VizWizDataset(args.test_image_dir, args.test_annotations)

    predictions = []
    references = []
    bleu_scores = []
    smoothing_fn = SmoothingFunction().method1
    max_token_index = model.get_input_embeddings().weight.shape[0] - 1

    print("\n🧠 Evaluating Florence2 on VizWiz...\n")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        full_prompt = args.task_prompt + sample["prompt"]
        image = sample["image"]

        # Tokenize prompt only (truncate to safe length)
        tokenized = processor.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_token_index)
        safe_prompt = processor.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

        # Process prompt + image
        inputs = processor(text=safe_prompt, images=image, return_tensors="pt").to(device)
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=max_token_index)

        # Generate answer
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=10,
            num_beams=2
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        pred_str = processor.post_process_generation(
            generated_text,
            task=args.task_prompt,
            image_size=(image.width, image.height)
        )
        if not isinstance(pred_str, str):
            pred_str = str(pred_str)
        ref_str = sample["label"]

        predictions.append(pred_str)
        references.append(ref_str)

        bleu = sentence_bleu(
            [word_tokenize(ref_str)],
            word_tokenize(pred_str),
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing_fn
        )
        bleu_scores.append(bleu)

    accuracy = np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)])
    mean_bleu = np.mean(bleu_scores)

    print(f"\n✅ Overall Accuracy: {accuracy:.4f}")
    print(f"📝 Average BLEU-1 Score: {mean_bleu:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence2 VQA on VizWiz.")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Path to test images.")
    parser.add_argument("--test_annotations", type=str, required=True, help="Path to annotation JSON file.")
    parser.add_argument("--model_path", type=str, default="models/florence2-finetuned", help="Path to Florence2 model.")
    parser.add_argument("--task_prompt", type=str, default="Describe the scene: ", help="Prompt prefix for the model.")
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
