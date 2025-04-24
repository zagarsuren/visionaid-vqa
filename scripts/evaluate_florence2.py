#!/usr/bin/env python
import os
import json
import argparse
from collections import defaultdict, Counter
from PIL import Image
import pytesseract
import torch
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# Disable tokenizer parallelism warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Adjust path if necessary.

# Download required NLTK data.
nltk.download('punkt')

def load_model_and_processor(model_path):
    """
    Load the Florence2 model and processor using Auto classes with trust_remote_code=True.
    Also adjust config.vision_config.model_type to 'davit' if needed.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config.vision_config, "model_type", None) != "davit":
        config.vision_config.model_type = "davit"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    ).to(device)
    return model, processor

class VizWizDataset:
    """
    Loads VizWiz samples from the JSON file.
    For each sample, an image is opened, OCR is applied, and a combined prompt (question+OCR context)
    is formed. The reference answer is chosen as the most common answer (or "unanswerable").
    """
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
        # Apply OCR to get text context.
        ocr_text = pytesseract.image_to_string(image).strip()
        ocr_text = " ".join(ocr_text.split()) if ocr_text else "no visible text"
        prompt = f"{sample['question'].strip()} Context: {ocr_text}"
        answers = [a["answer"].strip().lower() for a in sample.get("answers", []) if a["answer"].strip()]
        ref_answer = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"
        return {"image": image, "prompt": prompt, "label": ref_answer}

def get_answer_type(answer: str):
    if answer == "unanswerable":
        return "unanswerable"
    elif answer in ["yes", "no"]:
        return "yes/no"
    elif answer.replace('.', '', 1).isdigit():
        return "number"
    else:
        return "other"

# Choose device: CUDA if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence2 on the VizWiz VQA task")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Directory containing test images.")
    parser.add_argument("--test_annotations", type=str, required=True, help="Path to the JSON annotations file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or identifier of the Florence2 model.")
    parser.add_argument("--task_prompt", type=str, default="Describe the scene: ",
                        help="Task prompt to prepend to the sample prompt.")
    args = parser.parse_args()

    # Load model and processor.
    model, processor = load_model_and_processor(args.model_path)
    # Load dataset.
    dataset = VizWizDataset(args.test_image_dir, args.test_annotations)

    # For evaluation metrics.
    predictions = []
    references = []
    bleu_scores = []
    answer_type_preds = defaultdict(list)
    answer_type_labels = defaultdict(list)
    smoothing_fn = SmoothingFunction().method1

    print("\nEvaluating the Florence2 model on the test set...\n")
    
    # Determine allowed total token length for this model.
    config_max = getattr(model.config, "max_position_embeddings", 512)
    # Try to get the positional offset (if available) from the language model embed_positions.
    try:
        offset = int(model.language_model.embed_positions.offset)
    except Exception:
        offset = 0
    allowed_length = config_max - offset

    # Define a small generation budget for VQA answers.
    generation_budget = 20
    # Calculate maximum allowed input tokens.
    max_input_tokens = allowed_length - generation_budget

    for idx in range(len(dataset)):
        sample = dataset[idx]
        # Prepend the task prompt.
        # Pre-tokenize and truncate text prompt if needed
        combined_prompt = args.task_prompt + sample["prompt"]
        text_inputs = processor.tokenizer(combined_prompt, return_tensors="pt")
        input_ids = text_inputs["input_ids"][0]

        if input_ids.shape[0] > max_input_tokens:
            input_ids = input_ids[:max_input_tokens]
            combined_prompt = processor.tokenizer.decode(input_ids, skip_special_tokens=True)

        # Tokenize again with image using truncated prompt
        inputs = processor(text=combined_prompt, images=sample["image"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        
        # Truncate the input tokens if necessary.
        input_length = inputs["input_ids"].shape[1]
        if input_length > max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, :max_input_tokens]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_tokens]
            input_length = max_input_tokens
        
        remaining_tokens = allowed_length - input_length
        current_gen_budget = min(generation_budget, remaining_tokens)
        # Do not pass any max_length parameter; use only max_new_tokens.
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=current_gen_budget,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        pred_str = processor.post_process_generation(
            generated_text,
            task=args.task_prompt,
            image_size=(sample["image"].width, sample["image"].height)
        )
        if not isinstance(pred_str, str):
            pred_str = str(pred_str)
        ref_str = sample["label"]

        predictions.append(pred_str)
        references.append(ref_str)
        a_type = get_answer_type(ref_str)
        answer_type_preds[a_type].append(pred_str)
        answer_type_labels[a_type].append(ref_str)

        # Compute BLEU-1 score.
        reference_tokens = [word_tokenize(ref_str)]
        prediction_tokens = word_tokenize(pred_str)
        bleu = sentence_bleu(reference_tokens, prediction_tokens, weights=(1, 0, 0, 0),
                             smoothing_function=smoothing_fn)
        bleu_scores.append(bleu)

    overall_acc = np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)])
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.4f}")
    print("\n🔍 Accuracy by Answer Type:")
    for a_type in ["yes/no", "number", "other", "unanswerable"]:
        preds = answer_type_preds[a_type]
        labels = answer_type_labels[a_type]
        if preds:
            acc = np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(preds, labels)])
            print(f"  {a_type:>13}: {acc:.4f} ({len(labels)} samples)")
        else:
            print(f"  {a_type:>13}: No samples")
    mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"\n🧠 Average BLEU-1 Score: {mean_bleu:.4f}")

if __name__ == "__main__":
    main()
