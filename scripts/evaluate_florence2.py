import os
import json
import argparse
from collections import Counter, defaultdict
from PIL import Image
import pytesseract
import torch
from torch.utils.data import Dataset
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set tesseract path as needed (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Download required NLTK resources
nltk.download('punkt')

from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

class VizWizDataset(Dataset):
    """
    A dataset class to load VizWiz samples.
    
    Each sample includes:
     - An image loaded and resized to 384x384.
     - OCR text extracted from the image.
     - A combined question: original question plus OCR-derived context.
     - The reference answer computed as the most common answer from multiple annotations;
       if no answer is available, "unanswerable" is used.
    """
    def __init__(self, image_dir, annotation_file, processor, max_length=40):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB").resize((384, 384))

        # Extract OCR text; use a fallback if OCR returns an empty string.
        ocr_text = pytesseract.image_to_string(image).strip()
        ocr_text = " ".join(ocr_text.split()) if ocr_text else "no visible text"
        # Combine the sample question with OCR-derived context.
        question = f"{sample['question'].strip()} Context: {ocr_text}"

        # Determine the reference answer from sample answers.
        answers = [a["answer"].strip().lower() for a in sample.get("answers", []) if a["answer"].strip()]
        reference = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"

        # Process the image and text.
        inputs = self.processor(text=question, images=image, return_tensors="pt", 
                                  truncation=True, padding="max_length", max_length=self.max_length)
        # Remove the extra batch dimension.
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["reference"] = reference
        inputs["image_obj"] = image  # Save the PIL image if needed for further processing.
        return inputs

def get_answer_type(answer: str):
    if answer == "unanswerable":
        return "unanswerable"
    elif answer in ["yes", "no"]:
        return "yes/no"
    elif answer.replace('.', '', 1).isdigit():
        return "number"
    else:
        return "other"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--test_annotations", type=str, required=True, help="Path to the test annotations JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the locally fine-tuned Florence2 model")
    args = parser.parse_args()

    # Set up the device: use MPS (Apple Silicon) if available, then CUDA, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configuration and override vision_config.model_type if necessary.
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        repo_type="model"
    )
    if getattr(config.vision_config, "model_type", None) != "davit":
        print(f"Warning: Overriding vision_config.model_type from {config.vision_config.model_type} to 'davit'")
        config.vision_config.model_type = "davit"

    # Load Florence2 model and processor from the local folder.
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    # Build the dataset.
    dataset = VizWizDataset(args.test_image_dir, args.test_annotations, processor)

    # Evaluation metrics containers.
    model.eval()
    predictions = []
    references = []
    bleu_scores = []
    answer_type_preds = defaultdict(list)
    answer_type_labels = defaultdict(list)
    smoothing_fn = SmoothingFunction().method1

    with torch.no_grad():
        for sample in dataset:
            # Prepare input batch: add a batch dimension and move tensors to device.
            inputs = {k: sample[k].unsqueeze(0).to(device) for k in ["input_ids", "pixel_values"]}
            # Use Florence2 to generate an answer.
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=3
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            reference = sample["reference"]

            predictions.append(generated_text.lower())
            references.append(reference.lower())
            
            answer_type = get_answer_type(reference)
            answer_type_preds[answer_type].append(generated_text.lower())
            answer_type_labels[answer_type].append(reference.lower())

            # Compute BLEU-1 score.
            reference_tokens = [word_tokenize(reference)]
            prediction_tokens = word_tokenize(generated_text)
            bleu = sentence_bleu(reference_tokens, prediction_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_fn)
            bleu_scores.append(bleu)

    # Calculate overall accuracy (exact string match).
    overall_acc = np.mean([p == r for p, r in zip(predictions, references)])
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.4f}")

    # Accuracy broken down by answer type.
    print("\n🔍 Accuracy by Answer Type:")
    for a_type in ["yes/no", "number", "other", "unanswerable"]:
        preds = answer_type_preds[a_type]
        labels = answer_type_labels[a_type]
        if preds:
            acc = np.mean([p == l for p, l in zip(preds, labels)])
            print(f"  {a_type:>13}: {acc:.4f} ({len(labels)} samples)")
        else:
            print(f"  {a_type:>13}: No samples")

    # Print average BLEU-1 score.
    mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"\n🧠 Average BLEU-1 Score: {mean_bleu:.4f}")

if __name__ == "__main__":
    main()
