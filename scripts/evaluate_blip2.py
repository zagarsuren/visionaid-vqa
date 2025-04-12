import os
import json
import argparse
from collections import Counter, defaultdict
from PIL import Image
import pytesseract
import torch
from torch.utils.data import Dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Adjust path if needed

# Download required NLTK resources if not already available
nltk.download('punkt')
nltk.download('punkt_tab')

class VizWizDataset(Dataset):
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

        # OCR context: extract text; fallback if empty.
        ocr_text = pytesseract.image_to_string(image).strip()
        ocr_text = " ".join(ocr_text.split()) if ocr_text else "no visible text"
        question = f"{sample['question'].strip()} Context: {ocr_text}"

        # Get the most common answer from annotations or default to "unanswerable".
        answers = [a["answer"].strip().lower() for a in sample.get("answers", []) if a["answer"].strip()]
        label = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"

        # Process the image and text using BLIP-2's processor.
        inputs = self.processor(image, question, return_tensors="pt",
                                  truncation=True, padding="max_length", max_length=self.max_length)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["label"] = label  # Store reference answer
        return inputs

def get_answer_type(answer: str):
    """Categorize the answer type for detailed reporting."""
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
    parser.add_argument("--test_image_dir", type=str, required=True, help="Directory with test images.")
    parser.add_argument("--test_annotations", type=str, required=True, help="Path to JSON annotations.")
    parser.add_argument("--model_path", type=str, required=True, help="Local folder path for BLIP-2 model weights.")
    args = parser.parse_args()

    # Sanity check for the provided model path.
    if "modes" in args.model_path:
        corrected_path = args.model_path.replace("modes", "models")
        print(f"Warning: It appears your model path contains 'modes'. Did you mean '{corrected_path}'?")
        args.model_path = corrected_path

    if not os.path.isdir(args.model_path):
        raise OSError(
            f"The specified model path '{args.model_path}' does not exist. "
            f"Please verify that the folder exists and that you provided the correct path. "
            f"For example: '/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/local_blip2'."
        )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the BLIP-2 model and processor from the local path.
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True).to(device)
    processor = Blip2Processor.from_pretrained(args.model_path, local_files_only=True)

    # Load the dataset.
    dataset = VizWizDataset(args.test_image_dir, args.test_annotations, processor)

    model.eval()
    predictions = []
    references = []
    bleu_scores = []
    answer_type_preds = defaultdict(list)
    answer_type_labels = defaultdict(list)
    smoothing_fn = SmoothingFunction().method1

    with torch.no_grad():
        for sample in dataset:
            # Prepare inputs for generation (skip the 'label' key).
            inputs = {k: sample[k].unsqueeze(0).to(device)
                      for k in sample if k != "label"}
            outputs = model.generate(**inputs)
            # Decode the generated answer.
            pred_str = processor.decode(outputs[0], skip_special_tokens=True).strip().lower()
            label_str = sample["label"].strip().lower()

            predictions.append(pred_str)
            references.append(label_str)
            answer_type = get_answer_type(label_str)
            answer_type_preds[answer_type].append(pred_str)
            answer_type_labels[answer_type].append(label_str)

            # Compute BLEU-1 score for the prediction.
            reference_tokens = [word_tokenize(label_str)]
            prediction_tokens = word_tokenize(pred_str)
            bleu = sentence_bleu(reference_tokens, prediction_tokens,
                                 weights=(1, 0, 0, 0),
                                 smoothing_function=smoothing_fn)
            bleu_scores.append(bleu)

            # Optional: Debug message for unanswerable cases.
            if label_str == "unanswerable":
                print(f"✅ Unanswerable example: pred={pred_str}, label={label_str}")

    # Compute overall exact-match accuracy.
    overall_acc = np.mean([p == l for p, l in zip(predictions, references)])
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.4f}")

    # Accuracy breakdown by answer type.
    print("\n🔍 Accuracy by Answer Type:")
    for a_type in ["yes/no", "number", "other", "unanswerable"]:
        preds = answer_type_preds[a_type]
        labels = answer_type_labels[a_type]
        if preds:
            acc = np.mean([p == l for p, l in zip(preds, labels)])
            print(f"  {a_type:>13}: {acc:.4f} ({len(labels)} samples)")
        else:
            print(f"  {a_type:>13}: No samples")

    # Compute and print the average BLEU-1 score.
    mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"\n🧠 Average BLEU-1 Score: {mean_bleu:.4f}")

if __name__ == "__main__":
    main()
