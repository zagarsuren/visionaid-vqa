import os
import json
import argparse
from collections import Counter, defaultdict
from PIL import Image
import pytesseract
import torch
from torch.utils.data import Dataset
from transformers import ViltProcessor, ViltForQuestionAnswering
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
    def __init__(self, image_dir, annotation_file, processor, answer2id, max_length=40):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.answer2id = answer2id
        self.max_length = max_length
        self.num_labels = len(answer2id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB").resize((384, 384))

        # OCR context: get text from image; use a fallback if empty
        ocr_text = pytesseract.image_to_string(image).strip()
        ocr_text = " ".join(ocr_text.split()) if ocr_text else "no visible text"
        question = f"{sample['question'].strip()} Context: {ocr_text}"

        # Get the most common answer, or default to "unanswerable"
        answers = [a["answer"].strip().lower() for a in sample.get("answers", []) if a["answer"].strip()]
        answer = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"
        label = self.answer2id.get(answer, -100)

        inputs = self.processor(image, question, return_tensors="pt", truncation=True,
                                  padding="max_length", max_length=self.max_length)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["label"] = label
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
    parser.add_argument("--test_image_dir", type=str, required=True)
    parser.add_argument("--test_annotations", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model = ViltForQuestionAnswering.from_pretrained(args.model_path).to(device)
    processor = ViltProcessor.from_pretrained(args.model_path)

    # Build answer2id and id2answer mappings from the model config
    answer2id = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    id2answer = {int(k): v.lower() for k, v in model.config.id2label.items()}

    # Ensure "unanswerable" exists in the label mapping
    if "unanswerable" not in answer2id:
        next_id = max(answer2id.values()) + 1
        answer2id["unanswerable"] = next_id
        id2answer[next_id] = "unanswerable"

    # Load dataset
    dataset = VizWizDataset(args.test_image_dir, args.test_annotations, processor, answer2id)

    # Evaluation
    model.eval()
    predictions = []
    references = []
    bleu_scores = []
    answer_type_preds = defaultdict(list)
    answer_type_labels = defaultdict(list)
    smoothing_fn = SmoothingFunction().method1

    with torch.no_grad():
        for sample in dataset:
            inputs = {k: sample[k].unsqueeze(0).to(device) for k in ["input_ids", "attention_mask", "pixel_values"]}
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            label = sample["label"]

            if label in id2answer:
                pred_str = id2answer[pred]
                label_str = id2answer[label]
                answer_type = get_answer_type(label_str)

                predictions.append(pred_str)
                references.append(label_str)
                answer_type_preds[answer_type].append(pred_str)
                answer_type_labels[answer_type].append(label_str)

                # Compute BLEU-1 score
                reference_tokens = [word_tokenize(label_str)]
                prediction_tokens = word_tokenize(pred_str)
                bleu = sentence_bleu(reference_tokens, prediction_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_fn)
                bleu_scores.append(bleu)

                # Optional: Debug unanswerable cases
                # if label_str == "unanswerable":
                #    print(f"✅ Unanswerable example: pred={pred_str}, label={label_str}")

    # Overall accuracy
    overall_acc = np.mean([p == l for p, l in zip(predictions, references)])
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.4f}")

    # Accuracy by answer type
    print("\n🔍 Accuracy by Answer Type:")
    for a_type in ["yes/no", "number", "other", "unanswerable"]:
        preds = answer_type_preds[a_type]
        labels = answer_type_labels[a_type]
        if preds:
            acc = np.mean([p == l for p, l in zip(preds, labels)])
            print(f"  {a_type:>13}: {acc:.4f} ({len(labels)} samples)")
        else:
            print(f"  {a_type:>13}: No samples")

    # BLEU score
    mean_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"\n🧠 Average BLEU-1 Score: {mean_bleu:.4f}")

if __name__ == "__main__":
    main()
