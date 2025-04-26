#!/usr/bin/env python
import os
import json
import argparse
from collections import Counter, defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViltProcessor, ViltForQuestionAnswering
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# ─── SETUP ──────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download("punkt", quiet=True)


class VizWizDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor, answer2id, max_length=40):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.answer2id = answer2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["image"])
        try:
            image = Image.open(img_path).convert("RGB").resize((384, 384))
        except Exception as e:
            raise RuntimeError(f"Error opening {img_path}: {e}")

        # raw reference answer (majority vote; or 'unanswerable')
        answers = [
            a["answer"].strip().lower()
            for a in sample.get("answers", [])
            if a.get("answer", "").strip()
        ]
        ref_answer = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"

        # map to label id (fall back to 'unanswerable')
        label = self.answer2id.get(ref_answer, self.answer2id["unanswerable"])

        # prepare model inputs (only question text)
        question = sample["question"].strip()
        inputs = self.processor(
            image,
            question,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        # squeeze out batch dim
        for k in ["input_ids", "attention_mask", "pixel_values"]:
            inputs[k] = inputs[k].squeeze(0)

        # attach ground-truth info for evaluation
        inputs["label"] = label
        inputs["ref_answer"] = ref_answer
        return inputs


def get_answer_type(answer: str):
    if answer == "unanswerable":
        return "unanswerable"
    if answer in ("yes", "no"):
        return "yes/no"
    if answer.replace(".", "", 1).isdigit():
        return "number"
    return "other"


def evaluate(model, processor, dataset, id2answer):
    smoothing_fn = SmoothingFunction().method1
    N = len(dataset)

    bleu_scores = []
    total_correct = 0
    type_counts = defaultdict(int)
    type_correct = defaultdict(int)

    print("\n🧠 Evaluating ViLT on VizWiz...\n")

    model.eval()
    with torch.no_grad():
        for idx in range(N):
            sample = dataset[idx]

            # send inputs to device
            inputs = {
                k: sample[k].unsqueeze(0).to(model.device)
                for k in ("input_ids", "attention_mask", "pixel_values")
            }

            # model forward
            logits = model(**inputs).logits
            pred_idx = int(logits.argmax(dim=-1).item())

            # decode strings
            pred_str = id2answer.get(pred_idx, "unanswerable")
            ref_str = sample["ref_answer"]

            # BLEU-1
            bleu = sentence_bleu(
                [word_tokenize(ref_str)],
                word_tokenize(pred_str),
                weights=(1, 0, 0, 0),
                smoothing_function=smoothing_fn
            )
            bleu_scores.append(bleu)

            # accuracy
            correct = (pred_str.strip().lower() == ref_str.strip().lower())
            if correct:
                total_correct += 1

            # per-type tallies based on raw JSON
            t = get_answer_type(ref_str)
            type_counts[t] += 1
            if correct:
                type_correct[t] += 1

    overall_acc = total_correct / N if N > 0 else 0.0
    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0

    # print results
    print(f"\n✅ Overall Accuracy: {overall_acc:.4f} ({total_correct}/{N})")
    print(f"📝 Average BLEU-1 Score: {avg_bleu:.4f}\n")
    print("🎯 Accuracy by Answer Type:")
    for t in ("unanswerable", "yes/no", "number", "other"):
        cnt = type_counts[t]
        corr = type_correct[t]
        acc = corr / cnt if cnt > 0 else 0.0
        print(f" - {t:12s}: {corr}/{cnt} = {acc:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViLT VQA on VizWiz.")
    parser.add_argument("--test_image_dir", type=str, required=True,
                        help="Directory of test images.")
    parser.add_argument("--test_annotations", type=str, required=True,
                        help="JSON file with test annotations.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned ViLT model.")
    args = parser.parse_args()

    # device setup
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # load model & processor
    model = ViltForQuestionAnswering.from_pretrained(args.model_path).to(device)
    processor = ViltProcessor.from_pretrained(args.model_path)

    # build model label mappings (for decoding predictions)
    answer2id = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    id2answer = {int(k): v.lower() for k, v in model.config.id2label.items()}

    # ensure 'unanswerable' exists
    if "unanswerable" not in answer2id:
        next_id = max(answer2id.values()) + 1
        answer2id["unanswerable"] = next_id
        id2answer[next_id] = "unanswerable"

    # prepare dataset (uses raw JSON for type counts)
    dataset = VizWizDataset(
        args.test_image_dir,
        args.test_annotations,
        processor,
        answer2id,
    )

    # run eval
    evaluate(model, processor, dataset, id2answer)


if __name__ == "__main__":
    main()