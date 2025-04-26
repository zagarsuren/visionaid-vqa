#!/usr/bin/env python
import os
import json
import argparse
from collections import Counter, defaultdict

from PIL import Image

import torch
import numpy as np

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# ─── SETUP ──────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download("punkt", quiet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_answer_type(answer: str):
    """Classify each reference answer into: unanswerable, yes/no, number, or other."""
    if answer == "unanswerable":
        return "unanswerable"
    if answer in ("yes", "no"):
        return "yes/no"
    if answer.replace(".", "", 1).isdigit():
        return "number"
    return "other"


def load_model_and_processor(model_path: str):
    # load config and ensure the Florence2 vision_config if needed
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config.vision_config, "model_type", None) != "davit":
        config.vision_config.model_type = "davit"

    # processor handles text+image inputs
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # load the custom Florence2 generation class via the CausalLM entrypoint
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    ).to(device)

    return model, processor


class VizWizDataset:
    def __init__(self, image_dir: str, annotation_file: str):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error opening {img_path}: {e}")

        # only the question text
        prompt = sample["question"].strip()

        # majority‐vote reference answer (or 'unanswerable')
        answers = [
            a["answer"].strip().lower()
            for a in sample.get("answers", [])
            if a.get("answer", "").strip()
        ]
        ref_answer = Counter(answers).most_common(1)[0][0] if answers else "unanswerable"

        return {"image": image, "prompt": prompt, "label": ref_answer}


def evaluate(args):
    model, processor = load_model_and_processor(args.model_path)
    dataset = VizWizDataset(args.test_image_dir, args.test_annotations)

    smoothing_fn = SmoothingFunction().method1
    max_token_idx = model.get_input_embeddings().weight.shape[0] - 1

    # tracking
    bleu_scores = []
    type_counts = defaultdict(int)
    type_correct = defaultdict(int)
    total_correct = 0

    print("\n🧠 Evaluating Florence2 on VizWiz...\n")

    for idx in range(len(dataset)):
        sample = dataset[idx]
        question = sample["prompt"]
        image = sample["image"]
        ref = sample["label"]

        # full prompt
        full_prompt = args.task_prompt + question

        # truncate safely
        tok = processor.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=max_token_idx
        )
        safe_prompt = processor.tokenizer.decode(tok["input_ids"][0], skip_special_tokens=True)

        # prepare inputs
        inputs = processor(text=safe_prompt, images=image, return_tensors="pt").to(device)
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=max_token_idx)

        # generate
        gen_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=40,
            num_beams=2,
            early_stopping=True,
        )

        # decode & post-process
        dec = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        post = processor.post_process_generation(
            dec,
            task=args.task_prompt,
            image_size=(image.width, image.height)
        )
        # if a dict is returned, extract its sole value; else stringify
        if isinstance(post, dict):
            pred = next(iter(post.values()))
        else:
            pred = str(post)

        # BLEU-1
        bleu = sentence_bleu(
            [word_tokenize(ref)],
            word_tokenize(pred),
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing_fn
        )
        bleu_scores.append(bleu)

        # accuracy
        correct = (pred.strip().lower() == ref.strip().lower())
        if correct:
            total_correct += 1

        # per-type
        t = get_answer_type(ref)
        type_counts[t] += 1
        if correct:
            type_correct[t] += 1

    N = len(dataset)
    overall_acc = total_correct / N if N > 0 else 0.0
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

    print(f"\n✅ Overall Accuracy: {overall_acc:.4f} ({total_correct}/{N})")
    print(f"📝 Average BLEU-1 Score: {avg_bleu:.4f}")

    print("\n🎯 Accuracy by Answer Type:")
    for t in ("unanswerable", "yes/no", "number", "other"):
        cnt = type_counts[t]
        corr = type_correct[t]
        acc = corr / cnt if cnt > 0 else 0.0
        print(f" - {t:12s}: {corr}/{cnt} = {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence2 VQA on VizWiz.")
    parser.add_argument(
        "--test_image_dir", type=str, required=True,
        help="Directory of test images."
    )
    parser.add_argument(
        "--test_annotations", type=str, required=True,
        help="JSON file with test annotations."
    )
    parser.add_argument(
        "--model_path", type=str, default="models/florence2-finetuned",
        help="Path to the Florence2 model."
    )
    parser.add_argument(
        "--task_prompt", type=str, default="Describe the scene: ",
        help="Prompt prefix for the model."
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()