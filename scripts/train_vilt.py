import argparse
import os
import json
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    ViltProcessor,
    ViltForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
    TrainerCallback,
)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

"""
Finetune a ViLT model on the VizWiz Visual Question Answering dataset.

This script defines:
- VizWizDataset: a PyTorch Dataset for image-question-answer samples.
- StepLoggerCallback: a TrainerCallback for logging metrics and saving plots.
- main(): argument parsing, dataset and model setup, and training loop.

Usage:
    python vilt_vqa_finetune.py \
        --train_image_dir PATH_TO_TRAIN_IMAGES \
        --val_image_dir PATH_TO_VAL_IMAGES \
        --train_annotations PATH_TO_TRAIN_JSON \
        --val_annotations PATH_TO_VAL_JSON
"""

class VizWizDataset(Dataset):
    """
    PyTorch Dataset for the VizWiz VQA dataset.

    Each sample consists of:
    - An image (resized to 384x384 RGB).
    - A question string.
    - A one-hot encoded label vector corresponding to the most common answer.
    """
    def __init__(self, image_dir, annotation_file, processor, answer2id, max_length=40):
        # Load annotation JSON (list of sample dicts)
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.answer2id = answer2id
        self.max_length = max_length
        self.num_labels = len(answer2id)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve and process a single sample by index.

        Returns a dict suitable for the HuggingFace Trainer:
        - pixel_values: image tensor
        - input_ids, attention_mask: tokenized question
        - labels: one-hot answer tensor
        """
        sample = self.samples[idx]

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB").resize((384, 384))
        question = sample["question"]

        # Gather all non-empty answers, find the most common
        answers_list = sample.get("answers", [])
        answer_candidates = [ans["answer"].strip().lower() for ans in answers_list if ans.get("answer", "").strip()]
        answer = Counter(answer_candidates).most_common(1)[0][0] if answer_candidates else "unanswerable"
        label = self.answer2id.get(answer, -100)

        # Create one-hot label vector; -100 indicates ignored sample
        one_hot = torch.zeros(self.num_labels, dtype=torch.float)
        if label != -100:
            one_hot[label] = 1.0

        # Tokenize question and process image jointly
        inputs = self.processor(
            image, question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove batch dimension (Trainer expects single-sample tensors)
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        # Attach labels for VQA training
        inputs["labels"] = one_hot
        return inputs

class StepLoggerCallback(TrainerCallback):
    """
    TrainerCallback that logs training and evaluation metrics to TensorBoard,
    and saves loss/accuracy plots at the end of training.
    """    
    def __init__(self):
        # Directory for TensorBoard logs
        self.writer = SummaryWriter(log_dir="runs/vilt-vqa")
        self.train_steps = []
        self.train_loss = []
        self.eval_steps = []
        self.eval_loss = []
        self.eval_accuracy = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            # Record training loss
            if "loss" in logs:
                self.train_steps.append(step)
                self.train_loss.append(logs["loss"])
                self.writer.add_scalar("Train/Loss", logs["loss"], step)
            # Record evaluation loss    
            if "eval_loss" in logs:
                self.eval_steps.append(step)
                self.eval_loss.append(logs["eval_loss"])
                self.writer.add_scalar("Eval/Loss", logs["eval_loss"], step)
            # Record evaluation accuracy
            if "eval_accuracy" in logs:
                self.writer.add_scalar("Eval/Accuracy", logs["eval_accuracy"], step)
                self.eval_accuracy.append(logs["eval_accuracy"])

    def on_train_end(self, args, state, control, **kwargs):
        """
        At training end, close TensorBoard writer and save plots to 'outputs/'.
        """        
        self.writer.close()

        os.makedirs("outputs", exist_ok=True)

        # Training Loss Plot
        if self.train_steps and self.train_loss:
            plt.figure()
            plt.plot(self.train_steps, self.train_loss, label="Train Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.savefig("outputs/train_loss.png")
            plt.close()

        # Validation Loss Plot
        if self.eval_steps and self.eval_loss:
            plt.figure()
            plt.plot(self.eval_steps, self.eval_loss, label="Validation Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Validation Loss")
            plt.legend()
            plt.savefig("outputs/val_loss.png")
            plt.close()

        # Validation Accuracy Plot
        if self.eval_steps and self.eval_accuracy:
            plt.figure()
            plt.plot(self.eval_steps, self.eval_accuracy, label="Validation Accuracy")
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.title("Validation Accuracy")
            plt.legend()
            plt.savefig("outputs/val_accuracy.png")
            plt.close()

def main():
    """
    Entry point for training.
    Parses arguments, initializes processor, model, datasets, and trainer,
    then trains and saves the fine-tuned model.
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--train_annotations", type=str, required=True)
    parser.add_argument("--val_annotations", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--output_dir", type=str, default="models/vilt_finetuned_vizwiz")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    # Select device (MPS for Mac, CPU otherwise)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load processor and model, move model to device
    processor = ViltProcessor.from_pretrained(args.model_name)
    model = ViltForQuestionAnswering.from_pretrained(args.model_name).to(device)
    # Map label text to ID (lowercased)
    id2label = model.config.id2label
    answer2id = {v.lower(): int(k) for k, v in id2label.items()}

    # Initialize datasets
    train_dataset = VizWizDataset(args.train_image_dir, args.train_annotations, processor, answer2id)
    val_dataset = VizWizDataset(args.val_image_dir, args.val_annotations, processor, answer2id)

    # Metric computation for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        valid_preds, valid_true = [], []
        for pred, lab in zip(preds, labels):
            if lab.sum() == 0:
                continue
            valid_preds.append(pred.item())
            valid_true.append(lab.argmax().item())
        if not valid_true:
            return {"accuracy": 0.0}
        accuracy = (np.array(valid_preds) == np.array(valid_true)).mean().item()
        return {"accuracy": accuracy}

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_dir="runs/vilt-vqa",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        report_to=["tensorboard"]
    )
    # Initialize Trainer with custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    trainer.add_callback(StepLoggerCallback())
    trainer.train()
    # Save final model and processor
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()