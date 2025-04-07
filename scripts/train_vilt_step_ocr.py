import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from collections import Counter
from PIL import Image
import pytesseract
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
import matplotlib.pyplot as plt

# Path for macOS with Homebrew
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

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

        # ✅ OCR extraction using PyTesseract
        ocr_text = pytesseract.image_to_string(image)
        ocr_text = " ".join(ocr_text.split())  # remove newlines and compress whitespaces

        original_question = sample["question"]
        question = f"{original_question.strip()} Context: {ocr_text}"

        answers_list = sample.get("answers", [])
        answer_candidates = [ans["answer"].strip().lower() for ans in answers_list if ans.get("answer", "").strip()]
        answer = Counter(answer_candidates).most_common(1)[0][0] if answer_candidates else "unanswerable"

        label = self.answer2id.get(answer, -100)
        one_hot = torch.zeros(self.num_labels, dtype=torch.float)
        if label != -100:
            one_hot[label] = 1.0

        inputs = self.processor(
            image, question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        inputs["labels"] = one_hot
        return inputs


# Custom callback to log training metrics at each logging step and evaluation metrics at each eval step.
class TrainEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.step_train_metrics = []  # To store training metrics per logging step (from logs)
        self.step_eval_metrics = []   # To store evaluation metrics per eval step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step} - Training metrics (log): {logs}")
        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Only logs validation metrics
        eval_record = {
            "step": state.global_step,
            "eval_loss": metrics.get("eval_loss"),
            "eval_accuracy": metrics.get("eval_accuracy"),
        }
        self.step_eval_metrics.append(eval_record)
        print(f"Step {state.global_step} - Validation metrics: {metrics}")
        return control

def main():
    parser = argparse.ArgumentParser()
    # Arguments for train, validation, (and optional test) splits.
    parser.add_argument("--train_image_dir", type=str, required=True, help="Directory containing train images")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Directory containing validation images")
    parser.add_argument("--test_image_dir", type=str, required=False, help="Directory containing test images")
    parser.add_argument("--train_annotations", type=str, required=True, help="Path to train annotation JSON file")
    parser.add_argument("--val_annotations", type=str, required=True, help="Path to validation annotation JSON file")
    parser.add_argument("--test_annotations", type=str, required=False, help="Path to test annotation JSON file")
    
    parser.add_argument("--model_name", type=str, default="dandelin/vilt-b32-finetuned-vqa", help="Pretrained ViLT model name")
    parser.add_argument("--output_dir", type=str, default="models/vilt_finetuned_vizwiz", help="Output directory for the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    # Use MPS (Apple Silicon) if available.
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    processor = ViltProcessor.from_pretrained(args.model_name)
    model = ViltForQuestionAnswering.from_pretrained(args.model_name).to(device)
    
    # Build answer mapping using id2label from model configuration.
    id2label = model.config.id2label
    answer2id = {v.lower(): int(k) for k, v in id2label.items()}
    
    # Create datasets for training and validation (and test if provided)
    train_dataset = VizWizDataset(args.train_image_dir, args.train_annotations, processor, answer2id, max_length=40)
    val_dataset = VizWizDataset(args.val_image_dir, args.val_annotations, processor, answer2id, max_length=40)
    if args.test_image_dir and args.test_annotations:
        test_dataset = VizWizDataset(args.test_image_dir, args.test_annotations, processor, answer2id, max_length=40)
    else:
        test_dataset = None
    
    # Define compute_metrics function to convert one-hot targets back to scalars.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        valid_preds = []
        valid_true = []
        # Iterate over each sample in the batch
        for pred, lab in zip(preds, labels):
            # Skip samples with no valid label (i.e. one-hot vector is all zeros)
            if lab.sum() == 0:
                continue
            valid_preds.append(pred.item())
            valid_true.append(lab.argmax().item())
        if len(valid_true) == 0:
            return {"accuracy": 0.0}
        valid_preds = np.array(valid_preds)
        valid_true = np.array(valid_true)
        accuracy = (valid_preds == valid_true).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    
    # Set training arguments with a linear learning rate scheduler.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,      # Log training metrics every 50 steps
        eval_steps=100,        # Evaluate every 100 steps
        eval_strategy="steps",
        save_steps=500,
        warmup_steps=100,      # Warmup steps for the learning rate scheduler
        lr_scheduler_type="linear",  # Use a linear learning rate scheduler
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,  # Pads the inputs appropriately.
    )
    
    # Create and add our custom callback for logging both training and evaluation metrics.
    train_eval_callback = TrainEvalCallback(trainer)
    trainer.add_callback(train_eval_callback)
    
    # Train the model.
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")
    
    # Optionally evaluate on the test dataset if provided.
    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test metrics: {test_metrics}")
    
    # --- Plotting metrics ---
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Plot Evaluation Loss and Accuracy over Steps (from evaluation callbacks)
    if train_eval_callback.step_eval_metrics:
        eval_steps = [m["step"] for m in train_eval_callback.step_eval_metrics]
        eval_loss = [m["eval_loss"] for m in train_eval_callback.step_eval_metrics]
        eval_accuracy = [m["eval_accuracy"] for m in train_eval_callback.step_eval_metrics]
        # train_loss_eval = [m["train_loss"] for m in train_eval_callback.step_eval_metrics]
        # train_accuracy_eval = [m["train_accuracy"] for m in train_eval_callback.step_eval_metrics]

        plt.figure()
        plt.plot(eval_steps, eval_loss, marker="o", label="Validation Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Validation Loss over Steps")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "loss_plot_steps.png"))
        plt.close()

        plt.figure()
        plt.plot(eval_steps, eval_accuracy, marker="o", label="Validation Accuracy")
        plt.xlabel("Global Step")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy over Steps")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "accuracy_plot_steps.png"))
        plt.close()

        print(f"Evaluation plots saved in directory: {outputs_dir}")

if __name__ == "__main__":
    main()