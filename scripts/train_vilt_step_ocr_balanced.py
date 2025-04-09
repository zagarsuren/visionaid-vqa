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
    EarlyStoppingCallback,
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

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

        ocr_text = pytesseract.image_to_string(image)
        ocr_text = " ".join(ocr_text.split())

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

class TrainEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.step_eval_metrics = []
        self.writer = SummaryWriter(log_dir=os.path.join(trainer.args.output_dir, "runs"))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step} - Training logs: {logs}")
            for k, v in logs.items():
                self.writer.add_scalar(f"train/{k}", v, state.global_step)
        return control

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.step_eval_metrics.append({
            "step": state.global_step,
            "eval_loss": metrics.get("eval_loss"),
            "eval_accuracy": metrics.get("eval_accuracy"),
        })
        print(f"Step {state.global_step} - Validation metrics: {metrics}")
        for k, v in metrics.items():
            self.writer.add_scalar(f"eval/{k}", v, state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    valid_preds = []
    valid_true = []

    for pred, lab in zip(preds, labels):
        if lab.sum() == 0:
            continue
        valid_preds.append(pred.item())
        valid_true.append(lab.argmax().item())

    if len(valid_true) == 0:
        return {"accuracy": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(valid_true, valid_preds, average="weighted")
    acc = (np.array(valid_preds) == np.array(valid_true)).astype(np.float32).mean().item()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--test_image_dir", type=str, required=False)
    parser.add_argument("--train_annotations", type=str, required=True)
    parser.add_argument("--val_annotations", type=str, required=True)
    parser.add_argument("--test_annotations", type=str, required=False)
    parser.add_argument("--model_name", type=str, default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--output_dir", type=str, default="models/vilt_finetuned_vizwiz")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    processor = ViltProcessor.from_pretrained(args.model_name)
    model = ViltForQuestionAnswering.from_pretrained(args.model_name).to(device)

    id2label = model.config.id2label
    answer2id = {v.lower(): int(k) for k, v in id2label.items()}

    train_dataset = VizWizDataset(args.train_image_dir, args.train_annotations, processor, answer2id)
    val_dataset = VizWizDataset(args.val_image_dir, args.val_annotations, processor, answer2id)
    test_dataset = VizWizDataset(args.test_image_dir, args.test_annotations, processor, answer2id) if args.test_image_dir and args.test_annotations else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=500,
        warmup_steps=100,
        lr_scheduler_type="linear",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    trainer.add_callback(TrainEvalCallback(trainer))
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")

    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test metrics: {test_metrics}")

    # Plotting
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    callback = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, TrainEvalCallback)][0]

    if callback.step_eval_metrics:
        eval_steps = [m["step"] for m in callback.step_eval_metrics]
        eval_loss = [m["eval_loss"] for m in callback.step_eval_metrics]
        eval_accuracy = [m["eval_accuracy"] for m in callback.step_eval_metrics]

        plt.figure()
        plt.plot(eval_steps, eval_loss, marker="o", label="Validation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "loss_plot_steps.png"))

        plt.figure()
        plt.plot(eval_steps, eval_accuracy, marker="o", label="Validation Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "accuracy_plot_steps.png"))

        print(f"Evaluation plots saved to {outputs_dir}")

if __name__ == "__main__":
    main()