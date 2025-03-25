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
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt

class VizWizDataset(Dataset):
    """
    VizWiz-VQA dataset.
    Each sample should have:
      - "image": image filename.
      - "question": the question string.
      - "answers": a list of answer objects (each with keys "answer_confidence" and "answer").
    
    This implementation extracts answer strings from "answers", computes the most common answer (mode),
    and then maps it to a label using answer2id. The scalar label is then converted into a one-hot vector.
    Additionally, images are explicitly resized to (384, 384) to ensure a consistent shape.
    """
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
        image = Image.open(image_path).convert("RGB")
        # Resize image to a fixed resolution (384x384)
        image = image.resize((384, 384))
        question = sample["question"]

        # Extract answer strings from the "answers" list
        answers_list = sample.get("answers", [])
        answer_candidates = [ans["answer"].strip().lower() for ans in answers_list if ans.get("answer", "").strip()]
        if answer_candidates:
            # Use the most common answer as the ground truth.
            answer = Counter(answer_candidates).most_common(1)[0][0]
        else:
            answer = "unanswerable"
            
        # Map answer to label (if not found, use -100 so that it's ignored during loss computation)
        label = self.answer2id.get(answer, -100)
        
        # Convert scalar label to one-hot vector.
        one_hot = torch.zeros(self.num_labels, dtype=torch.float)
        if label != -100:
            one_hot[label] = 1.0

        # Process image and question with max_length=40
        inputs = self.processor(
            image, question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Remove the batch dimension from all inputs
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        inputs["labels"] = one_hot  # one-hot target vector
        return inputs

# Custom callback to evaluate and record metrics on both train and validation sets.
class TrainEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.epoch_metrics = []  # To store metrics for plotting

    def on_epoch_end(self, args, state, control, **kwargs):
        # Evaluate on the training dataset
        train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        # Evaluate on the validation dataset
        eval_metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
        epoch_record = {
            "epoch": state.epoch,
            "train_loss": train_metrics.get("eval_loss", None),
            "train_accuracy": train_metrics.get("eval_accuracy", None),
            "train_bleu": train_metrics.get("eval_bleu", None),
            "eval_loss": eval_metrics.get("eval_loss", None),
            "eval_accuracy": eval_metrics.get("eval_accuracy", None),
            "eval_bleu": eval_metrics.get("eval_bleu", None)
        }
        self.epoch_metrics.append(epoch_record)
        print(f"Epoch {state.epoch} - Training metrics: {train_metrics}")
        print(f"Epoch {state.epoch} - Validation metrics: {eval_metrics}")
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
    
    # Build answer mapping using id2label (instead of id2answer)
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
            return {"accuracy": 0.0, "bleu": 0.0}
        valid_preds = np.array(valid_preds)
        valid_true = np.array(valid_true)
        accuracy = (valid_preds == valid_true).astype(np.float32).mean().item()
        # Convert predictions and true labels back to answer strings using id2label,
        # defaulting to "unknown" if the key is not found.
        pred_strings = [id2label.get(str(int(p)), "unknown").lower() for p in valid_preds]
        true_strings = [id2label.get(str(int(t)), "unknown").lower() for t in valid_true]
        # Prepare tokens for BLEU score computation
        pred_tokens = [s.split() for s in pred_strings]
        true_tokens = [[s.split()] for s in true_strings]
        bleu_score = corpus_bleu(true_tokens, pred_tokens)
        return {"accuracy": accuracy, "bleu": bleu_score}
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="epoch",  # Evaluate on the validation set at the end of each epoch.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,  # Pads the inputs appropriately.
    )
    
    # Create and add our custom callback for logging metrics.
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
    # Retrieve the recorded metrics from the callback.
    metrics = train_eval_callback.epoch_metrics
    if metrics:
        epochs = [m["epoch"] for m in metrics]
        train_loss = [m["train_loss"] for m in metrics]
        eval_loss = [m["eval_loss"] for m in metrics]
        train_accuracy = [m["train_accuracy"] for m in metrics]
        eval_accuracy = [m["eval_accuracy"] for m in metrics]
        train_bleu = [m["train_bleu"] for m in metrics]
        eval_bleu = [m["eval_bleu"] for m in metrics]

        # Create outputs directory if it doesn't exist.
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)

        # Plot Loss
        plt.figure()
        plt.plot(epochs, train_loss, marker="o", label="Train Loss")
        plt.plot(epochs, eval_loss, marker="o", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "loss_plot.png"))
        plt.close()

        # Plot Accuracy
        plt.figure()
        plt.plot(epochs, train_accuracy, marker="o", label="Train Accuracy")
        plt.plot(epochs, eval_accuracy, marker="o", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "accuracy_plot.png"))
        plt.close()

        # Plot BLEU Score
        plt.figure()
        plt.plot(epochs, train_bleu, marker="o", label="Train BLEU")
        plt.plot(epochs, eval_bleu, marker="o", label="Validation BLEU")
        plt.xlabel("Epoch")
        plt.ylabel("BLEU Score")
        plt.title("BLEU Score over Epochs")
        plt.legend()
        plt.savefig(os.path.join(outputs_dir, "bleu_plot.png"))
        plt.close()
        
        print(f"Plots saved in directory: {outputs_dir}")

if __name__ == "__main__":
    main()