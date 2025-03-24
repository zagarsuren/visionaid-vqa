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
)

class VizWizDataset(Dataset):
    """
    VizWiz-VQA dataset.
    Each sample should have:
      - "image": image filename.
      - "question": the question string.
      - "answers": a list of answer objects (each with keys "answer_confidence" and "answer").
    
    This implementation extracts answer strings from "answers", computes the most common answer (mode),
    and then maps it to a label using answer2id. The scalar label is then converted to a one-hot vector.
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
        # Resize the image to a fixed resolution (384x384)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing VizWiz images")
    parser.add_argument("--annotations", type=str, required=True, help="Path to VizWiz annotation JSON file")
    parser.add_argument("--model_name", type=str, default="dandelin/vilt-b32-finetuned-vqa", help="Pretrained ViLT model name")
    parser.add_argument("--output_dir", type=str, default="models/vilt_finetuned_vizwiz", help="Output directory for the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    
    processor = ViltProcessor.from_pretrained(args.model_name)
    model = ViltForQuestionAnswering.from_pretrained(args.model_name)
    
    # Build answer mapping using id2label (instead of id2answer)
    id2label = model.config.id2label
    answer2id = {v.lower(): int(k) for k, v in id2label.items()}
    
    dataset = VizWizDataset(args.image_dir, args.annotations, processor, answer2id, max_length=40)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,  # default collator pads the inputs appropriately
    )
    
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()