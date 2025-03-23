import argparse
import os
import json
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

class VizWizDatasetBlip2(Dataset):
    """
    VizWiz-VQA dataset for BLIP2 fine-tuning with enhanced visual preprocessing.
    Each sample is expected to have:
      - "image": image filename.
      - "question": the question string.
      - "answers": a list of answer objects (each with keys "answer_confidence" and "answer").

    This implementation extracts the answer strings from "answers", computes the most common answer (mode),
    and uses that answer as the target text. In addition, it applies a series of image augmentations to help
    the model become more robust to real-world variations in image quality and appearance.
    """
    def __init__(self, image_dir, annotation_file, processor, max_length=50, augment=True):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

        # Define a transformation pipeline for data augmentation.
        if augment:
            self.transform = T.Compose([
                T.Resize((896, 896)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomRotation(10),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((896, 896))
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")
        # Apply visual data augmentation
        image = self.transform(image)
        question = sample["question"]

        # Extract answer strings from the "answers" list.
        answers_list = sample.get("answers", [])
        answer_candidates = [ans["answer"].strip() for ans in answers_list if ans.get("answer", "").strip()]
        if answer_candidates:
            # Use the most common answer as the target.
            answer = Counter(answer_candidates).most_common(1)[0][0]
        else:
            answer = "unanswerable"

        # Define the target text for generation.
        target_text = answer

        # Process image and question.
        inputs = self.processor(
            image,
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Tokenize the target text using the processor's target mode.
        with self.processor.as_target_processor():
            labels = self.processor.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ).input_ids

        # Remove the batch dimension.
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        labels = labels.squeeze(0)
        inputs["labels"] = labels

        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing VizWiz images")
    parser.add_argument("--annotations", type=str, required=True, help="Path to VizWiz annotation JSON file")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-flan-t5-xl", help="Pretrained BLIP2 model name")
    parser.add_argument("--output_dir", type=str, default="blip2_finetuned_vizwiz", help="Output directory for the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--augment", action="store_true", help="Apply visual data augmentation")
    args = parser.parse_args()

    # Load BLIP2 processor and model.
    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_name)

    # Create dataset.
    dataset = VizWizDatasetBlip2(args.image_dir, args.annotations, processor, max_length=50, augment=args.augment)

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
        data_collator=default_data_collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned BLIP2 model saved to {args.output_dir}")

if __name__ == "__main__":
    main()