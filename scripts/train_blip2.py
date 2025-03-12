import argparse
import os
import json
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

class VizWizDatasetBlip2(Dataset):
    """
    VizWiz-VQA dataset for BLIP2 fine-tuning.
    Each sample is expected to have:
      - "image": image filename.
      - "question": the question string.
      - "answers": a list of answer objects (each with keys "answer_confidence" and "answer").
    
    This class extracts the answer strings from "answers", computes the most common answer (mode),
    and then uses that answer as the target text.
    """
    def __init__(self, image_dir, annotation_file, processor, max_length=50):
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
        image = Image.open(image_path).convert("RGB")
        # Resize the image to a fixed resolution (e.g., 384x384) for consistency.
        image = image.resize((384, 384))
        question = sample["question"]

        # Extract answer strings from the "answers" list
        answers_list = sample.get("answers", [])
        answer_candidates = [ans["answer"].strip() for ans in answers_list if ans.get("answer", "").strip()]
        if answer_candidates:
            # Use the most common answer as the target.
            answer = Counter(answer_candidates).most_common(1)[0][0]
        else:
            answer = "unanswerable"
        
        # Define the target text for generation.
        target_text = answer

        # Process image and question
        inputs = self.processor(
            image,
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Tokenize target text using the processor's target mode.
        with self.processor.as_target_processor():
            labels = self.processor.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ).input_ids

        # Remove the batch dimension from all fields.
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
    parser.add_argument("--output_dir", type=str, default="models/blip2_finetuned_vizwiz", help="Output directory for the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    
    # Load the BLIP2 processor and model.
    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Create the dataset.
    dataset = VizWizDatasetBlip2(args.image_dir, args.annotations, processor, max_length=50)
    
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
        data_collator=default_data_collator,  # Handles dynamic padding.
    )
    
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned BLIP2 model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
