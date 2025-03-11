import argparse
import os
import json
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainingArguments, default_data_collator

from modules.resnet_backbone import ResNetBackbone
from modules.bert_backbone import BertBackbone
from modules.multimodal_attention import MultimodalVQAWithAttention

class VizWizDatasetCustom(Dataset):
    """
    Custom VizWiz dataset for our multimodal model.
    Each sample should include:
      - "image": filename,
      - "question": question string,
      - "answers": a list of answer objects.
    This dataset extracts the most common answer and converts it to a label.
    """
    def __init__(self, image_dir, annotation_file, answer2id, max_length=40):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.answer2id = answer2id
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # shape: [3, 384, 384]
        question = sample["question"]
        
        # Extract answer strings and choose the most common
        answers_list = sample.get("answers", [])
        answer_candidates = [ans["answer"].strip().lower() for ans in answers_list if ans.get("answer", "").strip()]
        if answer_candidates:
            answer = Counter(answer_candidates).most_common(1)[0][0]
        else:
            answer = "unanswerable"
        
        label = self.answer2id.get(answer, -100)
        return {"pixel_values": image, "question": question, "labels": torch.tensor(label, dtype=torch.long)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with VizWiz images")
    parser.add_argument("--annotations", type=str, required=True, help="Path to VizWiz annotation JSON file")
    parser.add_argument("--output_dir", type=str, default="multimodal_vqa_finetuned", help="Output directory for the model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    
    # Build answer mapping (using a dummy mapping for illustration)
    # Build this mapping from the dataset or use a pre-defined answer vocabulary.
    dummy_answer_list = ["yes", "no", "basil", "soda", "coke", "unanswerable"]
    answer2id = {ans: idx for idx, ans in enumerate(dummy_answer_list)}
    
    # Instantiate backbones
    visual_backbone = ResNetBackbone(pretrained=True)
    text_backbone = BertBackbone(model_name="bert-base-uncased")
    
    # Number of answer classes (dummy value, update as needed)
    num_answers = len(answer2id)
    
    # Instantiate the multimodal model with attention.
    model = MultimodalVQAWithAttention(visual_backbone, text_backbone, num_answers)
    
    # Create dataset.
    dataset = VizWizDatasetCustom(args.image_dir, args.annotations, answer2id, max_length=40)
    
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
    print(f"Finetuned multimodal VQA model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
