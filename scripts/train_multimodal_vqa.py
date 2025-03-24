import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainingArguments

from modules.resnet_backbone import ResNetBackbone
from modules.bert_backbone import BertBackbone
from modules.multimodal_attention import MultimodalVQAWithAttention

class VizWizDatasetCustom(Dataset):
    """
    Custom VizWiz dataset for the multimodal VQA model.
    Each sample should include:
      - "image": the image filename,
      - "question": the question string,
      - "answers": a list of answer objects.
    
    This implementation extracts answer strings from the "answers" list,
    selects the most common answer as ground truth, maps it to a label, and
    returns a dictionary with keys expected by the model.
    """
    def __init__(self, image_dir, annotation_file, answer2id, max_length=40):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.answer2id = answer2id
        self.max_length = max_length
        # Resize images to 384x384 and convert to tensor.
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
        image = self.transform(image)  # tensor shape: [3, 384, 384]
        question = sample["question"]
        
        # Extract answer strings and choose the most common answer.
        answers_list = sample.get("answers", [])
        answer_candidates = [
            ans["answer"].strip().lower() for ans in answers_list if ans.get("answer", "").strip()
        ]
        if answer_candidates:
            answer = Counter(answer_candidates).most_common(1)[0][0]
        else:
            answer = "unanswerable"
        
        # Map answer to label (if not found, use -100 so it's ignored during loss computation)
        label = self.answer2id.get(answer, -100)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            "image": image,
            "question": question,
            "label": label_tensor,      # ground truth label for loss computation
            "label_ids": label_tensor   # duplicate key if needed elsewhere
        }

def custom_data_collator(features):
    """
    Collates a list of samples into a batch.
    - Stacks tensor fields ("image", "label", and "label_ids").
    - Keeps the raw "question" strings in a list.
    """
    batch = {
        "image": torch.stack([f["image"] for f in features]),
        "label": torch.stack([f["label"] for f in features]),
        "label_ids": torch.stack([f["label_ids"] for f in features]),
        "question": [f["question"] for f in features]
    }
    return batch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop the "label" key so it isn't passed to model.forward.
        labels = inputs.pop("label")
        logits, attn_weights = model(**inputs)
        # Ensure labels are on the same device as logits (required for MPS)
        labels = labels.to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, (logits, attn_weights)) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with VizWiz images")
    parser.add_argument("--annotations", type=str, required=True, help="Path to VizWiz annotation JSON file")
    parser.add_argument("--output_dir", type=str, default="models/multimodal_vqa_finetuned", help="Output directory for the model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    
    # Build a dummy answer mapping. Update this mapping as needed.
    dummy_answer_list = ["yes", "no", "basil", "soda", "coke", "unanswerable"]
    answer2id = {ans: idx for idx, ans in enumerate(dummy_answer_list)}
    
    # Use MPS (Apple Silicon) if available.
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Instantiate backbones.
    visual_backbone = ResNetBackbone(pretrained=True)
    text_backbone = BertBackbone(model_name="bert-base-uncased")
    num_answers = len(answer2id)
    
    # Instantiate your custom multimodal VQA model with attention.
    model = MultimodalVQAWithAttention(visual_backbone, text_backbone, num_answers).to(device)
    
    # Create dataset.
    dataset = VizWizDatasetCustom(args.image_dir, args.annotations, answer2id, max_length=40)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no"  # (Note: evaluation_strategy is deprecated in favor of eval_strategy in future versions)
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=custom_data_collator
    )
    
    trainer.train()
    # model.save_pretrained(args.output_dir)
    # print(f"Finetuned multimodal VQA model saved to {args.output_dir}")

    # Use torch.save to save the state dictionary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"Finetuned multimodal VQA model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
