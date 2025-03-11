# scripts/inference_robust_vilt.py

import argparse
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned ViLT model directory")
    args = parser.parse_args()
    
    processor = ViltProcessor.from_pretrained(args.model_path)
    model = ViltForQuestionAnswering.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = Image.open(args.image).convert("RGB")
    # Resize image to a fixed resolution
    image = image.resize((384, 384))
    inputs = processor(image, args.question, return_tensors="pt", padding="max_length", truncation=True, max_length=40)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_id = logits.argmax(-1).item()
    answer = model.config.id2label[predicted_id]
    
    print("Question:", args.question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
