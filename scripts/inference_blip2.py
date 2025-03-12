import argparse
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to the BLIP2 model directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model from the local path without re-downloading
    processor = Blip2Processor.from_pretrained(args.model_path, local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True).to(device)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=args.question, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=40)
    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Question:", args.question)
    print("Answer:", generated_text)

if __name__ == "__main__":
    main()