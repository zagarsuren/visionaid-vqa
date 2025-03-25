import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to the PaliGemma model directory")
    args = parser.parse_args()


    # Use MPS (Apple Silicon) if available.
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load processor and model from the local path
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True).to(device)

    # Load and preprocess the image
    image = Image.open(args.image).convert("RGB")

    # Format the prompt
    prompt = f"Question: {args.question}\nAnswer:"

    # Tokenize inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Generate answer
    generated_ids = model.generate(**inputs, max_new_tokens=40)
    answer = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Question:", args.question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()