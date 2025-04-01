import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import torch
import time
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model_path", type=str, default="./models/local_gemma3", help="Path to local Gemma 3 model directory")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor from local directory
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True).to(device).eval()

    # Load and prepare image
    image = Image.open(args.image).convert("RGB")

    # Create message using chat template
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": args.question}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

    input_len = inputs["input_ids"].shape[-1]

    # Run inference
    with torch.inference_mode():
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        end_time = time.time()

    # Decode result
    generation = outputs[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)

    print("Question:", args.question)
    print("Answer:", decoded)
    print(f"Inference Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
