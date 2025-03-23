import argparse
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", type=str, required=True, help="URL of the input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it", help="Gemma 3 model name (instruction-tuned variant)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--use_auth_token", action="store_true", help="Use your Hugging Face authentication token")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    auth_flag = True if args.use_auth_token else None

    # Load Gemma 3 model and processor.
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_name, 
        use_auth_token=auth_flag, 
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_name, use_auth_token=auth_flag)

    # Construct chat messages as per the Gemma 3 instructions.
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_url},
                {"type": "text", "text": args.question}
            ]
        }
    ]
    
    # Apply the chat template to prepare inputs.
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]

    # Generate output (using inference mode).
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        # Remove the prompt portion of the generated tokens.
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    print("Question:", args.question)
    print("Answer:", decoded)

if __name__ == "__main__":
    main()
