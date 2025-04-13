import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
import argparse

# Choose the device: prefer CUDA, then MPS (Apple Silicon), else CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
    else "cpu"
)

def load_model_and_processor(model_path="models/florence2-finetuned"):
    """
    Load the locally fine-tuned Florence2 model and its processor.
    This function checks the model configuration's vision_config.model_type and overrides
    it to 'davit' if necessary to bypass the assertion error.
    """
    # Load the configuration first.
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Check the expected vision model type.
    if getattr(config.vision_config, "model_type", None) != "davit":
        print(f"Warning: Overriding vision_config.model_type from {config.vision_config.model_type} to 'davit'")
        config.vision_config.model_type = "davit"
    
    # Load processor and model using the (possibly modified) config.
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True).to(device)
    return model, processor

def run_example(task_prompt, text_input, image, processor, model):
    """
    Run inference by combining the task prompt and text input with the provided image.
    
    Parameters:
        task_prompt (str): The task instruction for the model (e.g., "Describe the scene: ").
        text_input (str): Additional text appended to the prompt.
        image (PIL.Image): The input image.
        processor: The model processor.
        model: The Florence2 model.
    
    Returns:
        str: The parsed answer from the model.
    """
    # Concatenate the task prompt and text input.
    prompt = task_prompt + text_input

    # Convert image to RGB mode if necessary.
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Process the image and text input.
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate the output with beam search.
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )

    # Decode the generated tokens.
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process the generated text to obtain the final answer.
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run inference using a locally fine-tuned Florence2 model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--task_prompt", type=str, default="Describe the scene: ", help="Task prompt for the model.")
    parser.add_argument("--text_input", type=str, default="What is happening in the image? ", help="Additional text input for the prompt.")
    parser.add_argument("--model_path", type=str, default="models/florence2-finetuned", help="Path to the locally fine-tuned Florence2 model.")
    args = parser.parse_args()

    # Load the model and processor.
    model, processor = load_model_and_processor(args.model_path)

    # Open the image.
    try:
        image = Image.open(args.image)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Run inference.
    answer = run_example(args.task_prompt, args.text_input, image, processor, model)
    print("Generated Answer:")
    print(answer)

if __name__ == "__main__":
    main()
