import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image

class Florence2Model:
    """
    Florence2 Vision-Language Model for Visual Question Answering (VQA)
    using a locally fine-tuned checkpoint.
    
    The model and processor are loaded from a local checkpoint (default: "models/florence2-finetuned").
    """
    def __init__(self, model_path="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/florence2-finetuned"):
        # Load the configuration.
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Override vision_config.model_type if necessary.
        if getattr(config.vision_config, "model_type", None) != "davit":
            print(f"Warning: Overriding vision_config.model_type from {config.vision_config.model_type} to 'davit'")
            config.vision_config.model_type = "davit"
            
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
        
        # Set device to CUDA, then MPS (Apple Silicon), else CPU.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            # else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
            else "cpu"
        )
        self.model.to(self.device)

    def generate_answer(self, image, task_prompt, text_input):
        """
        Generate an answer for the provided image and question using Florence2.
        
        This method replicates the behavior from the working command-line inference script:
          - It concatenates the task prompt and the additional text input to form the full prompt.
          - It processes the input image and prompt.
          - It generates a prediction using beam search.
          - It decodes and post-processes the output, using the task_prompt or full prompt to guide post-processing.
        
        Parameters:
            image (PIL.Image): The input image.
            task_prompt (str): The primary task instruction (e.g., "Describe the scene: ").
            text_input (str): Additional text to append to the prompt (e.g., "What is happening in the image? ").
        
        Returns:
            str: The generated answer.
        """
        # Concatenate task prompt and additional text to form the full prompt.
        prompt = task_prompt + text_input

        # Ensure the image is in RGB mode.
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize the image to 384x384 pixels if necessary.
        if image.size != (384, 384):
            image = image.resize((384, 384))
        
        # Process the text and image input.
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # Generate output tokens using beam search.
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=2
        )
        
        # Decode the generated tokens.
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Post-process the output.
        answer = self.processor.post_process_generation(
            generated_text,
            task=prompt,  # The task prompt is used for post-processing.
            image_size=(image.width, image.height)
        )
        return answer
