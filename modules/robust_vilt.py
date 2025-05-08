import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

class RobustViLT:
    """
    Robust Vision-and-Language Transformer (ViLT) for VQA.
    Uses a pretrained checkpoint (default "dandelin/vilt-b32-finetuned-vqa").
    """
    def __init__(self, model_name="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/vilt_finetuned_vizwiz"):
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model.to(self.device)

    def generate_answer(self, image, question):
        """
        Given a PIL image and a question string, return the predicted answer.
        """
        # Resize the image to 384x384 for consistency
        if image.size != (384, 384):
            image = image.resize((384, 384))
        inputs = self.processor(image, question, return_tensors="pt", padding="max_length", truncation=True, max_length=40).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape: [batch_size, num_labels]
        predicted_id = logits.argmax(-1).item()
        answer = self.model.config.id2label[predicted_id]
        return answer