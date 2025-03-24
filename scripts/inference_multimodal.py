import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
from PIL import Image
import torch
from torchvision import transforms

# Import custom modules.
from modules.resnet_backbone import ResNetBackbone
from modules.bert_backbone import BertBackbone
from modules.multimodal_attention import MultimodalVQAWithAttention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned multimodal model directory")
    args = parser.parse_args()
    
    # Define the answer mapping as used during training.
    dummy_answer_list = ["yes", "no", "basil", "soda", "coke", "unanswerable"]
    id2answer = {idx: ans for idx, ans in enumerate(dummy_answer_list)}
    num_answers = len(dummy_answer_list)
    
    # Instantiate backbones.
    # Use the same configuration as during training.
    visual_backbone = ResNetBackbone(pretrained=True)
    text_backbone = BertBackbone(model_name="bert-base-uncased")
    
    # Instantiate your multimodal VQA model.
    model = MultimodalVQAWithAttention(visual_backbone, text_backbone, num_answers)
    
    # Load the saved model state.
    state_dict_path = os.path.join(args.model_path, "pytorch_model.bin")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Preprocess the input image.
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    image_tensor = image_tensor.to(device)
    
    # For the question, we assume the text backbone handles tokenization internally.
    # In this simple example, we pass the raw question string.
    question = args.question
    
    # Run inference.
    with torch.no_grad():
        logits, _ = model(image_tensor, question)
        predicted_id = logits.argmax(dim=-1).item()
        answer = id2answer.get(predicted_id, "unanswerable")
    
    print("Question:", question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()