import argparse
from PIL import Image
from modules.llama32 import Llama32Model

def main(args):
    model = Llama32Model(ollama_url=args.ollama_url)
    image = Image.open(args.image).convert("RGB")
    answer = model.infer(image, args.question)
    print("[Llama 3.2 VLM via Ollama Inference]")
    print("Question:", args.question)
    print("Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11411")
    args = parser.parse_args()
    main(args)
