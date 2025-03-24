# VisionAid-VQA: Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms

This repository implements an inclusive and context-aware Visual Question Answering (VQA) system that leverages advanced multimodal attention mechanisms. We fine-tune a ViLT model on the VizWiz dataset and compare its performance with BLIP2 and LLaMA 3.2 (accessed via Ollama). An interactive Streamlit app enables real-time queries.

## Project Structure

```graphql
visionaid-vqa/
├── modules/
│   ├── robust_vilt.py           # ViLT module (fine‑tunable on VizWiz)
│   ├── multimodal_attention.py  # Custom cross‑modal attention enhancement module
│   ├── resnet_backbone.py       # CNN feature extraction module
│   ├── bert_backbone.py         # Text feature extraction module   
│   ├── blip2.py                 # BLIP2 wrapper for inference (pretrained weights)
│   └── llama32.py               # LLaMA 3.2 wrapper via Ollama (placeholder)
├── scripts/
│   ├── train_robust_vilt.py     # Fine‑tuning script for ViLT on VizWiz
│   ├── train_multimodal_vqa.py  # Custom training script for multimodel attention on VizWiz
│   ├── inference_robust_vilt.py # Inference script for the fine‑tuned ViLT model
│   ├── inference_blip2.py       # Inference script for BLIP2
│   └── inference_llama32.py     # Inference script for LLaMA 3.2 via Ollama
├── app.py                       # Interactive Streamlit web app (model selection)
├── requirements.txt
└── README.md
```


## Setup

1. Create a Virtual Environment
Upgrade pip:
```bash
pip install --upgrade pip
```

Create and activate Virtual Environment (Windows):
```bash
python -m venv vqa
vqa\Scripts\activate
```

Create and activate Virtual Environment (Linux):
```bash
python3 -m venv vqa
source vqa/bin/activate
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Fine-Tuning

Prepare the VizWiz-VQA dataset with an annotation JSON file where each entry has:
```json
{
  "image": "image_filename.jpg",
  "question": "What is in the picture?",
  "answer": "expected answer"
}
```
Then run:

```bash
python scripts/train_robust_vilt.py --image_dir path/to/images --annotations path/to/annotations.json
```

## Inference
Test the fine-tuned model:
```bash
python scripts/inference_robust_vilt.py --image path/to/test_image.jpg --question "What is in the image?" --model_path ./models/vilt_finetuned_vizwiz
```

Test the BLIP2 model:
```bash
python scripts/inference_blip2.py --image path/to/image.jpg --question "What is in this picture?" --model_path ./models/local_blip2
```


## Interactive Web App
```bash
streamlit run app.py
```

## License
This project is licensed under the MIT License.
