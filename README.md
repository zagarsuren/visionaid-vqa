# VisionAid-VQA: Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms

This Streamlit web application allows users to upload or capture an image, ask a question about it, and receive an **audio response** using advanced **Visual Question Answering (VQA)** models. The system is designed for accessibility, especially supporting visually impaired users.

---

## 🧠 Features

- 🔍 Supports multiple VQA models:
  - `RobustViLT` (ViLT-finetuned with VizWiz)
  - `Florence2-finetuned` (Finetuned with VizWiz)
  - `BLIP2` (multi-modal reasoning model)
- 📷 Accepts image input from upload or camera
- ❓ Accepts natural language questions 
- 🔊 Accepts audio input and converts audio to text
- 🔊 Converts answers to speech using `gTTS`
- 🎧 Auto-plays audio response in the app

---

## 📁 Project Structure

```graphql
project_root/
├── data  
├── app.py  
├── modules/
│   ├── robust_vilt.py
│   ├── florence2.py
│   └── blip2.py
├── models
│   ├── vilt_finetuned_vizwiz_ocr
│   ├── florence2-finetuned
│   ├── local_blip2
├── scripts
├── assets/
│   └── audio/
│       └── recording.mp3  ← recorded question audio
│       └── speech.mp3  ← generated answer audio

```


## Setup
---
1. Clone the repository:
```bash
git clone https://github.com/your-username/inclusive-vqa-app.git
cd visionaid-vqa
```

2. Create a Virtual Environment
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

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## 🧠 Model Weights

- `RobustViLT` → `/models/vilt_finetuned_vizwiz_ocr`
- `Florence2Model` → `/models/florence2-finetuned`
- `BLIP2Model` → `local_blip2` Uses HuggingFace or local model by default.

## To run the Web App
```bash
streamlit run app.py
```

## License
This project is licensed under the MIT License.