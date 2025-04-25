# VisionAid-VQA: Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms

This Streamlit web application allows users to upload or capture an image, ask a question about it, and receive an **audio response** using advanced **Visual Question Answering (VQA)** models. The system is designed for accessibility, especially supporting visually impaired users.

---

## 🧠 Features

- 🔍 Supports VQA models:
  - `vilt_finetuned_vizwiz` (Vision Language Transformer model finetuned with VizWiz)
  - `florence2-finetuned` (Multimodal reasoning model finetuned with VizWiz)
- 📷 Accepts image input from upload or camera
- ❓ Accepts natural language questions 
- 🔊 Accepts audio input and converts audio to text (appstt.py)
- 🔊 Converts answers to speech using `gTTS` (appstt.py)
- 🎧 Auto-plays audio response in the app (appstt.py)

---

## 📁 Project Structure

```graphql
project_root/
├── data  
├── modules/
│   ├── robust_vilt.py
│   ├── florence2.py
├── models
│   ├── vilt_finetuned_vizwiz
│   ├── florence2-finetuned
├── scripts
├── appchat.py # App with chat like UI
├── appstt.py # App with speech recognition and TTS module

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

- `RobustViLT` → `/models/vilt_finetuned_vizwiz`
- `Florence2Model` → `/models/florence2-finetuned`

## To run the Web App
```bash
streamlit run app.py
```

## To see the log
```
tensorboard --logdir=runs
```

## License
This project is licensed under the MIT License.