# app.py

import streamlit as st
from PIL import Image
import torch
from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.llama32 import Llama32Model

def main():
    st.title("Inclusive VQA System for Visually Impaired Users")
    st.write("Choose a model and ask a question about an image.")
    
    model_option = st.sidebar.selectbox(
        "Select Model",
        ("ViLT (Fine-Tuned on VizWiz)", "BLIP2", "LLaMA 3.2 (via Ollama)")
    )
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    question = st.text_input("Enter your question:")
    model_path = st.sidebar.text_input("ViLT Model Path (if using ViLT)", value="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/vilt_finetuned_vizwiz")
    
    if st.button("Get Answer"):
        if uploaded_file is None:
            st.warning("Please upload an image.")
            return
        if not question:
            st.warning("Please enter a question.")
            return
        
        image = Image.open(uploaded_file).convert("RGB")
        # Optional: resize image if necessary
        if image.size != (384, 384):
            image = image.resize((384, 384))
        
        try:
            if model_option == "ViLT (Fine-Tuned on VizWiz)":
                model = RobustViLT(model_name=model_path)
                answer = model.generate_answer(image, question)
            elif model_option == "BLIP2":
                model = BLIP2Model()
                answer = model.generate_answer(image, question)
            else:  # LLaMA 3.2 via Ollama
                model = Llama32Model()
                answer = model.generate_answer(image, question)
        except KeyError as e:
            st.error("An error occurred during prediction. Returning fallback answer.")
            answer = "Unknown"
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer ({model_option}):** {answer}")

if __name__ == "__main__":
    main()
