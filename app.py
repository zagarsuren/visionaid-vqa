import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
import torch
from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.florence2 import Florence2Model  # Florence2Model with three-argument generate_answer

def main():
    st.title("Inclusive VQA System for Visually Impaired Users")
    st.write("Choose a model and ask a question about an image.")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ("ViLT (Fine-Tuned on VizWiz)", "BLIP2", "FLORENCE2")
    )
    
    # Option for image source: Upload or Camera Capture
    image_source = st.sidebar.radio("Image Source", options=["Upload", "Camera Capture"])
    
    if image_source == "Upload":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capture an image")


    question = st.text_input("Enter your question:")
    # Retain the ViLT model path input for ViLT; Florence2 uses a fixed path.
    model_path = st.sidebar.text_input("ViLT Model Path (if using ViLT)", 
                                       value="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/vilt_finetuned_vizwiz3")

    # Display uploaded image immediately
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded/Captured Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying the image: {e}")
            return
    else:
        image = None

    if st.button("Get Answer"):
        if uploaded_file is None:
            st.warning("Please upload or capture an image.")
            return
        if not question:
            st.warning("Please enter a question.")
            return

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            return
        
        # Resize image if necessary
        if image.size != (384, 384):
            image = image.resize((384, 384))
        
        try:
            if model_option == "ViLT (Fine-Tuned on VizWiz)":
                model = RobustViLT(model_name=model_path)
                answer = model.generate_answer(image, question)
                inference_time = None
                gpu_memory = None
            elif model_option == "BLIP2":
                model = BLIP2Model()
                answer = model.generate_answer(image, question)
                inference_time = None
                gpu_memory = None
            elif model_option == "FLORENCE2":
                model = Florence2Model(model_path="models/florence2-finetuned")
                # For Florence2, use a fixed task prompt and treat the user question as additional input.
                task_prompt = ""
                answer = model.generate_answer(image, task_prompt, question)
                inference_time = None
                gpu_memory = None
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            answer = "Unknown"
            inference_time = None
            gpu_memory = None
        
        st.image(image, caption="Captured/Uploaded Image", use_column_width=True)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer ({model_option}):** {answer}")
        
        if inference_time is not None:
            st.write(f"⏱️ Inference Time: {inference_time:.2f} seconds")
        if gpu_memory is not None:
            st.write(f"🖥️ GPU Memory Used: {gpu_memory:.2f} MB")

if __name__ == "__main__":
    main()
