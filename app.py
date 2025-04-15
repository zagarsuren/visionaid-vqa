import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
import torch
from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.florence2 import Florence2Model  # Florence2Model with three-argument generate_answer

from gtts import gTTS
import base64

def autoplay_audio(file_path: str):
    """
    Reads an audio file, encodes it in base64 and returns an HTML string that embeds the audio.
    The audio is set to auto-play.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        # Encode audio data with base64
        b64 = base64.b64encode(data).decode()
        # Construct an HTML audio element with base64 source
        html_audio = f"""
            <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
        st.markdown(html_audio, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load the audio: {e}")

def main():
    st.title("Inclusive VQA System for Visually Impaired Users")
    st.write("Choose a model and ask a question about an image.")
    
    # Model selection in the sidebar
    model_option = st.sidebar.selectbox(
        "Select Model",
        ("Florence2-finetuned", "ViLT-finetuned", "BLIP2")
    )
    
    # Option for image source: Upload or Camera Capture
    image_source = st.sidebar.radio("Image Source", options=["Upload", "Camera Capture"])
    
    if image_source == "Upload":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capture an image")
    
    question = st.text_input("Enter your question:")
    
    # Display the uploaded or captured image
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
            if model_option == "ViLT-finetuned":
                model = RobustViLT(model_name="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/vilt_finetuned_vizwiz_ocr")
                answer = model.generate_answer(image, question)
            elif model_option == "Florence2-finetuned":
                model = Florence2Model(model_path="models/florence2-finetuned")
                # Use a fixed task prompt for Florence2.
                task_prompt = "Describe the photo in detail."
                answer = model.generate_answer(image, task_prompt, question)
            elif model_option == "BLIP2":
                model = BLIP2Model()
                answer = model.generate_answer(image, question)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            answer = "Unknown"
        
        # Ensure the answer is a string
        if not isinstance(answer, str):
            answer = str(answer)
        
        st.image(image, caption="Captured/Uploaded Image", use_column_width=True)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer ({model_option}):** {answer}")
        
        # ---------- Text-to-Speech: Clean Answer and Convert to Audio, Save it, and Autoplay ----------
        try:
            # If the answer contains the task prompt, remove it.
            prompt_to_remove = task_prompt
            if prompt_to_remove in answer:
                answer = answer.replace(prompt_to_remove, "").strip()
            
            audio_dir = os.path.join("assets", "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_file_path = os.path.join(audio_dir, "speech.mp3")
            
            tts = gTTS(answer, lang='en')
            tts.save(audio_file_path)
            
            st.success("Audio generated and saved successfully!")
            autoplay_audio(audio_file_path)
        except Exception as e:
            st.error(f"Text-to-speech conversion failed: {e}")

if __name__ == "__main__":
    main()
