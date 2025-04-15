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
import speech_recognition as sr
import io

# Import the streamlit_audiorecorder component and pydub
from audiorecorder import audiorecorder
from pydub import AudioSegment

# Initialize a session state variable for the voice question if not already set
if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""

def autoplay_audio(file_path: str):
    """
    Reads the audio file from file_path, encodes it as base64, and injects HTML
    to play the audio automatically.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
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
    
    # Model selection via sidebar.
    model_option = st.sidebar.selectbox(
        "Select Model",
        ("Florence2-finetuned", "ViLT-finetuned", "BLIP2")
    )
    
    # Image source selection: Upload or Camera Capture.
    image_source = st.sidebar.radio("Image Source", options=["Upload", "Camera Capture"])
    if image_source == "Upload":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capture an image")
    
    # Choose question input method.
    st.subheader("How would you like to input your question?")
    input_mode = st.radio("Choose input method:", ["Text", "Voice (Microphone)"])
    
    # If Text mode is selected: simply show a text input.
    if input_mode == "Text":
        question = st.text_input("Enter your question:")
    else:
        # If Voice mode is selected, record the audio and let the user save it.
        st.info("Step 1: Record your question using your microphone.")
        audio_recording = audiorecorder("Record audio")  # returns a pydub.AudioSegment
        if audio_recording is not None:
            # Button to save the recorded audio to assets/recording.mp3.
            if st.button("Save Recording"):
                audio_dir = os.path.join("assets/audio")
                os.makedirs(audio_dir, exist_ok=True)
                audio_file_path = os.path.join(audio_dir, "recording.mp3")
                audio_recording.export(audio_file_path, format="mp3")
                st.success(f"Recording saved to {audio_file_path}")
            
            st.info("Step 2: Transcribe the saved recording.")
            if st.button("Transcribe Recording"):
                audio_file_path = os.path.join("assets/audio", "recording.mp3")
                try:
                    # Load the saved MP3 file using pydub and convert it to WAV.
                    audio = AudioSegment.from_file(audio_file_path, format="mp3")
                    buf = io.BytesIO()
                    # Export to WAV for compatibility with SpeechRecognition.
                    audio.export(buf, format="wav")
                    buf.seek(0)
                    
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(buf) as source:
                        audio_data = recognizer.record(source)
                    transcribed_text = recognizer.recognize_google(audio_data)
                    st.session_state.voice_question = transcribed_text
                    st.write("Transcribed Question:", transcribed_text)
                except Exception as e:
                    st.error("Transcription error: " + str(e))
        
        # Use the transcribed voice question (if available) as the question.
        question = st.session_state.voice_question

    # Display the uploaded or captured image.
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded/Captured Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying the image: {e}")
            return
    else:
        image = None

    # When the user clicks "Get Answer", process the image and question with the VQA model.
    if st.button("Get Answer"):
        if uploaded_file is None:
            st.warning("Please upload or capture an image.")
            return
        if not question:
            st.warning("Please provide a question (either via text or by transcribing your recorded voice).")
            return

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
            return
        
        # Resize the image if necessary.
        if image.size != (384, 384):
            image = image.resize((384, 384))
        
        try:
            if model_option == "ViLT-finetuned":
                model = RobustViLT(model_name="/path/to/vilt_finetuned_model")
                answer = model.generate_answer(image, question)
            elif model_option == "Florence2-finetuned":
                model = Florence2Model(model_path="models/florence2-finetuned")
                task_prompt = "Describe the answer in detail."
                answer = model.generate_answer(image, task_prompt, question)
            elif model_option == "BLIP2":
                model = BLIP2Model()
                answer = model.generate_answer(image, question)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            answer = "Unknown"
        
        # Ensure the answer is a string.
        if not isinstance(answer, str):
            answer = str(answer)
        
        st.image(image, caption="Captured/Uploaded Image", use_column_width=True)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer ({model_option}):** {answer}")
        
        # ---------- Text-to-Speech: Convert the answer to audio, save it, and autoplay ----------
        try:
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
