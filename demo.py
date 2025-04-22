import os
import io
import base64
import warnings

# --- HACK: disable Blocks.__exit__ so Gradio won't crash building the JSON schema ---
import gradio as gr
import gradio.blocks
gradio.blocks.Blocks.__exit__ = lambda self, exc_type, exc_value, traceback: None
# ------------------------------------------------------------------------------

from PIL import Image
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.florence2 import Florence2Model

# Initialize models once
vilt_model = RobustViLT(model_name="models/vilt_finetuned_vizwiz")
blip2_model = BLIP2Model()
florence2_model = Florence2Model(model_path="models/florence2-finetuned")

def tts_autoplay(answer: str) -> str:
    """Generate base64‐encoded <audio autoplay> HTML."""
    tts = gTTS(answer, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"""
    <audio controls autoplay>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    """

def transcribe_audio(audio_filepath: str) -> str:
    """Transcribe recorded audio file to text via SpeechRecognition."""
    audio = AudioSegment.from_file(audio_filepath)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

def predict(chat_history, img, text_q, model_name, use_voice, audio_path):
    # Choose which question to use
    question = ""
    if use_voice and audio_path:
        try:
            question = transcribe_audio(audio_path)
        except Exception as e:
            question = ""
    else:
        question = text_q.strip()

    if img is None or not question:
        return chat_history, "", "", None

    # Prepare image
    image = img.convert("RGB")
    if image.size != (384, 384):
        image = image.resize((384, 384))

    # Run the selected model
    try:
        if model_name == "ViLT-finetuned":
            answer = vilt_model.generate_answer(image, question)
        elif model_name == "Florence2-finetuned":
            answer = florence2_model.generate_answer(
                image,
                "Describe the answer in detail.",
                question
            )
        else:  # BLIP2
            answer = blip2_model.generate_answer(image, question)
    except Exception as e:
        answer = f"Error: {e}"
    answer = str(answer)

    # Generate autoplaying TTS snippet
    audio_html = tts_autoplay(answer)

    # Append to chat
    chat_history = chat_history + [(question, answer)]
    # Clear inputs
    return chat_history, "", audio_html, None

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Inclusive VQA Chat Interface")

    with gr.Row():
        model_choice = gr.Radio(
            ["Florence2-finetuned", "ViLT-finetuned", "BLIP2"],
            label="Select Model",
            value="Florence2-finetuned"
        )
        img_input = gr.Image(
            type="pil",
            label="Upload or Capture Image"
        )

    with gr.Row():
        use_voice = gr.Checkbox(label="Use Voice Input")
        text_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question here..."
        )
        audio_input = gr.Audio(
            label="Record Question",
            type="filepath",
            interactive=True
        )

    history = gr.Chatbot(label="Conversation History")
    html_output = gr.HTML()

    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")

    send_btn.click(
        fn=predict,
        inputs=[history, img_input, text_input, model_choice, use_voice, audio_input],
        outputs=[history, text_input, html_output, audio_input]
    )
    clear_btn.click(
        fn=lambda: ([], "", "", None),
        outputs=[history, text_input, html_output, audio_input]
    )

demo.launch()