import os
import time
import shutil
import sqlite3
import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition as sr
import io
from pydub import AudioSegment
from datetime import datetime
from PIL import Image
from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.florence2 import Florence2Model

# ─── CONFIG ─────────────────────────────────────────────────────────────
LOG_DIR = "logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
TIMESTAMP = datetime.now().strftime("%d-%b-%Y-%H-%M")
DB_PATH = os.path.join(LOG_DIR, f"{TIMESTAMP}_chat.db")
os.makedirs(IMG_DIR, exist_ok=True)

# ─── CONNECT TO DB ─────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    role TEXT,
    content TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ─── UTILS ─────────────────────────────────────────────────────────────
def save_message(role: str, content: str, image_file=None):
    img_path = None
    if image_file:
        ts = int(time.time()*1000)
        img_path = os.path.join(IMG_DIR, f"{ts}.png")
        Image.open(image_file).convert("RGB").save(img_path)
    c.execute(
        "INSERT INTO messages (role, content, image_path) VALUES (?, ?, ?)",
        (role, content, img_path)
    )
    conn.commit()

def clear_history():
    c.execute("DELETE FROM messages")
    conn.commit()
    shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR, exist_ok=True)
    for key in ["messages", "upload_count", "pending_image", "pending_question", "uploaded_file_id"]:
        st.session_state[key] = [] if key == "messages" else None
    st.rerun()

def list_logs():
    return sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".db")])

def display_log(db_file):
    st.sidebar.markdown(f"### Log: `{db_file}`")
    db_path = os.path.join(LOG_DIR, db_file)
    conn_log = sqlite3.connect(db_path)
    cursor = conn_log.cursor()
    cursor.execute("SELECT role, content, image_path, timestamp FROM messages ORDER BY id")
    logs = cursor.fetchall()
    for role, content, image_path, timestamp in logs:
        st.sidebar.markdown(f"**{role.title()} ({timestamp}):**")
        if image_path and os.path.exists(image_path):
            st.sidebar.image(image_path, width=200)
        if content:
            st.sidebar.markdown(f"> {content}")
    conn_log.close()

# ─── SESSION STATE INIT ────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "pending_question": None,
    "pending_image": None,
    "upload_count": 0,
    "uploaded_file_id": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.title("Settings")

# Model selection
model_option = st.sidebar.selectbox("Model", ["Florence2-finetuned", "ViLT-finetuned", "BLIP2"])

# Upload or Camera
image_source = st.sidebar.radio("Image Source", ["Upload", "Camera"])
upload_key = f"img_uploader_{st.session_state.upload_count}"

if image_source == "Upload":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"], key=upload_key
    )
else:
    uploaded_file = st.sidebar.camera_input("Take a picture")

# View Logs
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Chat Logs")
log_files = list_logs()
selected_log = st.sidebar.selectbox("View previous logs", log_files)
if selected_log:
    display_log(selected_log)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Current Log"):
    clear_history()

# ─── CHAT UI ───────────────────────────────────────────────────────────
st.title("🧑‍💻 Inclusive VQA Chat")

# Show current chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], use_container_width=True)
        if msg["content"]:
            st.write(msg["content"])

# Prevent duplicate image injection using uploaded_file_id (filename + size)
if uploaded_file:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if file_id != st.session_state.uploaded_file_id:
        st.session_state.messages.append({
            "role": "user",
            "content": "",
            "image": uploaded_file
        })
        save_message("user", "", image_file=uploaded_file)
        st.session_state.pending_image = uploaded_file
        st.session_state.uploaded_file_id = file_id
        st.rerun()

# ─── Inference ─────────────────────────────────────────────────────────
if st.session_state.pending_question:
    q = st.session_state.pending_question
    img_file = st.session_state.pending_image

    pil_img = Image.open(img_file).convert("RGB").resize((384,384))
    try:
        if model_option == "ViLT-finetuned":
            model = RobustViLT("models/vilt_finetuned_vizwiz")
            answer = model.generate_answer(pil_img, q)
        elif model_option == "Florence2-finetuned":
            model = Florence2Model("models/florence2-finetuned")
            answer = model.generate_answer(pil_img, "Describe in detail.", q)
        else:
            model = BLIP2Model()
            answer = model.generate_answer(pil_img, q)
    except Exception as e:
        answer = f"❗ Error: {e}"

    if isinstance(answer, dict):
        answer = answer.get("", str(answer))

    st.session_state.messages[-1] = {
        "role": "assistant",
        "content": answer,
        "image": None
    }
    save_message("assistant", answer)
    st.session_state.pending_question = None
    st.session_state.pending_image = None
    st.session_state.upload_count += 1
    st.rerun()

# ─── User Question ─────────────────────────────────────────────────────
question = st.chat_input("Ask a question about the image…")

# ─── Audio Input ─────────────────────────────────────────────────────
st.markdown("### Or ask using your voice:")
audio = audiorecorder("🎙️ Record Voice", "Stop Recording")

if audio:
    st.success("Voice recorded. Transcribing...")
    audio.export("temp_audio.wav", format="wav")
    
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = recognizer.record(source)

    try:
        voice_text = recognizer.recognize_google(audio_data)
        st.success(f"You said: {voice_text}")

        # Inject voice input as a regular question
        if st.session_state.pending_image is None:
            st.error("❗ Please upload an image before asking a question.")
        else:
            st.session_state.messages.append({"role": "user", "content": voice_text, "image": None})
            save_message("user", voice_text)
            st.session_state.messages.append({"role": "assistant", "content": "Agent is working...", "image": None})
            save_message("assistant", "Agent is working...")
            st.session_state.pending_question = voice_text
            st.rerun()

    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech Recognition failed: {e}")


if question:
    if st.session_state.pending_image is None:
        st.error("❗ Please upload an image before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question, "image": None})
        save_message("user", question)
        st.session_state.messages.append({"role": "assistant", "content": "Agent is working...", "image": None})
        save_message("assistant", "Agent is working...")
        st.session_state.pending_question = question
        st.rerun()