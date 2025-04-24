import os
import time
import base64
import json
import streamlit as st
from datetime import datetime
from PIL import Image

from modules.robust_vilt import RobustViLT
from modules.florence2 import Florence2Model

# ─── CONFIG ─────────────────────────────────────────────────────────────
IMG_DIR = "images"
SESSION_DIR = "chat_sessions"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

# ─── SESSION STATE INIT ─────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "pending_question": None,
    "pending_image": None,
    "last_uploaded_image": None,
    "upload_count": 0,
    "uploaded_file_id": None,
    "current_session_filename": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── AUTO-SAVE FUNCTION ─────────────────────────────────────────────────
def save_session():
    if st.session_state.messages and st.session_state.current_session_filename:
        with open(os.path.join(SESSION_DIR, st.session_state.current_session_filename), "w") as f:
            json.dump(st.session_state.messages, f, indent=2)

# ─── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Model", ["Florence2-finetuned", "ViLT-finetuned"])
image_source = st.sidebar.radio("Image Source", ["Upload", "Camera"])
upload_key = f"img_uploader_{st.session_state.upload_count}"

if image_source == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key=upload_key)
else:
    uploaded_file = st.sidebar.camera_input("Take a picture")

if st.sidebar.button("🗑️ Clear Chat"):
    for key in ["messages", "pending_image", "pending_question", "uploaded_file_id", "last_uploaded_image", "current_session_filename"]:
        st.session_state[key] = None if key != "messages" else []
    st.rerun()

# ─── SESSION HISTORY: LOAD OLD CHAT ─────────────────────────────────────
st.sidebar.markdown("### 🗂️ Chat Sessions")
session_files = sorted(os.listdir(SESSION_DIR), reverse=True)
selected_session = st.sidebar.selectbox("Load previous session", ["-- Select --"] + session_files)

if selected_session and selected_session != "-- Select --":
    if selected_session != st.session_state.get("current_session_filename"):
        with open(os.path.join(SESSION_DIR, selected_session), "r") as f:
            st.session_state.messages = json.load(f)
        st.session_state.current_session_filename = selected_session
        st.session_state.pending_image = None
        st.session_state.pending_question = None
        st.session_state.uploaded_file_id = None
        st.session_state.last_uploaded_image = None
        st.rerun()

# ─── DOWNLOAD CHAT HISTORY ──────────────────────────────────────────────
st.sidebar.markdown("---")
if st.session_state.messages:
    markdown_lines = ["# 🧑‍💻 Inclusive VQA Chat History\n"]
    for i, msg in enumerate(st.session_state.messages, start=1):
        role = msg["role"].capitalize()
        content = msg.get("content", "")
        image_path = msg.get("image")
        image_md = f"![Image uploaded]({image_path})" if image_path and os.path.exists(image_path) else ""
        markdown_lines.append(f"### {i}. {role}\n{image_md}\n\n{content}\n")
    md_text = "\n".join(markdown_lines)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    st.sidebar.download_button("📥 Download Chat as Markdown", md_text, f"chat_{timestamp}.md", "text/markdown")

    html_lines = ["<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Chat</title></head><body><h1>Chat History</h1>"]
    for i, msg in enumerate(st.session_state.messages, start=1):
        role = msg["role"].capitalize()
        content = msg.get("content", "").replace("\n", "<br>")
        image_path = msg.get("image")
        img_tag = ""
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                b64_img = base64.b64encode(img_file.read()).decode("utf-8")
                img_tag = f"<img src='data:image/png;base64,{b64_img}' width='300'>"
        html_lines.append(f"<h3>{i}. {role}</h3>{img_tag}<p>{content}</p><hr>")
    html_lines.append("</body></html>")
    st.sidebar.download_button("🌐 Download Chat as HTML", "\n".join(html_lines), f"chat_{timestamp}.html", "text/html")


# ─── FOOTER IN SIDEBAR ───────────────────────────────────────────────────────────── 
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <small>Developed by:</small><br>
            <small>Zagarsuren Sukhbaatar</small><br>
            <small><a href="mailto:zagarsuren.sukhbaatar@student.uts.edu.au">zagarsuren.sukhbaatar<br>@student.uts.edu.au</a></small><br>
        </div>
        """,
        unsafe_allow_html=True
    )


# ─── CHAT UI ───────────────────────────────────────────────────────────
st.title("🧑‍💻 Inclusive VQA Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") and os.path.exists(msg["image"]):
            st.image(msg["image"], use_container_width=True)
        if msg["content"]:
            st.write(msg["content"])

if st.session_state.pending_question and st.session_state.pending_image:
    with st.chat_message("assistant"):
        st.write("Agent is working...")

# ─── IMAGE HANDLING ────────────────────────────────────────────────────
if uploaded_file:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}" if hasattr(uploaded_file, "name") else str(time.time())
    if file_id != st.session_state.uploaded_file_id:
        ts = int(time.time() * 1000)
        image_path = os.path.join(IMG_DIR, f"{ts}.png")
        Image.open(uploaded_file).convert("RGB").save(image_path)

        if st.session_state.current_session_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            st.session_state.current_session_filename = f"session_{timestamp}.json"

        st.session_state.messages.append({"role": "user", "content": "", "image": image_path})
        st.session_state.pending_image = image_path
        st.session_state.last_uploaded_image = image_path
        st.session_state.uploaded_file_id = file_id
        save_session()
        st.rerun()

# ─── INFERENCE ─────────────────────────────────────────────────────────
if st.session_state.pending_question:
    q = st.session_state.pending_question
    img_path = st.session_state.pending_image
    pil_img = Image.open(img_path).convert("RGB").resize((384, 384))

    try:
        if model_option == "ViLT-finetuned":
            model = RobustViLT("models/vilt_finetuned_vizwiz")
            answer = model.generate_answer(pil_img, q)
        elif model_option == "Florence2-finetuned":
            model = Florence2Model("models/florence2-finetuned")
            answer = model.generate_answer(pil_img, "Describe in detail.", q)
    except Exception as e:
        answer = f"❗ Error: {e}"

    if isinstance(answer, dict):
        answer = next(iter(answer.values()), str(answer))

    st.session_state.messages.append({"role": "assistant", "content": answer, "image": None})
    st.session_state.last_uploaded_image = st.session_state.pending_image
    st.session_state.pending_question = None
    st.session_state.pending_image = None
    st.session_state.upload_count += 1
    save_session()
    st.rerun()

# ─── CHAT INPUT ────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about the image…")
if question:
    image_to_use = st.session_state.pending_image or st.session_state.last_uploaded_image
    if image_to_use is None:
        st.error("❗ Please upload an image before asking a question.")
    else:
        if st.session_state.current_session_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            st.session_state.current_session_filename = f"session_{timestamp}.json"

        st.session_state.pending_image = image_to_use
        st.session_state.messages.append({"role": "user", "content": question, "image": None})
        st.session_state.pending_question = question
        save_session()
        st.rerun()
