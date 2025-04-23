import os
import time
import streamlit as st
from datetime import datetime
from PIL import Image

from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.florence2 import Florence2Model

# ─── CONFIG ─────────────────────────────────────────────────────────────
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# ─── SESSION STATE INIT ─────────────────────────────────────────────────
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
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key=upload_key)
else:
    uploaded_file = st.sidebar.camera_input("Take a picture")

if st.sidebar.button("🗑️ Clear Chat"):
    for key in ["messages", "pending_image", "pending_question", "uploaded_file_id"]:
        st.session_state[key] = None if key != "messages" else []
    st.rerun()

# ─── Download Chat History ─────────────────────────────────────────────
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
    md_filename = f"chat_history_{timestamp}.md"

    st.sidebar.download_button(
        label="📥 Download Chat as Markdown",
        data=md_text,
        file_name=md_filename,
        mime="text/markdown"
    )

import base64

# ─── Download Chat History as HTML (Base64 embedded images) ──────────────
if st.session_state.messages:
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'><title>Inclusive VQA Chat</title></head><body>",
        "<h1>🧑‍💻 Inclusive VQA Chat History</h1>"
    ]

    for i, msg in enumerate(st.session_state.messages, start=1):
        role = msg["role"].capitalize()
        content = msg.get("content", "").replace("\n", "<br>")
        image_path = msg.get("image")
        image_tag = ""

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = "image/png"
                image_tag = f"<img src='data:{mime_type};base64,{img_base64}' width='300'>"

        html_lines.append(f"<h3>{i}. {role}</h3>{image_tag}<p>{content}</p><hr>")

    html_lines.append("</body></html>")
    html_text = "\n".join(html_lines)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    html_filename = f"chat_history_{timestamp}.html"

    st.sidebar.download_button(
        label="🌐 Download Chat as HTML",
        data=html_text,
        file_name=html_filename,
        mime="text/html"
    )


# ─── CHAT UI ───────────────────────────────────────────────────────────
st.title("🧑‍💻 Inclusive VQA Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") and os.path.exists(msg["image"]):
            st.image(msg["image"], use_container_width=True)
        if msg["content"]:
            st.write(msg["content"])

# ─── Image Upload Handling ─────────────────────────────────────────────
if uploaded_file:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}" if hasattr(uploaded_file, "name") else str(time.time())
    if file_id != st.session_state.uploaded_file_id:
        ts = int(time.time() * 1000)
        image_path = os.path.join(IMG_DIR, f"{ts}.png")
        Image.open(uploaded_file).convert("RGB").save(image_path)

        st.session_state.messages.append({
            "role": "user",
            "content": "",
            "image": image_path
        })
        st.session_state.pending_image = image_path
        st.session_state.uploaded_file_id = file_id
        st.rerun()

# ─── Inference ─────────────────────────────────────────────────────────
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
        else:
            model = BLIP2Model()
            answer = model.generate_answer(pil_img, q)
    except Exception as e:
        answer = f"❗ Error: {e}"

    if isinstance(answer, dict):
        answer = next(iter(answer.values()), str(answer))  # get the first value only

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image": None
    })

    st.session_state.pending_question = None
    st.session_state.pending_image = None
    st.session_state.upload_count += 1
    st.rerun()

# ─── User Text Input ───────────────────────────────────────────────────
question = st.chat_input("Ask a question about the image…")

if question:
    if st.session_state.pending_image is None:
        st.error("❗ Please upload an image before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question, "image": None})
        st.session_state.messages.append({"role": "assistant", "content": "Agent is working...", "image": None})
        st.session_state.pending_question = question
        st.rerun()
