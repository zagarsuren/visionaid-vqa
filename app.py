import os
import time
import base64
import json
import streamlit as st
from datetime import datetime
from PIL import Image
import sqlite3
import bcrypt

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

# ─── DATABASE FUNCTIONS ─────────────────────────────────────────────────
def create_connection():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    return conn

def create_table(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        login_time TEXT
    )
    """)
    conn.commit()

def insert_user(conn, email, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn.execute("INSERT INTO users (email, password, login_time) VALUES (?, ?, ?)", 
                 (email, hashed.decode(), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()

def verify_user(conn, email, password):
    cursor = conn.execute("SELECT password FROM users WHERE email=?", (email,))
    row = cursor.fetchone()
    if row and bcrypt.checkpw(password.encode(), row[0].encode()):
        return True
    return False

# ─── EMAIL VALIDATION ──────────────────────────────────────────────────────
import re

def is_valid_email(email):
    # Simple regex for standard email format
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
    return re.match(pattern, email) is not None


# ─── DEMO PAGE ──────────────────────────────────────────────────────────
def demo_page():
    # st.markdown('<div class="section-title">Demo: Inclusive AI Chat</div>', unsafe_allow_html=True)
    # st.write("""
    # Try out our inclusive Visual Question Answering system powered by AI. Upload an image and ask any question about it.
    # """)
    st.sidebar.markdown("---")
    # st.sidebar.title("VQA Assistant")
    model_option = st.sidebar.selectbox("Model", ["Florence2", "ViLT"])
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

    # Display Chat Sessions
    st.sidebar.markdown("### 🗂️ Chat Sessions")

    # Get session filenames and remove .json for display
    session_files = sorted(os.listdir(SESSION_DIR), reverse=True)
    session_display_names = [f.replace(".json", "") for f in session_files]

    # Add "-- Select --" option at the beginning
    selected_display_name = st.sidebar.selectbox("Load previous session", ["-- Select --"] + session_display_names)

    # Match display name back to real filename
    if selected_display_name and selected_display_name != "-- Select --":
        real_filename = selected_display_name + ".json"
        if real_filename != st.session_state.get("current_session_filename"):
            with open(os.path.join(SESSION_DIR, real_filename), "r") as f:
                st.session_state.messages = json.load(f)
            st.session_state.current_session_filename = real_filename
            st.session_state.pending_image = None
            st.session_state.pending_question = None
            st.session_state.uploaded_file_id = None
            st.session_state.last_uploaded_image = None
            st.rerun()


    st.sidebar.markdown("---")
    if st.session_state.messages:
        markdown_lines = ["# 🧑‍💻 VQA Chat History\n"]
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

    st.title("🧑‍💻 Inclusive VQA Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image") and os.path.exists(msg["image"]):
                img = Image.open(msg["image"]).resize((400, 400))
                st.image(img)
            if msg["content"]:
                st.write(msg["content"])

    if st.session_state.pending_question and st.session_state.pending_image:
        with st.chat_message("assistant"):
            st.write("VQA assistant is working...")

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

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        img_path = st.session_state.pending_image
        pil_img = Image.open(img_path).convert("RGB").resize((384, 384))

        try:
            if model_option == "ViLT":
                model = RobustViLT("models/vilt_finetuned_vizwiz")
                answer = model.generate_answer(pil_img, q)
            elif model_option == "Florence2":
                model = Florence2Model("models/florence2-finetuned")
                answer = model.generate_answer(pil_img, "Describe the answer in detail.", q)
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

# ─── HOME PAGE ──────────────────────────────────────────────────────────
def home_page():
    st.title("🏠 Welcome to VQA Assistant")
    st.write("""
    This portfolio showcases interactive AI tools and research projects developed by me. 
    Explore the Demo to test an AI-powered visual question answering chatbot, or learn more about the purpose behind this work in the About section.
    
    Technologies used include:
    - Streamlit for frontend
    - ViLT, Florence2 for model inference
    """)

# ─── ABOUT PAGE ─────────────────────────────────────────────────────────
def about_page():
    st.title("ℹ️ About This Project")
    st.write("""
    This web app was built as part of an inclusive AI research initiative to improve digital accessibility for people with visual impairments. 
    The chatbot uses cutting-edge Visual Question Answering (VQA) models to interpret images and respond to user questions in natural language.

    Features:
    - Upload or capture images
    - Ask questions using text
    - Receive contextual answers powered by Florence2 and ViLT models
    
    Developed by: Zagarsuren Sukhbaatar  
    Research focus: Assistive AI, Computer Vision, Responsible Technology
    """)
    # Profile Image
    st.image("/Users/zagaraa/Documents/GitHub/visionaid-vqa/assets/profile/Photo-grey.png", use_container_width=False, width=300)

# ─── MAIN FUNCTION ───────────────────────────────────────────────────────
def main():
    # DB Setup
    conn = create_connection()
    create_table(conn)

    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""

    # Sidebar Logo
    st.sidebar.image("/Users/zagaraa/Documents/GitHub/visionaid-vqa/assets/logo/vqa-logo.png", use_container_width=True)

    # Sidebar Title
    st.sidebar.title("Inclusive VQA Assistant")

    st.sidebar.markdown("""
    <style>
    .menu-button {
        display: block;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        margin: 8px 0;
        text-align: center;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        text-decoration: none;
    }
    .menu-button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    # ─── SIDEBAR MENU ─────────────────────────────────────────────────────
    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as {st.session_state.user_email}")
        if st.sidebar.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("ℹ️ About"):
            st.session_state.page = "About"
        if st.sidebar.button("🧪 Demo"):
            st.session_state.page = "Demo"
        if st.sidebar.button("🔓 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "Home"
            st.rerun()
    else:
        st.sidebar.subheader("🔐 Login to Continue")
        login_email = st.sidebar.text_input("Email", key="login_email")
        login_password = st.sidebar.text_input("Password", type="password", key="login_password")

        if st.sidebar.button("Login"):
            if verify_user(conn, login_email, login_password):
                st.session_state.logged_in = True
                st.session_state.user_email = login_email
                st.sidebar.success(f"Welcome {login_email}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid email or password")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🚀 Create New Account")
        new_email = st.sidebar.text_input("New Email", key="new_email")
        new_password = st.sidebar.text_input("New Password", type="password", key="new_password")

        if st.sidebar.button("Create Account"):
            if not is_valid_email(new_email):
                st.sidebar.error("❌ Please enter a valid email address.")
            elif len(new_password) < 6:
                st.sidebar.error("❌ Password must be at least 6 characters long.")
            else:
                try:
                    insert_user(conn, new_email, new_password)
                    st.sidebar.success("✅ Account created successfully! You can now log in.")
                except sqlite3.IntegrityError:
                    st.sidebar.error("❌ Account with this email already exists.")


    # ─── PAGE RENDERING ─────────────────────────────────────────────────────
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "About":
        about_page()
    elif st.session_state.page == "Demo":
        if st.session_state.logged_in:
            demo_page()
        else:
            st.warning("⚠️ Please log in to access the Demo page.")
    elif st.session_state.page == "Login":
        st.title("🔐 Login Page (Use sidebar to login)")

if __name__ == "__main__":
    main()