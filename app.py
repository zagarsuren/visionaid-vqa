import os
import time
import json
import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
from PIL import Image
import re
import io
from gtts import gTTS
import base64

from modules.robust_vilt import RobustViLT
from modules.florence2 import Florence2Model

# ─── CONFIG ─────────────────────────────────────────────────────────────
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)
DB_PATH = 'users.db'

# ─── DATABASE FUNCTIONS ─────────────────────────────────────────────────
def create_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_tables(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            login_time TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            created_at TEXT NOT NULL,
            messages TEXT NOT NULL
        )
        """
    )
    conn.commit()

def insert_user(conn, email, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn.execute(
        "INSERT INTO users (email, password, login_time) VALUES (?, ?, ?)",
        (email, hashed.decode(), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()

def verify_user(conn, email, password):
    cursor = conn.execute("SELECT password FROM users WHERE email=?", (email,))
    row = cursor.fetchone()
    return bool(row and bcrypt.checkpw(password.encode(), row[0].encode()))

def insert_session(conn, user_email, messages):
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute(
        "INSERT INTO sessions (user_email, created_at, messages) VALUES (?, ?, ?)",
        (user_email, created_at, json.dumps(messages))
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

def update_session(conn, session_id, messages):
    conn.execute(
        "UPDATE sessions SET messages=? WHERE id=?",
        (json.dumps(messages), session_id)
    )
    conn.commit()

def get_user_sessions(conn, user_email):
    return conn.execute(
        "SELECT id, created_at FROM sessions WHERE user_email=? ORDER BY created_at DESC",
        (user_email,)
    ).fetchall()

def get_session_messages(conn, session_id):
    row = conn.execute("SELECT messages FROM sessions WHERE id=?", (session_id,)).fetchone()
    return json.loads(row[0]) if row else []

# ─── EMAIL VALIDATION ──────────────────────────────────────────────────────
def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
    return re.match(pattern, email) is not None

# ─── SESSION SAVE ─────────────────────────────────────────────────────────
def save_session():
    messages = st.session_state.messages
    if st.session_state.session_id:
        update_session(conn, st.session_state.session_id, messages)
    else:
        sid = insert_session(conn, st.session_state.user_email, messages)
        st.session_state.session_id = sid

# ─── AUDIO PLAY ─────────────────────────────────────────────────────────
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


# ─── DEMO PAGE ───────────────────────────────────────────────────────────
def demo_page():
    st.sidebar.markdown("---")
    model_option = st.sidebar.selectbox("Model", ["Florence2", "ViLT"])
    image_source = st.sidebar.radio("Image Source", ["Upload", "Camera"])
    upload_key = f"img_uploader_{st.session_state.upload_count}"

    uploaded_file = (
        st.sidebar.file_uploader("Upload Image", type=["jpg","jpeg","png"], key=upload_key)
        if image_source == "Upload" else st.sidebar.camera_input("Take a picture")
    )

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.pending_image = None
        st.session_state.pending_question = None
        st.session_state.last_uploaded_image = None
        st.session_state.uploaded_file_id = None
        st.session_state.session_id = None
        st.session_state.upload_count = 0
        st.rerun()

    st.sidebar.markdown("### 🗂️ Chat Sessions")
    sessions = get_user_sessions(conn, st.session_state.user_email)
    sessions_dict = {sid: ts for sid, ts in sessions}
    options = [None] + list(sessions_dict.keys())
    selected_sid = st.sidebar.selectbox(
        "Load previous session", options,
        format_func=lambda x: "-- Select --" if x is None else sessions_dict[x]
    )
    if selected_sid and selected_sid != st.session_state.session_id:
        st.session_state.messages = get_session_messages(conn, selected_sid)
        st.session_state.session_id = selected_sid
        for key in ["pending_image","pending_question","uploaded_file_id","last_uploaded_image"]:
            st.session_state[key] = None
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


    st.title("🧑‍💻 Inclusive VQA Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image") and os.path.exists(msg["image"]):
                st.image(Image.open(msg["image"]).resize((400,400)))
            if msg.get("content"):
                st.write(msg["content"])

    if st.session_state.pending_question and st.session_state.pending_image:
        with st.chat_message("assistant"):
            st.write("VQA assistant is working...")

    if uploaded_file:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}" if hasattr(uploaded_file,"name") else str(time.time())
        if file_id != st.session_state.uploaded_file_id:
            ts = int(time.time()*1000)
            image_path = os.path.join(IMG_DIR, f"{ts}.png")
            Image.open(uploaded_file).convert("RGB").save(image_path)
            st.session_state.messages.append({"role":"user","content":"","image":image_path})
            st.session_state.pending_image = image_path
            st.session_state.last_uploaded_image = image_path
            st.session_state.uploaded_file_id = file_id
            save_session()
            st.rerun()

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        img = Image.open(st.session_state.pending_image).convert("RGB").resize((384,384))
        try:
            if model_option == "ViLT":
                model = RobustViLT("models/vilt_finetuned_vizwiz")
                answer = model.generate_answer(img, q)
            else:
                model = Florence2Model("models/florence2-finetuned")
                answer = model.generate_answer(img, "Describe the answer in detail.", q)
        except Exception as e:
            answer = f"❗ Error: {e}"
        if isinstance(answer, dict):
            answer = next(iter(answer.values()), answer)
        st.session_state.messages.append({"role":"assistant","content":answer,"image":None})
        st.session_state.last_uploaded_image = st.session_state.pending_image
        st.session_state.pending_question = None
        st.session_state.pending_image = None
        st.session_state.upload_count += 1
        save_session()
        # st.rerun()

        # display answer
        with st.chat_message("assistant"):
            st.write(answer)

        # ---------- Text-to-Speech: Clean Answer and Convert to Audio, Save it, and Autoplay ----------
        try:
            prompt_to_remove = "{'Describe the answer in detail.'"
            if prompt_to_remove in answer:
                answer = answer.replace(prompt_to_remove, "").strip()
            audio_dir = os.path.join("assets", "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_file_path = os.path.join(audio_dir, "speech.mp3")
            tts = gTTS(answer, lang='en')
            tts.save(audio_file_path)
            # st.success("Audio generated and saved successfully!")
            autoplay_audio(audio_file_path)
        except Exception as e:
            st.error(f"Text-to-speech conversion failed: {e}")

        time.sleep(5)  # Wait for 5 seconds before clearing the assistant message to allow TTS to finish
        st.rerun() # Rerun to clear the assistant message. Comment if want to use full TTS.

    question = st.chat_input("Upload or Capture an image, then ask a question…")
    if question:
        if not (st.session_state.pending_image or st.session_state.last_uploaded_image):
            st.error("❗ Please upload an image before asking a question.")
        else:
            st.session_state.pending_image = st.session_state.pending_image or st.session_state.last_uploaded_image
            st.session_state.messages.append({"role":"user","content":question,"image":None})
            st.session_state.pending_question = question
            save_session()
            st.rerun()

# ─── HOME PAGE ───────────────────────────────────────────────────────────
def home_page():
    st.title("Welcome to the Inclusive VQA Assistant")
    st.subheader("From pixels to answers—let your images speak!")  # H3 under the main title

    # Display the selected image
    st.image("assets/demo/app.jpeg", use_container_width=True)

    st.write("""
    This is your gateway to accessible AI-powered image understanding.
    """)

    st.markdown("**What You Can Do Here:**")
    st.markdown("""
    - **Upload or Snap a Photo**: Provide an image via uploader or camera.  
    - **Ask Natural-Language Questions**: Anything from “What objects do you see?” to “Describe the scene in detail.”  
    - **Get Instant, Contextual Answers**: Powered by ViLT & Florence2 models.  
    """)

    st.markdown("**Key Features:**")
    st.markdown("""
    - Model choice: switch between ViLT and Florence2  
    - Session management: save, revisit, and export your Q&A history  
    - Exportable chats: download as Markdown or HTML  
    """)

    st.markdown("**Get Started:**")
    st.markdown("""
    1. Create an account (use the sidebar)  
    2. Log in and click **Demo**  
    3. Upload an image → Ask a question → Get answer about the image  
    """)

# ─── ABOUT PAGE ──────────────────────────────────────────────────────────
def about_page():
    st.title("About This Project")

    st.markdown("**Our Mission**")
    st.write("""
    To lower barriers to visual information by harnessing cutting-edge AI, making image understanding seamless and democratic for everyone—especially those who rely on assistive technologies.
    """)

    st.markdown("**How It Works**")
    st.write("""
    1. **Preprocessing:** images are resized & normalized.  
    2. **Model Inference:**  
       - **ViLT:** transformer-based vision+language encoder  
       - **Florence2:** multimodal fusion for richer context  
    3. **Answer Generation:** a language head streams back contextual answers.
    """)

    st.markdown("**Technology Stack**")
    st.markdown("""
    - **Frontend:** Streamlit  
    - **Models:** ViLT-finetuned, Florence2-finetuned  
    - **Backend:** SQLite for secure user/session management  
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='color:steelblue;'>Developer</h3>", unsafe_allow_html=True)
        # st.image("assets/profile/Photo-grey.png", width=200)
        st.markdown("<strong>Zagarsuren Sukhbaatar</strong>", unsafe_allow_html=True)
        st.write("""
        Master of AI, Sub-Major in Computer Vision, University of Technology Sydney (UTS).
        """ )
    with col2:
        st.markdown("<h3 style='color:steelblue;'>Supervisor</h3>", unsafe_allow_html=True)
        # st.image("assets/profile/Photo-grey.png", width=200)
        st.markdown("<strong>Nabin Sharma</strong>", unsafe_allow_html=True)
        st.write("""
        Associate Professor, University of Technology Sydney (UTS).
        """ )
# ─── MAIN FUNCTION ───────────────────────────────────────────────────────
def main():
    global conn
    conn = create_connection()
    create_tables(conn)

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    for key, val in {
        "messages": [],
        "pending_question": None,
        "pending_image": None,
        "last_uploaded_image": None,
        "upload_count": 0,
        "uploaded_file_id": None,
        "session_id": None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Sidebar Logo
    st.sidebar.image("/Users/zagaraa/Documents/GitHub/visionaid-vqa/assets/logo/vqa-logo.png", use_container_width=True)
    # Sidebar Title
    st.sidebar.title("Inclusive VQA Assistant")
    st.sidebar.markdown("---")
    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as {st.session_state.user_email}")
        if st.sidebar.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("ℹ️ About"):
            st.session_state.page = "About"
        if st.sidebar.button("🤖 Demo"):
            st.session_state.page = "Demo"
            st.session_state.messages = []
            st.session_state.session_id = None
        if st.sidebar.button("🔓 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "Home"
            st.session_state.user_email = ""
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.pending_image = None
            st.session_state.last_uploaded_image = None
            st.session_state.upload_count = 0
            st.session_state.uploaded_file_id = None
            st.session_state.session_id = None
            st.rerun()
    else:
        st.sidebar.subheader("🔐 Login")
        login_email = st.sidebar.text_input("Email", key="login_email")
        login_password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            if verify_user(conn, login_email, login_password):
                st.session_state.logged_in = True
                st.session_state.user_email = login_email
                st.session_state.messages = []
                st.session_state.session_id = None
                st.session_state.upload_count = 0
                st.sidebar.success(f"Welcome {login_email}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚀 Create Account")
        new_email = st.sidebar.text_input("New Email", key="new_email")
        new_password = st.sidebar.text_input("New Password", type="password", key="new_password")
        if st.sidebar.button("Sign Up"):
            if not is_valid_email(new_email):
                st.sidebar.error("Enter a valid email.")
            elif len(new_password) < 6:
                st.sidebar.error("Password must be ≥6 chars.")
            else:
                try:
                    insert_user(conn, new_email, new_password)
                    st.sidebar.success("Account created! Log in above.")
                except sqlite3.IntegrityError:
                    st.sidebar.error("Email already registered.")

    # Render pages
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "About":
        about_page()
    elif st.session_state.page == "Demo":
        if st.session_state.logged_in:
            demo_page()
        else:
            st.warning("⚠️ Log in to access Demo.")

if __name__ == "__main__":
    main()