import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
import torch
from modules.robust_vilt import RobustViLT
from modules.blip2 import BLIP2Model
from modules.florence2 import Florence2Model  # Florence2Model with three-argument generate_answer

def run_vqa_demo():
    st.title("Inclusive VQA System for Visually Impaired Users")
    st.write("Choose a model and ask a question about an image.")
    
    # Model selection options
    model_option = st.selectbox(
        "Select Model",
        ("ViLT-finetuned", "Florence2-finetuned", "BLIP2", )
    )
    
    # Option for image source: Upload or Camera Capture
    image_source = st.radio("Image Source", options=["Upload", "Camera Capture"])
    
    if image_source == "Upload":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capture an image")
    
    question = st.text_input("Enter your question:")
    
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
            if model_option == "ViLT-finetuned":
                model = RobustViLT(model_name="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/vilt_finetuned_vizwiz_ocr")
                answer = model.generate_answer(image, question)
            elif model_option == "Florence2-finetuned":
                model = Florence2Model(model_path="/Users/zagaraa/Documents/GitHub/visionaid-vqa/models/florence2-finetuned")
                # For Florence2, use a fixed task prompt and treat the user question as additional input.
                task_prompt = "Answer the questions in detail:"
                answer = model.generate_answer(image, task_prompt, question)
            elif model_option == "BLIP2":
                model = BLIP2Model()
                answer = model.generate_answer(image, question)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            answer = "Unknown"
        
        st.image(image, caption="Captured/Uploaded Image", use_column_width=True)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer ({model_option}):** {answer}")

def main():
    # Create top menus using tabs
    tabs = st.tabs(["Home", "Demo", "About"])
    
    with tabs[0]:
        st.title("Welcome to the Inclusive VQA System")
        st.markdown("""
        **Inclusive Visual Question Answering (VQA) for Visually Impaired Users**

        This application is designed to assist visually impaired users in interpreting images by asking questions about their content. Our system utilizes advanced computer vision models to generate detailed answers, helping users gain better insight into the visual world around them.

        **How It Works:**
        - **Home:** An overview of our mission and the underlying technology.
        - **Demo:** Try out the system by uploading or capturing an image and asking a question.
        - **About:** Get to know the developer and find contact information.

        Use the tabs above to navigate through the app.
        """)

    with tabs[1]:
        run_vqa_demo()
        
    with tabs[2]:
        st.title("About")
        st.markdown("""
        - **Developer:** Zagasuren Sukhbaatar       
        - **University:** Univsity of Technology Sydney (UTS)   
        - **Degree:** Master of AI, Sub-major in Computer Vision                 
        - **Project:** Inclusive Visual Question Answering (VQA) System             
        - **Supervisor:** Dr. Nabin Sharma              
        - **Contact:** zagarsuren.sukhbaatar@student.uts.edu.au
        """)
        # Attempt to display a profile photo; make sure 'profile_photo.jpg' exists in your working directory.
        try:
            profile_image = Image.open("/Users/zagaraa/Documents/GitHub/visionaid-vqa/assets/profile/Photo-grey.png")
            st.image(profile_image, caption="Developer: Zagarsuren Sukhbaatar", width=200)
        except Exception as e:
            st.warning("Profile photo not found. Please add a 'profile_photo.jpg' file to the working directory.")

    # Footer using HTML/CSS styling to fix it at the bottom
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #333;
        text-align: center;
        padding: 10px 0;
        border-top: 1px solid #e2e2e2;
    }
    </style>
    <div class="footer">
        <p>Contact: zagarsuren.sukhbaatar@student.uts.edu.au | &copy; 2025 University of Technology Sydney (UTS). All rights reserved.</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()