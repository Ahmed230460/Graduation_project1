# -*- coding: utf-8 -*-
"""
app.py

This is the main application file for the Video Story Generator.
It processes uploaded videos, detects anomalies, generates a story, provides translation options,
and displays detailed information about the video, including frame counts and detected objects.
"""

import streamlit as st
import cv2
import numpy as np
import av
import torch
from ultralytics import YOLO
from transformers import VivitForVideoClassification, VivitImageProcessor
from transformers import MarianMTModel, MarianTokenizer  # Imports for translation
import google.generativeai as genai
import os
import io
from PIL import Image
from google.api_core.exceptions import ResourceExhausted
from collections import Counter

# Add custom CSS for styling the UI
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #FF4B4B;
        text-align: center;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader label {
        color: white !important;
    }
    .logo-container {
        position: absolute;  /* استخدام absolute لتحديد الموقع بدقة */
        top: 20px;  /* المسافة من أعلى الصفحة */
        left: 300px;  /* المسافة من الشمال (تحريك اللوجو لليمين) */
        display: flex;
        flex-direction: column;
        z-index: 999;  /* التأكد إن اللوجو فوق العناصر التانية */
    }
    .center-image {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .center-image img {
        width: 300px;  /* Set a fixed width */
        height: auto;  /* Maintain aspect ratio */
        max-width: 100%;  /* Ensure it scales down on smaller screens */
    }
    .center-text {
        text-align: center;
        width: 100%;
    }
    .content-container {
        width: 100%;
        max-width: 1200px;  /* Limit the width of the main content */
        margin: 0 auto;  /* Center the content within the main area */
        padding: 0 20px;
        margin-top: 150px;  /* مسافة كافية عشان المحتوى ما يتداخلش مع اللوجو */
    }
    .sidebar .sidebar-content {
        background-color: #2A2A2A;
        text-align: center;
    }
    .developer-card {
        background-color: #2A2A2A;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .developer-card img {
        border-radius: 50%;
        width: 100px;
        height: 100px;
        object-fit: cover;
    }
    .footer {
        text-align: center;
        padding: 20px;
        background-color: #2A2A2A;
        margin-top: 40px;
        border-radius: 10px 10px 0 0;
    }
    a {
        color: #FF4B4B;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    /* Hover effect for Usage Instructions */
    .instructions-container {
        position: relative;
        display: inline-block;
    }
    .instructions-title {
        color: #FF4B4B;
        cursor: pointer;
    }
    .instructions-tooltip {
        visibility: hidden;
        width: 300px;
        background-color: #2A2A2A;
        color: white;
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        position: absolute;
        z-index: 1;
        top: 100%;
        left: 0;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .instructions-container:hover .instructions-tooltip {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation Sidebar (without the logo)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Us", "Contact Us"])

# Function to display the logo at the top of each page
def display_logo():
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.markdown('<div class="center-image">', unsafe_allow_html=True)
    st.image("logo_transparent.png")  # Width is controlled via CSS
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="center-text">Echolens: Turning Videos into Stories with AI</h3>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Display the logo on every page
display_logo()

# Home Page
if page == "Home":
    # Wrap the rest of the content in a centered container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    # Display user instructions with hover effect
    st.markdown("""
    <div class="instructions-container">
        <h3 class="instructions-title">Usage Instructions</h3>
        <span class="instructions-tooltip">
            1. Upload a video (up to 200 MB) in MP4 or AVI format.<br>
            2. Wait for the video to be processed.<br>
            3. The results (classification, story, and video details) will appear below in English.<br>
            4. Use the buttons to translate the story into other languages.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Set the title of the app
    st.title("Video Story Generator")

    # File uploader for video input
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    # Main function to handle the app logic
    def main():
        if uploaded_file is not None:
            # Save the uploaded video
            with open("input_video.mp4", "wb") as f:
                f.write(uploaded_file.read())

            # Process the video with a spinner
            with st.spinner("Processing..."):
                yolo_model, processor, vivit_model, gemini_model = load_models()
                
                # Calculate original video duration and frame count
                original_duration, original_frame_count = get_video_info("input_video.mp4")
                
                # Preprocess the video
                preprocessed_video = preprocess_video("input_video.mp4", "preprocessed_video.mp4")
                
                # Extract keyframes and collect event timings and objects
                keyframes_video, event_timestamps, object_counts, saved_frames = extract_keyframes(preprocessed_video, yolo_model)
                
                # Calculate keyframes video duration and frame count
                keyframes_duration, keyframes_frame_count = get_video_info(keyframes_video) if keyframes_video else (0, 0)
                
                # Detect anomaly
                prediction = anomaly_detection(keyframes_video, processor, vivit_model)
                
                # Generate story in English based on the keyframes video
                story = generate_story(keyframes_video, prediction, gemini_model)

            # Display the processed video
            st.video("input_video.mp4")

            # Display video details in an expander
            with st.expander("Video Details"):
                st.subheader("Video Information")
                st.write(f"**Original Video Duration**: {original_duration:.2f} seconds")
                st.write(f"**Original Video Frame Count**: {original_frame_count}")
                st.write(f"**Keyframes Video Duration**: {keyframes_duration:.2f} seconds")
                st.write(f"**Keyframes Video Frame Count**: {keyframes_frame_count}")
                st.write("**Event Timings (seconds)**:")
                if event_timestamps:
                    for i, timestamp in enumerate(event_timestamps, 1):
                        st.write(f"- Event {i}: {timestamp:.2f} seconds")
                else:
                    st.write("- No significant events detected.")
                st.write("**Detected Objects (by YOLO)**:")
                if object_counts:
                    for obj, count in object_counts.items():
                        st.write(f"- {obj}: {count}")
                else:
                    st.write("- No objects detected.")

            # Display results in an expander
            with st.expander("Processing Results"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Classification")
                    st.write(prediction)
                with col2:
                    st.subheader("Story (English)")
                    st.write(story)

            # Add translation buttons for the story
            st.subheader("Translate Story")
            target_languages = ["es", "fr", "de", "ar"]
            language_names = {"es": "Spanish", "fr": "French", "de": "German", "ar": "Arabic"}
            cols = st.columns(len(target_languages))  # Create columns for buttons
            translations = {}  # Store translations to avoid re-computing

            for idx, lang in enumerate(target_languages):
                with cols[idx]:
                    if st.button(f"To {language_names[lang]}"):
                        if lang not in translations:
                            # Translate the story to the target language
                            with st.spinner(f"Translating to {language_names[lang]}..."):
                                translated_story = translate_text(story, lang)
                                if translated_story:
                                    translations[lang] = translated_story
                        # Display the translated story if available
                        if lang in translations:
                            st.write(f"Story ({language_names[lang]}):")
                            st.write(translations[lang])

            # Add a download button for the story (English version)
            st.download_button(
                label="Download Story (English)",
                data=story,
                file_name="video_description.txt",
                mime="text/plain"
            )

    # Function to load models with caching to avoid reloading
    @st.cache_resource
    def load_models():
        # Load YOLOv8 model
        yolo_model = YOLO("yolov8n.pt")  # Load directly from Ultralytics
        
        # Load ViViT model for anomaly detection
        processor = VivitImageProcessor.from_pretrained("prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime", token=os.getenv("HF_TOKEN"))
        vivit_model = VivitForVideoClassification.from_pretrained("prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime", token=os.getenv("HF_TOKEN"))

        # Configure Gemini model with API key
        genai.configure(api_key="AIzaSyCZFf2r-fmE9uRQjKebHfF_MZhDKwiZP7A")
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        return yolo_model, processor, vivit_model, gemini_model

    # Function to calculate video duration and frame count
    def get_video_info(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps == 0:
            return 0, frame_count
        duration = frame_count / fps
        return duration, frame_count

    # Function to preprocess the video (resize and normalize)
    def preprocess_video(input_path, output_path, target_size=(224, 224)):
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            normalized_frame = resized_frame / 255.0
            output_frame = (normalized_frame * 255).astype(np.uint8)
            out.write(output_frame)

        cap.release()
        out.release()
        return output_path

    # Function to extract keyframes from the video and collect additional information
    def extract_keyframes(video_path, yolo_model, output_path="significant_keyframes_output.mp4"):
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            return None, [], {}, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        event_active = False
        motion_history = []
        saved_frames = 0
        peak_frame = None
        event_timestamps = []  # To store timestamps of detected events
        all_detected_objects = []  # To store detected objects across all frames

        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret:
                if event_active and peak_frame is not None:
                    out.write(peak_frame)
                    saved_frames += 1
                    timestamp = frame_count / fps
                    event_timestamps.append(timestamp)
                    # Run YOLO on the peak frame to detect objects
                    results = yolo_model(peak_frame)
                    for result in results:
                        for box in result.boxes:
                            label = result.names[int(box.cls)]
                            all_detected_objects.append(label)
                break

            frame_count += 1
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            diff_gray = cv2.absdiff(prev_gray, current_gray)
            _, thresh_gray = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            non_zero_count = cv2.countNonZero(thresh_gray)
            motion_history.append(non_zero_count)
            if len(motion_history) > 50:
                motion_history.pop(0)
            adaptive_threshold = max(100, np.mean(motion_history) * 2)

            motion_detected = non_zero_count > adaptive_threshold

            if motion_detected:
                if not event_active:
                    event_active = True
                peak_frame = current_frame
            elif event_active and frame_count > 30:
                event_active = False
                if peak_frame is not None:
                    out.write(peak_frame)
                    saved_frames += 1
                    timestamp = frame_count / fps
                    event_timestamps.append(timestamp)
                    # Run YOLO on the peak frame to detect objects
                    results = yolo_model(peak_frame)
                    for result in results:
                        for box in result.boxes:
                            label = result.names[int(box.cls)]
                            all_detected_objects.append(label)

            prev_gray = current_gray
            prev_frame = current_frame.copy()

        cap.release()
        out.release()

        # Count the occurrences of each object type
        object_counts = dict(Counter(all_detected_objects))

        return output_path, event_timestamps, object_counts, saved_frames

    # Function to detect anomalies in the video
    def anomaly_detection(video_path, processor, vivit_model):
        container = av.open(video_path)
        frames = [frame.to_image() for frame in container.decode(video=0)]

        required_frames = 32
        if len(frames) < required_frames:
            repeat_factor = (required_frames + len(frames) - 1) // len(frames)
            frames = (frames * repeat_factor)[:required_frames]
        elif len(frames) > required_frames:
            frames = frames[:required_frames]

        inputs = processor(frames, return_tensors="pt")
        with torch.no_grad():
            outputs = vivit_model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()

        return vivit_model.config.id2label[predicted_class]

    # Function to generate a story based on the keyframes video (in English by default)
    def generate_story(video_path, prediction, gemini_model):
        cap = cv2.VideoCapture(video_path)  # Use the keyframes video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return f"A video classified as '{prediction}' was processed, but no story could be generated due to an empty video."

        # Reduce the number of frames to describe to avoid quota issues
        step = max(1, total_frames // 3)  # Extract up to 3 frames
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0 and len(frames) < 3:  # Limit to 3 frames
                frame_path = f"frame_{frame_idx}.jpg"
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            frame_idx += 1
        cap.release()

        descriptions = {}
        for frame_path in frames:
            prompt = f"This frame is from a video classified as '{prediction}'. Describe the event in one sentence in English."
            try:
                with open(frame_path, "rb") as img_file:
                    image_data = Image.open(io.BytesIO(img_file.read()))
                response = gemini_model.generate_content([prompt, image_data])
                descriptions[frame_path] = response.text
            except ResourceExhausted as e:
                st.error("Gemini API quota exceeded. Please try again later or upgrade your API plan.")
                return f"A video classified as '{prediction}' was processed, but the story could not be generated due to API quota limits."
            except Exception as e:
                st.error(f"Error generating description for frame: {str(e)}")
                descriptions[frame_path] = f"Could not describe this frame due to an error."

        # Generate the summary
        try:
            summary_prompt = "Here are multiple descriptions of frames from a video in English:\n" + "\n".join(descriptions.values()) + "\nProvide a concise summary of the overall event in English."
            summary_response = gemini_model.generate_content(summary_prompt)
            return summary_response.text
        except ResourceExhausted as e:
            st.error("Gemini API quota exceeded while summarizing. Please try again later or upgrade your API plan.")
            return f"A video classified as '{prediction}' was processed, but the story could not be fully generated due to API quota limits."
        except Exception as e:
            st.error(f"Error generating story summary: {str(e)}")
            return f"A video classified as '{prediction}' was processed, but the story could not be summarized due to an error."

    # Function to translate text using MarianMT
    @st.cache_resource
    def translate_text(text, target_lang):
        # Model names for different languages (from English)
        model_mapping = {
            "es": "Helsinki-NLP/opus-mt-en-es",  # Spanish
            "fr": "Helsinki-NLP/opus-mt-en-fr",  # French
            "de": "Helsinki-NLP/opus-mt-en-de",  # German
            "ar": "Helsinki-NLP/opus-mt-en-ar",  # Arabic
        }

        if target_lang not in model_mapping:
            raise ValueError(f"Unsupported target language: {target_lang}")

        try:
            # Load tokenizer and model for the target language
            model_name = model_mapping[target_lang]
            tokenizer = MarianTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))
            translation_model = MarianMTModel.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))

            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Generate translation
            translated = translation_model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

            return translated_text
        except Exception as e:
            st.error(f"Error during translation to {target_lang}: {str(e)}")
            return None

    if __name__ == "__main__":
        main()

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

# About Us Page
elif page == "About Us":
    # Wrap the rest of the content in a centered container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    st.markdown("""
    <h1 style="text-align: center;">About Us</h1>
    We are the team behind **Echolens**, a project dedicated to transforming videos into meaningful stories using AI. Our team consists of passionate developers and researchers working together to push the boundaries of video analysis and storytelling.
    """, unsafe_allow_html=True)

    # Developer Profiles (Placeholder Data)
    st.subheader("Meet Our Team")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="developer-card">
            <img src="https://via.placeholder.com/100" alt="Developer 1">
            <h4>Ahmed Mohamed</h4>
            <p><b>AI Model Developer</b></p>
            <p>Ahmed is a senior computer science student specializing in machine learning.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="developer-card">
            <img src="https://via.placeholder.com/100" alt="Developer 2">
            <h4>Sara Ali</h4>
            <p><b>Frontend Developer</b></p>
            <p>Sara designed the user interface and ensured a seamless user experience.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="developer-card">
            <img src="https://via.placeholder.com/100" alt="Developer 3">
            <h4>Mohamed Khaled</h4>
            <p><b>Backend Developer</b></p>
            <p>Mohamed handled the integration of AI models and backend processing.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

# Contact Us Page
elif page == "Contact Us":
    # Wrap the rest of the content in a centered container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    st.markdown("""
    <h1 style="text-align: center;">Contact Us</h1>
    Have questions or feedback? We'd love to hear from you! Reach out to us through the following channels.
    """, unsafe_allow_html=True)

    st.subheader("Get in Touch")
    st.write("📧 **Email**: contact@echolens.com")
    st.write("🌐 **GitHub**: [Echolens Project](https://github.com/echolens)")
    st.write("🔗 **LinkedIn**: [Echolens Team](https://linkedin.com/company/echolens)")

    st.subheader("Send Us a Message")
    with st.form(key="contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button(label="Send Message")
        if submit_button:
            st.success(f"Thank you, {name}! We've received your message and will get back to you at {email} soon.")

    st.markdown('</div>', unsafe_allow_html=True)  # Close content-container

# Footer
st.markdown('<div class="content-container">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>© 2025 Echolens. All rights reserved.</p>
    <p><a href="https://github.com/echolens">GitHub</a> | <a href="mailto:contact@echolens.com">Email Us</a></p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # Close content-container
