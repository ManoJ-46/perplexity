import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import time
import mediapipe as mp
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import random
import threading
import queue
from CNNModel import CNNModel

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load models with caching
@st.cache_resource
def load_models():
    model_alpha = CNNModel()
    model_alpha.load_state_dict(torch.load("trained.pth", map_location="cpu"))
    model_alpha.eval()
    return model_alpha

model_alpha = load_models()

# Alphabet classes mapping
alphabet_classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Mobile-responsive CSS
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.2rem;
        }
        
        .stSelectbox > div > div > div {
            font-size: 1.1rem;
        }
        
        .game-container {
            padding: 1rem;
        }
        
        .prediction-box {
            font-size: 1.5rem;
            padding: 1rem;
            text-align: center;
            background: #f0f2f6;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .video-container {
            width: 100%;
            max-width: 100%;
        }
        
        .alphabet-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
            padding: 10px;
        }
        
        .alphabet-item {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: bold;
        }
    }
    
    @media screen and (min-width: 769px) {
        .alphabet-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 15px;
            padding: 15px;
        }
        
        .alphabet-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            font-size: 1.4rem;
            font-weight: bold;
        }
    }
    
    /* General styling */
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stSidebar {
        background: #f8f9fa;
    }
    
    .prediction-display {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 15px;
        border: 3px solid #1f77b4;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1rem 0;
    }
    
    .game-stats {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Video processor class for real-time sign language recognition
class SignLanguageVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.prediction_queue = queue.Queue(maxsize=10)
        self.current_prediction = None
        self.prediction_confidence = 0.0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame for sign language detection
        processed_frame, prediction = self.process_frame(img)
        
        # Store prediction
        if prediction:
            try:
                self.prediction_queue.put_nowait(prediction)
            except queue.Full:
                pass  # Skip if queue is full
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
    
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)
        
        predicted_character = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract coordinates
                coordinates = []
                x_coords, y_coords, z_coords = [], [], []
                
                for i in range(len(hand_landmarks.landmark)):
                    lm = hand_landmarks.landmark[i]
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
                    z_coords.append(lm.z)
                
                # Create data dictionary
                data = {}
                for i, landmark in enumerate(mp_hands.HandLandmark):
                    lm = hand_landmarks.landmark[i]
                    data[f'{landmark.name}_x'] = lm.x - min(x_coords)
                    data[f'{landmark.name}_y'] = lm.y - min(y_coords)
                    data[f'{landmark.name}_z'] = lm.z - min(z_coords)
                
                coordinates.append(data)
                
                # Get bounding box
                h, w, _ = frame.shape
                x1 = int(min(x_coords) * w) - 10
                y1 = int(min(y_coords) * h) - 10
                x2 = int(max(x_coords) * w) + 10
                y2 = int(max(y_coords) * h) + 10
                
                # Make prediction
                coordinates_df = pd.DataFrame(coordinates)
                coords_reshaped = np.reshape(coordinates_df.values, (coordinates_df.shape[0], 63, 1))
                coords_tensor = torch.from_numpy(coords_reshaped).float()
                
                with torch.no_grad():
                    outputs = model_alpha(coords_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    pred_idx = predicted.cpu().numpy()[0]
                    conf_score = confidence.cpu().numpy()[0]
                    
                    if conf_score > 0.7:  # Confidence threshold
                        predicted_character = alphabet_classes[pred_idx]
                        self.current_prediction = predicted_character
                        self.prediction_confidence = conf_score
                
                # Draw bounding box and prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                if predicted_character:
                    # Draw prediction text
                    label = f"{predicted_character} ({conf_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return frame, predicted_character
    
    def get_latest_prediction(self):
        predictions = []
        while not self.prediction_queue.empty():
            try:
                predictions.append(self.prediction_queue.get_nowait())
            except queue.Empty:
                break
        return predictions[-1] if predictions else None

# Function to load sign images
def load_sign_image(letter):
    image_path = f"alphabets/{letter}.jpg"
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        # Create placeholder image
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()
        
        text_width = draw.textlength(letter, font=font)
        x = (200 - text_width) // 2
        y = 50
        draw.text((x, y), letter, fill='black', font=font)
        return img

# Game logic for guessing characters
def guess_the_character_game():
    st.markdown('<div class="section-header"><h2>🎮 Guess the Character Game</h2></div>', unsafe_allow_html=True)
    
    # Initialize game state
    if 'target_letter' not in st.session_state:
        st.session_state.target_letter = random.choice(list(alphabet_classes.values()))
        st.session_state.score = 0
        st.session_state.attempts = 0
        st.session_state.game_started = False
    
    # Game controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎯 New Challenge", key="new_game_btn"):
            st.session_state.target_letter = random.choice(list(alphabet_classes.values()))
            st.session_state.game_started = True
            st.rerun()
    
    if st.session_state.game_started:
        # Display target letter
        st.markdown(f"""
        <div class="prediction-display">
            Show the sign for: <strong>{st.session_state.target_letter}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reference image
        target_image = load_sign_image(st.session_state.target_letter)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(target_image, caption=f"Reference: {st.session_state.target_letter}", width=200)
        
        # Game stats
        st.markdown(f"""
        <div class="game-stats">
            <p><strong>Score:</strong> {st.session_state.score} | <strong>Attempts:</strong> {st.session_state.attempts}</p>
        </div>
        """, unsafe_allow_html=True)

# Main application
def main():
    st.markdown('<h1 class="main-title">🤟 Sign Language Recognition System</h1>', unsafe_allow_html=True)
    
    # Sidebar for mode selection
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        app_mode = st.selectbox(
            "Select Mode:",
            ["Live Detection", "English to Sign Language", "Guess the Character"],
            key="mode_selector"
        )
        
        st.markdown("---")
        st.markdown("### 📱 Mobile Optimized")
        st.markdown("This app is fully responsive and works great on mobile devices!")
        
        # Add some info about the app
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        if app_mode == "Live Detection":
            st.markdown("1. Click 'Start Camera' below\n2. Allow camera permissions\n3. Show sign language gestures\n4. See real-time predictions")
        elif app_mode == "English to Sign Language":
            st.markdown("1. Enter text below\n2. View corresponding sign language images\n3. Practice the gestures")
        else:
            st.markdown("1. Click 'New Challenge'\n2. Practice the displayed letter\n3. Show it to the camera\n4. Get instant feedback")
    
    # Main content area
    if app_mode == "Live Detection":
        live_detection_mode()
    elif app_mode == "English to Sign Language":
        text_to_sign_mode()
    else:
        game_mode()

def live_detection_mode():
    st.markdown('<div class="section-header"><h2>📹 Live Sign Language Detection</h2></div>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div style="background: #e8f4fd; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4>📋 Instructions:</h4>
        <ul>
            <li>Click the <strong>START</strong> button to begin</li>
            <li>Allow camera permissions when prompted</li>
            <li>Position your hand clearly in front of the camera</li>
            <li>The system will detect and predict your sign language gesture</li>
            <li>Green boxes indicate detected hands with predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # WebRTC configuration for better connectivity
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })
        
        # Video processor
        video_processor = SignLanguageVideoProcessor()
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="sign-language-detection",
            video_processor_factory=lambda: video_processor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"min": 10, "ideal": 15, "max": 30}
                },
                "audio": False
            },
            async_processing=True,
        )
    
    with col2:
        # Real-time prediction display
        st.markdown("### 🎯 Current Prediction")
        
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        # Update predictions in real-time
        if webrtc_ctx.video_processor:
            latest_prediction = webrtc_ctx.video_processor.get_latest_prediction()
            if latest_prediction:
                prediction_placeholder.markdown(f"""
                <div class="prediction-display">
                    {latest_prediction}
                </div>
                """, unsafe_allow_html=True)
                
                confidence_placeholder.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <strong>Confidence:</strong> {webrtc_ctx.video_processor.prediction_confidence:.2%}
                </div>
                """, unsafe_allow_html=True)
            else:
                prediction_placeholder.markdown("""
                <div class="prediction-display" style="background: #f8f9fa; color: #6c757d;">
                    No gesture detected
                </div>
                """, unsafe_allow_html=True)
        
        # Alphabet reference
        st.markdown("### 📚 Alphabet Reference")
        if st.checkbox("Show alphabet grid", key="show_alphabet"):
            display_alphabet_grid()

def text_to_sign_mode():
    st.markdown('<div class="section-header"><h2>✍️ English to Sign Language</h2></div>', unsafe_allow_html=True)
    
    # Text input
    user_text = st.text_input("Enter text to convert to sign language:", 
                             placeholder="Type here...", 
                             key="text_input")
    
    if user_text:
        # Filter only alphabetic characters
        letters = [char.upper() for char in user_text if char.isalpha()]
        
        if letters:
            st.markdown(f"### 🔤 Sign Language for: **{' '.join(letters)}**")
            
            # Display images in a responsive grid
            cols_per_row = 4 if len(letters) <= 8 else 6
            
            for i in range(0, len(letters), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, letter in enumerate(letters[i:i+cols_per_row]):
                    with cols[j]:
                        image = load_sign_image(letter)
                        st.image(img, caption=..., use_container_width=True)
        else:
            st.warning("Please enter at least one alphabetic character.")

def game_mode():
    guess_the_character_game()
    
    # Add WebRTC for game mode
    st.markdown("### 📹 Camera Feed")
    
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    video_processor = SignLanguageVideoProcessor()
    
    webrtc_ctx = webrtc_streamer(
        key="sign-language-game",
        video_processor_factory=lambda: video_processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"min": 320, "ideal": 640, "max": 1280},
                "height": {"min": 240, "ideal": 480, "max": 720}
            },
            "audio": False
        },
        async_processing=True,
    )
    
    # Game logic
    if webrtc_ctx.video_processor and st.session_state.game_started:
        latest_prediction = webrtc_ctx.video_processor.get_latest_prediction()
        if latest_prediction:
            st.markdown(f"**Your gesture:** {latest_prediction}")
            
            if latest_prediction == st.session_state.target_letter:
                st.success("🎉 Correct! Great job!")
                st.session_state.score += 1
                st.session_state.attempts += 1
                st.session_state.target_letter = random.choice(list(alphabet_classes.values()))
                time.sleep(1)
                st.rerun()
            else:
                st.info(f"Try again! Target: {st.session_state.target_letter}, Your gesture: {latest_prediction}")

def display_alphabet_grid():
    """Display alphabet reference grid"""
    st.markdown("""
    <div class="alphabet-grid">
    """, unsafe_allow_html=True)
    
    # Create responsive grid
    cols = st.columns(6)
    letters = list(alphabet_classes.values())
    
    for i, letter in enumerate(letters):
        with cols[i % 6]:
            if i % 6 == 0 and i > 0:
                cols = st.columns(6)
            
            try:
                image = load_sign_image(letter)
                st.image(image, caption=letter, use_column_width=True)
            except:
                st.markdown(f"""
                <div class="alphabet-item">
                    {letter}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
