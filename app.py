# Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import io
import base64

# =============================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# =============================================
st.set_page_config(
    page_title="Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the parent directory of the current file
ROOT = FILE.parent

# Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

# Image Config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'image2.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage2.jpg'

# Model Configurations
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolo11n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolo11n-seg.pt'

# =============================================
# UI 
# =============================================

# CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stSlider {
            margin-bottom: 20px;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        }
        .header-text {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        .subheader-text {
            font-size: 1.2rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .detection-button-container {
            margin-top: 2rem;
            text-align: center;
        }
        .paste-container {
            border: 2px dashed #ccc;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Header with title 
st.markdown('<p class="header-text">System for Object Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Real-time object detection and segmentation</p>', unsafe_allow_html=True)

# SideBar
with st.sidebar:
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-image: linear-gradient(#343a40,#2c3e50);
                color: white;
            }
            .sidebar .stRadio label {
                color: white;
            }
            .sidebar .stSlider label {
                color: white;
            }
            .sidebar .stFileUploader label {
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("Model Configuration")
    
    # Choose Model: Detection or Segmentation 
    model_type = st.radio(
        "Select Task Type:",
        ["Detection", "Segmentation",],
        index=0,
        help="Choose between object detection and instance segmentation"
    )
    
    st.markdown("---")
    
    # Select Confidence Value
    confidence_value = st.slider(
        "Confidence Threshold", 
        min_value=0, 
        max_value=100, 
        value=40,
        help="Adjust the minimum confidence level for detections"
    )
    confidence_value = float(confidence_value) / 100
    
    # Visual indicator for confidence level
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #e74c3c {confidence_value*100}%, #ecf0f1 {confidence_value*100}%);
                    height: 8px; 
                    border-radius: 4px;
                    margin-bottom: 20px;"></div>
    """, unsafe_allow_html=True)
    
    # Class Selection
    CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Image Source Selection
    st.subheader("Image Source")
    image_source = st.radio(
        "Select image source:",
        ["Upload an image", "Paste from clipboard"],
        index=0,
        help="Choose how to provide the image for detection"
    )
    
    # Image Upload/Paste Section
    st.subheader("Image Configuration")
    source_image = None
    
    if image_source == "Upload an image":
        source_image = st.file_uploader(
            "Upload an image",
            type=("jpg", "png", "jpeg", "bmp", "webp"),
            help="Upload an image for object detection",
            key="file_uploader"
        )
    else:
        paste_data = st.text_area("Paste image here (as base64 or URL)", "", height=100, key="paste_area")
        
        if paste_data:
            try:
                # Try to decode as base64
                if paste_data.startswith("data:image"):
                    # Extract the base64 data
                    header, encoded = paste_data.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    source_image = io.BytesIO(image_data)
                elif paste_data.startswith(("http://", "https://")):
                    # Handle URL
                    import requests
                    from io import BytesIO
                    response = requests.get(paste_data)
                    source_image = BytesIO(response.content)
                else:
                    # Try direct base64 decode
                    image_data = base64.b64decode(paste_data)
                    source_image = io.BytesIO(image_data)
            except:
                st.error("Could not process the pasted image. Please try another method.")

# Selecting Detection or Segmentation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)

# Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(e)

# Main Content Area
tab1, tab2 = st.tabs(["Image Detection", "Statistics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, 
                         caption="Default Image - Upload or paste your own image to see detection results", 
                         use_container_width=True)
            else:
                if isinstance(source_image, io.BytesIO):
                    # Reset pointer to start if it's a BytesIO object
                    source_image.seek(0)
                uploaded_image = Image.open(source_image)
                st.image(uploaded_image, 
                         caption="Input Image - Click 'Detect Objects' to process", 
                         use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)
    
    with col2:
        st.subheader("Detection Results")
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, 
                         caption="Sample Detection - Upload or paste your own image to see live results", 
                         use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Processing the Image")
            st.error(e)

# Detection Button
st.markdown("---")

if source_image is not None:
    # Create a centered container for the detection button
    st.markdown('<div class="detection-button-container">', unsafe_allow_html=True)
    if st.button(f"Detect Objects ({model_type})", key="detect_button"):
        with st.spinner(f"Processing {model_type}..."):
            try:
                if isinstance(source_image, io.BytesIO):
                    source_image.seek(0)
                uploaded_image = Image.open(source_image)
                
                # Run detection
                result = model.predict(uploaded_image, conf=confidence_value)
                boxes = result[0].boxes
                result_plotted = result[0].plot()[:, :, ::-1]
                
                # Display results in the right column
                with tab1:
                    with col2:
                        st.image(result_plotted, 
                                 caption=f"{model_type} Results (Confidence: {confidence_value*100}%)", 
                                 use_container_width=True)
                
                # Object Counting
                class_counts = {}
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = CLASSES[class_id]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                
                # Store results in session state for the statistics tab
                st.session_state.detection_results = {
                    "image": result_plotted,
                    "counts": class_counts,
                    "boxes": boxes,
                    "model_type": model_type
                }
            except Exception as e:
                st.error("Error during detection:")
                st.error(e)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if 'detection_results' not in st.session_state:
        st.warning("Run a detection first to see statistics")
    else:
        results = st.session_state.detection_results
        st.subheader(f"{results['model_type']} Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-title">Total Objects Detected</div>'
                        f'<div class="metric-value">{sum(results["counts"].values())}</div></div>', 
                        unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-title">Unique Classes</div>'
                        f'<div class="metric-value">{len(results["counts"])}</div></div>', 
                        unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-title">Model Confidence</div>'
                        f'<div class="metric-value">{confidence_value*100:.0f}%</div></div>', 
                        unsafe_allow_html=True)
        
        # Display Object Counts in a Table with sorting
        st.subheader("Object Counts")
        count_df = pd.DataFrame(list(results["counts"].items()), columns=["Class", "Count"])
        st.dataframe(
            count_df.sort_values("Count", ascending=False),
            use_container_width=True,
            height=min(400, 50 + 35 * len(count_df))
        )
        # Show detection details in an expander
        with st.expander("Detailed Detection Data"):
            st.write("Raw detection data from the model:")
            for i, box in enumerate(results["boxes"]):
                st.json({
                    "object_id": i,
                    "class": CLASSES[int(box.cls)],
                    "confidence": float(box.conf),
                    "coordinates": box.xywh.tolist()
                })