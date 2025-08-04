import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.applications.efficientnet import preprocess_input

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(""" 
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .custom-link {
        color: #1f77b4;
        font-weight: bold;
        text-decoration: none;
        margin-left: 0.5em;
    }
    .footer {
        text-align: center;
        color: gray;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_INFO = {
    'Glioma': 'Tumor ganas yang berasal dari sel glial di otak atau tulang belakang.',
    'Meningioma': 'Tumor jinak yang tumbuh di meninges (selaput pelindung otak dan sumsum tulang belakang).',
    'No Tumor': 'Kondisi otak normal tanpa adanya pertumbuhan tumor.',
    'Pituitary': 'Tumor yang tumbuh di kelenjar pituitari (hipofisis), bagian kecil di dasar otak yang mengatur hormon..'
}

@st.cache_resource
def load_efficientnet_model():
    try:
        model_path = "model/finetune_model_tune.keras"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading EfficientNet B2 model: {str(e)}")
        return None

@st.cache_resource
def load_efficientvit_model():
    if not TIMM_AVAILABLE:
        st.error("Error: 'timm' library is required for EfficientViT model.")
        return None
        
    try:
        model_path = "model/finetune_transformer_best.pth"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = timm.create_model('efficientvit_b1', pretrained=False, num_classes=4)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state_dict into the model
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Error loading EfficientViT B1 model: {str(e)}")
        return None

def preprocess_image_efficientnet(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((260, 260))
    img_array = np.array(image, dtype=np.float32)
    
    if len(img_array.shape) == 2:
        # Grayscale image --> convert to RGB (by repeating channels)
        img_array = np.stack([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 1:
            # Single channel --> repeat to make 3 channels
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.shape[2] == 4:
            # RGBA --> take only RGB channels
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (H, W, 3)")
    else:
        raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (H, W) or (H, W, C)")
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    return img_array

def preprocess_image_efficientvit(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image)
    
    if img_tensor.shape[0] != 3:
        # If grayscale --> repeat to make 3 channels
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        else:
            raise ValueError(f"Unexpected number of channels: {img_tensor.shape[0]}. Expected 3.")
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def predict_efficientnet(model, image):
    if model is None:
        return None, None
    
    processed_image = preprocess_image_efficientnet(image)
    predictions = model.predict(processed_image, verbose=0)
    
    # Get probabilities
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities

def predict_efficientvit(model, image):
    if model is None:
        return None, None
    
    try:
        processed_image = preprocess_image_efficientvit(image)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processed_image = processed_image.to(device)
        model = model.to(device)
        
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    except Exception as e:
        st.error(f"Error EfficientViT prediction: {str(e)}")
        return None, None

def create_confidence_chart(probabilities, class_names):
    fig = go.Figure(go.Bar(
        x=probabilities * 100,
        y=class_names,
        orientation='h',
        marker=dict(
            color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
            line=dict(color='rgba(0,0,0,0.8)', width=1)
        )
    ))
    
    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="Tumor Type",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_accuracy_comparison():
    models = ['EfficientNet B2', 'EfficientViT B1']
    accuracies = [89.65, 98.05]
    
    fig = go.Figure(go.Bar(
        x=models,
        y=accuracies,
        marker=dict(
            color=['#ff6b6b', '#4ecdc4'],
            line=dict(color='rgba(0,0,0,0.8)', width=1)
        ),
        text=[f'{acc}%' for acc in accuracies],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Model Accuracy Comparison",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def load_sample_images():
    sample_images = {}
    samples_folder = "samples"
    
    if os.path.exists(samples_folder):
        for class_name in CLASS_NAMES:
            image_path = os.path.join(samples_folder, f"{class_name}.jpg")
            if os.path.exists(image_path):
                sample_images[class_name] = image_path
    
    return sample_images

def resize_image_to_square(image, size=300):
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    width, height = image.size
    max_dim = max(width, height)
    
    # Create a new square image with white background
    square_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    x = (max_dim - width) // 2
    y = (max_dim - height) // 2
    
    # Paste original image to the square background
    square_image.paste(image, (x, y))
    
    return square_image.resize((size, size))

# === Sidebar ===
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")

    st.markdown(
        """
        <div style="margin-bottom:0.2em">
            <span>Check Out the other app:</span>
            <br>
            <a href="#" target="_blank" class="custom-link">
                Brain Tumor Object Detection
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    st.subheader("🎯 Model Selection")
    selected_model = st.radio(
        "Choose a model for prediction:",
        ["EfficientNet B2", "EfficientViT B1"]
    )
    
    st.markdown("---")
    
    st.subheader("📋 Model Information")
    if selected_model == "EfficientNet B2":
        st.info("""
        - **Architecture:** EfficientNet B2
        - **Parameters:** 7.7 million
        - **Accuracy:** 89.65% - test set
        """)
    else:
        st.info("""
        - **Architecture:** EfficientViT B1
        - **Parameters:** 7.5 million
        - **Accuracy:** 98.05% - test set
        """)
    
    st.markdown("---")
    
    st.subheader("✅ Model Performance")
    accuracy_fig = create_accuracy_comparison()
    st.plotly_chart(accuracy_fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📖 About Tumor Types")
    selected_info = st.selectbox("Select tumor type for info:", CLASS_NAMES)
    st.write(CLASS_INFO[selected_info])

# === Main ===
st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Classification System</h1>', unsafe_allow_html=True)

st.markdown("""
This application uses deep learning models to classify brain MRI scans into four categories: 
**Glioma**, **Meningioma**, **No Tumor**, and **Pituitary** tumor.
""")

# === Load Models ===
with st.spinner("Loading models..."):
    efficientnet_model = load_efficientnet_model()
    efficientvit_model = load_efficientvit_model()

# === Sample Images ===
st.subheader("📷 Sample Images")
st.write("Click on any sample image below to use it for prediction:")

sample_images = load_sample_images()
if sample_images:
    cols = st.columns(4)
    selected_sample = None
    
    for idx, (class_name, image_path) in enumerate(sample_images.items()):
        with cols[idx]:
            if os.path.exists(image_path):
                sample_img = Image.open(image_path)
                sample_img_resized = resize_image_to_square(sample_img, 150)
                
                if st.button(f"Use {class_name}", key=f"sample_{class_name}"):
                    selected_sample = sample_img
                    st.session_state['selected_sample'] = selected_sample
                    st.session_state['sample_name'] = class_name
                
                st.image(sample_img_resized, caption=class_name, use_container_width=True)
else:
    st.warning("Sample images not found. Please ensure the 'samples' folder exists with the required images.")

st.markdown("---")

st.subheader("📤 Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose a brain MRI scan image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a brain MRI scan image in JPG, JPEG, or PNG format"
)

image_to_predict = None
image_source = ""

if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file).convert('RGB')
    image_source = "Uploaded Image"
elif 'selected_sample' in st.session_state:
    image_to_predict = st.session_state['selected_sample']
    image_source = f"Sample: {st.session_state['sample_name']}"

# === Prediction ===
if image_to_predict is not None:
    st.markdown("---")
    st.subheader("🔍 Prediction Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"**Source:** {image_source}")
        display_image = resize_image_to_square(image_to_predict, 300)
        st.image(display_image, caption="Image for Prediction", use_container_width=True)
    
    with col2:
        with st.spinner(f"Making prediction with {selected_model}..."):
            try:
                if selected_model == "EfficientNet B2":
                    if efficientnet_model is not None:
                        predicted_class, probabilities = predict_efficientnet(efficientnet_model, image_to_predict)
                    else:
                        st.error("EfficientNet B2 model is not loaded properly.")
                        predicted_class, probabilities = None, None
                elif selected_model == "EfficientViT B1":
                    if efficientvit_model is not None:
                        predicted_class, probabilities = predict_efficientvit(efficientvit_model, image_to_predict)
                    else:
                        st.error("EfficientViT B1 model is not loaded properly.")
                        predicted_class, probabilities = None, None
                else:
                    st.error(f"Unknown model selected: {selected_model}")
                    predicted_class, probabilities = None, None
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                predicted_class, probabilities = None, None
        
        if predicted_class is not None and probabilities is not None:
            predicted_label = CLASS_NAMES[predicted_class]
            confidence = probabilities[predicted_class] * 100
            
            st.success(f"**Prediction:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # Display confidence chart
            confidence_fig = create_confidence_chart(probabilities, CLASS_NAMES)
            st.plotly_chart(confidence_fig, use_container_width=True)
            
            with st.expander("ℹ️ About this condition"):
                st.write(CLASS_INFO[predicted_label])
        else:
            st.error("Prediction failed. Please try again.")

# === Footer ===
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Richard Dean Tanjaya | Github: <a href="https://github.com/RichardDeanTan/Brain-Tumor-Image-Classification" target="_blank">@RichardDeanTan</a></p>
</div>
""", unsafe_allow_html=True)