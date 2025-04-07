import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# ======= Load Trained Model =======
def load_model(model_path, num_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ======= Image Preprocessing =======
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ======= Label Simplification =======
def simplify_label(label):
    parts = label.split("___")
    plant = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else ""
    return f"{plant} - Healthy" if "healthy" in disease.lower() else f"{plant} - {disease}"

# ======= Original Labels =======
original_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

class_labels = [simplify_label(label) for label in original_labels]

# ======= Prediction Function =======
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = class_labels[predicted.item()]
        confidence_score = confidence.item()
        return predicted_label, confidence_score, probabilities

# ======= Streamlit UI =======
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

st.title("🌱 Plant Disease Detection")
st.write("Upload an image of a leaf 🌿 to detect possible diseases.")

uploaded_file = st.file_uploader("📤 Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Uploaded Image', use_container_width=True)

    model_path = "plant_disease_mobilenetv2.pth"
    if os.path.exists(model_path):
        model = load_model(model_path, num_classes=len(class_labels))
        input_tensor = preprocess_image(image)
        prediction, confidence, probabilities = predict(model, input_tensor)

        st.success(f"🧠 Prediction: **{prediction}**")
        st.info(f"🔍 Confidence: **{confidence * 100:.2f}%**")

        # Display top 3 predictions
        st.subheader("🔝 Top 3 Predictions")
        top3_probabilities, top3_indices = torch.topk(probabilities, 3)
        for i, (label_idx, prob) in enumerate(zip(top3_indices[0], top3_probabilities[0])):
            st.write(f"{i+1}. {class_labels[label_idx.item()]}: **{prob.item() * 100:.2f}%**")

    else:
        st.error("🚫 Model file not found. Please upload the trained model (.pth).")
