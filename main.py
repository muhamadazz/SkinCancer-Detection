import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn.functional as F  # Import untuk softmax

# Muat model PyTorch
model = torch.load('best.pt')
model.eval()

# Definisikan transformasi untuk input gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fungsi untuk memprediksi gambar
def predict_image(image):
    img = Image.open(image)
    img = transform(img).unsqueeze(0)  # Menambahkan batch dimension
    
    with torch.no_grad():
        output = model(img)
        
        # Menghitung probabilitas menggunakan softmax
        prob = F.softmax(output, dim=1)
        
        # Ambil prediksi kelas dan confidence
        confidence, predicted_class = torch.max(prob, 1)
        
        return predicted_class.item(), confidence.item()

# Antarmuka Streamlit
st.title("Computer Vision Model Deployment")
st.write("Upload an image to make a prediction")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diupload
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Prediksi
    predicted_class, confidence = predict_image(uploaded_file)

    # Tampilkan hasil prediksi dan confidence
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
