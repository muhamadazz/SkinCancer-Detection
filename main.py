import streamlit as st
from roboflow import Roboflow
import requests
from PIL import Image
from io import BytesIO

# Inisialisasi Roboflow API dengan API key Anda
rf = Roboflow(api_key="nZGnQ7JmEmth0YRBNrT6")

# Ganti dengan model Anda
project = rf.workspace("uascv-5uxrj").project("skin_disease-y20fz")
model = project.version(1).model  # Pastikan versi model sesuai

# Membuat tampilan Streamlit
st.title("Skin Disease Detection")
st.write("Upload an image to classify")

# Upload gambar oleh pengguna
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Menampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert gambar ke format yang diterima oleh model Roboflow
    img_array = np.array(image)

    # Menggunakan model untuk inferensi
    prediction = model.predict(img_array, confidence=40, overlap=30).json()

    # Menampilkan hasil inferensi
    st.write("Prediction Results:")
    st.write(prediction)

    # Menampilkan label dan confidence
    for prediction_result in prediction["predictions"]:
        st.write(f"Class: {prediction_result['class']} - Confidence: {prediction_result['confidence']*100:.2f}%")
