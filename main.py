import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Skin Disease Detection", layout="centered")

# Display project information
st.markdown("""
    ## Project Information
    This is a project for the final exam (UAS) of Group 2, TI22I class, **Computer Vision** course at **Nusaputra University**.
    The project aims to build an image-based skin disease detection system using Roboflow's computer vision model.
""")

# Title of the app
st.title("Skin Disease Detection")

# Upload image or enter URL
option = st.selectbox("Choose Input Method", ("Upload Image", "Enter Image URL"))

if option == "Upload Image":
    image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
else:
    image_url = st.text_input("Enter Image URL")

if option == "Upload Image" and image is not None:
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

elif option == "Enter Image URL" and image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Image from URL", use_column_width=True)
    except requests.exceptions.RequestException as e:
        st.error("There was an error fetching the image. Please check the URL.")

# Set up Roboflow model
roboflow = Roboflow(api_key="nZGnQ7JmEmth0YRBNrT6")  # Replace with your API key
model = roboflow.workspace("uascv-5uxrj").project("skin_disease-y20fz").version(1).model

# Inference
if option == "Upload Image" and image is not None or (option == "Enter Image URL" and image_url):
    with st.spinner("Running inference..."):
        if option == "Upload Image":
            img = np.array(img)
        else:
            img = np.array(Image.open(BytesIO(response.content)))

        result = model.predict(img, confidence=40, overlap=30).json()

    st.subheader("Inference Result")
    st.write(result)

    # Display additional information
    if result:
        st.write("Prediction details:")
        for prediction in result["predictions"]:
            st.write(f"Class: {prediction['class']}")
            st.write(f"Confidence: {prediction['confidence']*100:.2f}%")
            st.write("---")
