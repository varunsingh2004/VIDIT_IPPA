import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.util import random_noise

st.set_page_config(page_title="Image Processor", layout="wide")

def main():
    st.title("📷 Advanced Image Processing Tool")
    st.write("Upload an image and apply various processing techniques")

    # Sidebar for upload and options
    with st.sidebar:
        st.header("Upload & Settings")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display original image in sidebar
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Processing options
            st.header("Processing Options")
            sharpen = st.checkbox("Sharpen Image")
            smooth = st.checkbox("Smooth Image")
            interpolate = st.checkbox("Apply Interpolation")
            make_blue = st.checkbox("Make Image Blue")
            add_noise = st.checkbox("Add Noise")
            rotate = st.checkbox("Rotate Image")
            edge_detect = st.checkbox("Canny Edge Detection")
            
            # Parameters
            if smooth:
                smoothing_kernel = st.slider("Smoothing Kernel Size", 3, 21, 5, step=2)
            
            if sharpen:
                sharpen_amount = st.slider("Sharpen Amount", 0.1, 3.0, 1.0, step=0.1)
            
            if add_noise:
                noise_amount = st.slider("Noise Amount", 0.01, 0.2, 0.05)
                noise_type = st.selectbox("Noise Type", ["gaussian", "salt", "pepper", "s&p"])
            
            if rotate:
                rotation_angle = st.slider("Rotation Angle (degrees)", -180, 180, 45)
            
            if edge_detect:
                threshold1 = st.slider("Canny Threshold 1", 0, 255, 100)
                threshold2 = st.slider("Canny Threshold 2", 0, 255, 200)
    
    # Main content area
    if uploaded_file is not None:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Apply processing based on selected options
        processed_img = img_array.copy()
        
        if smooth:
            processed_img = cv2.GaussianBlur(processed_img, (smoothing_kernel, smoothing_kernel), 0)
        
        if sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5*sharpen_amount, -1], [0, -1, 0]])
            processed_img = cv2.filter2D(processed_img, -1, kernel)
        
        if interpolate:
            height, width = processed_img.shape[:2]
            processed_img = cv2.resize(processed_img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        if make_blue:
            # Increase blue channel intensity
            processed_img[:, :, 2] = np.clip(processed_img[:, :, 2] * 1.5, 0, 255)
        
        if add_noise:
            processed_img = (random_noise(processed_img, mode=noise_type, amount=noise_amount) * 255).astype(np.uint8)
        
        if rotate:
            (h, w) = processed_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            processed_img = cv2.warpAffine(processed_img, M, (w, h))
        
        if edge_detect:
            if len(processed_img.shape) == 3:  # Convert to grayscale if color
                gray_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = processed_img
            processed_img = cv2.Canny(gray_img, threshold1, threshold2)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(processed_img, caption="Processed Image", use_column_width=True)
            
            # Download button for processed image
            buf = BytesIO()
            processed_pil = Image.fromarray(processed_img)
            processed_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )
        
        # Show processing steps
        st.subheader("Processing Pipeline")
        st.write("Applied the following operations in order:")
        operations = []
        if smooth: operations.append(f"Smoothing (kernel size: {smoothing_kernel})")
        if sharpen: operations.append(f"Sharpening (amount: {sharpen_amount})")
        if interpolate: operations.append("Interpolation (bicubic 2x)")
        if make_blue: operations.append("Blue tint")
        if add_noise: operations.append(f"Noise ({noise_type}, amount: {noise_amount})")
        if rotate: operations.append(f"Rotation ({rotation_angle}°)")
        if edge_detect: operations.append(f"Canny Edge Detection (thresholds: {threshold1}/{threshold2})")
        
        if operations:
            for op in operations:
                st.write(f"- {op}")
        else:
            st.write("No operations selected - showing original image")

if __name__ == "_main_":
    main()