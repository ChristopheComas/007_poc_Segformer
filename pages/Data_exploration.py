import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import cv2
import plotly.graph_objects as go
import glob 
import os 

st.set_page_config(
    page_title="Exploratory data analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
) 

st.title("Image Segmentation Dashboard")
st.write(
    """
    This dashboard provides tools for exploring image segmentation models, 
    visualizing masks, and comparing model performance across categories.
    """
)
# Define the color mappings for segmentation classes
color_class = {
    'void': [0, 0, 255],         # Blue
    'road': [255, 255, 0],       # Yellow
    'construction': [255, 0, 0], # Red
    'object': [0, 255, 255],     # Cyan
    'vegetation': [0, 255, 0],   # Green
    'sky': [255, 0, 255],        # Magenta
    'human': [255, 100, 100],    # Pale pink
    'vehicle': [255, 125, 0]     # Orange
}

# Numeric mappings for segmentation classes
num_class = {cls: idx for idx, cls in enumerate(color_class.keys())}
example_images = glob.glob(os.path.join(".", '*left*'))


selected_img = st.selectbox("Select a file", example_images)

selected_mask = "_".join(selected_img.split('_')[:3])+'_gtFine_labelIds_cat.png'


# Define layout
main_layout = st.columns((1, 0.2, 1), gap='small')

### Data Transformations Section
with main_layout[0]:
    st.header("Data Transformations")
    eda_layout = st.columns((1, 2), gap='medium')

    with eda_layout[0]:
        # Transformation selection
        transformation = st.selectbox("Select Transformation", ["Original", "Blurring"])

    with eda_layout[1]:
        # Apply transformations and display image
        img = cv2.imread(selected_img)
        if transformation == "Blurring":
            img_blur = cv2.GaussianBlur(img, (25, 25), 0)
            st.image(img_blur, caption="Blurred Image", use_container_width=True)
        else:
            st.image(img, caption="Original Image", use_container_width=True)

### Interactive Mask Viewer Section
    st.header("Interactive Mask Viewer")
    mask_viewer_layout = st.columns((1, 2), gap='medium')

    with mask_viewer_layout[0]:

        # Checkbox for toggling classes
        st.subheader("Class Selection")
        selected_classes = [
            cls for cls in color_class if st.checkbox(cls, value=True)
        ]

    with mask_viewer_layout[1]:
        # Load images and masks
        original_image = cv2.imread(selected_img)
        mask = cv2.imread(selected_mask, cv2.IMREAD_GRAYSCALE)
        # Convert the grayscale mask to RGB
        mask_rgb = np.zeros((*original_image.shape,), dtype=np.uint8)
        for i, (cls, color) in enumerate(color_class.items()):
            mask_rgb[mask == i] = color

        if selected_classes:
            combined_mask = np.zeros_like(mask_rgb)
            for cls in selected_classes:
                combined_mask[np.where(mask == num_class[cls])] = color_class[cls]

            # Blend original image with mask
            blended_image = cv2.addWeighted(original_image, 0.7, combined_mask, 0.3, 0)
            st.image(blended_image, caption="Filtered Segmentation", use_container_width=True)
        else:
            st.write("No classes selected.")

### Sidebar Information
st.sidebar.header("References")
st.sidebar.write("For more information, visit:")
st.sidebar.write("[The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)")
st.sidebar.write("[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)")
st.sidebar.write("[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)")
st.sidebar.header("Accessibility Features")
st.sidebar.write(
    "This dashboard is designed to comply with WCAG standards for visualizations, including colorblind-friendly graphs."
)
